#!/usr/bin/env python3
import rospy
import numpy as np
import onnxruntime as ort
import pickle
import os
import rospkg
from collections import deque
from sensor_msgs.msg import Imu
import sys
import time
import socket
import struct

# --- defaults (overridable via ROS params) ---
WINDOW_SIZE = 500  # Number of IMU samples before first inference
STEP_SIZE = 350    # Number of samples the sliding window advances between inferences
MODEL_INTERVAL = 9 # Model interval (outputs start from index 9)


class IMUBuffer:
    """Buffer for storing IMU data and managing inference windows."""

    def __init__(self, window_size: int, step_size: int):
        self.window_size = window_size
        self.step_size = step_size
        self.time_buf = deque()
        self.acc_buf = deque()
        self.gyro_buf = deque()

    def add(self, msg: Imu):
        self.time_buf.append(msg.header.stamp.to_sec())
        self.acc_buf.append(
            (
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            )
        )
        self.gyro_buf.append(
            (
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            )
        )

    def ready(self) -> bool:
        return len(self.time_buf) >= self.window_size

    def get_window(self):
        if not self.ready():
            raise ValueError("Attempted to get window before buffer was full")

        time_arr = np.asarray(self.time_buf, dtype=np.float32)
        acc = np.asarray(self.acc_buf, dtype=np.float32)
        gyro = np.asarray(self.gyro_buf, dtype=np.float32)

        # Only the most recent window_size samples are used for inference.
        time_arr = time_arr[-self.window_size:]
        acc = acc[-self.window_size:]
        gyro = gyro[-self.window_size:]
        return time_arr, acc, gyro

    def slide_window(self):
        remove_count = min(self.step_size, len(self.time_buf))
        for _ in range(remove_count):
            self.time_buf.popleft()
            self.acc_buf.popleft()
            self.gyro_buf.popleft()


class CorrectedIMUPublisher:
    """Publishes corrected IMU messages to UDP in PX4 FRD format."""

    def __init__(self, udp_host="127.0.0.1", udp_port=14560):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target = (udp_host, udp_port)
        self.accel_device_id = 0xA14ACC01
        self.gyro_device_id = 0xA14A7701
        self.last_stamp = None
        rospy.loginfo(f"[UDP Publisher] Initialized to {udp_host}:{udp_port}")

    @staticmethod
    def _enu_to_frd(vec):
        return [vec[1], vec[0], -vec[2]]

    def publish(self, corrected_acc, corrected_gyro, stamp=None):
        """Publish corrected IMU data over UDP in vehicle_imu_ai format."""
        if stamp is None:
            stamp = rospy.Time.now().to_sec()

        if self.last_stamp is None:
            self.last_stamp = stamp
            dt = 0.005  # Assume 200 Hz for the first sample
        else:
            dt = stamp - self.last_stamp
            if dt <= 0.0 or dt > 0.04:  # Clamp to a reasonable dt range
                dt = 0.005
            self.last_stamp = stamp

        # Convert ENU to FRD
        acc_frd = self._enu_to_frd(corrected_acc)
        gyro_frd = self._enu_to_frd(corrected_gyro)

        # Integrate to deltas
        delta_angle = [gyro_frd[i] * dt for i in range(3)]
        delta_velocity = [acc_frd[i] * dt for i in range(3)]

        # Pack struct matching AIBridgePacketV1 (see imu_ai_bridge.cpp)
        now_us = int(rospy.Time.now().to_sec() * 1e6)
        timestamp_us = now_us
        timestamp_sample = int(stamp * 1e6)
        delta_angle_dt = int(round(dt * 1e6))
        delta_velocity_dt = int(round(dt * 1e6))

        # Saturate to uint16
        delta_angle_dt = max(0, min(0xFFFF, delta_angle_dt))
        delta_velocity_dt = max(0, min(0xFFFF, delta_velocity_dt))

        delta_angle_clipping = 0
        delta_velocity_clipping = 0
        accel_calibration_count = 0
        gyro_calibration_count = 0

        fmt = '<QQII3f3fHHBBBB'
        payload = struct.pack(
            fmt,
            timestamp_us,
            timestamp_sample,
            self.accel_device_id,
            self.gyro_device_id,
            delta_angle[0], delta_angle[1], delta_angle[2],
            delta_velocity[0], delta_velocity[1], delta_velocity[2],
            delta_angle_dt,
            delta_velocity_dt,
            delta_angle_clipping,
            delta_velocity_clipping,
            accel_calibration_count,
            gyro_calibration_count,
        )

        try:
            self.sock.sendto(payload, self.target)
        except Exception as e:
            rospy.logwarn(f"[UDP Publisher] Send failed: {e}")


class IMUInferenceNode:
    """Main node for IMU inference and publishing corrected data."""

    def __init__(self):
        rp = rospkg.RosPack()
        self.pkg_path = rp.get_path("imu_listener_pkg")
        self.onnx_path = os.path.join(self.pkg_path, "models", "airimu_euroc.onnx")
        self.pickle_path = os.path.join(self.pkg_path, "results", "timeit_sim_new_net_output.pickle")
        self.log_path = os.path.join(self.pkg_path, "results", "inference_log.txt")

        # Params
        self.imu_topic = rospy.get_param("~imu_topic", "/imu_data")
        window_size = int(rospy.get_param("~window_size", WINDOW_SIZE))
        step_size = int(rospy.get_param("~step_size", STEP_SIZE))
        self.model_interval = int(rospy.get_param("~model_interval", MODEL_INTERVAL))
        udp_host = rospy.get_param("~udp_host", "127.0.0.1")
        udp_port = int(rospy.get_param("~udp_port", 14560))
        self.send_raw_until_ready = bool(rospy.get_param("~send_raw_until_ready", True))

        # Runtime
        self.buffer = IMUBuffer(window_size, step_size)
        self.corrected_imu_pub = CorrectedIMUPublisher(udp_host, udp_port)
        self.results = []
        self.correction_counter = 0
        self.inference_counter = 0
        self.output_sample_counter = 0
        self.onnx_model = None

        # Timing for rate calculation
        self.last_inference_time = None
        self.inference_times = deque(maxlen=50)
        self.start_time = None
        self.log_file = None
        self._last_progress_log_n = -1

    def check_files(self):
        if not os.path.isfile(self.onnx_path):
            rospy.logerr(f"ONNX model file not found: {self.onnx_path}")
            return False
        return True

    def load_model(self):
        """Load the ONNX model for inference."""
        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = os.cpu_count()
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            self.onnx_model = ort.InferenceSession(
                self.onnx_path, sess_options=session_options, providers=["CPUExecutionProvider"]
            )
            rospy.loginfo(
                f"[READY] Model loaded at ROS time: {rospy.Time.now().to_sec():.2f}; "
                f"model interval: {self.model_interval}, min sequence length: {self.model_interval + 1}"
            )
        except Exception as e:
            rospy.logerr(f"Failed to load ONNX model: {e}")
            rospy.signal_shutdown("Fatal error: Model loading failed.")

    def init_log_file(self):
        """Initialize the log file for predictions."""
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self.log_file = open(self.log_path, 'w')
            self.log_file.write("=" * 100 + "\n")
            self.log_file.write("IMU INFERENCE LOG\n")
            self.log_file.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write(
                f"Configuration: WINDOW_SIZE={self.buffer.window_size}, STEP_SIZE={self.buffer.step_size}, "
                f"MODEL_INTERVAL={self.model_interval}\n"
            )
            self.log_file.write("=" * 100 + "\n")
            self.log_file.write(f"{'Counter':<10} {'Timestamp':<20} {'ROS Time':<15} {'Acc X':<12} {'Acc Y':<12} {'Acc Z':<12} {'Gyro X':<12} {'Gyro Y':<12} {'Gyro Z':<12}\n")
            self.log_file.write("-" * 100 + "\n")
            self.log_file.flush()
            rospy.loginfo(f"[LOG] Initialized log file: {self.log_path}")
        except Exception as e:
            rospy.logerr(f"Failed to initialize log file: {e}")
            self.log_file = None

    def log_prediction(self, counter, corrected_acc, corrected_gyro, ros_time=None):
        """Log a single prediction to the text file."""
        if self.log_file is None:
            return

        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            ros_time_str = f"{ros_time:.6f}" if ros_time else "N/A"

            log_line = (
                f"{counter:<10} {timestamp:<20} {ros_time_str:<15} "
                f"{corrected_acc[0]:<12.6f} {corrected_acc[1]:<12.6f} {corrected_acc[2]:<12.6f} "
                f"{corrected_gyro[0]:<12.6f} {corrected_gyro[1]:<12.6f} {corrected_gyro[2]:<12.6f}\n"
            )
            self.log_file.write(log_line)
            self.log_file.flush()  # Ensure it's written immediately
        except Exception as e:
            rospy.logerr(f"Failed to log prediction: {e}")

    def save_results(self):
        """Save inference results to a pickle file."""
        try:
            os.makedirs(os.path.dirname(self.pickle_path), exist_ok=True)
            with open(self.pickle_path, "wb") as f:
                pickle.dump(self.results, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            rospy.logerr(f"Failed to save results: {e}")

    def run_inference(self):
        try:
            # Track timing
            current_time = time.time()
            if self.last_inference_time is not None:
                time_diff = current_time - self.last_inference_time
                self.inference_times.append(time_diff)
            self.last_inference_time = current_time

            time_buf, acc, gyro = self.buffer.get_window()

            # Validation
            if time_buf.shape[0] != self.buffer.window_size:
                raise ValueError(
                    f"Expected time buffer of length {self.buffer.window_size}, got {time_buf.shape[0]}"
                )

            if acc.shape != gyro.shape:
                raise ValueError(
                    f"Acceleration shape {acc.shape} does not match gyro shape {gyro.shape}"
                )

            if acc.shape[0] != time_buf.shape[0]:
                raise ValueError(
                    "Time and IMU data length mismatch: "
                    f"{time_buf.shape[0]} timestamps vs {acc.shape[0]} samples"
                )

            if acc.shape[1] != 3:
                raise ValueError(f"Expected IMU vectors of length 3, got {acc.shape[1]}")

            # Prepare input for ONNX model: shape (1, window_size, 3)
            acc_input = acc.astype(np.float32, copy=False)[None, ...]
            gyro_input = gyro.astype(np.float32, copy=False)[None, ...]

            if acc_input.shape != (1, self.buffer.window_size, 3) or gyro_input.shape != (1, self.buffer.window_size, 3):
                raise ValueError(
                    f"Input shapes incorrect: acc {acc_input.shape}, gyro {gyro_input.shape}, "
                    f"expected (1, {self.buffer.window_size}, 3)"
                )

            # Run ONNX inference
            # The model returns corrections starting from position MODEL_INTERVAL (9) onwards
            corr_acc, corr_gyro = self.onnx_model.run(None, {"acc": acc_input, "gyro": gyro_input})

            # Validate output shapes
            if corr_acc.shape != corr_gyro.shape:
                raise ValueError(
                    f"Correction shapes do not match: acc {corr_acc.shape}, gyro {corr_gyro.shape}"
                )

            if corr_acc.ndim != 3:
                raise ValueError(f"Expected 3D correction tensors, got {corr_acc.ndim} dimensions")

            if corr_acc.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got {corr_acc.shape[0]}")

            if corr_acc.shape[2] != 3:
                raise ValueError(f"Expected 3D features, got {corr_acc.shape[2]}")

            # Expected output length: window_size - model_interval
            expected_output_len = self.buffer.window_size - self.model_interval
            if corr_acc.shape[1] != expected_output_len:
                raise ValueError(
                    f"Correction sequence length {corr_acc.shape[1]} != expected {expected_output_len}"
                )

            # Apply corrections to the corresponding portion of the input
            # Input: acc[0:WINDOW_SIZE], Output corrections: corr[0:WINDOW_SIZE-MODEL_INTERVAL]
            # Corrections apply to input[model_interval:window_size]
            corrected_acc = acc[self.model_interval:, :] + corr_acc[0]
            corrected_gyro = gyro[self.model_interval:, :] + corr_gyro[0]

            # Validate corrected shapes
            if corrected_acc.shape != (expected_output_len, 3):
                raise ValueError(
                    f"Corrected acc shape {corrected_acc.shape} != expected ({expected_output_len}, 3)"
                )
            if corrected_gyro.shape != (expected_output_len, 3):
                raise ValueError(
                    f"Corrected gyro shape {corrected_gyro.shape} != expected ({expected_output_len}, 3)"
                )

            self.correction_counter += 1
            self.inference_counter += 1

            # Calculate and log inference rate
            if len(self.inference_times) > 0:
                avg_time = np.mean(self.inference_times)
                inference_rate = 1.0 / avg_time if avg_time > 0 else 0.0

                # Calculate overall output rate
                if self.start_time is None:
                    self.start_time = current_time
                elapsed_time = current_time - self.start_time
                overall_output_rate = self.output_sample_counter / elapsed_time if elapsed_time > 0 else 0.0

                # Log every 5 inferences to reduce spam
                if self.inference_counter % 5 == 0:
                    rospy.loginfo(
                        f"[Inference #{self.inference_counter}] "
                        f"Inference Rate: {inference_rate:.2f} Hz (period: {avg_time*1000:.2f} ms) | "
                        f"Output: {corrected_acc.shape[0]} samples | "
                        f"Overall Output Rate: {overall_output_rate:.2f} Hz | "
                        f"Total Published: {self.output_sample_counter}"
                    )
            else:
                rospy.loginfo(f"[Inference #{self.inference_counter}] (first inference, {corrected_acc.shape[0]} samples)")

            # Publish ALL corrected IMU samples from this inference window to UDP
            # Extract corresponding timestamps for corrected samples (accounting for model_interval offset)
            # Use time_buf from buffer.get_window() which is already aligned with acc/gyro data
            corrected_timestamps = time_buf[self.model_interval:self.model_interval + expected_output_len]

            for i in range(corrected_acc.shape[0]):
                self.corrected_imu_pub.publish(
                    corrected_acc[i],
                    corrected_gyro[i],
                    corrected_timestamps[i]
                )
                self.output_sample_counter += 1

                # Log each prediction to file
                self.log_prediction(
                    self.output_sample_counter,
                    corrected_acc[i],
                    corrected_gyro[i],
                    corrected_timestamps[i]
                )

            # Compute dt for the corrected portion
            dt = np.diff(time_buf[self.model_interval - 1:])  # dt aligns with corrected samples

            if dt.shape[0] != expected_output_len:
                raise ValueError(
                    f"dt length {dt.shape[0]} does not match corrected length {expected_output_len}"
                )

            # Store results
            self.results.append({
                "correction_acc":  corr_acc[0],
                "correction_gyro": corr_gyro[0],
                "corrected_acc":   corrected_acc,
                "corrected_gyro":  corrected_gyro,
                "dt":              dt,
            })

            self.save_results()

        except Exception as e:
            rospy.logerr(f"Inference error: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

    def imu_callback(self, msg: Imu):
        """Add IMU sample to buffer, optionally stream raw deltas, and run inference when ready."""
        # If model isn't loaded, always stream raw deltas and skip buffering/inference
        if self.onnx_model is None:
            stamp = msg.header.stamp.to_sec() if msg.header.stamp and msg.header.stamp.to_sec() > 0 else rospy.Time.now().to_sec()
            acc = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
            gyro = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            self.corrected_imu_pub.publish(acc, gyro, stamp)
            return

        # Otherwise, optionally stream raw IMU deltas so the bridge receives data immediately
        if self.send_raw_until_ready and not self.buffer.ready():
            stamp = msg.header.stamp.to_sec() if msg.header.stamp and msg.header.stamp.to_sec() > 0 else rospy.Time.now().to_sec()
            acc = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
            gyro = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            self.corrected_imu_pub.publish(acc, gyro, stamp)

            # Buffer-fill progress (log every ~50 samples)
            n = len(self.buffer.time_buf)
            if n // 50 != self._last_progress_log_n:
                self._last_progress_log_n = n // 50
                rospy.loginfo(f"[BUFFER] {n}/{self.buffer.window_size} samples collected ({100.0 * n / self.buffer.window_size:.1f}%)")

        # Always add to buffer after optional raw publish
        self.buffer.add(msg)

        # Run inference when buffer is full
        if self.buffer.ready():
            self.run_inference()
            self.buffer.slide_window()

    def start(self):
        rospy.init_node("imu_inference_node")
        rospy.loginfo("=" * 70)
        rospy.loginfo(f"[INIT] IMU Inference Node Starting")
        rospy.loginfo(f"[INIT] ROS time: {rospy.Time.now().to_sec():.2f}")
        rospy.loginfo(f"[INFO] Python version: {sys.version.split()[0]}")
        rospy.loginfo(f"[CONFIG] IMU topic: {self.imu_topic}")
        rospy.loginfo(f"[CONFIG] Window Size: {self.buffer.window_size} samples")
        rospy.loginfo(f"[CONFIG] Step Size: {self.buffer.step_size} samples")
        rospy.loginfo(f"[CONFIG] Model Interval: {self.model_interval}")
        rospy.loginfo(f"[CONFIG] Expected output per inference: {self.buffer.window_size - self.model_interval} samples")
        rospy.loginfo("=" * 70)

        if not self.check_files():
            rospy.signal_shutdown("Required files missing.")
            return

        self.load_model()
        self.init_log_file()  # Initialize logging

        rospy.loginfo(f"[STATUS] Subscribing to {self.imu_topic} ...")
        rospy.loginfo(f"[STATUS] Will publish corrected IMU via UDP to PX4")
        rospy.loginfo(f"[STATUS] Waiting for {self.buffer.window_size} samples to fill buffer...")
        rospy.Subscriber(self.imu_topic, Imu, self.imu_callback, queue_size=1000)

        # Register shutdown hook to close log file
        rospy.on_shutdown(self.shutdown_hook)
        rospy.spin()

    def shutdown_hook(self):
        """Clean up resources on shutdown."""
        if self.log_file is not None:
            try:
                self.log_file.write("\n" + "=" * 100 + "\n")
                self.log_file.write(f"Session ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.log_file.write(f"Total predictions logged: {self.output_sample_counter}\n")
                self.log_file.write(f"Total inferences: {self.inference_counter}\n")
                self.log_file.write("=" * 100 + "\n")
                self.log_file.close()
                rospy.loginfo(f"[LOG] Closed log file. Total predictions: {self.output_sample_counter}")
            except Exception as e:
                rospy.logerr(f"Error closing log file: {e}")

if __name__ == "__main__":
    node = IMUInferenceNode()
    node.start()
