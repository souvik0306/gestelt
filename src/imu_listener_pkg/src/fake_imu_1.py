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

# --- configuration ---
WINDOW_SIZE = 500  # Number of IMU samples collected before triggering first inference
STEP_SIZE = 350     # Number of samples the sliding window advances between inferences
                   # At 200 Hz: STEP_SIZE=10 → ~20 Hz inference rate
                   # At 200 Hz: STEP_SIZE=20 → ~40-50 Hz inference rate
                   # At 200 Hz: STEP_SIZE=50 → ~100+ Hz inference rate
# The CodeNet model has an interval of 9, requiring at least 10 input samples.
# It outputs corrections starting from position 9 onwards.
MODEL_INTERVAL = 9
MIN_SEQUENCE_LENGTH = MODEL_INTERVAL + 1  # Minimum samples needed (10)

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
        """Check if enough samples are collected for inference."""

        return len(self.time_buf) >= self.window_size

    def get_window(self):
        """Get the current window of buffered IMU data."""

        if not self.ready():
            raise ValueError("Attempted to get window before buffer was full")

        time = np.asarray(self.time_buf, dtype=np.float32)
        acc = np.asarray(self.acc_buf, dtype=np.float32)
        gyro = np.asarray(self.gyro_buf, dtype=np.float32)

        # Only the most recent window_size samples are used for inference.
        time = time[-self.window_size :]
        acc = acc[-self.window_size :]
        gyro = gyro[-self.window_size :]
        return time, acc, gyro

    def slide_window(self):
        """Advance the window by the configured step size."""

        remove_count = min(self.step_size, len(self.time_buf))
        for _ in range(remove_count):
            self.time_buf.popleft()
            self.acc_buf.popleft()
            self.gyro_buf.popleft()

class CorrectedIMUPublisher:
    """Publishes corrected IMU messages to a ROS topic."""
    def __init__(self, topic_name="/corrected_imu"):
        self.pub = rospy.Publisher(topic_name, Imu, queue_size=100)

    def publish(self, corrected_acc, corrected_gyro):
        imu_msg = Imu()

        # timestamp
        imu_msg.header.stamp = rospy.Time.now()
        imu_msg.header.frame_id = "imu_link"

        # Corrected acceleration
        imu_msg.linear_acceleration.x = float(corrected_acc[0])
        imu_msg.linear_acceleration.y = float(corrected_acc[1])
        imu_msg.linear_acceleration.z = float(corrected_acc[2])

        # Corrected gyroscope
        imu_msg.angular_velocity.x = float(corrected_gyro[0])
        imu_msg.angular_velocity.y = float(corrected_gyro[1])
        imu_msg.angular_velocity.z = float(corrected_gyro[2])

        self.pub.publish(imu_msg)

class IMUInferenceNode:
    """Main node for IMU inference and publishing corrected data."""
    def __init__(self):
        rp = rospkg.RosPack()
        self.pkg_path = rp.get_path("imu_listener_pkg")
        self.onnx_path   = os.path.join(self.pkg_path, "models", "airimu_euroc.onnx")
        self.pickle_path = os.path.join(self.pkg_path, "results", "timeit_sim_new_net_output.pickle")
        self.log_path = os.path.join(self.pkg_path, "results", "inference_log.txt")
        self.buffer = IMUBuffer(WINDOW_SIZE, STEP_SIZE)
        self.results = []
        self.correction_counter = 0
        self.inference_counter = 0
        self.output_sample_counter = 0  # Track total output samples published
        self.onnx_model = None
        self.corrected_imu_pub = CorrectedIMUPublisher("/corrected_imu")
        self.model_interval = MODEL_INTERVAL
        
        # Timing for rate calculation
        self.last_inference_time = None
        self.inference_times = deque(maxlen=50)  # Store last 50 inference times
        self.start_time = None  # Track overall start time
        self.log_file = None  # File handle for logging

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
                f"model interval: {self.model_interval}, min sequence length: {MIN_SEQUENCE_LENGTH}"
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
            self.log_file.write(f"Configuration: WINDOW_SIZE={WINDOW_SIZE}, STEP_SIZE={STEP_SIZE}, MODEL_INTERVAL={MODEL_INTERVAL}\n")
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
            if time_buf.shape[0] != WINDOW_SIZE:
                raise ValueError(
                    f"Expected time buffer of length {WINDOW_SIZE}, got {time_buf.shape[0]}"
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

            # Prepare input for ONNX model: shape (1, WINDOW_SIZE, 3)
            acc_input = acc.astype(np.float32, copy=False)[None, ...]
            gyro_input = gyro.astype(np.float32, copy=False)[None, ...]

            if acc_input.shape != (1, WINDOW_SIZE, 3) or gyro_input.shape != (1, WINDOW_SIZE, 3):
                raise ValueError(
                    f"Input shapes incorrect: acc {acc_input.shape}, gyro {gyro_input.shape}, "
                    f"expected (1, {WINDOW_SIZE}, 3)"
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

            # Expected output length: WINDOW_SIZE - MODEL_INTERVAL
            expected_output_len = WINDOW_SIZE - MODEL_INTERVAL
            if corr_acc.shape[1] != expected_output_len:
                raise ValueError(
                    f"Correction sequence length {corr_acc.shape[1]} != expected {expected_output_len}"
                )

            # Apply corrections to the corresponding portion of the input
            # Input: acc[0:WINDOW_SIZE], Output corrections: corr[0:WINDOW_SIZE-MODEL_INTERVAL]
            # Corrections apply to input[MODEL_INTERVAL:WINDOW_SIZE]
            corrected_acc = acc[MODEL_INTERVAL:, :] + corr_acc[0]
            corrected_gyro = gyro[MODEL_INTERVAL:, :] + corr_gyro[0]

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
                
                # Log every 10 inferences to reduce spam
                if self.inference_counter % 10 == 0:
                    rospy.loginfo(
                        f"[Inference #{self.inference_counter}] "
                        f"Inference Rate: {inference_rate:.2f} Hz (period: {avg_time*1000:.2f} ms) | "
                        f"Output: {corrected_acc.shape[0]} samples | "
                        f"Overall Output Rate: {overall_output_rate:.2f} Hz | "
                        f"Total Published: {self.output_sample_counter}"
                    )
            else:
                rospy.loginfo(f"[Inference #{self.inference_counter}] (first inference, {corrected_acc.shape[0]} samples)")

            # Publish ALL corrected IMU samples from this inference window
            # This increases output rate without increasing inference rate
            for i in range(corrected_acc.shape[0]):
                self.corrected_imu_pub.publish(
                    corrected_acc[i], corrected_gyro[i]
                )
                self.output_sample_counter += 1
                
                # Log each prediction to file
                self.log_prediction(
                    self.output_sample_counter,
                    corrected_acc[i],
                    corrected_gyro[i],
                    rospy.Time.now().to_sec()
                )

            # Compute dt for the corrected portion
            dt = np.diff(time_buf[MODEL_INTERVAL-1:])  # dt aligns with corrected samples
            
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
        """Add IMU sample to buffer and run inference when ready."""
        self.buffer.add(msg)
        
        # Run inference when buffer is full
        if self.buffer.ready():
            self.run_inference()
            # Slide window by STEP_SIZE for next inference
            self.buffer.slide_window()

    def start(self):
        rospy.init_node("imu_inference_node")
        rospy.loginfo("=" * 70)
        rospy.loginfo(f"[INIT] IMU Inference Node Starting")
        rospy.loginfo(f"[INIT] ROS time: {rospy.Time.now().to_sec():.2f}")
        rospy.loginfo(f"[INFO] Python version: {sys.version.split()[0]}")
        rospy.loginfo(f"[CONFIG] Window Size: {WINDOW_SIZE} samples")
        rospy.loginfo(f"[CONFIG] Step Size: {STEP_SIZE} samples")
        rospy.loginfo(f"[CONFIG] Model Interval: {MODEL_INTERVAL}")
        rospy.loginfo(f"[CONFIG] Expected output per inference: {WINDOW_SIZE - MODEL_INTERVAL} samples")
        rospy.loginfo("=" * 70)

        if not self.check_files():
            rospy.signal_shutdown("Required files missing.")
            return

        self.load_model()
        self.init_log_file()  # Initialize logging
        
        rospy.loginfo(f"[STATUS] Subscribing to /imu_data topic...")
        rospy.loginfo(f"[STATUS] Will publish corrected IMU to /corrected_imu topic")
        rospy.loginfo(f"[STATUS] Waiting for {WINDOW_SIZE} samples to fill buffer...")
        rospy.Subscriber("/imu_data", Imu, self.imu_callback, queue_size=1000)
        # rospy.Subscriber("/snappy_imu", Imu, self.imu_callback, queue_size=1000)
        # rospy.Subscriber("/mavros/imu/data", Imu, self.imu_callback, queue_size=1000)
        
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