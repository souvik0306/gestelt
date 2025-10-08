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

# --- configuration ---
WINDOW_SIZE = 100  # Number of IMU samples per inference window
STEP_SIZE   = 2    # Number of samples the sliding window advances between inferences

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
        self.buffer = IMUBuffer(WINDOW_SIZE, STEP_SIZE)
        self.results = []
        self.correction_counter = 0
        self.onnx_model = None
        self.corrected_imu_pub = CorrectedIMUPublisher("/corrected_imu")

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
            rospy.loginfo(f"[READY] Model loaded at ROS time: {rospy.Time.now().to_sec():.2f}")
        except Exception as e:
            rospy.logerr(f"Failed to load ONNX model: {e}")
            rospy.signal_shutdown("Fatal error: Model loading failed.")

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
            time, acc, gyro = self.buffer.get_window()

            if time.shape[0] != self.buffer.window_size:
                raise ValueError(
                    f"Expected time buffer of length {self.buffer.window_size}, got {time.shape[0]}"
                )

            if acc.shape != gyro.shape:
                raise ValueError(
                    f"Acceleration shape {acc.shape} does not match gyro shape {gyro.shape}"
                )

            if acc.shape[0] != time.shape[0]:
                raise ValueError(
                    "Time and IMU data length mismatch: "
                    f"{time.shape[0]} timestamps vs {acc.shape[0]} samples"
                )

            if acc.shape[1] != 3:
                raise ValueError(f"Expected IMU vectors of length 3, got {acc.shape[1]}")

            dt = np.diff(time, prepend=time[0])[..., None]
            dt = dt[1:, :]  # drop the artificial first element
            acc = acc[:-1]
            gyro = gyro[:-1]

            acc_b = acc[None, ...]
            gyro_b = gyro[None, ...]

            corr_acc, corr_gyro = self.onnx_model.run(None, {"acc": acc_b, "gyro": gyro_b})

            if corr_acc.shape != corr_gyro.shape:
                raise ValueError(
                    f"Correction shapes do not match: acc {corr_acc.shape}, gyro {corr_gyro.shape}"
                )

            if corr_acc.ndim != 3:
                raise ValueError(f"Expected 3D correction tensors, got {corr_acc.ndim} dimensions")

            if corr_acc.shape[0] != acc_b.shape[0]:
                raise ValueError(
                    f"Batch dimension mismatch: input {acc_b.shape[0]} vs correction {corr_acc.shape[0]}"
                )

            if corr_acc.shape[2] != acc_b.shape[2]:
                raise ValueError(
                    f"Feature dimension mismatch: input {acc_b.shape[2]} vs correction {corr_acc.shape[2]}"
                )

            start_index = acc_b.shape[1] - corr_acc.shape[1]
            if start_index < 0:
                raise ValueError(
                    f"Correction sequence longer than input sequence: "
                    f"input length {acc_b.shape[1]}, correction length {corr_acc.shape[1]}"
                )

            corrected_acc = acc_b[:, start_index:, :] + corr_acc
            corrected_gyro = gyro_b[:, start_index:, :] + corr_gyro
            dt_trim = dt[start_index:, :]

            self.correction_counter += 1
            rospy.loginfo(f"[Correction #{self.correction_counter}]")
            rospy.loginfo(f"Corrected accel: {corrected_acc[0, -1]}")
            rospy.loginfo(f"Corrected gyro:  {corrected_gyro[0, -1]}")
            rospy.loginfo("-----------------------")

            # Publish the last corrected IMU message
            self.corrected_imu_pub.publish(
                corrected_acc[0, -1], corrected_gyro[0, -1]
            )

            self.results.append({
                "correction_acc":  corr_acc[0],
                "correction_gyro": corr_gyro[0],
                "corrected_acc":   corrected_acc[0],
                "corrected_gyro":  corrected_gyro[0],
                "dt":              dt_trim,
            })

            self.save_results()
        except Exception as e:
            rospy.logerr(f"Inference error: {e}")

    def imu_callback(self, msg: Imu):
        self.buffer.add(msg)
        if self.buffer.ready():
            self.run_inference()
            self.buffer.slide_window()

    def start(self):
        rospy.init_node("imu_inference_node")
        rospy.loginfo(f"[INIT] Node started at ROS time: {rospy.Time.now().to_sec():.2f}")
        rospy.loginfo(f"[INFO] Python version: {sys.version}")

        if not self.check_files():
            rospy.signal_shutdown("Required files missing.")
            return

        self.load_model()
        rospy.Subscriber("/imu_data", Imu, self.imu_callback, queue_size=1000)
        # rospy.Subscriber("/snappy_imu", Imu, self.imu_callback, queue_size=1000)
        # rospy.Subscriber("mavros/imu/data", Imu, self.imu_callback, queue_size=1000)
        rospy.spin()

if __name__ == "__main__":
    node = IMUInferenceNode()
    node.start()
