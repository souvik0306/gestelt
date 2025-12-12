#!/usr/bin/env python3
"""
Real-time IMU Inference ROS Node

This ROS node:
1. Subscribes to /imu_data topic (from rosbag or live data)
2. Processes each IMU sample through ONNX model using RealtimeIMUInference
3. Publishes corrected IMU data to /corrected_imu topic
4. Logs inference results to file

Usage:
    rosrun imu_listener_pkg fake_imu.py
    
    # With rosbag:
    rosbag play your_imu_data.bag
"""

import rospy
import numpy as np
import os
import rospkg
import sys
from sensor_msgs.msg import Imu
from realtime_imu_inference import RealtimeIMUInference


class CorrectedIMUPublisher:
    """Publishes corrected IMU messages to a ROS topic."""
    
    def __init__(self, topic_name="/corrected_imu"):
        self.pub = rospy.Publisher(topic_name, Imu, queue_size=100)
        rospy.loginfo(f"[Publisher] Publishing corrected IMU to: {topic_name}")

    def publish(self, corrected_acc, corrected_gyro, original_header):
        """
        Publish corrected IMU data.
        
        Args:
            corrected_acc: Corrected acceleration [ax, ay, az]
            corrected_gyro: Corrected gyroscope [gx, gy, gz]  
            original_header: Original message header (preserves timestamp and frame)
        """
        imu_msg = Imu()

        # Keep original timestamp and frame
        imu_msg.header = original_header

        # Corrected acceleration
        imu_msg.linear_acceleration.x = float(corrected_acc[0])
        imu_msg.linear_acceleration.y = float(corrected_acc[1])
        imu_msg.linear_acceleration.z = float(corrected_acc[2])

        # Corrected gyroscope
        imu_msg.angular_velocity.x = float(corrected_gyro[0])
        imu_msg.angular_velocity.y = float(corrected_gyro[1])
        imu_msg.angular_velocity.z = float(corrected_gyro[2])

        # Keep original orientation (if available)
        imu_msg.orientation = rospy.get_param('~preserve_orientation', True) and hasattr(self, 'orientation')
        
        self.pub.publish(imu_msg)


class RealtimeIMUNode:
    """Main ROS node for real-time IMU inference."""
    
    def __init__(self):
        # Get package paths
        rp = rospkg.RosPack()
        self.pkg_path = rp.get_path("imu_listener_pkg")
        
        # Model and log paths
        model_path = os.path.join(self.pkg_path, "models", "airimu_euroc.onnx")
        log_path = os.path.join(self.pkg_path, "results", "realtime_imu_inference.txt")
        
        # Initialize inference module
        self.inference = RealtimeIMUInference(model_path, log_path)
        
        # Initialize publisher
        self.corrected_imu_pub = CorrectedIMUPublisher("/corrected_imu")
        
        # Statistics
        self.sample_count = 0
        self.start_time = None
        
        print("=" * 70)
        print("[INIT] Realtime IMU Inference Node Starting")
        print(f"[INFO] Python version: {sys.version.split()[0]}")
        print(f"[INFO] Model path: {model_path}")
        print(f"[INFO] Log path: {log_path}")
        print("=" * 70)
    
    def check_inference_ready(self) -> bool:
        """Check if inference module is ready."""
        if not hasattr(self.inference, 'onnx_model') or self.inference.onnx_model is None:
            rospy.logwarn("[WARNING] ONNX model not loaded. Running in pass-through mode.")
            return False
        return True
    
    def imu_callback(self, msg: Imu):
        """
        Process incoming IMU message.
        
        Args:
            msg: ROS IMU message from /imu_data topic
        """
        try:
            # Initialize timing on first sample
            if self.start_time is None:
                self.start_time = rospy.Time.now().to_sec()
                rospy.loginfo("[READY] Receiving IMU data, starting inference...")
            
            # Extract IMU data
            input_acc = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ], dtype=np.float32)
            
            input_gyro = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ], dtype=np.float32)
            
            # Get timestamp for logging
            timestamp_us = int(msg.header.stamp.to_sec() * 1e6)
            
            # Process through inference module
            corrected_acc, corrected_gyro = self.inference.process(
                input_acc, 
                input_gyro,
                timestamp_us=timestamp_us,
                sequence=self.sample_count
            )
            
            # Publish corrected IMU data
            self.corrected_imu_pub.publish(corrected_acc, corrected_gyro, msg.header)
            
            # Update statistics
            self.sample_count += 1
            
            # Log progress periodically
            if self.sample_count % 100 == 0:
                elapsed = rospy.Time.now().to_sec() - self.start_time
                rate = self.sample_count / elapsed if elapsed > 0 else 0
                rospy.loginfo(f"[STATS] Processed {self.sample_count} samples, Rate: {rate:.1f} Hz")
            
            # Log first few samples with more detail
            elif self.sample_count <= 5:
                rospy.loginfo(f"[Sample {self.sample_count}] Input acc: [{input_acc[0]:.3f}, {input_acc[1]:.3f}, {input_acc[2]:.3f}]")
                rospy.loginfo(f"[Sample {self.sample_count}] Output acc: [{corrected_acc[0]:.3f}, {corrected_acc[1]:.3f}, {corrected_acc[2]:.3f}]")
                
        except Exception as e:
            rospy.logerr(f"[ERROR] Failed to process IMU sample: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def print_final_statistics(self):
        """Print final statistics on shutdown."""
        if self.start_time and self.sample_count > 0:
            elapsed = rospy.Time.now().to_sec() - self.start_time
            avg_rate = self.sample_count / elapsed if elapsed > 0 else 0
            
            rospy.loginfo("=" * 70)
            rospy.loginfo("[FINAL STATS]")
            rospy.loginfo(f"Total samples processed: {self.sample_count}")
            rospy.loginfo(f"Total time: {elapsed:.2f} seconds")
            rospy.loginfo(f"Average rate: {avg_rate:.2f} Hz")
            rospy.loginfo("=" * 70)
    
    def shutdown_hook(self):
        """Clean up resources on shutdown."""
        rospy.loginfo("[SHUTDOWN] Cleaning up...")
        
        # Print final statistics
        self.print_final_statistics()
        
        # Close inference module (handles log file)
        if hasattr(self, 'inference'):
            self.inference.close()
            
        rospy.loginfo("[SHUTDOWN] Cleanup complete")
    
    def start(self):
        """Start the ROS node."""
        # Check if inference is ready
        inference_ready = self.check_inference_ready()
        if not inference_ready:
            rospy.logwarn("[WARNING] Continuing without model - data will pass through unchanged")
        
        # Set up subscriber
        input_topic = rospy.get_param('~input_topic', '/imu_data')
        queue_size = rospy.get_param('~queue_size', 1000)
        
        rospy.loginfo("=" * 70)
        rospy.loginfo(f"[INIT] ROS time: {rospy.Time.now().to_sec():.2f}")
        rospy.loginfo(f"[STATUS] Subscribing to: {input_topic}")
        rospy.loginfo(f"[STATUS] Queue size: {queue_size}")
        rospy.loginfo(f"[STATUS] Publishing to: /corrected_imu")
        rospy.loginfo(f"[STATUS] Waiting for IMU data...")
        rospy.loginfo("=" * 70)
        
        # Subscribe to IMU data
        rospy.Subscriber(input_topic, Imu, self.imu_callback, queue_size=queue_size)
        
        # Register shutdown hook
        rospy.on_shutdown(self.shutdown_hook)
        
        # Spin
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("[INFO] Keyboard interrupt received")
        except Exception as e:
            rospy.logerr(f"[ERROR] Node error: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())


def main():
    """Main entry point."""
    try:
        # Initialize ROS node first
        rospy.init_node("realtime_imu_inference_node", anonymous=True)
        
        # Create and start node
        node = RealtimeIMUNode()
        node.start()
        
    except Exception as e:
        print(f"[FATAL] Failed to start node: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
