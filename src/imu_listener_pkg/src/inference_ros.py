#!/usr/bin/env python3
"""
Real-time IMU correction using ONNX AirIMU model for ROS.

Pipeline:
1. Subscribe to IMU messages from MAVROS
2. Buffer incoming samples (SEQLEN real samples)
3. Auto-generate 9 padding frames (stationary with gravity)
4. Run ONNX inference: [1, SEQLEN+9, 3] -> [1, SEQLEN, 3]
5. Publish corrected IMU data
6. Log results to pickle file

Requirements:
- ONNX model exported with fixed GRU initialization 
- Model expects pre-padded input: [batch, N+9, 3]
- Outputs corrections: [batch, N, 3]
"""
import rospy
import numpy as np
import onnxruntime as ort
import pickle
import os
import rospkg
import sys
import time
from collections import deque
from sensor_msgs.msg import Imu

# --- Configuration ---
SEQLEN = 1          # Number of real IMU samples per window
INTERVAL = 9        # Padding frames (must match model training)
GRAVITY = 9.81007   # Gravity magnitude (m/s^2)


class PaddingGenerator:
    """
    Generates synthetic padding frames for ONNX inference.
    
    Padding simulates stationary IMU at the start of each window:
    - acc: gravity vector in body frame (assumes upright orientation)
    - gyro: zeros (stationary)
    """
    
    def __init__(self, interval=9, gravity=9.81007):
        self.interval = interval
        self.gravity = np.array([0.0, 0.0, gravity], dtype=np.float32)
    
    def generate(self, acc_samples, gyro_samples):
        """
        Generate padding and concatenate with real samples.
        
        Args:
            acc_samples: [N, 3] real accelerometer data
            gyro_samples: [N, 3] real gyroscope data
        
        Returns:
            acc_padded: [N+interval, 3] padded accelerometer
            gyro_padded: [N+interval, 3] padded gyroscope
        """
        N = acc_samples.shape[0]
        
        # Padding: assume identity rotation (upright start)
        # This means gravity = [0, 0, 9.81] in body frame
        pad_acc = np.tile(self.gravity, (self.interval, 1))
        pad_gyro = np.zeros((self.interval, 3), dtype=np.float32)
        
        # Concatenate: [padding] + [real data]
        acc_padded = np.vstack([pad_acc, acc_samples])
        gyro_padded = np.vstack([pad_gyro, gyro_samples])
        
        return acc_padded, gyro_padded


class IMUBuffer:
    """
    Circular buffer for incoming IMU messages.
    
    Accumulates SEQLEN samples before triggering inference.
    """
    
    def __init__(self, seqlen):
        self.seqlen = seqlen
        self.time_buf = deque(maxlen=seqlen)
        self.acc_buf = deque(maxlen=seqlen)
        self.gyro_buf = deque(maxlen=seqlen)
    
    def add(self, msg: Imu):
        """Add new IMU sample to buffer."""
        self.time_buf.append(msg.header.stamp.to_sec())
        self.acc_buf.append([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        self.gyro_buf.append([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
    
    def ready(self):
        """Check if buffer has enough samples for inference."""
        return len(self.time_buf) >= self.seqlen
    
    def get_arrays(self):
        """
        Get buffered data as numpy arrays.
        
        Returns:
            time: [N] timestamps
            acc: [N, 3] accelerometer
            gyro: [N, 3] gyroscope
        """
        time = np.array(self.time_buf, dtype=np.float64)
        acc = np.array(self.acc_buf, dtype=np.float32)
        gyro = np.array(self.gyro_buf, dtype=np.float32)
        return time, acc, gyro
    
    def clear(self):
        """Clear buffer after inference."""
        self.time_buf.clear()
        self.acc_buf.clear()
        self.gyro_buf.clear()


class IMUInferenceNode:
    """
    ROS node for real-time IMU correction using ONNX model.
    """
    
    def __init__(self):
        # Paths
        rp = rospkg.RosPack()
        self.pkg_path = rp.get_path("imu_listener_pkg")
        self.onnx_path = os.path.join(self.pkg_path, "models", "airimu_cpu_fp32.onnx")
        self.results_path = os.path.join(self.pkg_path, "results", "corrected_imu.npz")
        
        # Components
        self.buffer = IMUBuffer(SEQLEN)
        self.padding_gen = PaddingGenerator(INTERVAL, GRAVITY)
        self.onnx_model = None
        
        # Pre-allocate numpy arrays for results (estimate max size)
        self.max_samples = 100000  # Adjust based on expected rosbag length
        self.raw_acc_buffer = np.zeros((self.max_samples, 3), dtype=np.float32)
        self.raw_gyro_buffer = np.zeros((self.max_samples, 3), dtype=np.float32)
        self.corrected_acc_buffer = np.zeros((self.max_samples, 3), dtype=np.float32)
        self.corrected_gyro_buffer = np.zeros((self.max_samples, 3), dtype=np.float32)
        
        # Counters
        self.inference_count = 0
        self.total_samples_processed = 0
        
        # Statistics (use fixed-size circular buffer)
        self.inference_times = np.zeros(1000, dtype=np.float32)  # Keep last 1000
        self.time_idx = 0
    
    def check_files(self):
        """Verify ONNX model exists."""
        if not os.path.isfile(self.onnx_path):
            rospy.logerr(f"ONNX model not found: {self.onnx_path}")
            rospy.logerr("Please export model using: python export_onnx_new.py")
            return False
        return True
    
    def load_model(self):
        """Load ONNX model with optimized settings."""
        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = os.cpu_count()
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            
            self.onnx_model = ort.InferenceSession(
                self.onnx_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"]
            )
            
            # Print model info
            rospy.loginfo("="*70)
            rospy.loginfo("ONNX Model Loaded Successfully")
            rospy.loginfo("="*70)
            rospy.loginfo(f"Model path: {self.onnx_path}")
            
            for inp in self.onnx_model.get_inputs():
                rospy.loginfo(f"  Input: {inp.name}, shape: {inp.shape}, dtype: {inp.type}")
            for out in self.onnx_model.get_outputs():
                rospy.loginfo(f"  Output: {out.name}, shape: {out.shape}, dtype: {out.type}")
            
            rospy.loginfo(f"Configuration:")
            rospy.loginfo(f"  SEQLEN: {SEQLEN} (real samples per window)")
            rospy.loginfo(f"  INTERVAL: {INTERVAL} (padding frames)")
            rospy.loginfo(f"  Input shape: [1, {SEQLEN + INTERVAL}, 3]")
            rospy.loginfo(f"  Output shape: [1, {SEQLEN}, 3]")
            rospy.loginfo("="*70)
            
        except Exception as e:
            rospy.logerr(f"Failed to load ONNX model: {e}")
            rospy.signal_shutdown("Fatal error: Model loading failed")
    
    def save_results(self):
        """Save results to numpy compressed format (NPZ) - much faster than pickle."""
        try:
            os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
            
            # Trim arrays to actual size
            n = self.inference_count
            
            # Save as compressed numpy format (10-100x faster than pickle)
            np.savez_compressed(
                self.results_path,
                # Metadata
                seqlen=SEQLEN,
                interval=INTERVAL,
                total_inferences=self.inference_count,
                total_samples=self.total_samples_processed,
                avg_inference_time_ms=np.mean(self.inference_times[:min(n, 1000)]) * 1000,
                # Data arrays
                raw_acc=self.raw_acc_buffer[:n],
                raw_gyro=self.raw_gyro_buffer[:n],
                corrected_acc=self.corrected_acc_buffer[:n],
                corrected_gyro=self.corrected_gyro_buffer[:n]
            )
                
            rospy.logdebug(f"Results saved: {n} inferences")
            
        except Exception as e:
            rospy.logerr(f"Failed to save results: {e}")
    
    def run_inference(self):
        """
        Run ONNX inference on buffered IMU data.
        
        Pipeline:
        1. Get real samples from buffer [N, 3]
        2. Generate padding [9, 3]
        3. Concatenate: [9+N, 3]
        4. Add batch dim: [1, 9+N, 3]
        5. Run ONNX: [1, 9+N, 3] -> [1, N, 3]
        6. Extract corrections and compute corrected IMU
        """
        try:
            t_start = time.time()
            
            # Get buffered data
            timestamps, acc, gyro = self.buffer.get_arrays()
            N = acc.shape[0]
            
            # Generate padding and concatenate
            acc_padded, gyro_padded = self.padding_gen.generate(acc, gyro)
            
            # Add batch dimension: [N+9, 3] -> [1, N+9, 3]
            acc_batch = acc_padded[np.newaxis, ...]
            gyro_batch = gyro_padded[np.newaxis, ...]
            
            # Run ONNX inference
            # Input: [1, N+9, 3] -> Output: [1, N, 3]
            corr_acc, corr_gyro = self.onnx_model.run(
                None,  # Return all outputs
                {"acc": acc_batch, "gyro": gyro_batch}
            )
            
            # Apply corrections: corrected = original + correction
            # Note: corr_acc/gyro are [1, N, 3], acc/gyro are [N, 3]
            corrected_acc = acc + corr_acc[0]
            corrected_gyro = gyro + corr_gyro[0]
            
            inference_time = time.time() - t_start
            self.inference_times.append(inference_time)
            
            # Update counters
            self.inference_count += 1
            self.total_samples_processed += N
            
            # Store results
            result = {
                'raw_acc': acc,
                'raw_gyro': gyro,
                'corrected_acc': corrected_acc,
                'corrected_gyro': corrected_gyro,
            }
            self.results.append(result)
            
            # Log inference summary every 100 inferences
            if self.inference_count % 100 == 0:
                avg_time_recent = np.mean(self.inference_times[-100:]) * 1000 if len(self.inference_times) >= 100 else np.mean(self.inference_times) * 1000
                rospy.loginfo(f"[Progress] Inferences: {self.inference_count}, Samples: {self.total_samples_processed}, Avg time: {avg_time_recent:.2f}ms")

            # Save results periodically (every 500 inferences)
            if self.inference_count % 500 == 0:
                self.save_results()
            
        except Exception as e:
            rospy.logerr(f"Inference error: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def _log_inference_result(self, n_samples, inference_time, raw_acc, raw_gyro, 
                             corr_acc, corr_gyro, corrected_acc, corrected_gyro):
        """Log inference results in a structured format."""
        rospy.loginfo(f"[Inference #{self.inference_count}]")
        rospy.loginfo(f"  Samples: {n_samples}, Time: {inference_time*1000:.2f}ms")
        rospy.loginfo(f"  Raw acc:       [{raw_acc[-1, 0]:.3f}, {raw_acc[-1, 1]:.3f}, {raw_acc[-1, 2]:.3f}]")
        rospy.loginfo(f"  Correction:    [{corr_acc[-1, 0]:.3f}, {corr_acc[-1, 1]:.3f}, {corr_acc[-1, 2]:.3f}]")
        rospy.loginfo(f"  Corrected acc: [{corrected_acc[-1, 0]:.3f}, {corrected_acc[-1, 1]:.3f}, {corrected_acc[-1, 2]:.3f}]")
        rospy.loginfo(f"  Raw gyro:      [{raw_gyro[-1, 0]:.3f}, {raw_gyro[-1, 1]:.3f}, {raw_gyro[-1, 2]:.3f}]")
        rospy.loginfo(f"  Corrected gyro:[{corrected_gyro[-1, 0]:.3f}, {corrected_gyro[-1, 1]:.3f}, {corrected_gyro[-1, 2]:.3f}]")
        rospy.loginfo("-" * 70)
    
    def imu_callback(self, msg: Imu):
        """
        IMU message callback.
        
        Buffers incoming samples and triggers inference when ready.
        """
        try:
            # Add sample to buffer
            self.buffer.add(msg)
            
            # Check if ready for inference
            if self.buffer.ready():
                self.run_inference()
                self.buffer.clear()
                
        except Exception as e:
            rospy.logerr(f"Callback error: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def shutdown_hook(self):
        """Called on node shutdown."""
        rospy.loginfo("="*70)
        rospy.loginfo("Shutting down IMU Inference Node")
        rospy.loginfo("="*70)
        rospy.loginfo(f"Total inferences: {self.inference_count}")
        rospy.loginfo(f"Total samples processed: {self.total_samples_processed}")
        
        if self.inference_times:
            avg_time = np.mean(self.inference_times) * 1000
            min_time = np.min(self.inference_times) * 1000
            max_time = np.max(self.inference_times) * 1000
            rospy.loginfo(f"Inference time (ms): avg={avg_time:.2f}, min={min_time:.2f}, max={max_time:.2f}")
        
        # Final save
        self.save_results()
        rospy.loginfo(f"Results saved to: {self.pickle_path}")
        rospy.loginfo("="*70)
    
    def start(self):
        """Start the ROS node."""
        rospy.init_node("imu_inference_node", log_level=rospy.INFO)
        rospy.loginfo("="*70)
        rospy.loginfo("AirIMU Real-time Inference Node")
        rospy.loginfo("="*70)
        rospy.loginfo(f"Node started at: {rospy.Time.now().to_sec():.3f}")
        rospy.loginfo(f"Python version: {sys.version}")
        
        # Check files
        if not self.check_files():
            rospy.signal_shutdown("Required files missing")
            return
        
        # Load model
        self.load_model()
        
        # Register shutdown hook
        rospy.on_shutdown(self.shutdown_hook)
        
        # Subscribe to IMU topic
        rospy.loginfo("Subscribing to: /imu_data")
        rospy.Subscriber("/imu_data", Imu, self.imu_callback, queue_size=1000)
        
        rospy.loginfo("Node ready! Waiting for IMU messages...")
        rospy.loginfo("="*70)
        
        # Spin
        rospy.spin()


if __name__ == "__main__":
    try:
        node = IMUInferenceNode()
        node.start()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
