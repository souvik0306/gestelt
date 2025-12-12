#!/usr/bin/env python3
"""
AI IMU Subscriber and Feedback Client for PX4 TCP Pipeline

This script:
1. Connects to PX4 TCP publisher on port 14567
2. Receives IMU samples from PX4
3. Performs AI preprocessing/processing with ONNX model
4. Sends processed samples back to PX4 on port 14568

Architecture:
- Client connects to PX4 TCP server on port 14567 (receives IMU data)
- Client connects to PX4 TCP server on port 14568 (sends processed data back)
- Implements same packet format as PX4 for compatibility
- Non-blocking I/O with proper error handling
"""

import socket
import struct
import time
import signal
import sys
import os
import subprocess
import numpy as np
from collections import deque
from typing import Optional, Tuple

# ROS imports
import rospy
from sensor_msgs.msg import Imu

# Import the inference module
from realtime_imu_inference import RealtimeIMUInference

# Packet format: matches ImuUdpPacket structure in PX4
# uint64_t timestamp_us      (8 bytes)
# uint32_t sequence          (4 bytes)
# float gyro_x, gyro_y, gyro_z     (12 bytes)
# float accel_x, accel_y, accel_z  (12 bytes)
# float delta_ang_dt         (4 bytes)
# float delta_vel_dt         (4 bytes)
# uint16_t crc16             (2 bytes)
# Total: 46 bytes
PACKET_FORMAT = '<QI3f3f2fH'  # Little-endian: Q=uint64, I=uint32, 3f=gyro, 3f=accel, 2f=deltas, H=uint16
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

# TCP connection settings
PX4_PUBLISHER_HOST = '127.0.0.1'
PX4_PUBLISHER_PORT = 14567
PX4_SUBSCRIBER_HOST = '127.0.0.1'
PX4_SUBSCRIBER_PORT = 14568

# Processing settings
STATS_INTERVAL_S = 1.0
RECONNECT_DELAY_S = 1.0


class ImuSample:
    """IMU sample data structure"""

    def __init__(self, timestamp_us: int, sequence: int,
                 gyro_x: float, gyro_y: float, gyro_z: float,
                 accel_x: float, accel_y: float, accel_z: float,
                 delta_ang_dt: float, delta_vel_dt: float,
                 crc16: int):
        self.timestamp_us = timestamp_us
        self.sequence = sequence
        self.gyro_x = gyro_x
        self.gyro_y = gyro_y
        self.gyro_z = gyro_z
        self.accel_x = accel_x
        self.accel_y = accel_y
        self.accel_z = accel_z
        self.delta_ang_dt = delta_ang_dt
        self.delta_vel_dt = delta_vel_dt
        self.crc16 = crc16

    def __repr__(self):
        return (f"ImuSample(ts={self.timestamp_us}, seq={self.sequence}, "
                f"gyro=[{self.gyro_x:.3f},{self.gyro_y:.3f},{self.gyro_z:.3f}], "
                f"accel=[{self.accel_x:.3f},{self.accel_y:.3f},{self.accel_z:.3f}])")


class AIClient:
    """AI IMU processing client"""

    def __init__(self):
        self.running = False

        # Initialize ROS node
        rospy.init_node('ai_imu_client', anonymous=True, disable_signals=True)
        
        # ROS Publishers for raw and corrected IMU data
        self.pub_raw_imu = rospy.Publisher('/imu/raw', Imu, queue_size=100)
        self.pub_corrected_imu = rospy.Publisher('/imu/corrected', Imu, queue_size=100)
        
        rospy.loginfo("ROS Publishers initialized:")
        rospy.loginfo("  - Raw IMU: /imu/raw")
        rospy.loginfo("  - Corrected IMU: /imu/corrected")

        # TCP sockets
        self.rx_socket: Optional[socket.socket] = None
        self.tx_socket: Optional[socket.socket] = None
        
        # Rosbag recording process
        self.rosbag_process: Optional[subprocess.Popen] = None

        # Statistics
        self.stats = {
            'samples_received': 0,
            'samples_processed': 0,
            'samples_sent': 0,
            'rx_errors': 0,
            'tx_errors': 0,
            'crc_failures': 0,
            'sequence_gaps': 0,
            'corrupted_outputs': 0,  # Track NaN/Inf from model
            'last_sequence': None,
            'start_time': None,
            'last_stats_time': None,
            # For instantaneous rate calculation
            'last_samples_received': 0,
            'last_samples_processed': 0,
            'last_samples_sent': 0,
            # Timing measurements
            'total_ai_time_ms': 0.0,  # Time spent in AI inference only
            'total_pipeline_time_ms': 0.0,  # Time spent in RX + AI + TX
        }

        # Processing queue
        self.process_queue = deque(maxlen=1000)

        # Sample buffer for TCP recv
        self.rx_buffer = b''

        # Get model path (use INT8 optimized model)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pkg_path = os.path.dirname(script_dir)  # imu_listener_pkg root
        
        # Try INT8 model first, fall back to FP32 if not available
        model_path = os.path.join(pkg_path, "models", "airimu_cpu_fp32_new.onnx")
        # fp32_model_path = os.path.join(pkg_path, "models", "airimu_cpu_fp32.onnx")
        
        # if os.path.isfile(int8_model_path):
        #     model_path = int8_model_path
        #     print(f"Using INT8 quantized model: {model_path}")
        # elif os.path.isfile(fp32_model_path):
        #     model_path = fp32_model_path
        #     print(f"INT8 model not found, using FP32 model: {model_path}")
        # else:
        #     raise FileNotFoundError(f"No model found at {int8_model_path} or {fp32_model_path}")

        # Initialize inference module (INT8 handled automatically)
        self.inference = RealtimeIMUInference(
            model_path, 
            verbose=False
        )

    def start_rosbag_recording(self) -> bool:
        """Start rosbag recording"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            pkg_path = os.path.dirname(script_dir)
            bags_dir = os.path.join(pkg_path, 'bags')
            os.makedirs(bags_dir, exist_ok=True)
            
            # Generate timestamped filename
            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
            bag_file = os.path.join(bags_dir, f'imu_comparison_{timestamp}.bag')
            
            # Start rosbag record process
            cmd = [
                'rosbag', 'record',
                '-O', bag_file,
                '/imu/raw',
                '/imu/corrected',
                '/mavros/local_position/pose'
            ]
            
            self.rosbag_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group for clean shutdown
            )
            
            print(f"Started rosbag recording: {bag_file}")
            rospy.loginfo(f"Recording to: {bag_file}")
            return True
            
        except Exception as e:
            print(f"Failed to start rosbag recording: {e}")
            rospy.logerr(f"Failed to start rosbag recording: {e}")
            return False
    
    def stop_rosbag_recording(self):
        """Stop rosbag recording gracefully"""
        if self.rosbag_process:
            try:
                print("Stopping rosbag recording...")
                rospy.loginfo("Stopping rosbag recording...")
                
                # Send SIGINT for graceful shutdown
                self.rosbag_process.send_signal(signal.SIGINT)
                
                # Wait for process to finish (with timeout)
                try:
                    self.rosbag_process.wait(timeout=5.0)
                    print("Rosbag recording stopped successfully")
                    rospy.loginfo("Rosbag recording stopped successfully")
                except subprocess.TimeoutExpired:
                    print("Rosbag process did not stop gracefully, forcing termination...")
                    self.rosbag_process.kill()
                    self.rosbag_process.wait()
                    
            except Exception as e:
                print(f"Error stopping rosbag: {e}")
                rospy.logerr(f"Error stopping rosbag: {e}")
            finally:
                self.rosbag_process = None

    def calculate_crc16(self, data: bytes) -> int:
        """Calculate CRC16-CCITT"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        return crc

    def validate_packet(self, sample: ImuSample, raw_data: bytes) -> bool:
        """Validate packet CRC"""
        # CRC covers all fields except crc16 itself
        crc_data = raw_data[:-2]
        calculated_crc = self.calculate_crc16(crc_data)

        if calculated_crc != sample.crc16:
            self.stats['crc_failures'] += 1
            return False

        return True

    def connect_rx(self) -> bool:
        """Connect to PX4 publisher (receive IMU data)"""
        try:
            self.rx_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.rx_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.rx_socket.settimeout(5.0)

            print(f"Connecting to PX4 publisher at {PX4_PUBLISHER_HOST}:{PX4_PUBLISHER_PORT}...")
            self.rx_socket.connect((PX4_PUBLISHER_HOST, PX4_PUBLISHER_PORT))

            # Set to non-blocking after connect
            self.rx_socket.setblocking(False)

            print(f"Connected to PX4 publisher")
            return True

        except Exception as e:
            print(f"Failed to connect to PX4 publisher: {e}")
            if self.rx_socket:
                self.rx_socket.close()
                self.rx_socket = None
            return False

    def connect_tx(self) -> bool:
        """Connect to PX4 subscriber (send processed data)"""
        try:
            self.tx_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tx_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.tx_socket.settimeout(5.0)

            print(f"Connecting to PX4 subscriber at {PX4_SUBSCRIBER_HOST}:{PX4_SUBSCRIBER_PORT}...")
            self.tx_socket.connect((PX4_SUBSCRIBER_HOST, PX4_SUBSCRIBER_PORT))

            # Set to non-blocking after connect
            self.tx_socket.setblocking(False)

            print(f"Connected to PX4 subscriber")
            return True

        except Exception as e:
            print(f"Failed to connect to PX4 subscriber: {e}")
            if self.tx_socket:
                self.tx_socket.close()
                self.tx_socket = None
            return False

    def disconnect(self):
        """Close all connections"""
        # Stop rosbag recording first
        self.stop_rosbag_recording()
        
        if self.rx_socket:
            self.rx_socket.close()
            self.rx_socket = None
        if self.tx_socket:
            self.tx_socket.close()
            self.tx_socket = None

        # Close inference module
        if hasattr(self, 'inference'):
            self.inference.close()

        print("Disconnected from PX4")

    def receive_samples(self) -> int:
        """Receive IMU samples from PX4 publisher"""
        if not self.rx_socket:
            return 0

        samples_received = 0

        try:
            # Non-blocking receive
            while True:
                chunk = self.rx_socket.recv(4096)
                if not chunk:
                    # Connection closed
                    print("PX4 publisher closed connection")
                    self.disconnect()
                    return samples_received

                self.rx_buffer += chunk

                # Process complete packets
                while len(self.rx_buffer) >= PACKET_SIZE:
                    packet_data = self.rx_buffer[:PACKET_SIZE]
                    self.rx_buffer = self.rx_buffer[PACKET_SIZE:]

                    # Unpack packet
                    values = struct.unpack(PACKET_FORMAT, packet_data)
                    sample = ImuSample(*values)

                    # Validate CRC
                    if not self.validate_packet(sample, packet_data):
                        self.stats['rx_errors'] += 1
                        continue

                    # Check sequence
                    if self.stats['last_sequence'] is not None:
                        expected_seq = (self.stats['last_sequence'] + 1) & 0xFFFFFFFF
                        if sample.sequence != expected_seq:
                            gap = (sample.sequence - expected_seq) & 0xFFFFFFFF
                            self.stats['sequence_gaps'] += gap

                    self.stats['last_sequence'] = sample.sequence
                    self.stats['samples_received'] += 1
                    samples_received += 1

                    # Log first few samples
                    if self.stats['samples_received'] <= 4:
                        print(f"RX Sample {self.stats['samples_received']}: {sample}")

                    # Publish raw IMU data to ROS
                    self.publish_raw_imu(sample)

                    # Add to processing queue
                    self.process_queue.append(sample)

        except BlockingIOError:
            # No more data available
            pass
        except Exception as e:
            print(f"RX error: {e}")
            self.stats['rx_errors'] += 1
            self.disconnect()

        return samples_received

    def publish_raw_imu(self, sample: ImuSample):
        """Publish raw IMU data to ROS topic"""
        try:
            imu_msg = Imu()
            imu_msg.header.stamp = rospy.Time.from_sec(sample.timestamp_us / 1e6)
            imu_msg.header.frame_id = "imu_raw"
            
            # Angular velocity (gyro)
            imu_msg.angular_velocity.x = sample.gyro_x
            imu_msg.angular_velocity.y = sample.gyro_y
            imu_msg.angular_velocity.z = sample.gyro_z
            
            # Linear acceleration
            imu_msg.linear_acceleration.x = sample.accel_x
            imu_msg.linear_acceleration.y = sample.accel_y
            imu_msg.linear_acceleration.z = sample.accel_z
            
            self.pub_raw_imu.publish(imu_msg)
        except Exception as e:
            rospy.logwarn(f"Failed to publish raw IMU: {e}")

    def publish_corrected_imu(self, sample: ImuSample):
        """Publish corrected IMU data to ROS topic"""
        try:
            imu_msg = Imu()
            imu_msg.header.stamp = rospy.Time.from_sec(sample.timestamp_us / 1e6)
            imu_msg.header.frame_id = "imu_corrected"
            
            # Angular velocity (gyro)
            imu_msg.angular_velocity.x = sample.gyro_x
            imu_msg.angular_velocity.y = sample.gyro_y
            imu_msg.angular_velocity.z = sample.gyro_z
            
            # Linear acceleration
            imu_msg.linear_acceleration.x = sample.accel_x
            imu_msg.linear_acceleration.y = sample.accel_y
            imu_msg.linear_acceleration.z = sample.accel_z
            
            self.pub_corrected_imu.publish(imu_msg)
        except Exception as e:
            rospy.logwarn(f"Failed to publish corrected IMU: {e}")

    def process_sample(self, sample: ImuSample) -> ImuSample:
        """
        AI processing of IMU sample using ONNX model.

        Args:
            sample: Input IMU sample from PX4

        Returns:
            Processed IMU sample with corrections applied
        """
        # TEMPORARY: AI processing disabled - simple loopback
        return sample
        
        # ai_start_time = time.time()
        # 
        # try:
        #     # Extract IMU data
        #     input_acc = np.array([sample.accel_x, sample.accel_y, sample.accel_z], dtype=np.float32)
        #     input_gyro = np.array([sample.gyro_x, sample.gyro_y, sample.gyro_z], dtype=np.float32)
        # 
        #     # Validate input data
        #     if not (np.isfinite(input_acc).all() and np.isfinite(input_gyro).all()):
        #         print(f"[AI Client] WARNING: Invalid input data, passing through unchanged")
        #         return sample
        # 
        #     # Run inference (measure this specifically)
        #     inference_start = time.time()
        #     corrected_acc, corrected_gyro = self.inference.inference_airimu(input_acc, input_gyro)
        #     ai_time_ms = (time.time() - inference_start) * 1000
        #     self.stats['total_ai_time_ms'] += ai_time_ms
        # 
        #     # Validate output data (CRITICAL: check for NaN/Inf corruption)
        #     if not (np.isfinite(corrected_acc).all() and np.isfinite(corrected_gyro).all()):
        #         print(f"[AI Client] ERROR: Model produced invalid output (NaN/Inf), passing through original")
        #         print(f"  Input acc: {input_acc}")
        #         print(f"  Input gyro: {input_gyro}")
        #         print(f"  Output acc: {corrected_acc}")
        #         print(f"  Output gyro: {corrected_gyro}")
        #         self.stats['corrupted_outputs'] += 1
        #         return sample
        # 
        #     # Create corrected sample
        #     processed = ImuSample(
        #         timestamp_us=sample.timestamp_us,
        #         sequence=sample.sequence,
        #         gyro_x=float(corrected_gyro[0]),
        #         gyro_y=float(corrected_gyro[1]),
        #         gyro_z=float(corrected_gyro[2]),
        #         accel_x=float(corrected_acc[0]),
        #         accel_y=float(corrected_acc[1]),
        #         accel_z=float(corrected_acc[2]),
        #         delta_ang_dt=sample.delta_ang_dt,
        #         delta_vel_dt=sample.delta_vel_dt,
        #         crc16=sample.crc16
        #     )
        # 
        #     return processed
        # 
        # except Exception as e:
        #     print(f"[AI Client] Error processing sample: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     # On error, pass through unchanged
        #     return sample

    def send_sample(self, sample: ImuSample) -> bool:
        """Send processed sample back to PX4"""
        if not self.tx_socket:
            return False

        try:
            # CRITICAL: Validate all float values before sending to PX4
            values_to_check = [
                sample.gyro_x, sample.gyro_y, sample.gyro_z,
                sample.accel_x, sample.accel_y, sample.accel_z,
                sample.delta_ang_dt, sample.delta_vel_dt
            ]
            
            if not all(np.isfinite(v) for v in values_to_check):
                print(f"[AI Client] ERROR: Attempting to send corrupted data to PX4! Dropping sample.")
                print(f"  Sample: {sample}")
                self.stats['tx_errors'] += 1
                return False

            # Pack sample into binary format (must match C++ ImuUdpPacket structure)
            # First pack without CRC to calculate it
            packet_data_no_crc = struct.pack(
                '<QI3f3f2f',  # All fields except CRC
                sample.timestamp_us,
                sample.sequence,
                sample.gyro_x, sample.gyro_y, sample.gyro_z,
                sample.accel_x, sample.accel_y, sample.accel_z,
                sample.delta_ang_dt,
                sample.delta_vel_dt
            )

            # Calculate CRC for the modified data
            new_crc = self.calculate_crc16(packet_data_no_crc)

            # Now pack the complete packet with the new CRC
            packet_data = packet_data_no_crc + struct.pack('<H', new_crc)

            # Send (TCP handles partial sends automatically)
            self.tx_socket.sendall(packet_data)
            self.stats['samples_sent'] += 1
            self.stats['samples_processed'] += 1  # Count complete pipeline: RX + AI + TX

            # Log first few samples
            if self.stats['samples_sent'] <= 4:
                print(f"TX Sample {self.stats['samples_sent']}: {sample} (CRC: {new_crc:04X})")

            return True

        except Exception as e:
            print(f"TX error: {e}")
            self.stats['tx_errors'] += 1
            self.disconnect()
            return False

    def process_queue_samples(self) -> int:
        """Process samples from queue and send back to PX4"""
        processed_count = 0

        while self.process_queue:
            # Measure complete pipeline time: receive + AI + transmit
            pipeline_start_time = time.time()
            
            sample = self.process_queue.popleft()

            # AI processing
            processed_sample = self.process_sample(sample)

            # Publish corrected IMU data to ROS
            self.publish_corrected_imu(processed_sample)

            # Send back to PX4
            if self.send_sample(processed_sample):
                # Measure total pipeline time (queuing + AI + TX)
                pipeline_time_ms = (time.time() - pipeline_start_time) * 1000
                self.stats['total_pipeline_time_ms'] += pipeline_time_ms
                processed_count += 1
            else:
                # TX error, stop processing and try to reconnect
                break

        return processed_count

    def print_statistics(self):
        """Print performance statistics"""
        now = time.time()

        if self.stats['start_time'] is None:
            return

        uptime = now - self.stats['start_time']
        
        # Use actual elapsed time since last stats print for interval calculation
        if self.stats['last_stats_time'] is not None:
            interval = now - self.stats['last_stats_time']
        else:
            interval = uptime

        # Calculate instantaneous rates (samples per second during this interval)
        rx_interval = self.stats['samples_received'] - self.stats['last_samples_received']
        tx_interval = self.stats['samples_sent'] - self.stats['last_samples_sent']
        processed_interval = self.stats['samples_processed'] - self.stats['last_samples_processed']
        
        # Calculate instantaneous rates - use actual measured interval
        rx_rate_instant = rx_interval / interval if interval > 0 else 0
        tx_rate_instant = tx_interval / interval if interval > 0 else 0
        processed_rate_instant = processed_interval / interval if interval > 0 else 0

        # Calculate average rates (since start) - most reliable metric
        rx_rate_avg = self.stats['samples_received'] / uptime if uptime > 0 else 0
        tx_rate_avg = self.stats['samples_sent'] / uptime if uptime > 0 else 0
        processed_rate_avg = self.stats['samples_processed'] / uptime if uptime > 0 else 0

        print(f"\n{'='*80}")
        print(f"\nErrors & Status:")
        print(f"  Queue depth: {len(self.process_queue)}")
        print(f"  RX errors: {self.stats['rx_errors']}")
        print(f"  TX errors: {self.stats['tx_errors']}")
        print(f"  CRC failures: {self.stats['crc_failures']}")
        print(f"  Sequence gaps: {self.stats['sequence_gaps']}")
        print(f"  Corrupted outputs: {self.stats['corrupted_outputs']}")
        print(f"  Connected: RX={self.rx_socket is not None}, TX={self.tx_socket is not None}")

        print(f"\nAI Client Statistics")
        print(f"{'='*80}")
        print(f"Runtime: {uptime:.1f}s (last interval: {interval:.3f}s)")
        print(f"\nAI Model Inference Timing:")
        if hasattr(self, 'inference'):
            inference_stats = self.inference.get_statistics()
            print(f"  Total inferences: {inference_stats['inference_count']}")
            print(f"  Avg inference time: {inference_stats['avg_inference_time_ms']:.3f} ms/sample")
            print(f"  Max inference time: {inference_stats['max_inference_time_ms']:.3f} ms/sample")
        
        print(f"\nThroughput:")
        print(f"  RX:        {self.stats['samples_received']} samples | Avg: {rx_rate_avg:.1f} Hz | Current: {rx_rate_instant:.1f} Hz ({rx_interval} samples)")
        print(f"  TX:        {self.stats['samples_sent']} samples | Avg: {tx_rate_avg:.1f} Hz | Current: {tx_rate_instant:.1f} Hz ({tx_interval} samples)")
        print(f"  Processed: {self.stats['samples_processed']} samples | Avg: {processed_rate_avg:.1f} Hz | Current: {processed_rate_instant:.1f} Hz ({processed_interval} samples)")
        
        # Calculate timing averages
        if self.stats['samples_processed'] > 0:
            avg_ai_time = self.stats['total_ai_time_ms'] / self.stats['samples_processed']
            avg_pipeline_time = self.stats['total_pipeline_time_ms'] / self.stats['samples_processed']
            overhead_time = avg_pipeline_time - avg_ai_time
            print(f"\nPipeline Timing (per sample):")
            print(f"  AI inference only: {avg_ai_time:.3f} ms")
            print(f"  Total pipeline (RX+AI+TX): {avg_pipeline_time:.3f} ms")
            print(f"  Overhead (RX+TX+queue): {overhead_time:.3f} ms")

        print(f"{'='*80}\n")

        # Update counters for next interval
        self.stats['last_samples_received'] = self.stats['samples_received']
        self.stats['last_samples_processed'] = self.stats['samples_processed']
        self.stats['last_samples_sent'] = self.stats['samples_sent']
        self.stats['last_stats_time'] = now

    def run(self):
        """Main processing loop"""
        self.running = True
        self.stats['start_time'] = time.time()
        self.stats['last_stats_time'] = self.stats['start_time']

        print("AI IMU Client starting...")
        
        # Start rosbag recording
        self.start_rosbag_recording()

        while self.running:
            # Ensure connections are established
            if not self.rx_socket:
                if not self.connect_rx():
                    time.sleep(RECONNECT_DELAY_S)
                    continue

            if not self.tx_socket:
                if not self.connect_tx():
                    time.sleep(RECONNECT_DELAY_S)
                    continue

            # Receive samples from PX4
            self.receive_samples()

            # Process and send back to PX4
            self.process_queue_samples()

            # Print statistics periodically
            now = time.time()
            if now - self.stats['last_stats_time'] >= STATS_INTERVAL_S:
                self.print_statistics()

        self.disconnect()
        print("AI IMU Client stopped")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nShutdown signal received...")
    if 'client' in globals():
        client.running = False
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    client = AIClient()

    try:
        client.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect()
