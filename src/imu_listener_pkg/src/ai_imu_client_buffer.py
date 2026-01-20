#!/usr/bin/env python3
"""
AI IMU Subscriber and Feedback Client for PX4 TCP Pipeline (Buffered Version)

This script:
1. Connects to PX4 TCP publisher on port 14567
2. Receives IMU samples from PX4
3. Maintains a rolling buffer of 100 samples for AI context
4. Performs AI preprocessing/processing with ONNX model on full buffer
5. Sends ONLY the latest corrected sample back to PX4 on port 14568

Buffer Strategy:
- Buffer is initialized with 100 zero samples
- Each new IMU sample is added to the buffer (FIFO)
- Oldest sample is removed when buffer is full
- Entire buffer (100 samples) is fed to neural network for context
- Only the latest corrected sample is sent back to PX4

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
import numpy as np
from collections import deque
from typing import Optional

# Import the buffered inference module
from realtime_imu_inference_buffer import RealtimeIMUInference

# Packet format: matches ImuUdpPacket structure in PX4
# uint64_t timestamp_us      (8 bytes)
# uint32_t sequence          (4 bytes)
# float gyro_x, gyro_y, gyro_z     (12 bytes)
# float accel_x, accel_y, accel_z  (12 bytes)
# float delta_ang_dt         (4 bytes)
# float delta_vel_dt         (4 bytes)
# uint16_t crc16             (2 bytes)
# Total: 46 bytes
PACKET_FORMAT = '<QI3f3f2fH'  # Little-endian
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

# TCP connection settings
PX4_PUBLISHER_HOST = '127.0.0.1'
PX4_PUBLISHER_PORT = 14567
PX4_SUBSCRIBER_HOST = '127.0.0.1'
PX4_SUBSCRIBER_PORT = 14568

# Processing settings
STATS_INTERVAL_S = 1.0
RECONNECT_DELAY_S = 1.0

# Buffer settings
BUFFER_SIZE = 200  # Number of samples to maintain for context


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


class AIClientBuffer:
    """AI IMU processing client with buffered inference"""

    def __init__(self):
        self.running = False

        print("AI IMU Client (Buffered) initialized")

        # TCP sockets
        self.rx_socket: Optional[socket.socket] = None
        self.tx_socket: Optional[socket.socket] = None

        # Statistics
        self.stats = {
            'samples_received': 0,
            'samples_processed': 0,
            'samples_sent': 0,
            'samples_dropped': 0,  # Track dropped samples due to queue overflow
            'rx_errors': 0,
            'tx_errors': 0,
            'crc_failures': 0,
            'sequence_gaps': 0,
            'corrupted_outputs': 0,
            'last_sequence': None,
            'start_time': None,
            'last_stats_time': None,
            # For instantaneous rate calculation
            'last_samples_received': 0,
            'last_samples_processed': 0,
            'last_samples_sent': 0,
            # Timing measurements
            'total_ai_time_ms': 0.0,
            'total_pipeline_time_ms': 0.0,
        }

        # Processing queue (minimal - we process immediately, no batching)
        # Queue only exists to decouple RX from processing slightly
        self.process_queue = deque(maxlen=100)  # Small queue, process ASAP

        # Sample buffer for TCP recv
        self.rx_buffer = b''
        
        # Buffer logging
        self.buffer_log_file = None
        self.enable_buffer_logging = False  # DISABLED: Massive performance hit (375 KB/s disk I/O)
        if self.enable_buffer_logging:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            pkg_path = os.path.dirname(script_dir)
            logs_dir = os.path.join(pkg_path, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            log_path = os.path.join(logs_dir, f'buffer_log_{timestamp}.csv')
            self.buffer_log_file = open(log_path, 'w')
            self.buffer_log_file.write('timestamp_us,buffer_state\n')
            print(f"Buffer logging enabled: {log_path}")

        # Get model path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pkg_path = os.path.dirname(script_dir)
        model_path = os.path.join(pkg_path, "models", "airimu_cpu_fp32_200.onnx")
        
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Using model: {model_path}")

        # Initialize buffered inference module
        self.inference = RealtimeIMUInference(
            model_path, 
            seqlen=BUFFER_SIZE,  # 100 samples buffer
            verbose=False  # Disabled for performance
        )
        
        print(f"Inference module initialized with buffer size: {BUFFER_SIZE}")

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
        if self.rx_socket:
            self.rx_socket.close()
            self.rx_socket = None
        if self.tx_socket:
            self.tx_socket.close()
            self.tx_socket = None

        # Close buffer log file
        if self.buffer_log_file:
            self.buffer_log_file.close()
            self.buffer_log_file = None
            print("Buffer log file closed")

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
                    if self.stats['samples_received'] <= 3:
                        print(f"RX Sample {self.stats['samples_received']}: {sample}")

                    # Add to processing queue
                    # Check if queue is full (will drop oldest)
                    if len(self.process_queue) >= self.process_queue.maxlen:
                        self.stats['samples_dropped'] += 1
                    self.process_queue.append(sample)

        except BlockingIOError:
            # No more data available
            pass
        except Exception as e:
            print(f"RX error: {e}")
            self.stats['rx_errors'] += 1
            self.disconnect()

        return samples_received

    def process_sample(self, sample: ImuSample) -> Optional[ImuSample]:
        """
        AI processing of IMU sample using buffered ONNX inference.
        
        IMPORTANT: This function processes EVERY incoming sample immediately!
        - Buffer state: Initially 99 zeros + 1 real sample, then 98 zeros + 2 real samples, etc.
        - The buffer is ALWAYS fed to the NN (even when mostly zeros)
        - Only the latest corrected sample is returned and sent to PX4
        - This ensures 1:1 RX to TX sample ratio (no waiting for buffer to fill)
        
        Timeline:
        t1:  Buffer = [0, 0, ..., 0, imu1]     → Process → Send corrected_imu1
        t2:  Buffer = [0, 0, ..., imu1, imu2]  → Process → Send corrected_imu2
        ...
        t100: Buffer = [imu1, imu2, ..., imu100] → Process → Send corrected_imu100

        Args:
            sample: Input IMU sample from PX4

        Returns:
            Processed IMU sample with corrections applied (only latest sample)
        """
        try:
            # Extract IMU data
            input_acc = np.array([sample.accel_x, sample.accel_y, sample.accel_z], dtype=np.float32)
            input_gyro = np.array([sample.gyro_x, sample.gyro_y, sample.gyro_z], dtype=np.float32)

            # Validate input data
            if not (np.isfinite(input_acc).all() and np.isfinite(input_gyro).all()):
                print(f"[AI Client] WARNING: Invalid input data, skipping")
                return None

            # Run buffered inference
            # This adds the sample to the buffer and runs inference on the full buffer
            # Returns only the corrected value for the latest sample
            inference_start = time.time()
            corrected_acc, corrected_gyro = self.inference.inference_airimu(input_acc, input_gyro)
            ai_time_ms = (time.time() - inference_start) * 1000
            self.stats['total_ai_time_ms'] += ai_time_ms
            
            # Log buffer state if enabled
            if self.buffer_log_file:
                self.log_buffer_state(sample.timestamp_us)

            # Validate output data
            if not (np.isfinite(corrected_acc).all() and np.isfinite(corrected_gyro).all()):
                print(f"[AI Client] ERROR: Model produced invalid output (NaN/Inf)")
                print(f"  Input acc: {input_acc}")
                print(f"  Input gyro: {input_gyro}")
                print(f"  Output acc: {corrected_acc}")
                print(f"  Output gyro: {corrected_gyro}")
                self.stats['corrupted_outputs'] += 1
                return None

            # Create corrected sample (only for the latest input)
            processed = ImuSample(
                timestamp_us=sample.timestamp_us,
                sequence=sample.sequence,
                gyro_x=float(corrected_gyro[0]),
                gyro_y=float(corrected_gyro[1]),
                gyro_z=float(corrected_gyro[2]),
                accel_x=float(corrected_acc[0]),
                accel_y=float(corrected_acc[1]),
                accel_z=float(corrected_acc[2]),
                delta_ang_dt=sample.delta_ang_dt,
                delta_vel_dt=sample.delta_vel_dt,
                crc16=sample.crc16
            )

            return processed

        except Exception as e:
            print(f"[AI Client] Error processing sample: {e}")
            import traceback
            traceback.print_exc()
            return None

    def log_buffer_state(self, timestamp_us: int):
        """
        Log the current buffer state to CSV file.
        
        Args:
            timestamp_us: Timestamp in microseconds
        """
        try:
            # Get current buffer state
            buffer = self.inference.buffer
            acc_buf = buffer.acc_buf
            gyro_buf = buffer.gyro_buf
            
            # Format buffer as string: flatten and concatenate accel and gyro
            # Format: [acc0_x,acc0_y,acc0_z,...,acc99_x,acc99_y,acc99_z,gyro0_x,gyro0_y,gyro0_z,...,gyro99_x,gyro99_y,gyro99_z]
            acc_flat = acc_buf.flatten()
            gyro_flat = gyro_buf.flatten()
            buffer_array = np.concatenate([acc_flat, gyro_flat])
            buffer_str = '[' + ','.join([f'{v:.6f}' for v in buffer_array]) + ']'
            
            # Write to CSV
            self.buffer_log_file.write(f'{timestamp_us},"{buffer_str}"\n')
            
        except Exception as e:
            print(f"[AI Client] Error logging buffer state: {e}")

    def send_sample(self, sample: ImuSample) -> bool:
        """Send processed sample back to PX4"""
        if not self.tx_socket:
            return False

        try:
            # Validate all float values before sending
            values_to_check = [
                sample.gyro_x, sample.gyro_y, sample.gyro_z,
                sample.accel_x, sample.accel_y, sample.accel_z,
                sample.delta_ang_dt, sample.delta_vel_dt
            ]
            
            if not all(np.isfinite(v) for v in values_to_check):
                print(f"[AI Client] ERROR: Attempting to send corrupted data to PX4! Dropping sample.")
                self.stats['tx_errors'] += 1
                return False

            # Pack sample into binary format
            packet_data_no_crc = struct.pack(
                '<QI3f3f2f',
                sample.timestamp_us,
                sample.sequence,
                sample.gyro_x, sample.gyro_y, sample.gyro_z,
                sample.accel_x, sample.accel_y, sample.accel_z,
                sample.delta_ang_dt,
                sample.delta_vel_dt
            )

            # Calculate CRC
            new_crc = self.calculate_crc16(packet_data_no_crc)
            packet_data = packet_data_no_crc + struct.pack('<H', new_crc)

            # Send
            self.tx_socket.sendall(packet_data)
            self.stats['samples_sent'] += 1
            self.stats['samples_processed'] += 1

            # Log first few samples
            if self.stats['samples_sent'] <= 3:
                print(f"TX Sample {self.stats['samples_sent']}: {sample} (CRC: {new_crc:04X})")

            return True

        except Exception as e:
            print(f"TX error: {e}")
            self.stats['tx_errors'] += 1
            self.disconnect()
            return False

    def process_queue_samples(self) -> int:
        """
        Process ALL samples from queue and send back to PX4.
        
        This processes every sample immediately - no batching or waiting!
        Each incoming IMU sample triggers one inference and one TX.
        """
        processed_count = 0

        while self.process_queue:
            sample = self.process_queue.popleft()

            # AI processing with buffering
            # This will add sample to buffer and run inference immediately
            processed_sample = self.process_sample(sample)
            
            if processed_sample is None:
                # Skip if processing failed (but this shouldn't happen often)
                continue

            # Send back to PX4 immediately
            if self.send_sample(processed_sample):
                processed_count += 1
            else:
                # TX error, stop processing
                break

        return processed_count

    def print_statistics(self):
        """Print performance statistics"""
        now = time.time()

        if self.stats['start_time'] is None:
            return

        uptime = now - self.stats['start_time']
        
        if self.stats['last_stats_time'] is not None:
            interval = now - self.stats['last_stats_time']
        else:
            interval = uptime

        # Calculate instantaneous rates
        rx_interval = self.stats['samples_received'] - self.stats['last_samples_received']
        tx_interval = self.stats['samples_sent'] - self.stats['last_samples_sent']
        processed_interval = self.stats['samples_processed'] - self.stats['last_samples_processed']
        
        rx_rate_instant = rx_interval / interval if interval > 0 else 0
        tx_rate_instant = tx_interval / interval if interval > 0 else 0
        processed_rate_instant = processed_interval / interval if interval > 0 else 0

        # Calculate average rates
        rx_rate_avg = self.stats['samples_received'] / uptime if uptime > 0 else 0
        tx_rate_avg = self.stats['samples_sent'] / uptime if uptime > 0 else 0
        processed_rate_avg = self.stats['samples_processed'] / uptime if uptime > 0 else 0

        print(f"\n{'='*80}")
        print(f"AI Client Statistics (Buffered Inference)")
        print(f"{'='*80}")
        print(f"Runtime: {uptime:.1f}s (last interval: {interval:.3f}s)")
        
        # Buffer status
        buffer_stats = self.inference.get_statistics()
        print(f"\nBuffer Status:")
        print(f"  Buffer fill: {self.inference.buffer.get_fill_percentage():.1f}%")
        print(f"  Samples in buffer: {min(self.inference.buffer.sample_count, BUFFER_SIZE)}/{BUFFER_SIZE}")
        print(f"  Queue depth: {len(self.process_queue)} (should be near 0 - processing immediately)")
        
        print(f"\nAI Model Inference Timing:")
        print(f"  Total inferences: {buffer_stats['inference_count']}")
        print(f"  Avg inference time: {buffer_stats['avg_inference_time_ms']:.3f} ms/sample")
        print(f"  Max inference time: {buffer_stats['max_inference_time_ms']:.3f} ms/sample")
        
        print(f"\nThroughput:")
        print(f"  RX:        {self.stats['samples_received']} samples | Avg: {rx_rate_avg:.1f} Hz | Current: {rx_rate_instant:.1f} Hz")
        print(f"  TX:        {self.stats['samples_sent']} samples | Avg: {tx_rate_avg:.1f} Hz | Current: {tx_rate_instant:.1f} Hz")
        print(f"  Processed: {self.stats['samples_processed']} samples | Avg: {processed_rate_avg:.1f} Hz | Current: {processed_rate_instant:.1f} Hz")
        
        # Check for RX/TX mismatch
        rx_tx_diff = self.stats['samples_received'] - self.stats['samples_sent']
        if abs(rx_tx_diff) > 10:
            print(f"  ⚠ WARNING: RX/TX mismatch of {rx_tx_diff} samples (should be ~0 for immediate processing)")
        
        # Timing averages
        if self.stats['samples_processed'] > 0:
            avg_ai_time = self.stats['total_ai_time_ms'] / self.stats['samples_processed']
            avg_pipeline_time = self.stats['total_pipeline_time_ms'] / self.stats['samples_processed']
            overhead_time = avg_pipeline_time - avg_ai_time
            print(f"\nPipeline Timing (per sample):")
            print(f"  AI inference only: {avg_ai_time:.3f} ms")
            print(f"  Total pipeline (RX+AI+TX): {avg_pipeline_time:.3f} ms")
            print(f"  Overhead (RX+TX+queue): {overhead_time:.3f} ms")

        print(f"\nErrors & Status:")
        print(f"  Queue depth: {len(self.process_queue)} (max: {self.process_queue.maxlen})")
        print(f"  Samples DROPPED: {self.stats['samples_dropped']} ⚠ (inference too slow!)")
        print(f"  RX errors: {self.stats['rx_errors']}")
        print(f"  TX errors: {self.stats['tx_errors']}")
        print(f"  Sequence gaps: {self.stats['sequence_gaps']}")
        print(f"  Corrupted outputs: {self.stats['corrupted_outputs']}")
        print(f"  Connected: RX={self.rx_socket is not None}, TX={self.tx_socket is not None}")
        print(f"{'='*80}\n")

        # Update counters
        self.stats['last_samples_received'] = self.stats['samples_received']
        self.stats['last_samples_processed'] = self.stats['samples_processed']
        self.stats['last_samples_sent'] = self.stats['samples_sent']
        self.stats['last_stats_time'] = now

    def run(self):
        """Main processing loop"""
        self.running = True
        self.stats['start_time'] = time.time()
        self.stats['last_stats_time'] = self.stats['start_time']

        print("AI IMU Client (Buffered) starting...")
        print(f"Buffer configuration: {BUFFER_SIZE} samples for context")
        print("Note: Buffer starts with zeros, fills progressively with real data")

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
        print("AI IMU Client (Buffered) stopped")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nShutdown signal received...")
    if 'client' in globals():
        client.running = False
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    client = AIClientBuffer()

    try:
        client.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect()
