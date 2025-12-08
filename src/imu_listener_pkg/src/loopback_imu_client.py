#!/usr/bin/env python3
"""
AI IMU Subscriber and Feedback Client for PX4 TCP Pipeline

This script:
1. Connects to PX4 TCP publisher on port 14567
2. Receives IMU samples from PX4
3. Performs AI preprocessing/processing
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
from collections import deque
from typing import Optional, Tuple

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
STATS_INTERVAL_S = 5.0
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

        # TCP sockets
        self.rx_socket: Optional[socket.socket] = None
        self.tx_socket: Optional[socket.socket] = None

        # Statistics
        self.stats = {
            'samples_received': 0,
            'samples_processed': 0,
            'samples_sent': 0,
            'rx_errors': 0,
            'tx_errors': 0,
            'crc_failures': 0,
            'sequence_gaps': 0,
            'last_sequence': None,
            'start_time': None,
            'last_stats_time': None,
        }

        # Processing queue
        self.process_queue = deque(maxlen=256)

        # Sample buffer for TCP recv
        self.rx_buffer = b''

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
        if self.rx_socket:
            self.rx_socket.close()
            self.rx_socket = None
        if self.tx_socket:
            self.tx_socket.close()
            self.tx_socket = None
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

    def process_sample(self, sample: ImuSample) -> ImuSample:
        """
        AI preprocessing/processing of IMU sample

        This is where you would implement:
        - Noise filtering
        - Bias correction
        - ML-based feature extraction
        - Anomaly detection
        - Sensor fusion

        For now, we pass through unchanged (identity transform)
        """
        # # DEBUG: Log every 250th sample to check for sign issues
        # if sample.sequence % 250 == 0:
        #     print(f"[AI CLIENT] Sample {sample.sequence}: "
        #           f"accel=[{sample.accel_x:.3f}, {sample.accel_y:.3f}, {sample.accel_z:.3f}] "
        #           f"gyro=[{sample.gyro_x:.3f}, {sample.gyro_y:.3f}, {sample.gyro_z:.3f}]")

        # TODO: Implement actual AI processing here
        # For now, just pass through
        processed = sample
        self.stats['samples_processed'] += 1
        return processed

    def send_sample(self, sample: ImuSample) -> bool:
        """Send processed sample back to PX4"""
        if not self.tx_socket:
            return False

        try:
            # Pack sample into binary format (must match C++ ImuUdpPacket structure)
            packet_data = struct.pack(
                PACKET_FORMAT,
                sample.timestamp_us,
                sample.sequence,
                sample.gyro_x, sample.gyro_y, sample.gyro_z,
                sample.accel_x, sample.accel_y, sample.accel_z,
                sample.delta_ang_dt,
                sample.delta_vel_dt,
                sample.crc16
            )

            # Send (TCP handles partial sends automatically)
            self.tx_socket.sendall(packet_data)
            self.stats['samples_sent'] += 1

            # Log first few samples
            if self.stats['samples_sent'] <= 4:
                print(f"TX Sample {self.stats['samples_sent']}: {sample}")

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
            sample = self.process_queue.popleft()

            # AI processing
            processed_sample = self.process_sample(sample)

            # Send back to PX4
            if self.send_sample(processed_sample):
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
        interval = now - self.stats['last_stats_time'] if self.stats['last_stats_time'] else uptime

        rx_rate = self.stats['samples_received'] / uptime if uptime > 0 else 0
        tx_rate = self.stats['samples_sent'] / uptime if uptime > 0 else 0

        print(f"\n{'='*80}")
        print(f"AI Client Statistics (uptime: {uptime:.1f}s)")
        print(f"{'='*80}")
        print(f"RX: {self.stats['samples_received']} samples ({rx_rate:.1f} Hz)")
        print(f"TX: {self.stats['samples_sent']} samples ({tx_rate:.1f} Hz)")
        print(f"Processed: {self.stats['samples_processed']} samples")
        print(f"Queue depth: {len(self.process_queue)}")
        print(f"RX errors: {self.stats['rx_errors']}")
        print(f"TX errors: {self.stats['tx_errors']}")
        print(f"CRC failures: {self.stats['crc_failures']}")
        print(f"Sequence gaps: {self.stats['sequence_gaps']}")
        print(f"Connected: RX={self.rx_socket is not None}, TX={self.tx_socket is not None}")
        print(f"{'='*80}\n")

        self.stats['last_stats_time'] = now

    def run(self):
        """Main processing loop"""
        self.running = True
        self.stats['start_time'] = time.time()
        self.stats['last_stats_time'] = self.stats['start_time']

        print("AI IMU Client starting...")

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

            # Small sleep to prevent busy loop
            time.sleep(0.001)  # 1ms

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
