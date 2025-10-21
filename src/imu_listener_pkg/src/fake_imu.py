#!/usr/bin/env python3
"""
UDP delta sender for PX4 imu_ai_bridge.

Subscribes to /mavros/imu/data (ENU frame), applies simple low-pass
denoising, converts the vectors to the FRD body frame required by
vehicle_imu_ai, then sends the packed struct over UDP to the PX4
bridge.

Struct layout matching vehicle_imu_ai.msg:
    uint64  timestamp
    uint64  timestamp_sample
    uint32  accel_device_id
    uint32  gyro_device_id
    float32[3] delta_angle
    float32[3] delta_velocity
    uint16  delta_angle_dt           # microseconds
    uint16  delta_velocity_dt        # microseconds
    uint8   delta_velocity_clipping
    uint8   accel_calibration_count
    uint8   gyro_calibration_count
"""

import socket
import struct
import rospy
from sensor_msgs.msg import Imu


class AIDeltaSender:
    def __init__(self):
        self._load_params()
        self._setup_udp()
        self._init_state()
        rospy.Subscriber(self.imu_topic, Imu, self.on_imu)
        rospy.loginfo(
            "[ai_delta_sender] UDP -> %s:%d, alpha=%.3f, accel_id=0x%08X, gyro_id=0x%08X",
            self.udp_host,
            self.udp_port,
            self.alpha,
            self.accel_device_id,
            self.gyro_device_id,
        )

    def _load_params(self):
        """Load parameters from ROS or set defaults."""
        self.udp_host = rospy.get_param('~udp_host', '127.0.0.1')
        self.udp_port = int(rospy.get_param('~udp_port', 14560))
        self.alpha = float(rospy.get_param('~alpha', 0.2))  # Low Pass Filter coefficient [0.2]
        self.max_dt = float(rospy.get_param('~max_dt', 0.04))  # cap dt to 2x nominal 200 Hz
        self.min_dt = float(rospy.get_param('~min_dt', 5e-4))
        self.accel_device_id = int(rospy.get_param('~accel_device_id', 0xA14ACC01))
        self.gyro_device_id = int(rospy.get_param('~gyro_device_id', 0xA14A7701))
        self.gyro_bias = rospy.get_param('~gyro_bias', [0.0, 0.0, 0.001])
        self.accel_bias = rospy.get_param('~accel_bias', [0.0, 0.0, 0.001])
        self.imu_topic = rospy.get_param('~imu_topic', '/mavros/imu/data_raw')

    def _setup_udp(self):
        """Initialize UDP socket."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target = (self.udp_host, self.udp_port)

    def _init_state(self):
        """Initialize filter and state variables."""
        self.last_stamp = None
        self.filt_gyro = [0.0, 0.0, 0.0]
        self.filt_accel = [0.0, 0.0, 0.0]
        self.msg_count = 0
        self.start_time = rospy.Time.now().to_sec()
        self._last_short_dt_log = 0.0

    @staticmethod
    def _clamp(x, lo, hi):
        """Clamp value x to [lo, hi]."""
        return max(lo, min(hi, x))

    @staticmethod
    def _enu_to_frd(vec):
        """Convert ENU vector to PX4 FRD body frame."""
        return [vec[1], vec[0], -vec[2]]

    def _filter(self, new, prev):
        """First-order low-pass filter."""
        alp = self.alpha
        inv = 1.0 - alp
        return [alp * new[i] + inv * prev[i] for i in range(3)]

    def _pack_payload(self, da, dv, dt, stamp):
        """Pack IMU delta data into struct for UDP."""
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
        return struct.pack(
            fmt,
            timestamp_us,
            timestamp_sample,
            self.accel_device_id,
            self.gyro_device_id,
            da[0], da[1], da[2],
            dv[0], dv[1], dv[2],
            delta_angle_dt,
            delta_velocity_dt,
            delta_angle_clipping,
            delta_velocity_clipping,
            accel_calibration_count,
            gyro_calibration_count,
        )

    def on_imu(self, imu: Imu):
        """IMU callback: process, filter, convert, and send UDP."""
        stamp = imu.header.stamp.to_sec() if imu.header.stamp and imu.header.stamp.to_sec() > 0 else rospy.Time.now().to_sec()
        if self.last_stamp is None:
            self.last_stamp = stamp
            # Initialize filters with first sample (converted to FRD)
            self.filt_gyro = self._enu_to_frd([
                imu.angular_velocity.x,
                imu.angular_velocity.y,
                imu.angular_velocity.z,
            ])
            self.filt_accel = self._enu_to_frd([
                imu.linear_acceleration.x,
                imu.linear_acceleration.y,
                imu.linear_acceleration.z,
            ])
            return
        dt = stamp - self.last_stamp
        now_sec = rospy.Time.now().to_sec()
        # Drop samples with too short/negative dt
        if dt <= 0.0 or dt < self.min_dt:
            if now_sec - self._last_short_dt_log >= 5.0:
                rospy.logwarn(
                    f"[ai_delta_sender] skipping short/negative dt {dt * 1e6:.3f}us "
                    f"(min={self.min_dt * 1e6:.3f}us)"
                )
                self._last_short_dt_log = now_sec
            return
        self.last_stamp = stamp
        dt = self._clamp(dt, self.min_dt, self.max_dt)
        # Remove bias and convert to FRD
        g_enu = [imu.angular_velocity.x - self.gyro_bias[0],
                 imu.angular_velocity.y - self.gyro_bias[1],
                 imu.angular_velocity.z - self.gyro_bias[2]]
        a_enu = [imu.linear_acceleration.x - self.accel_bias[0],
                 imu.linear_acceleration.y - self.accel_bias[1],
                 imu.linear_acceleration.z - self.accel_bias[2]]
        g = self._enu_to_frd(g_enu)
        a = self._enu_to_frd(a_enu)
        # Low-pass filter
        self.filt_gyro = self._filter(g, self.filt_gyro)
        self.filt_accel = self._filter(a, self.filt_accel)
        # Integrate to deltas
        da = [self.filt_gyro[i] * dt for i in range(3)]
        dv = [self.filt_accel[i] * dt for i in range(3)]
        # Pack and send
        payload = self._pack_payload(da, dv, dt, stamp)
        try:
            self.sock.sendto(payload, self.target)
            self.msg_count += 1
            if self.msg_count % 200 == 0:
                elapsed = rospy.Time.now().to_sec() - self.start_time
                rate = self.msg_count / elapsed if elapsed > 0 else 0.0
                rospy.loginfo(f"[ai_delta_sender] sent={self.msg_count}, rate={rate:.1f} Hz, dt={dt*1e3:.2f} ms")
        except Exception as e:
            rospy.logwarn(f"[ai_delta_sender] UDP send failed: {e}")

    def spin(self):
        """Start ROS spin loop."""
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('ai_delta_sender')
    node = AIDeltaSender()
    node.spin()
