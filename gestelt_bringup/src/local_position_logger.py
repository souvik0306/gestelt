#!/usr/bin/env python3
"""ROS node for recording MAVROS local position pose data.

This node subscribes to ``geometry_msgs/PoseStamped`` messages on a configurable
``~topic`` (default ``/mavros/local_position/pose``).  Timestamps and XYZ
positions are buffered in-memory and persisted as NumPy ``.npy`` files when the
node shuts down.  The default output directory is
``$HOME/gestelt_ws/collected_poses`` but can be overridden with parameters.
"""

from __future__ import annotations

import threading
from array import array
from pathlib import Path
from typing import Optional

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped


class LocalPositionLogger:
    """Capture timestamps and XYZ positions from a PoseStamped stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._times = array("d")
        self._xs = array("d")
        self._ys = array("d")
        self._zs = array("d")

        default_output_dir = Path.home() / "gestelt_ws" / "collected_poses"
        output_directory = Path(
            rospy.get_param("~output_directory", str(default_output_dir))
        )
        output_directory.mkdir(parents=True, exist_ok=True)

        self._time_output_path = Path(
            rospy.get_param(
                "~time_output_path", output_directory / "timestamps.npy"
            )
        )
        self._position_output_path = Path(
            rospy.get_param(
                "~position_output_path", output_directory / "positions.npy"
            )
        )
        self._log_interval = rospy.get_param("~log_interval", 100)

        topic = rospy.get_param("~topic", "/mavros/local_position/pose")
        queue_size = rospy.get_param("~queue_size", 100)

        self._count = 0
        self._last_log_count = 0

        self._subscriber = rospy.Subscriber(
            topic, PoseStamped, self._callback, queue_size=queue_size
        )
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(
            "local_position_logger listening on %s; logging to %s and %s",
            topic,
            self._time_output_path,
            self._position_output_path,
        )

    def _callback(self, msg: PoseStamped) -> None:
        position = msg.pose.position
        stamp = msg.header.stamp
        time_value = stamp.to_sec() if stamp else rospy.get_time()

        with self._lock:
            self._times.append(time_value)
            self._xs.append(position.x)
            self._ys.append(position.y)
            self._zs.append(position.z)
            self._count += 1
            current_count = self._count

        if self._log_interval and current_count - self._last_log_count >= self._log_interval:
            self._last_log_count = current_count
            rospy.loginfo("local_position_logger captured %d samples", current_count)

    def _on_shutdown(self) -> None:
        with self._lock:
            times = np.frombuffer(self._times, dtype=np.float64).copy()
            xs = np.frombuffer(self._xs, dtype=np.float64).copy()
            ys = np.frombuffer(self._ys, dtype=np.float64).copy()
            zs = np.frombuffer(self._zs, dtype=np.float64).copy()

        if times.size == 0:
            rospy.logwarn("local_position_logger: no samples received; nothing saved")
            return

        positions = np.column_stack((xs, ys, zs))

        self._time_output_path.parent.mkdir(parents=True, exist_ok=True)
        self._position_output_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(self._time_output_path, times)
        np.save(self._position_output_path, positions)

        rospy.loginfo(
            "local_position_logger wrote %d samples to %s and %s",
            times.size,
            self._time_output_path,
            self._position_output_path,
        )


def main(argv: Optional[list[str]] = None) -> None:
    rospy.init_node("local_position_logger")
    LocalPositionLogger()
    rospy.spin()


if __name__ == "__main__":
    main()
