#!/usr/bin/env python3
import rospy
from mavros_msgs.msg import HilSensor
import random

def fake_imu_publisher():
    rospy.init_node("fake_imu_publisher")
    pub = rospy.Publisher("/mavros/hil/imu_ned", HilSensor, queue_size=10)
    rate = rospy.Rate(200)  # 200 Hz

    while not rospy.is_shutdown():
        msg = HilSensor()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "imu_link"

        # Random accelerometer around gravity
        msg.acc.x = random.uniform(-0.1, 0.1)
        msg.acc.y = random.uniform(-0.1, 0.1)
        msg.acc.z = 9.81 + random.uniform(-0.1, 0.1)

        # Random small gyro rates
        msg.gyro.x = random.uniform(-0.01, 0.01)
        msg.gyro.y = random.uniform(-0.01, 0.01)
        msg.gyro.z = random.uniform(-0.01, 0.01)

        # Zero magnetometer for now
        msg.mag.x = 0.0
        msg.mag.y = 0.0
        msg.mag.z = 0.0

        msg.abs_pressure = 0.0
        msg.diff_pressure = 0.0
        msg.pressure_alt = 0.0
        msg.temperature = 0.0
        msg.fields_updated = 63

        pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    try:
        fake_imu_publisher()
    except rospy.ROSInterruptException:
        pass
