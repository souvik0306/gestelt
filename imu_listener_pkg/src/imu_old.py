#!/usr/bin/env python3
import rospy
import numpy as np
import onnxruntime as ort
import pickle
from sensor_msgs.msg import Imu

# --- configuration ---
SEQLEN      = 200                      # samples per inference call (>=10)
ONNX_PATH   = "airimu_euroc.onnx"
PICKLE_PATH = "net_output.pickle"
INTERVAL    = 9                        # network context size

onnx_model = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

# buffers
time_buf, acc_buf, gyro_buf = [], [], []
results = []                           # accumulate inference chunks

def run_inference():
    """Prepare tensors, run ONNX model, print and store results."""
    global results

    # assemble numpy arrays
    time = np.array(time_buf, dtype=np.float64)
    acc  = np.array(acc_buf,  dtype=np.float32)
    gyro = np.array(gyro_buf, dtype=np.float32)

    # compute dt and trim to match acc/gyro length
    dt = np.diff(time)[..., None]
    acc  = acc[:-1]
    gyro = gyro[:-1]

    # expand batch dimension
    acc_b  = acc[None, ...]            # (1, T-1, 3)
    gyro_b = gyro[None, ...]

    # run ONNX model
    corr_acc, corr_gyro = onnx_model.run(None, {"acc": acc_b, "gyro": gyro_b})

    # compute corrected signals (skip initial INTERVAL samples)
    corrected_acc  = acc_b[:, INTERVAL:, :]  + corr_acc
    corrected_gyro = gyro_b[:, INTERVAL:, :] + corr_gyro
    dt_trim        = dt[INTERVAL:, :]

    # print last corrected sample
    rospy.loginfo(f"Corrected accel: {corrected_acc[0, -1]}")
    rospy.loginfo(f"Corrected gyro:  {corrected_gyro[0, -1]}")
    rospy.loginfo("-----------------------")

    # store chunk and dump pickle
    results.append({
        "correction_acc":  corr_acc[0],
        "correction_gyro": corr_gyro[0],
        "corrected_acc":   corrected_acc[0],
        "corrected_gyro":  corrected_gyro[0],
        "dt":              dt_trim,
    })
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

def imu_callback(msg: Imu):
    """Collect IMU samples and trigger inference when buffer is full."""
    t = msg.header.stamp.to_sec()
    acc = (msg.linear_acceleration.x,
           msg.linear_acceleration.y,
           msg.linear_acceleration.z)
    gyro = (msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z)

    time_buf.append(t)
    acc_buf.append(acc)
    gyro_buf.append(gyro)

    if len(time_buf) >= SEQLEN:
        run_inference()

        # keep last INTERVAL samples for next window
        time_buf[:] = time_buf[-INTERVAL:]
        acc_buf[:]  = acc_buf[-INTERVAL:]
        gyro_buf[:] = gyro_buf[-INTERVAL:]

def listener():
    rospy.init_node("imu_inference_node")
    rospy.Subscriber("/imu0", Imu, imu_callback, queue_size=1000)
    rospy.spin()

if __name__ == "__main__":
    listener()
