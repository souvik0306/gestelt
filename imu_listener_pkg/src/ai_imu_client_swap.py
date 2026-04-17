#!/usr/bin/env python3
"""
AI IMU Client — ROS Topic Edition

Subscribe /mavros/imu/data_raw → ONNX inference → publish ai_msgs/ImuNoise
on /mavros/ai/imu_noise
"""

import os, sys, time, signal
import numpy as np
from collections import deque
from typing import Optional

import rospy
from sensor_msgs.msg import Imu
from ai_msgs.msg import ImuNoise

from realtime_imu_inference_buffer import RealtimeIMUInference

# ── Config ────────────────────────────────────
BUFFER_SIZE      = 200
STATS_INTERVAL_S = 5.0
# Test mode: feed MAVROS Z as model X and MAVROS X as model Z.
# Output uncertainty is swapped back to MAVROS axis order before publish.


# ── Helpers ───────────────────────────────────
def _load_model(buffer_size: int) -> RealtimeIMUInference:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(os.path.dirname(script_dir),
                              "models", "airimu_cpu_fp32_cov_200.onnx")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    print(f"[AI Client] Model: {model_path}")
    return RealtimeIMUInference(model_path, seqlen=buffer_size, verbose=False)


# ── Main class ────────────────────────────────
class AIClientROS:
    def __init__(self):
        self.running          = False
        self.sequence_counter = 0
        self.process_queue    = deque(maxlen=100)

        self.stats = dict(
            samples_received=0,
            samples_processed=0,
            samples_published=0,
            corrupted_outputs=0,
            samples_dropped=0,
            start_time=None,
            last_stats_time=None,
            last_samples_received=0,
            last_samples_processed=0,
            total_ai_time_ms=0.0,
        )

        self.inference = _load_model(BUFFER_SIZE)

        rospy.init_node('ai_imu_client', anonymous=True)

        # Publisher (your custom ROS topic)
        self.ai_noise_pub = rospy.Publisher(
            '/mavros/ai/imu_noise',
            ImuNoise,
            queue_size=10
        )

        # Subscriber (raw IMU)
        self.imu_sub = rospy.Subscriber(
            '/mavros/imu/data_raw',
            Imu,
            self._imu_callback,
            queue_size=10
        )

        print(f"[AI Client] buffer={BUFFER_SIZE}")
        print("[AI Client] sub=/mavros/imu/data_raw")
        print("[AI Client] pub=/mavros/ai/imu_noise")

    # ── ROS callback ──────────────────────────
    def _imu_callback(self, msg: Imu):
        try:
            sample = {
                'timestamp_us': int(msg.header.stamp.to_sec() * 1e6),
                'seq': self.sequence_counter,
                'gyro': np.array([
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ], dtype=np.float32),
                'accel': np.array([
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ], dtype=np.float32),
            }

            self.sequence_counter += 1
            self.stats['samples_received'] += 1

            if len(self.process_queue) >= self.process_queue.maxlen:
                self.stats['samples_dropped'] += 1

            self.process_queue.append(sample)

        except Exception as e:
            rospy.logerr(f"[AI Client] callback error: {e}")

    # ── Inference ─────────────────────────────
    def _infer(self, sample: dict) -> Optional[dict]:
        acc, gyro = sample['accel'], sample['gyro']

        if not (np.isfinite(acc).all() and np.isfinite(gyro).all()):
            rospy.logwarn("[AI Client] non-finite input, skipping")
            return None

        # Remap MAVROS [x, y, z] -> model [z, y, x]
        acc_model = acc[[2, 1, 0]]
        gyro_model = gyro[[2, 1, 0]]

        t0 = time.time()
        _, _, acc_var_all, gyro_var_all = self.inference.inference_airimu(acc_model, gyro_model)
        ai_ms = (time.time() - t0) * 1000.0
        self.stats['total_ai_time_ms'] += ai_ms

        ai_acc_noise_model  = np.asarray(acc_var_all[-1],  dtype=np.float64) #this is the last element of the returned list, which corresponds to the most recent prediction
        ai_gyro_noise_model = np.asarray(gyro_var_all[-1], dtype=np.float64) #same for gyro

        # Map model output [z, y, x] back to MAVROS [x, y, z]
        ai_acc_noise = ai_acc_noise_model[[2, 1, 0]]
        ai_gyro_noise = ai_gyro_noise_model[[2, 1, 0]]

        if not (np.isfinite(ai_acc_noise).all() and np.isfinite(ai_gyro_noise).all()):
            rospy.logerr("[AI Client] NaN/Inf output, dropping")
            self.stats['corrupted_outputs'] += 1
            return None

        return {
            'ai_acc_noise': ai_acc_noise,
            'ai_gyro_noise': ai_gyro_noise,
            'ai_ms': ai_ms
        }

    # ── ROS publish ───────────────────────────
    def _publish_ros_noise(self, noise: dict):
        msg = ImuNoise()
        msg.accel_noise = noise['ai_acc_noise'].tolist()
        msg.gyro_noise  = noise['ai_gyro_noise'].tolist()

        self.ai_noise_pub.publish(msg)

        n = self.stats['samples_published']
        if n == 0 or n % 500 == 0:
            an, gn = noise['ai_acc_noise'], noise['ai_gyro_noise']
            print(f"[ROS #{n}] accel=[{an[0]:.3e},{an[1]:.3e},{an[2]:.3e}] "
                  f"gyro=[{gn[0]:.3e},{gn[1]:.3e},{gn[2]:.3e}] "
                  f"ai={noise['ai_ms']:.1f}ms")

        self.stats['samples_published'] += 1

    # ── Queue drain ───────────────────────────
    def _drain_queue(self):
        while self.process_queue:
            sample = self.process_queue.popleft()
            noise  = self._infer(sample)

            if noise:
                self._publish_ros_noise(noise)
                self.stats['samples_processed'] += 1

    # ── Stats ─────────────────────────────────
    def _print_stats(self):
        now    = time.time()
        uptime = now - self.stats['start_time']
        dt     = now - (self.stats['last_stats_time'] or now)

        rx   = self.stats['samples_received']
        proc = self.stats['samples_processed']
        pub  = self.stats['samples_published']

        rx_hz   = (rx   - self.stats['last_samples_received']) / dt if dt > 0 else 0
        proc_hz = (proc - self.stats['last_samples_processed']) / dt if dt > 0 else 0
        avg_ai  = self.stats['total_ai_time_ms'] / proc if proc > 0 else 0

        buf_pct = self.inference.buffer.get_fill_percentage()

        print(f"\n{'='*60}")
        print(f"[AI] up={uptime:.0f}s  rx={rx}({rx_hz:.0f}Hz)  "
              f"proc={proc}({proc_hz:.0f}Hz)  pub={pub}")
        print(f"     buf={buf_pct:.0f}%  ai={avg_ai:.1f}ms  "
              f"drop={self.stats['samples_dropped']}  "
              f"corrupt={self.stats['corrupted_outputs']}")
        print(f"{'='*60}")

        self.stats.update(
            last_samples_received=rx,
            last_samples_processed=proc,
            last_stats_time=now
        )

    # ── Run ───────────────────────────────────
    def run(self):
        self.running = True
        self.stats['start_time'] = self.stats['last_stats_time'] = time.time()

        print("[AI Client] running ...")

        rate = rospy.Rate(250)

        while self.running and not rospy.is_shutdown():
            self._drain_queue()

            if time.time() - self.stats['last_stats_time'] >= STATS_INTERVAL_S:
                self._print_stats()

            rate.sleep()

        self._shutdown()

    def _shutdown(self):
        if hasattr(self, 'inference'):
            self.inference.close()
        print("[AI Client] stopped.")


# ── Entry point ───────────────────────────────
def _on_signal(sig, frame):
    print("\n[AI Client] interrupted.")
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, _on_signal)

    client = AIClientROS()

    try:
        client.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"[AI Client] fatal: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client._shutdown()