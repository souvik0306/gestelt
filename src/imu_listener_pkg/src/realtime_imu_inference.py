#!/usr/bin/env python3
"""
Real-time IMU Inference Module (No ROS Dependencies)

This module provides a clean, modular interface for running AirIMU ONNX inference
on streaming IMU data without any ROS dependencies.

Features:
- Class-based design for easy integration
- Automatic padding generation for ONNX model
- Efficient buffering and batching
- INT8 quantized model support
- Optional logging for debugging
- Thread-safe design

Usage:
    from realtime_imu_inference import RealtimeIMUInference

    # Initialize
    inference = RealtimeIMUInference(model_path="airimu.onnx")

    # Process single sample
    corrected_acc, corrected_gyro = inference.process(acc, gyro)

    # Clean up
    inference.close()
"""

import numpy as np
import onnxruntime as ort
import os
from collections import deque
from typing import Tuple, Optional
import time


class PaddingGenerator:
    """
    Generates synthetic padding frames for ONNX inference.

    Padding simulates stationary IMU at the start of each window:
    - acc: gravity vector in body frame (assumes upright orientation)
    - gyro: zeros (stationary)
    """

    def __init__(self, interval: int = 9, gravity: float = 9.81007):
        """
        Initialize padding generator.

        Args:
            interval: Number of padding frames to generate
            gravity: Gravity magnitude in m/s^2
        """
        self.interval = interval
        self.gravity = np.array([0.0, 0.0, gravity], dtype=np.float32)

        # Pre-generate padding arrays for efficiency
        self.pad_acc = np.tile(self.gravity, (self.interval, 1))
        self.pad_gyro = np.zeros((self.interval, 3), dtype=np.float32)

    def generate(self, acc_samples: np.ndarray, gyro_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate padding and concatenate with real samples.

        Args:
            acc_samples: [N, 3] real accelerometer data
            gyro_samples: [N, 3] real gyroscope data

        Returns:
            Tuple of (acc_padded, gyro_padded), each [N+interval, 3]
        """
        # Concatenate: [padding] + [real data]
        acc_padded = np.vstack([self.pad_acc, acc_samples])
        gyro_padded = np.vstack([self.pad_gyro, gyro_samples])

        return acc_padded, gyro_padded


class IMUBuffer:
    """
    Circular buffer for accumulating IMU samples before inference.

    Accumulates SEQLEN samples before triggering inference.
    """

    def __init__(self, seqlen: int):
        """
        Initialize buffer.

        Args:
            seqlen: Number of samples to accumulate before inference
        """
        self.seqlen = seqlen
        self.acc_buf = deque(maxlen=seqlen)
        self.gyro_buf = deque(maxlen=seqlen)

    def add(self, acc: np.ndarray, gyro: np.ndarray):
        """
        Add new IMU sample to buffer.

        Args:
            acc: [3] acceleration vector
            gyro: [3] gyroscope vector
        """
        self.acc_buf.append(acc)
        self.gyro_buf.append(gyro)

    def ready(self) -> bool:
        """Check if buffer has enough samples for inference."""
        return len(self.acc_buf) >= self.seqlen

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get buffered data as numpy arrays.

        Returns:
            Tuple of (acc, gyro), each [N, 3]
        """
        acc = np.array(self.acc_buf, dtype=np.float32)
        gyro = np.array(self.gyro_buf, dtype=np.float32)
        return acc, gyro

    def clear(self):
        """Clear buffer after inference."""
        self.acc_buf.clear()
        self.gyro_buf.clear()


class RealtimeIMUInference:
    """
    Real-time IMU inference engine using ONNX model.

    This class manages the complete inference pipeline:
    1. Buffer incoming IMU samples
    2. Generate padding frames
    3. Run ONNX inference
    4. Return corrected IMU data
    """

    def __init__(self,
                 model_path: str,
                 log_path: Optional[str] = None,
                 seqlen: int = 1,
                 interval: int = 9,
                 gravity: float = 9.81007,
                 verbose: bool = False):
        """
        Initialize inference engine.

        Args:
            model_path: Path to ONNX model file
            log_path: Optional path to log file for debugging
            seqlen: Number of real samples per inference window
            interval: Number of padding frames (must match model training)
            gravity: Gravity magnitude in m/s^2
            verbose: Enable verbose logging
        """
        self.model_path = model_path
        self.seqlen = seqlen
        self.interval = interval
        self.verbose = verbose

        # Components
        self.buffer = IMUBuffer(seqlen)
        self.padding_gen = PaddingGenerator(interval, gravity)
        self.onnx_session = None

        # Statistics
        self.inference_count = 0
        self.total_samples_processed = 0
        self.total_inference_time_ms = 0.0
        self.max_inference_time_ms = 0.0

        # Initialize
        self._load_model()

    def _load_model(self):
        """Load ONNX model with optimized settings."""
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = os.cpu_count() or 4
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

            self.onnx_session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"]
            )

            if self.verbose:
                print("="*70)
                print("ONNX Model Loaded Successfully")
                print("="*70)
                print(f"Model path: {self.model_path}")

                for inp in self.onnx_session.get_inputs():
                    print(f"  Input: {inp.name}, shape: {inp.shape}, dtype: {inp.type}")
                for out in self.onnx_session.get_outputs():
                    print(f"  Output: {out.name}, shape: {out.shape}, dtype: {out.type}")

                print(f"Configuration:")
                print(f"  SEQLEN: {self.seqlen} (real samples per window)")
                print(f"  INTERVAL: {self.interval} (padding frames)")
                print(f"  Input shape: [1, {self.seqlen + self.interval}, 3]")
                print(f"  Output shape: [1, {self.seqlen}, 3]")
                print("="*70)

        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    def _run_inference(self, acc: np.ndarray, gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run ONNX inference on buffered IMU data.

        Args:
            acc: [N, 3] accelerometer data
            gyro: [N, 3] gyroscope data

        Returns:
            Tuple of (corrected_acc, corrected_gyro), each [N, 3]
        """
        # Start timing
        t_start = time.perf_counter()

        # Generate padding and concatenate
        acc_padded, gyro_padded = self.padding_gen.generate(acc, gyro)

        # Add batch dimension: [N+interval, 3] -> [1, N+interval, 3]
        acc_batch = acc_padded[np.newaxis, ...]
        gyro_batch = gyro_padded[np.newaxis, ...]

        # Run ONNX inference (ONNX Runtime handles int8 quantization internally)
        # Input: [1, N+interval, 3] -> Output: [1, N, 3]
        corr_acc, corr_gyro = self.onnx_session.run(
            None,  # Return all outputs
            {"acc": acc_batch, "gyro": gyro_batch}
        )

        # Apply corrections: corrected = original + correction
        # Note: corr_acc/gyro are [1, N, 3], acc/gyro are [N, 3]
        corrected_acc = acc + corr_acc[0]
        corrected_gyro = gyro + corr_gyro[0]

        # End timing
        t_end = time.perf_counter()
        inference_time_ms = (t_end - t_start) * 1000.0

        # Update statistics
        self.inference_count += 1
        self.total_samples_processed += acc.shape[0]
        self.total_inference_time_ms += inference_time_ms
        self.max_inference_time_ms = max(self.max_inference_time_ms, inference_time_ms)

        # Warn if inference is too slow (>5ms is risky for 250Hz IMU)
        if inference_time_ms > 5.0 and self.verbose:
            print(f"[WARNING] Slow inference: {inference_time_ms:.2f}ms")

        return corrected_acc, corrected_gyro

    def inference_airimu(self,
                        acc: np.ndarray,
                        gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process IMU sample and return corrected values.

        Args:
            acc: [3] acceleration vector (m/s^2)
            gyro: [3] gyroscope vector (rad/s)

        Returns:
            Tuple of (corrected_acc, corrected_gyro), each [3]
        """
        # Ensure inputs are numpy arrays with correct shape
        acc = np.asarray(acc, dtype=np.float32).reshape(3)
        gyro = np.asarray(gyro, dtype=np.float32).reshape(3)

        # Add to buffer
        self.buffer.add(acc, gyro)

        # Check if ready for inference
        if self.buffer.ready():
            # Get buffered data
            acc_batch, gyro_batch = self.buffer.get_arrays()

            # Run inference
            corrected_acc_batch, corrected_gyro_batch = self._run_inference(
                acc_batch, gyro_batch
            )

            # Clear buffer
            self.buffer.clear()

            # Return the last sample from the batch (most recent)
            return corrected_acc_batch[-1], corrected_gyro_batch[-1]

        else:
            # Not enough samples yet, return original (pass-through)
            return acc, gyro

    def get_statistics(self) -> dict:
        """
        Get inference statistics.

        Returns:
            Dictionary with performance metrics
        """
        avg_inference_time_ms = (
            self.total_inference_time_ms / self.inference_count
            if self.inference_count > 0 else 0.0
        )

        stats = {
            'inference_count': self.inference_count,
            'total_samples_processed': self.total_samples_processed,
            'avg_inference_time_ms': avg_inference_time_ms,
            'max_inference_time_ms': self.max_inference_time_ms,
        }

        return stats

    def print_statistics(self):
        """Print inference statistics."""
        stats = self.get_statistics()

        print("\n" + "="*70)
        print("AirIMU Inference Statistics")
        print("="*70)
        print(f"Total inferences: {stats['inference_count']}")
        print(f"Total samples processed: {stats['total_samples_processed']}")
        print(f"Average inference time: {stats['avg_inference_time_ms']:.3f} ms")
        print(f"Max inference time: {stats['max_inference_time_ms']:.3f} ms")
        print("="*70 + "\n")

    def close(self):
        """Clean up resources."""
        if self.verbose:
            self.print_statistics()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

