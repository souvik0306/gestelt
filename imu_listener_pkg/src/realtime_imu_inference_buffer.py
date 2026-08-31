#!/usr/bin/env python3
"""
Real-time IMU Inference Module (No ROS Dependencies)

This module provides a clean, modular interface for running AirIMU ONNX inference
on streaming IMU data without any ROS dependencies.

Features:
- Class-based design for easy integration
- Automatic padding generation for ONNX model
- Efficient buffering and batching
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

    def __init__(self, interval: int = 9, gravity: float = 9.81007, max_seqlen: int = 500):
        """
        Initialize padding generator with pre-allocated output buffer.

        Args:
            interval: Number of padding frames to generate
            gravity: Gravity magnitude in m/s^2
            max_seqlen: Maximum sequence length (for buffer pre-allocation)
        """
        self.interval = interval
        self.gravity = np.array([0.0, 0.0, gravity], dtype=np.float32)

        # Pre-generate padding arrays for efficiency
        self.pad_acc = np.tile(self.gravity, (self.interval, 1))
        self.pad_gyro = np.zeros((self.interval, 3), dtype=np.float32)
        
        # Pre-allocate output buffers to avoid vstack allocation
        total_len = interval + max_seqlen
        self.acc_padded_buf = np.zeros((total_len, 3), dtype=np.float32)
        self.gyro_padded_buf = np.zeros((total_len, 3), dtype=np.float32)
        # Copy padding to start of buffer once
        self.acc_padded_buf[:interval] = self.pad_acc
        self.gyro_padded_buf[:interval] = self.pad_gyro

    def generate(self, acc_samples: np.ndarray, gyro_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate padding and concatenate with real samples using pre-allocated buffer.
        
        Avoids vstack allocation by copying into pre-allocated array.

        Args:
            acc_samples: [N, 3] real accelerometer data
            gyro_samples: [N, 3] real gyroscope data

        Returns:
            Tuple of (acc_padded, gyro_padded), each [N+interval, 3]
        """
        N = acc_samples.shape[0]
        total_len = self.interval + N
        
        # Copy real samples after padding (padding already in buffer)
        self.acc_padded_buf[self.interval:total_len] = acc_samples
        self.gyro_padded_buf[self.interval:total_len] = gyro_samples
        
        # Return view of used portion
        return self.acc_padded_buf[:total_len], self.gyro_padded_buf[:total_len]


class IMUBuffer:
    """
    Efficient circular buffer for IMU samples using ring buffer design.

    Uses a head pointer to track newest sample position, eliminating expensive shifts.
    Maintains a rolling window of SEQLEN samples, initialized with zeros.
    
    Timeline:
    - Samples 1-99: Buffer partially filled with zeros at start
    - Sample 100: Buffer FULLY filled with real data (ready() returns True from now on)
    - Sample 101+: Rolling FIFO, oldest sample overwritten by newest
    """

    def __init__(self, seqlen: int):
        """
        Initialize circular buffer with zeros.

        Args:
            seqlen: Fixed buffer size (e.g., 250 samples)
        """
        self.seqlen = seqlen
        # Initialize with zeros: [seqlen, 3]
        self.acc_buf = np.zeros((seqlen, 3), dtype=np.float32)
        self.gyro_buf = np.zeros((seqlen, 3), dtype=np.float32)
        self.head = 0  # Index of next write position (circular)
        self.sample_count = 0  # Track how many real samples we've received
        
        # Pre-allocate linearization buffer to avoid vstack allocations
        self.acc_linear = np.zeros((seqlen, 3), dtype=np.float32)
        self.gyro_linear = np.zeros((seqlen, 3), dtype=np.float32)

    def add(self, acc: np.ndarray, gyro: np.ndarray):
        """
        Add new IMU sample to circular buffer (O(1) operation).
        
        Overwrites the oldest sample position with newest data.
        No memory copying or shifting required.

        Args:
            acc: [3] acceleration vector
            gyro: [3] gyroscope vector
        """
        # Write to current head position
        self.acc_buf[self.head] = acc
        self.gyro_buf[self.head] = gyro
        
        # Advance head pointer (circular wrap)
        self.head = (self.head + 1) % self.seqlen
        self.sample_count += 1

    def is_filled(self) -> bool:
        """Check if buffer has received all 100 real samples (initial fill complete)."""
        return self.sample_count >= self.seqlen

    def ready(self) -> bool:
        """Always ready since buffer is always available (with zeros initially)."""
        return True

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get buffered data as linearized numpy arrays in chronological order.
        
        Reorders circular buffer so oldest sample is at index 0, newest at index -1.
        This is required for correct temporal ordering in the neural network.

        Returns:
            Tuple of (acc, gyro), each [seqlen, 3] in chronological order
        """
        # Linearize circular buffer using pre-allocated buffer (avoids vstack allocation)
        if self.sample_count >= self.seqlen:
            # Buffer is full - reorder from head to create chronological sequence
            # Copy in two chunks: [head:end] then [0:head]
            tail_len = self.seqlen - self.head
            self.acc_linear[:tail_len] = self.acc_buf[self.head:]
            self.acc_linear[tail_len:] = self.acc_buf[:self.head]
            self.gyro_linear[:tail_len] = self.gyro_buf[self.head:]
            self.gyro_linear[tail_len:] = self.gyro_buf[:self.head]
            
            return self.acc_linear, self.gyro_linear
        else:
            # Buffer not full yet - samples are already in order [0:head]
            return self.acc_buf, self.gyro_buf

    def get_fill_percentage(self) -> float:
        """Get percentage of buffer filled with real samples."""
        return min(100.0, (self.sample_count / self.seqlen) * 100.0)


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
                 seqlen: int = 250,
                 interval: int = 9,
                 gravity: float = 9.81007,
                 verbose: bool = False):
        """
        Initialize inference engine.

        Args:
            model_path: Path to ONNX model file
            log_path: Optional path to log file for debugging
            seqlen: Fixed buffer size (default: 100 samples)
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
        self.padding_gen = PaddingGenerator(interval, gravity, max_seqlen=seqlen)
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
            session_options.intra_op_num_threads = 2  # Reduced from cpu_count() to avoid contention
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # Sequential for lower latency

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
        # Unpack all four outputs from ONNX model
        corr_acc, corr_gyro, acc_var, gyro_var = self.onnx_session.run(
            None,  # Return all outputs
            {"acc": acc_batch, "gyro": gyro_batch}
        )

        # Apply corrections: corrected = original + correction
        # Note: all outputs are [1, N, 3] — strip batch dim with [0]
        corrected_acc = acc + corr_acc[0]    # [N, 3]
        corrected_gyro = gyro + corr_gyro[0]  # [N, 3]
        acc_var  = acc_var[0]   # [N, 3]
        gyro_var = gyro_var[0]  # [N, 3]

        # End timing
        t_end = time.perf_counter()
        inference_time_ms = (t_end - t_start) * 1000.0

        # Update statistics
        self.inference_count += 1
        self.total_samples_processed += acc.shape[0]
        self.total_inference_time_ms += inference_time_ms
        self.max_inference_time_ms = max(self.max_inference_time_ms, inference_time_ms)

        # # Warn if inference is too slow (>5ms is risky for 250Hz IMU)
        # if inference_time_ms > 5.0 and self.verbose:
        #     print(f"[WARNING] Slow inference: {inference_time_ms:.2f}ms")

        # Return all four outputs for downstream use
        return corrected_acc, corrected_gyro, acc_var, gyro_var

    def add_sample(self, acc: np.ndarray, gyro: np.ndarray):
        """
        Add one IMU sample to the rolling buffer without running ONNX inference.

        This is useful when the input stream is high rate, but the model should
        run at a lower output rate, for example feeding 200 Hz samples while
        running inference every fifth sample.
        """
        acc = np.asarray(acc, dtype=np.float32).reshape(3)
        gyro = np.asarray(gyro, dtype=np.float32).reshape(3)
        self.buffer.add(acc, gyro)

    def infer_current_buffer(self):
        """
        Run ONNX inference on the current rolling buffer.

        Returns only the latest corrected sample, plus the full variance arrays,
        matching the return format of inference_airimu.
        """
        acc_batch, gyro_batch = self.buffer.get_arrays()

        corrected_acc_batch, corrected_gyro_batch, acc_var_all, gyro_var_all = self._run_inference(
            acc_batch, gyro_batch
        )

        return (
            corrected_acc_batch[-1],
            corrected_gyro_batch[-1],
            acc_var_all,
            gyro_var_all
        )

    def inference_airimu(self,
                        acc: np.ndarray,
                        gyro: np.ndarray):
        """
        Backward compatible single sample API.

        Adds one IMU sample to the rolling buffer, runs ONNX inference
        immediately, and returns the latest corrected sample plus variance arrays.
        """
        self.add_sample(acc, gyro)
        return self.infer_current_buffer()

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
