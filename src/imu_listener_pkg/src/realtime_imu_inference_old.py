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
from typing import Tuple, Optional
import time


class PaddingGenerator:
    """
    Generates synthetic padding frames for ONNX inference (optimized).

    Padding simulates stationary IMU at the start of each window.
    """

    def __init__(self, interval: int = 9, gravity: float = 9.81007):
        """Initialize padding generator with pre-allocated arrays."""
        self.interval = interval
        # Pre-generate padding arrays for maximum efficiency
        self.pad_acc = np.zeros((interval, 3), dtype=np.float32)
        self.pad_acc[:, 2] = gravity  # gravity in z-axis
        self.pad_gyro = np.zeros((interval, 3), dtype=np.float32)


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
        self.padding_gen = PaddingGenerator(interval, gravity)
        self.onnx_session = None

        # Pre-allocate arrays for zero-copy operations (seqlen=1 assumed)
        self._acc_batch = np.zeros((1, 1 + interval, 3), dtype=np.float32)
        self._gyro_batch = np.zeros((1, 1 + interval, 3), dtype=np.float32)

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
            # Maximum optimization
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Single thread is often faster for small models (less overhead)
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1

            # Sequential execution (less overhead for small models)
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            # Enable CPU optimizations
            session_options.enable_cpu_mem_arena = True
            session_options.enable_mem_pattern = True
            session_options.enable_mem_reuse = True

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
        Run ONNX inference on buffered IMU data (optimized for speed).

        Args:
            acc: [N, 3] accelerometer data
            gyro: [N, 3] gyroscope data

        Returns:
            Tuple of (corrected_acc, corrected_gyro), each [N, 3]
        """
        # Timing (only if verbose)
        if self.verbose:
            t_start = time.perf_counter()

        # Use pre-allocated arrays (zero-copy for seqlen=1)
        # Fill padding (first 9 rows)
        self._acc_batch[0, :self.interval, :] = self.padding_gen.pad_acc
        self._gyro_batch[0, :self.interval, :] = self.padding_gen.pad_gyro

        # Fill real data (last row for seqlen=1)
        self._acc_batch[0, self.interval:, :] = acc
        self._gyro_batch[0, self.interval:, :] = gyro

        # Run ONNX inference (no memory allocation)
        corr_acc, corr_gyro = self.onnx_session.run(
            None,
            {"acc": self._acc_batch, "gyro": self._gyro_batch}
        )

        # Apply corrections in-place
        acc += corr_acc[0]
        gyro += corr_gyro[0]

        # Statistics (only if verbose)
        if self.verbose:
            t_end = time.perf_counter()
            inference_time_ms = (t_end - t_start) * 1000.0
            self.inference_count += 1
            self.total_samples_processed += acc.shape[0]
            self.total_inference_time_ms += inference_time_ms
            self.max_inference_time_ms = max(self.max_inference_time_ms, inference_time_ms)

            if inference_time_ms > 5.0:
                print(f"[WARNING] Slow inference: {inference_time_ms:.2f}ms")

        return acc, gyro

    def inference_airimu(self,
                        acc: np.ndarray,
                        gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process single IMU sample and return corrected values (optimized for speed).

        Args:
            acc: [3] acceleration vector (m/s^2)
            gyro: [3] gyroscope vector (rad/s)

        Returns:
            Tuple of (corrected_acc, corrected_gyro), each [3]
        """
        # Ensure inputs are numpy arrays with correct shape (avoid copy if possible)
        if not isinstance(acc, np.ndarray):
            acc = np.asarray(acc, dtype=np.float32).reshape(1, 3)
        else:
            acc = acc.reshape(1, 3)

        if not isinstance(gyro, np.ndarray):
            gyro = np.asarray(gyro, dtype=np.float32).reshape(1, 3)
        else:
            gyro = gyro.reshape(1, 3)

        # Run inference directly (no buffering for seqlen=1)
        corrected_acc, corrected_gyro = self._run_inference(acc, gyro)

        # Return flattened results
        return corrected_acc[0], corrected_gyro[0]

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

