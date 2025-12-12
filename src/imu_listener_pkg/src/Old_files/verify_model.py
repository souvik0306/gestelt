#!/usr/bin/env python3
"""
Verify INT8 Quantized ONNX Model Format and Test Inference Speed
Using RealtimeIMUInference module for realistic benchmarking
"""

import numpy as np
import time
import os

# Import the actual inference module
from realtime_imu_inference import RealtimeIMUInference

# Get INT8 model path
script_dir = os.path.dirname(os.path.abspath(__file__))
pkg_path = os.path.dirname(script_dir)
int8_model_path = os.path.join(pkg_path, "models", "airimu_cpu_fp32_new.onnx")
# fp32_model_path = os.path.join(pkg_path, "models", "airimu_cpu_fp32.onnx")

print("="*80)
print("ONNX Model Verification Using RealtimeIMUInference")
print("="*80)
model_path = int8_model_path
# # Select model
# if os.path.exists(int8_model_path):
#     model_path = int8_model_path
#     print(f"✓ Using INT8 quantized model")
# elif os.path.exists(fp32_model_path):
#     model_path = fp32_model_path
#     print(f"⚠ Using FP32 model (INT8 not found)")
# else:
#     print("ERROR: No model found!")
#     print(f"  INT8 path: {int8_model_path}")
#     print(f"  FP32 path: {fp32_model_path}")
#     exit(1)

print(f"Model path: {model_path}")
print()

# Initialize inference module with verbose output
print("Initializing RealtimeIMUInference...")
print("-"*80)
inference = RealtimeIMUInference(
    model_path=model_path,
    seqlen=1,
    interval=9,
    verbose=False  # Disable verbose logging
)
print("✓ Model loaded successfully")
print("-"*80)
print()

# Test with realistic IMU data
print("="*80)
print("Testing Inference with Realistic IMU Data Stream")
print("="*80)

# Simulate realistic IMU samples
# These values simulate a drone hovering with small movements
test_samples = [
    # (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
    (0.05, -0.03, 9.81, 0.001, -0.002, 0.0005),
    (0.08, -0.02, 9.80, 0.002, -0.001, 0.0008),
    (-0.02, 0.04, 9.82, -0.001, 0.003, -0.0002),
    (0.10, 0.01, 9.79, 0.003, 0.001, 0.0010),
    (0.03, -0.05, 9.81, 0.001, -0.002, 0.0003),
]

print(f"Processing {len(test_samples)} IMU samples for functionality test...")

# Process samples (no detailed output, just verify it works)
for i, (ax, ay, az, gx, gy, gz) in enumerate(test_samples):
    acc_in = np.array([ax, ay, az], dtype=np.float32)
    gyro_in = np.array([gx, gy, gz], dtype=np.float32)
    
    corrected_acc, corrected_gyro = inference.inference_airimu(acc_in, gyro_in)

print("✓ All samples processed successfully")
print()
print("="*80)
print("Performance Benchmark (1000 iterations)")
print("="*80)

# Run inference multiple times to measure speed
num_runs = 1000
print(f"Running {num_runs} inferences...")
times = []

# Create test data for benchmarking
test_acc = np.array([0.1, -0.2, 9.8], dtype=np.float32)
test_gyro = np.array([0.01, -0.02, 0.005], dtype=np.float32)

# Warm-up runs (important for accurate timing)
print("Warming up...")
for _ in range(10):
    _ = inference.inference_airimu(test_acc, test_gyro)

print("Running benchmark...")
for i in range(num_runs):
    t_start = time.perf_counter()
    
    corrected_acc, corrected_gyro = inference.inference_airimu(test_acc, test_gyro)
    
    t_end = time.perf_counter()
    times.append((t_end - t_start) * 1000.0)  # Convert to ms

# Performance stats
times = np.array(times)
print()
print(f"Benchmark completed ({num_runs} runs)")
print("-"*80)
print(f"Average time:  {times.mean():.3f} ms")
print(f"Median time:   {np.median(times):.3f} ms")
print(f"Min time:      {times.min():.3f} ms")
print(f"Max time:      {times.max():.3f} ms")
print(f"Std dev:       {times.std():.3f} ms")
print(f"P95 time:      {np.percentile(times, 95):.3f} ms")
print(f"P99 time:      {np.percentile(times, 99):.3f} ms")
print()

# Calculate throughput
throughput = 1000.0 / times.mean()
print(f"Inference throughput: {throughput:.1f} Hz")
print()

# Check if fast enough for real-time
target_time_ms = 4.0  # For 250Hz IMU
safe_time_ms = 3.0    # Safety margin
print(f"Target time for 250Hz IMU: {target_time_ms:.1f} ms")
print(f"Recommended safe time: {safe_time_ms:.1f} ms (with margin)")
print()

if times.mean() < safe_time_ms:
    speedup = safe_time_ms / times.mean()
    print(f"✓✓ Model is EXCELLENT for real-time processing! ({speedup:.1f}x faster than needed)")
elif times.mean() < target_time_ms:
    print(f"✓ Model is FAST ENOUGH for real-time processing!")
else:
    print(f"✗ Model is TOO SLOW (need {target_time_ms/times.mean():.1f}x speedup)")
    print(f"  Consider:")
    print(f"  - Using INT8 quantized model")
    print(f"  - Hardware acceleration (NPU/Edge TPU)")
    print(f"  - Reduce seqlen or simplify architecture")

print()
print("="*80)
print("Module Statistics from RealtimeIMUInference")
print("="*80)

# Get inference module statistics
stats = inference.get_statistics()
print(f"Total inferences:        {stats['inference_count']}")
print(f"Total samples processed: {stats['total_samples_processed']}")
print(f"Avg inference time:      {stats['avg_inference_time_ms']:.3f} ms")
print(f"Max inference time:      {stats['max_inference_time_ms']:.3f} ms")

print()
print("="*80)
print("Verification Complete!")
print("="*80)
