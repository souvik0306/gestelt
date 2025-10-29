#!/usr/bin/env python3
"""
Verify INT8 Quantized ONNX Model Format and Test Inference Speed
"""

import numpy as np
import onnxruntime as ort
import time
import os

# Get INT8 model path
script_dir = os.path.dirname(os.path.abspath(__file__))
pkg_path = os.path.dirname(script_dir)
model_path = os.path.join(pkg_path, "models", "airimu_cpu_fp32_int8.onnx")

print("="*80)
print("INT8 Quantized ONNX Model Verification")
print("="*80)
print(f"Model path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")
print()

if not os.path.exists(model_path):
    print("ERROR: INT8 model not found!")
    print(f"Please ensure the model exists at: {model_path}")
    exit(1)

# Load model with INT8 optimizations
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
session_options.intra_op_num_threads = max(2, (os.cpu_count() or 4))
session_options.inter_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# Enable memory optimizations for INT8
session_options.enable_mem_pattern = True
session_options.enable_cpu_mem_arena = True

# Configure CPU execution provider with INT8 optimizations
cpu_options = {
    'enable_quantized_int8_kernels': '1',
    'arena_extend_strategy': 'kSameAsRequested',
}

session = ort.InferenceSession(
    model_path,
    sess_options=session_options,
    providers=['CPUExecutionProvider'],
    provider_options=[cpu_options]
)

print("MODEL INPUTS:")
input_dtype = np.float32  # Default
is_quantized = False

for inp in session.get_inputs():
    print(f"  Name: {inp.name}")
    print(f"  Shape: {inp.shape}")
    print(f"  Type: {inp.type}")
    
    # Detect quantization
    if 'int8' in inp.type.lower():
        is_quantized = True
        input_dtype = np.int8
        print(f"  ✓ INT8 Quantized Input Detected")
    elif 'uint8' in inp.type.lower():
        is_quantized = True
        input_dtype = np.uint8
        print(f"  ✓ UINT8 Quantized Input Detected")
    else:
        print(f"  ℹ FP32 Input (not quantized)")
    print()

print("MODEL OUTPUTS:")
for out in session.get_outputs():
    print(f"  Name: {out.name}")
    print(f"  Shape: {out.shape}")
    print(f"  Type: {out.type}")
    print()

print("QUANTIZATION STATUS:")
if is_quantized:
    print(f"  ✓ Model is INT8 quantized")
    print(f"  Input dtype: {input_dtype}")
else:
    print(f"  ℹ Model appears to be FP32 (not quantized)")
    print(f"  Note: This script is optimized for INT8 models")
print()

# Test with realistic IMU data
print("="*80)
print("Testing INT8 Inference with Realistic IMU Data")
print("="*80)

# Create test data (seqlen=1, interval=9, so total 10 samples)
# Shape: [batch=1, timesteps=10, features=3]
# Use input_dtype for proper INT8 compatibility
test_acc = np.zeros((1, 10, 3), dtype=input_dtype)
test_gyro = np.zeros((1, 10, 3), dtype=input_dtype)

# First 9 frames: padding (gravity in z-axis)
if input_dtype == np.float32:
    test_acc[0, :9, 2] = 9.81007  # gravity
    # Last frame: actual sample
    test_acc[0, 9, :] = np.array([0.1, -0.2, 9.8], dtype=input_dtype)
    test_gyro[0, 9, :] = np.array([0.01, -0.02, 0.005], dtype=input_dtype)
else:
    # For quantized models, data might need to stay in FP32 for ONNX Runtime
    # ONNX Runtime handles quantization internally
    test_acc = test_acc.astype(np.float32)
    test_gyro = test_gyro.astype(np.float32)
    test_acc[0, :9, 2] = 9.81007
    test_acc[0, 9, :] = np.array([0.1, -0.2, 9.8], dtype=np.float32)
    test_gyro[0, 9, :] = np.array([0.01, -0.02, 0.005], dtype=np.float32)

# Ensure C-contiguous for optimal INT8 performance
test_acc = np.ascontiguousarray(test_acc)
test_gyro = np.ascontiguousarray(test_gyro)

print(f"Input acc shape: {test_acc.shape}")
print(f"Input gyro shape: {test_gyro.shape}")
print(f"Input acc dtype: {test_acc.dtype}")
print(f"Input gyro dtype: {test_gyro.dtype}")
print(f"Memory layout: {'C-contiguous' if test_acc.flags['C_CONTIGUOUS'] else 'Not contiguous'}")
print(f"Input acc sample: {test_acc[0, 9, :]}")
print(f"Input gyro sample: {test_gyro[0, 9, :]}")
print()

# Run inference multiple times to measure speed
num_runs = 1000  # More runs for better INT8 statistics
print(f"Running {num_runs} inferences for performance testing...")
times = []

# Warm-up run (important for INT8 models to initialize kernels)
for _ in range(10):
    _ = session.run(None, {"acc": test_acc, "gyro": test_gyro})

for i in range(num_runs):
    t_start = time.perf_counter()

    outputs = session.run(
        None,
        {"acc": test_acc, "gyro": test_gyro}
    )

    t_end = time.perf_counter()
    times.append((t_end - t_start) * 1000.0)  # Convert to ms

# First run results
print("INFERENCE RESULTS (first run):")
print(f"Number of outputs: {len(outputs)}")
for i, out in enumerate(outputs):
    print(f"  Output {i} shape: {out.shape}")
    print(f"  Output {i} dtype: {out.dtype}")
    if out.size <= 10:
        print(f"  Output {i} values: {out}")
    else:
        print(f"  Output {i} sample: {out[0, 0, :]}")
print()

# Performance stats
times = np.array(times)
print("="*80)
print(f"INT8 INFERENCE PERFORMANCE ({num_runs} runs, after warm-up)")
print("="*80)
print(f"Average time: {times.mean():.3f} ms")
print(f"Median time:  {np.median(times):.3f} ms")
print(f"Min time:     {times.min():.3f} ms")
print(f"Max time:     {times.max():.3f} ms")
print(f"Std dev:      {times.std():.3f} ms")
print(f"P95 time:     {np.percentile(times, 95):.3f} ms")
print(f"P99 time:     {np.percentile(times, 99):.3f} ms")
print()

# Calculate throughput
throughput = 1000.0 / times.mean()
print(f"Inference throughput: {throughput:.1f} Hz")
print()

# Check if fast enough for real-time
target_time_ms = 4.0  # For 250Hz IMU
safe_time_ms = 3.0    # Safety margin for INT8 processing
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
    print(f"  - Further model optimization (pruning, distillation)")
    print(f"  - Hardware acceleration (NPU/Edge TPU)")
    print(f"  - Reduce seqlen or simplify architecture")

# INT8 specific performance analysis
if is_quantized:
    print(f"\nINT8 PERFORMANCE BENEFITS:")
    print(f"  - Expected 2-4x speedup vs FP32")
    print(f"  - ~4x memory reduction")
    print(f"  - Lower power consumption")
    print(f"  - Better cache utilization")
else:
    print(f"\nNOTE: Model does not appear to be INT8 quantized.")
    print(f"      Consider converting to INT8 for better performance.")

print("="*80)

# Verify output format matches expectation
if len(outputs) == 2:
    corr_acc, corr_gyro = outputs
    print("✓ Model returns 2 outputs (corrections for acc and gyro)")

    if corr_acc.shape[1] == 1:
        print(f"✓ Output timesteps = 1 (matches seqlen=1)")
    else:
        print(f"✗ WARNING: Output timesteps = {corr_acc.shape[1]} (expected 1)")

    if corr_acc.shape[2] == 3 and corr_gyro.shape[2] == 3:
        print(f"✓ Output features = 3 (correct for 3-axis IMU)")
    else:
        print(f"✗ WARNING: Output features mismatch")

    # Apply correction
    original_acc = test_acc[0, 9, :]
    original_gyro = test_gyro[0, 9, :]
    corrected_acc = original_acc + corr_acc[0, 0, :]
    corrected_gyro = original_gyro + corr_gyro[0, 0, :]

    print(f"\nCORRECTION EXAMPLE:")
    print(f"  Original acc:   {original_acc}")
    print(f"  Correction:     {corr_acc[0, 0, :]}")
    print(f"  Corrected acc:  {corrected_acc}")
    print(f"  Delta:          {corrected_acc - original_acc}")
    print()
    print(f"  Original gyro:  {original_gyro}")
    print(f"  Correction:     {corr_gyro[0, 0, :]}")
    print(f"  Corrected gyro: {corrected_gyro}")
    print(f"  Delta:          {corrected_gyro - original_gyro}")
else:
    print(f"✗ WARNING: Expected 2 outputs, got {len(outputs)}")

print("="*80)
