#!/usr/bin/env python3
"""
Simple inference speed test for AI IMU model.

Tests the model inference speed with dummy IMU data to determine
if it can keep up with real-time requirements (240 Hz = 4.2 ms per sample).
"""

import os
import time
import numpy as np
from realtime_imu_inference_buffer import RealtimeIMUInference

# Configuration
BUFFER_SIZE = 200
NUM_WARMUP = 10  # Warmup iterations (excluded from timing)
NUM_TESTS = 1000  # Number of test iterations

def main():
    # Get model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_path = os.path.dirname(script_dir)
    model_path = os.path.join(pkg_path, "models", "airimu_cpu_fp32_200.onnx")
    
    if not os.path.isfile(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return
    
    print("="*80)
    print("AI IMU Inference Speed Test")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Buffer size: {BUFFER_SIZE} samples")
    print(f"Warmup iterations: {NUM_WARMUP}")
    print(f"Test iterations: {NUM_TESTS}")
    print()
    
    # Initialize inference module
    print("Loading model...")
    inference = RealtimeIMUInference(
        model_path, 
        seqlen=BUFFER_SIZE,
        verbose=True
    )
    print("Model loaded successfully!")
    print()
    
    # Generate diverse IMU data patterns
    print("Generating diverse IMU test data...")
    np.random.seed(42)  # For reproducibility
    
    # Create realistic IMU data with variations
    # Accel: gravity (9.81) + small variations for tilt/movement
    accel_data = np.random.randn(NUM_TESTS + NUM_WARMUP, 3).astype(np.float32) * 0.5
    accel_data[:, 2] += 9.81  # Add gravity to Z-axis
    
    # Gyro: small rotation rates with variations
    gyro_data = np.random.randn(NUM_TESTS + NUM_WARMUP, 3).astype(np.float32) * 0.05
    
    print(f"Generated {NUM_TESTS + NUM_WARMUP} diverse IMU samples")
    print(f"  Accel range: [{accel_data.min():.3f}, {accel_data.max():.3f}] m/s²")
    print(f"  Gyro range:  [{gyro_data.min():.3f}, {gyro_data.max():.3f}] rad/s")
    print()
    
    # Log file for input/output shapes and buffer states
    log_file = os.path.join(script_dir, "inference_buffer_log.txt")
    with open(log_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AI IMU Inference Buffer State Log\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test configuration:\n")
        f.write(f"  Total samples: {NUM_TESTS + NUM_WARMUP}\n")
        f.write(f"  Buffer size: {BUFFER_SIZE}\n")
        f.write(f"  Input data: Diverse random IMU values\n\n")
    print(f"Logging buffer states to: {log_file}\n")
    
    # Warmup phase
    print(f"Warming up ({NUM_WARMUP} iterations)...")
    for i in range(NUM_WARMUP):
        _ = inference.inference_airimu(accel_data[i], gyro_data[i])
    print("Warmup complete!")
    print()
    
    # Test phase
    print(f"Running inference test ({NUM_TESTS} iterations)...")
    inference_times = []
    
    # Track specific timesteps for logging
    log_timesteps = {
        0: "t0 (first iteration)",
        NUM_TESTS // 2: "t0.5 (middle iteration)", 
        NUM_TESTS - 1: "t1 (last iteration)"
    }
    
    for i in range(NUM_TESTS):
        # Get diverse input for this iteration
        test_idx = NUM_WARMUP + i
        input_acc = accel_data[test_idx]
        input_gyro = gyro_data[test_idx]
        
        start_time = time.time()
        corrected_acc, corrected_gyro = inference.inference_airimu(input_acc, input_gyro)
        end_time = time.time()
        
        inference_time_ms = (end_time - start_time) * 1000
        inference_times.append(inference_time_ms)
        
        # Log buffer state at specific timesteps
        if i in log_timesteps:
            with open(log_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Iteration {i}: {log_timesteps[i]}\n")
                f.write(f"{'='*80}\n\n")
                
                # Get buffer data
                buffer = inference.buffer
                acc_buf = buffer.acc_buf
                gyro_buf = buffer.gyro_buf
                
                f.write(f"Input shapes:\n")
                f.write(f"  acc_buf shape:  {acc_buf.shape}\n")
                f.write(f"  gyro_buf shape: {gyro_buf.shape}\n")
                f.write(f"  Buffer fill:    {buffer.get_fill_percentage():.1f}%\n")
                f.write(f"  Sample count:   {buffer.sample_count}\n\n")
                
                f.write(f"Single input sample (latest):\n")
                f.write(f"  Input accel:  {input_acc}\n")
                f.write(f"  Input gyro:   {input_gyro}\n\n")
                
                f.write(f"Output sample (corrected):\n")
                f.write(f"  Output accel: {corrected_acc}\n")
                f.write(f"  Output gyro:  {corrected_gyro}\n\n")
                
                f.write(f"Full buffer contents (all {BUFFER_SIZE} samples):\n")
                f.write(f"\nAccelerometer buffer (shape {acc_buf.shape}):\n")
                for j in range(BUFFER_SIZE):
                    f.write(f"  Sample {j:3d}: [{acc_buf[j, 0]:9.6f}, {acc_buf[j, 1]:9.6f}, {acc_buf[j, 2]:9.6f}]\n")
                
                f.write(f"\nGyroscope buffer (shape {gyro_buf.shape}):\n")
                for j in range(BUFFER_SIZE):
                    f.write(f"  Sample {j:3d}: [{gyro_buf[j, 0]:9.6f}, {gyro_buf[j, 1]:9.6f}, {gyro_buf[j, 2]:9.6f}]\n")
                
                f.write(f"\n")
            
            print(f"  Logged buffer state at iteration {i} ({log_timesteps[i]})")
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{NUM_TESTS} iterations")
    
    print("Test complete!")
    print()
    
    # Calculate statistics
    inference_times = np.array(inference_times)
    mean_time = np.mean(inference_times)
    median_time = np.median(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    std_time = np.std(inference_times)
    p95_time = np.percentile(inference_times, 95)
    p99_time = np.percentile(inference_times, 99)
    
    # Calculate achievable rates
    mean_hz = 1000.0 / mean_time if mean_time > 0 else 0
    median_hz = 1000.0 / median_time if median_time > 0 else 0
    
    # Print results
    print("="*80)
    print("INFERENCE SPEED RESULTS")
    print("="*80)
    print(f"Total iterations: {NUM_TESTS}")
    print()
    print("Timing Statistics:")
    print(f"  Mean:     {mean_time:.3f} ms  ({mean_hz:.1f} Hz)")
    print(f"  Median:   {median_time:.3f} ms  ({median_hz:.1f} Hz)")
    print(f"  Min:      {min_time:.3f} ms")
    print(f"  Max:      {max_time:.3f} ms")
    print(f"  Std Dev:  {std_time:.3f} ms")
    print(f"  95th %:   {p95_time:.3f} ms")
    print(f"  99th %:   {p99_time:.3f} ms")
    print()
    
    # Real-time capability assessment
    print("Real-Time Capability Assessment:")
    target_rates = [250, 200, 150, 100, 50]
    for rate_hz in target_rates:
        required_ms = 1000.0 / rate_hz
        if mean_time < required_ms:
            status = "✓ CAN meet"
            margin_pct = ((required_ms - mean_time) / required_ms) * 100
            print(f"  {rate_hz} Hz (requires < {required_ms:.2f} ms): {status} (margin: {margin_pct:.1f}%)")
        else:
            status = "✗ CANNOT meet"
            deficit_pct = ((mean_time - required_ms) / required_ms) * 100
            print(f"  {rate_hz} Hz (requires < {required_ms:.2f} ms): {status} (too slow by {deficit_pct:.1f}%)")
    print()
    
    # Recommendations
    print("="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    if mean_time > 4.2:  # 240 Hz requirement
        print("WARNING: Inference is too slow for 240 Hz real-time processing!")
        print()
        print("Suggested optimizations:")
        print(f"  1. Reduce buffer size (current: {BUFFER_SIZE})")
        print("     - Try BUFFER_SIZE = 50 (expect ~2x speedup)")
        print("     - Try BUFFER_SIZE = 25 (expect ~4x speedup)")
        print("  2. Use quantized INT8 model (expect 2-4x speedup)")
        print("  3. Use GPU/CUDA acceleration (expect 10x+ speedup)")
        print("  4. Downsample input (process every Nth sample)")
    else:
        print("✓ Inference speed is adequate for real-time processing at 240 Hz!")
        max_rate = 1000.0 / p99_time if p99_time > 0 else 0
        print(f"  Can sustain up to ~{max_rate:.0f} Hz (99th percentile)")
    print("="*80)
    
    # Cleanup
    inference.close()
    print("\nTest complete.")
    print(f"Buffer state log saved to: {log_file}")


if __name__ == '__main__':
    main()
