#!/usr/bin/env python3
"""
Analyze ULog files and calculate trajectory metrics.
Extracts estimate and ground truth data, computes ATE, RMSE, velocity errors, and more.
"""

import sys
import os
import numpy as np
from pyulog import ULog
from datetime import datetime

# Try to import scipy, use numpy fallback if not available
try:
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except (ImportError, AttributeError) as e:
    print(f"Warning: scipy not available or incompatible ({e}). Using numpy interpolation fallback.")
    HAS_SCIPY = False


def extract_position_data(ulog, topic_name):
    """Extract position and velocity data from ULog topic."""
    try:
        data = ulog.get_dataset(topic_name)
        
        # Extract timestamps (convert from microseconds to seconds)
        timestamps = data.data['timestamp'] / 1e6
        
        # Extract position (x, y, z in meters)
        x = data.data['x']
        y = data.data['y']
        z = data.data['z']
        
        # Extract velocity (vx, vy, vz in m/s)
        vx = data.data['vx']
        vy = data.data['vy']
        vz = data.data['vz']
        
        return timestamps, x, y, z, vx, vy, vz
    except Exception as e:
        print(f"Error extracting data from {topic_name}: {e}")
        return None, None, None, None, None, None, None


def align_position_data(est_time, est_x, est_y, est_z, est_vx, est_vy, est_vz,
                       gt_time, gt_x, gt_y, gt_z, gt_vx, gt_vy, gt_vz):
    """
    Align estimate and ground truth data to common time range.
    Interpolates the sparser dataset (usually GT) to match the denser one (estimate).
    Uses OVERLAP-ONLY approach (no extrapolation).
    """
    # Find overlapping time range
    time_start = max(est_time.min(), gt_time.min())
    time_end = min(est_time.max(), gt_time.max())
    
    print(f"\nTime alignment:")
    print(f"  Estimate time range: {est_time.min():.3f}s to {est_time.max():.3f}s ({len(est_time)} samples)")
    print(f"  GT time range:       {gt_time.min():.3f}s to {gt_time.max():.3f}s ({len(gt_time)} samples)")
    print(f"  Overlap range:       {time_start:.3f}s to {time_end:.3f}s")
    
    # Filter estimate data to overlapping range
    est_mask = (est_time >= time_start) & (est_time <= time_end)
    est_time_filtered = est_time[est_mask]
    est_x_filtered = est_x[est_mask]
    est_y_filtered = est_y[est_mask]
    est_z_filtered = est_z[est_mask]
    est_vx_filtered = est_vx[est_mask]
    est_vy_filtered = est_vy[est_mask]
    est_vz_filtered = est_vz[est_mask]
    
    print(f"  Filtered estimate samples: {len(est_time_filtered)}")
    
    # Interpolate ground truth at estimate timestamps
    if HAS_SCIPY:
        # Create interpolation functions for ground truth (NO extrapolation)
        interp_gt_x = interp1d(gt_time, gt_x, kind='linear', bounds_error=False, fill_value=np.nan)
        interp_gt_y = interp1d(gt_time, gt_y, kind='linear', bounds_error=False, fill_value=np.nan)
        interp_gt_z = interp1d(gt_time, gt_z, kind='linear', bounds_error=False, fill_value=np.nan)
        interp_gt_vx = interp1d(gt_time, gt_vx, kind='linear', bounds_error=False, fill_value=np.nan)
        interp_gt_vy = interp1d(gt_time, gt_vy, kind='linear', bounds_error=False, fill_value=np.nan)
        interp_gt_vz = interp1d(gt_time, gt_vz, kind='linear', bounds_error=False, fill_value=np.nan)
        
        gt_x_interp = interp_gt_x(est_time_filtered)
        gt_y_interp = interp_gt_y(est_time_filtered)
        gt_z_interp = interp_gt_z(est_time_filtered)
        gt_vx_interp = interp_gt_vx(est_time_filtered)
        gt_vy_interp = interp_gt_vy(est_time_filtered)
        gt_vz_interp = interp_gt_vz(est_time_filtered)
    else:
        # Numpy fallback - simple linear interpolation
        gt_x_interp = np.interp(est_time_filtered, gt_time, gt_x)
        gt_y_interp = np.interp(est_time_filtered, gt_time, gt_y)
        gt_z_interp = np.interp(est_time_filtered, gt_time, gt_z)
        gt_vx_interp = np.interp(est_time_filtered, gt_time, gt_vx)
        gt_vy_interp = np.interp(est_time_filtered, gt_time, gt_vy)
        gt_vz_interp = np.interp(est_time_filtered, gt_time, gt_vz)
    
    return (est_time_filtered, 
            est_x_filtered, est_y_filtered, est_z_filtered,
            est_vx_filtered, est_vy_filtered, est_vz_filtered,
            gt_x_interp, gt_y_interp, gt_z_interp,
            gt_vx_interp, gt_vy_interp, gt_vz_interp)


def calculate_ate(est_pos, gt_pos):
    """
    Calculate Absolute Trajectory Error (ATE).
    
    CORRECT FORMULA: ATE = mean(sqrt(dx^2 + dy^2 + dz^2))
    This is the AVERAGE of the Euclidean distances at each point.
    
    NOT RMSE: sqrt(mean(dx^2 + dy^2 + dz^2)) - that's a different metric!
    """
    # Calculate Euclidean distance at each timestamp
    euclidean_distances = np.sqrt(
        (est_pos[0] - gt_pos[0])**2 +
        (est_pos[1] - gt_pos[1])**2 +
        (est_pos[2] - gt_pos[2])**2
    )
    
    # ATE is the MEAN of these distances
    ate = np.mean(euclidean_distances)
    
    return ate, euclidean_distances


def calculate_rmse(est_pos, gt_pos):
    """
    Calculate Root Mean Square Error (RMSE).
    
    RMSE = sqrt(mean(dx^2 + dy^2 + dz^2))
    """
    squared_errors = (
        (est_pos[0] - gt_pos[0])**2 +
        (est_pos[1] - gt_pos[1])**2 +
        (est_pos[2] - gt_pos[2])**2
    )
    
    rmse = np.sqrt(np.mean(squared_errors))
    
    return rmse


def calculate_per_axis_errors(est_pos, gt_pos):
    """Calculate per-axis error metrics (mean absolute error, RMSE, and bias)."""
    errors = {
        'x': {
            'mean': np.mean(np.abs(est_pos[0] - gt_pos[0])),
            'rmse': np.sqrt(np.mean((est_pos[0] - gt_pos[0])**2)),
            'std': np.std(est_pos[0] - gt_pos[0]),
            'bias': np.mean(est_pos[0] - gt_pos[0])  # Systematic offset
        },
        'y': {
            'mean': np.mean(np.abs(est_pos[1] - gt_pos[1])),
            'rmse': np.sqrt(np.mean((est_pos[1] - gt_pos[1])**2)),
            'std': np.std(est_pos[1] - gt_pos[1]),
            'bias': np.mean(est_pos[1] - gt_pos[1])
        },
        'z': {
            'mean': np.mean(np.abs(est_pos[2] - gt_pos[2])),
            'rmse': np.sqrt(np.mean((est_pos[2] - gt_pos[2])**2)),
            'std': np.std(est_pos[2] - gt_pos[2]),
            'bias': np.mean(est_pos[2] - gt_pos[2])
        }
    }
    
    # Calculate total bias magnitude
    total_bias = np.sqrt(errors['x']['bias']**2 + errors['y']['bias']**2 + errors['z']['bias']**2)
    errors['total_bias'] = total_bias
    
    return errors


def calculate_velocity_errors(est_vel, gt_vel):
    """Calculate velocity error metrics."""
    # Per-axis velocity errors
    vel_errors = {
        'vx': {
            'mean': np.mean(np.abs(est_vel[0] - gt_vel[0])),
            'rmse': np.sqrt(np.mean((est_vel[0] - gt_vel[0])**2))
        },
        'vy': {
            'mean': np.mean(np.abs(est_vel[1] - gt_vel[1])),
            'rmse': np.sqrt(np.mean((est_vel[1] - gt_vel[1])**2))
        },
        'vz': {
            'mean': np.mean(np.abs(est_vel[2] - gt_vel[2])),
            'rmse': np.sqrt(np.mean((est_vel[2] - gt_vel[2])**2))
        }
    }
    
    # Total velocity magnitude errors
    est_vel_mag = np.sqrt(est_vel[0]**2 + est_vel[1]**2 + est_vel[2]**2)
    gt_vel_mag = np.sqrt(gt_vel[0]**2 + gt_vel[1]**2 + gt_vel[2]**2)
    
    vel_errors['total'] = {
        'mean': np.mean(np.abs(est_vel_mag - gt_vel_mag)),
        'rmse': np.sqrt(np.mean((est_vel_mag - gt_vel_mag)**2))
    }
    
    return vel_errors


def calculate_trajectory_length(x, y, z):
    """Calculate total trajectory length by summing segment distances."""
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    
    segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
    total_length = np.sum(segment_lengths)
    
    return total_length


def save_metrics(ulog_filename, metrics, output_dir='.'):
    """Save metrics to a text file with naming convention: prefix_metrics_interpolation_shape.txt"""
    # Extract base name without extension
    base_name = os.path.splitext(os.path.basename(ulog_filename))[0]
    
    # Determine prefix (RAW_ or AI_)
    if base_name.startswith('RAW_'):
        prefix = base_name  # Keep full RAW_ name
    elif base_name.startswith('AI_'):
        prefix = base_name  # Keep full AI_ name
    else:
        prefix = base_name
    
    output_filename = f"{prefix}_metrics_interpolation_shape.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ULog Trajectory Analysis Metrics\n")
        f.write("="*70 + "\n")
        f.write(f"Input file: {ulog_filename}\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write("FLIGHT DURATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Total flight time: {metrics['flight_time']:.3f} seconds\n\n")
        
        f.write("TRAJECTORY LENGTHS\n")
        f.write("-"*70 + "\n")
        f.write(f"Estimate trajectory length: {metrics['est_length']:.3f} meters\n")
        f.write(f"Ground truth trajectory length: {metrics['gt_length']:.3f} meters\n\n")
        
        f.write("POSITION ERRORS\n")
        f.write("-"*70 + "\n")
        f.write(f"ATE (Absolute Trajectory Error): {metrics['ate']:.6f} meters\n")
        f.write(f"RMSE (Root Mean Square Error):   {metrics['rmse']:.6f} meters\n")
        f.write(f"Min position error: {metrics['min_error']:.6f} meters\n")
        f.write(f"Max position error: {metrics['max_error']:.6f} meters\n")
        f.write(f"Std dev of errors:  {metrics['std_error']:.6f} meters\n\n")
        
        f.write("SYSTEMATIC BIAS\n")
        f.write("-"*70 + "\n")
        f.write(f"X-axis bias: {metrics['per_axis']['x']['bias']:+.6f} meters\n")
        f.write(f"Y-axis bias: {metrics['per_axis']['y']['bias']:+.6f} meters\n")
        f.write(f"Z-axis bias: {metrics['per_axis']['z']['bias']:+.6f} meters\n")
        f.write(f"Total bias magnitude: {metrics['per_axis']['total_bias']:.6f} meters\n\n")
        
        f.write("PER-AXIS POSITION ERRORS\n")
        f.write("-"*70 + "\n")
        for axis in ['x', 'y', 'z']:
            f.write(f"{axis.upper()}-Axis:\n")
            f.write(f"  Mean absolute error: {metrics['per_axis'][axis]['mean']:.6f} meters\n")
            f.write(f"  RMSE:                {metrics['per_axis'][axis]['rmse']:.6f} meters\n")
            f.write(f"  Std deviation:       {metrics['per_axis'][axis]['std']:.6f} meters\n")
            f.write(f"  Bias:                {metrics['per_axis'][axis]['bias']:+.6f} meters\n")
        f.write("\n")
        
        f.write("VELOCITY ERRORS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total velocity magnitude:\n")
        f.write(f"  Mean: {metrics['velocity']['total']['mean']:.6f} m/s\n")
        f.write(f"  RMSE: {metrics['velocity']['total']['rmse']:.6f} m/s\n\n")
        
        for axis in ['vx', 'vy', 'vz']:
            f.write(f"{axis.upper()}:\n")
            f.write(f"  Mean: {metrics['velocity'][axis]['mean']:.6f} m/s\n")
            f.write(f"  RMSE: {metrics['velocity'][axis]['rmse']:.6f} m/s\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"\nMetrics saved to: {output_path}")
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_ulog_metrics.py <ulog_file>")
        sys.exit(1)
    
    ulog_file = sys.argv[1]
    
    if not os.path.exists(ulog_file):
        print(f"Error: File '{ulog_file}' not found!")
        sys.exit(1)
    
    print(f"Analyzing ULog file: {ulog_file}")
    print("="*70)
    
    # Load ULog file
    try:
        ulog = ULog(ulog_file)
    except Exception as e:
        print(f"Error loading ULog file: {e}")
        sys.exit(1)
    
    # Extract estimate and ground truth data
    est_time, est_x, est_y, est_z, est_vx, est_vy, est_vz = \
        extract_position_data(ulog, 'vehicle_local_position')
    
    gt_time, gt_x, gt_y, gt_z, gt_vx, gt_vy, gt_vz = \
        extract_position_data(ulog, 'vehicle_local_position_groundtruth')
    
    if est_time is None or gt_time is None:
        print("Error: Could not extract required data from ULog file")
        sys.exit(1)
    
    # Align data using overlap-only interpolation
    aligned_data = align_position_data(
        est_time, est_x, est_y, est_z, est_vx, est_vy, est_vz,
        gt_time, gt_x, gt_y, gt_z, gt_vx, gt_vy, gt_vz
    )
    
    (time, est_x, est_y, est_z, est_vx, est_vy, est_vz,
     gt_x, gt_y, gt_z, gt_vx, gt_vy, gt_vz) = aligned_data
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # Position errors
    ate, euclidean_distances = calculate_ate((est_x, est_y, est_z), (gt_x, gt_y, gt_z))
    rmse = calculate_rmse((est_x, est_y, est_z), (gt_x, gt_y, gt_z))
    per_axis_errors = calculate_per_axis_errors((est_x, est_y, est_z), (gt_x, gt_y, gt_z))
    
    # Velocity errors
    velocity_errors = calculate_velocity_errors((est_vx, est_vy, est_vz), (gt_vx, gt_vy, gt_vz))
    
    # Trajectory lengths
    est_length = calculate_trajectory_length(est_x, est_y, est_z)
    gt_length = calculate_trajectory_length(gt_x, gt_y, gt_z)
    
    # Flight time
    flight_time = time[-1] - time[0]
    
    # Package metrics
    metrics = {
        'ate': ate,
        'rmse': rmse,
        'min_error': np.min(euclidean_distances),
        'max_error': np.max(euclidean_distances),
        'std_error': np.std(euclidean_distances),
        'per_axis': per_axis_errors,
        'velocity': velocity_errors,
        'est_length': est_length,
        'gt_length': gt_length,
        'flight_time': flight_time
    }
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Flight time: {flight_time:.3f} seconds")
    print(f"ATE (Absolute Trajectory Error): {ate:.6f} meters")
    print(f"RMSE: {rmse:.6f} meters")
    print(f"Systematic bias (offset): {metrics['per_axis']['total_bias']:.6f} meters")
    print(f"  X-bias: {per_axis_errors['x']['bias']:+.6f}m, "
          f"Y-bias: {per_axis_errors['y']['bias']:+.6f}m, "
          f"Z-bias: {per_axis_errors['z']['bias']:+.6f}m")
    print(f"Per-axis errors (mean): X={per_axis_errors['x']['mean']:.6f}m, "
          f"Y={per_axis_errors['y']['mean']:.6f}m, Z={per_axis_errors['z']['mean']:.6f}m")
    print(f"Trajectory lengths: Estimate={est_length:.3f}m, GT={gt_length:.3f}m")
    print("="*70)
    
    # Save to file
    output_file = save_metrics(ulog_file, metrics)
    
    print(f"\nAnalysis complete!")


if __name__ == "__main__":
    main()
