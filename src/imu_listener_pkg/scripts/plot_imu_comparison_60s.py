#!/usr/bin/env python3
"""
Plot IMU Comparison from Rosbag (First 60 seconds only)

This script reads a rosbag containing raw and corrected IMU data along with
position information and creates comparison plots for the first 60 seconds.

Usage:
    python3 plot_imu_comparison_60s.py <bagfile>
    
Example:
    python3 plot_imu_comparison_60s.py ../bags/imu_comparison_2025-12-08-14-21-51.bag
"""

import rosbag
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
import sys
import os
from datetime import datetime

# Time window to plot (seconds)
TIME_WINDOW = 35.0

def extract_imu_data(bag, topic):
    """Extract IMU data from a specific topic"""
    timestamps = []
    accel_x, accel_y, accel_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []
    
    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        timestamps.append(t.to_sec())
        accel_x.append(msg.linear_acceleration.x)
        accel_y.append(msg.linear_acceleration.y)
        accel_z.append(msg.linear_acceleration.z)
        gyro_x.append(msg.angular_velocity.x)
        gyro_y.append(msg.angular_velocity.y)
        gyro_z.append(msg.angular_velocity.z)
    
    return {
        'time': np.array(timestamps),
        'accel': np.array([accel_x, accel_y, accel_z]).T,
        'gyro': np.array([gyro_x, gyro_y, gyro_z]).T
    }

def extract_position_data(bag, topic):
    """Extract position data from pose topic"""
    timestamps = []
    pos_x, pos_y, pos_z = [], [], []
    
    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        timestamps.append(t.to_sec())
        pos_x.append(msg.pose.position.x)
        pos_y.append(msg.pose.position.y)
        pos_z.append(msg.pose.position.z)
    
    return {
        'time': np.array(timestamps),
        'position': np.array([pos_x, pos_y, pos_z]).T
    }

def calculate_statistics(raw_data, corrected_data):
    """Calculate statistics comparing raw and corrected data"""
    # Align data by interpolation if needed
    if len(raw_data) != len(corrected_data):
        print(f"Warning: Data length mismatch (raw: {len(raw_data)}, corrected: {len(corrected_data)})")
        min_len = min(len(raw_data), len(corrected_data))
        raw_data = raw_data[:min_len]
        corrected_data = corrected_data[:min_len]
    
    diff = corrected_data - raw_data
    
    stats = {
        'mean_diff': np.mean(diff, axis=0),
        'std_diff': np.std(diff, axis=0),
        'max_diff': np.max(np.abs(diff), axis=0),
        'rms_raw': np.sqrt(np.mean(raw_data**2, axis=0)),
        'rms_corrected': np.sqrt(np.mean(corrected_data**2, axis=0)),
    }
    
    return stats

def filter_time_window(data, time_window):
    """Filter data to only include samples within time window"""
    mask = data['time'] <= time_window
    filtered = {
        'time': data['time'][mask],
        'accel': data['accel'][mask],
        'gyro': data['gyro'][mask]
    }
    return filtered

def filter_position_time_window(data, time_window):
    """Filter position data to only include samples within time window"""
    mask = data['time'] <= time_window
    filtered = {
        'time': data['time'][mask],
        'position': data['position'][mask]
    }
    return filtered

def plot_comparison(bag_path):
    """Create comprehensive comparison plots"""
    
    print(f"Loading rosbag: {bag_path}")
    bag = rosbag.Bag(bag_path)
    
    # Get bag info
    info = bag.get_type_and_topic_info()
    topics = info.topics
    
    print("\nAvailable topics:")
    for topic_name, topic_info in topics.items():
        print(f"  {topic_name}: {topic_info.message_count} messages")
    
    # Extract data
    print("\nExtracting data...")
    raw_imu = extract_imu_data(bag, '/imu/raw')
    corrected_imu = extract_imu_data(bag, '/imu/corrected')
    
    # Try to get position data
    position = None
    if '/mavros/local_position/pose' in topics:
        position = extract_position_data(bag, '/mavros/local_position/pose')
    
    bag.close()
    
    # Normalize time to start at 0
    t0 = min(raw_imu['time'][0], corrected_imu['time'][0])
    raw_imu['time'] -= t0
    corrected_imu['time'] -= t0
    if position:
        position['time'] -= t0
    
    print(f"\nData extracted (full):")
    print(f"  Raw IMU: {len(raw_imu['time'])} samples")
    print(f"  Corrected IMU: {len(corrected_imu['time'])} samples")
    if position:
        print(f"  Position: {len(position['time'])} samples")
    
    # Filter to first 60 seconds
    raw_imu = filter_time_window(raw_imu, TIME_WINDOW)
    corrected_imu = filter_time_window(corrected_imu, TIME_WINDOW)
    if position:
        position = filter_position_time_window(position, TIME_WINDOW)
    
    print(f"\nData after filtering to first {TIME_WINDOW}s:")
    print(f"  Raw IMU: {len(raw_imu['time'])} samples")
    print(f"  Corrected IMU: {len(corrected_imu['time'])} samples")
    if position:
        print(f"  Position: {len(position['time'])} samples")
    
    # Calculate statistics
    accel_stats = calculate_statistics(raw_imu['accel'], corrected_imu['accel'])
    gyro_stats = calculate_statistics(raw_imu['gyro'], corrected_imu['gyro'])
    
    print("\n" + "="*80)
    print(f"ACCELEROMETER STATISTICS (First {TIME_WINDOW}s)")
    print("="*80)
    print(f"Mean difference (m/s²):     X: {accel_stats['mean_diff'][0]:8.4f}  Y: {accel_stats['mean_diff'][1]:8.4f}  Z: {accel_stats['mean_diff'][2]:8.4f}")
    print(f"Std deviation (m/s²):       X: {accel_stats['std_diff'][0]:8.4f}  Y: {accel_stats['std_diff'][1]:8.4f}  Z: {accel_stats['std_diff'][2]:8.4f}")
    print(f"Max absolute diff (m/s²):   X: {accel_stats['max_diff'][0]:8.4f}  Y: {accel_stats['max_diff'][1]:8.4f}  Z: {accel_stats['max_diff'][2]:8.4f}")
    print(f"RMS raw (m/s²):             X: {accel_stats['rms_raw'][0]:8.4f}  Y: {accel_stats['rms_raw'][1]:8.4f}  Z: {accel_stats['rms_raw'][2]:8.4f}")
    print(f"RMS corrected (m/s²):       X: {accel_stats['rms_corrected'][0]:8.4f}  Y: {accel_stats['rms_corrected'][1]:8.4f}  Z: {accel_stats['rms_corrected'][2]:8.4f}")
    
    print("\n" + "="*80)
    print(f"GYROSCOPE STATISTICS (First {TIME_WINDOW}s)")
    print("="*80)
    print(f"Mean difference (rad/s):    X: {gyro_stats['mean_diff'][0]:8.4f}  Y: {gyro_stats['mean_diff'][1]:8.4f}  Z: {gyro_stats['mean_diff'][2]:8.4f}")
    print(f"Std deviation (rad/s):      X: {gyro_stats['std_diff'][0]:8.4f}  Y: {gyro_stats['std_diff'][1]:8.4f}  Z: {gyro_stats['std_diff'][2]:8.4f}")
    print(f"Max absolute diff (rad/s):  X: {gyro_stats['max_diff'][0]:8.4f}  Y: {gyro_stats['max_diff'][1]:8.4f}  Z: {gyro_stats['max_diff'][2]:8.4f}")
    print(f"RMS raw (rad/s):            X: {gyro_stats['rms_raw'][0]:8.4f}  Y: {gyro_stats['rms_raw'][1]:8.4f}  Z: {gyro_stats['rms_raw'][2]:8.4f}")
    print(f"RMS corrected (rad/s):      X: {gyro_stats['rms_corrected'][0]:8.4f}  Y: {gyro_stats['rms_corrected'][1]:8.4f}  Z: {gyro_stats['rms_corrected'][2]:8.4f}")
    print("="*80 + "\n")
    
    # Create output directory with timestamp
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(pkg_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    colors_raw = '#FF6B6B'
    colors_corrected = '#0652DD'
    
    min_len = min(len(raw_imu['accel']), len(corrected_imu['accel']))
    
    # ========== Figure 1: Accelerometer X-axis ==========
    fig1 = plt.figure(figsize=(12, 6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(corrected_imu['time'][:min_len], corrected_imu['accel'][:min_len, 0], 
            label='Corrected', color=colors_corrected, linewidth=1.5, alpha=0.8)
    ax1.plot(raw_imu['time'][:min_len], raw_imu['accel'][:min_len, 0], 
            label='Raw', color=colors_raw, linewidth=2.0, linestyle='--')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Acceleration (m/s²)', fontsize=12)
    ax1.set_title(f'Accelerometer X-axis: Raw vs Corrected (First {TIME_WINDOW}s)', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, TIME_WINDOW])
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    # Ensure last timestamp is shown and filter out negative values
    xticks = [t for t in ax1.get_xticks() if 0 <= t <= TIME_WINDOW]
    if TIME_WINDOW not in xticks:
        xticks.append(TIME_WINDOW)
    ax1.set_xticks(sorted(xticks))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(results_dir, f'1_accel_x_60s_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig1)
    
    # ========== Figure 2: Accelerometer Y-axis ==========
    fig2 = plt.figure(figsize=(12, 6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(corrected_imu['time'][:min_len], corrected_imu['accel'][:min_len, 1], 
            label='Corrected', color=colors_corrected, linewidth=1.5, alpha=0.8)
    ax2.plot(raw_imu['time'][:min_len], raw_imu['accel'][:min_len, 1], 
            label='Raw', color=colors_raw, linewidth=2.0, linestyle='--')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Acceleration (m/s²)', fontsize=12)
    ax2.set_title(f'Accelerometer Y-axis: Raw vs Corrected (First {TIME_WINDOW}s)', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, TIME_WINDOW])
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    xticks = [t for t in ax2.get_xticks() if 0 <= t <= TIME_WINDOW]
    if TIME_WINDOW not in xticks:
        xticks.append(TIME_WINDOW)
    ax2.set_xticks(sorted(xticks))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(results_dir, f'2_accel_y_60s_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig2)
    
    # ========== Figure 3: Accelerometer Z-axis ==========
    fig3 = plt.figure(figsize=(12, 6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(corrected_imu['time'][:min_len], corrected_imu['accel'][:min_len, 2], 
            label='Corrected', color=colors_corrected, linewidth=1.5, alpha=0.8)
    ax3.plot(raw_imu['time'][:min_len], raw_imu['accel'][:min_len, 2], 
            label='Raw', color=colors_raw, linewidth=2.0, linestyle='--')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Acceleration (m/s²)', fontsize=12)
    ax3.set_title(f'Accelerometer Z-axis: Raw vs Corrected (First {TIME_WINDOW}s)', fontsize=14, fontweight='bold')
    ax3.set_xlim([0, TIME_WINDOW])
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    xticks = [t for t in ax3.get_xticks() if 0 <= t <= TIME_WINDOW]
    if TIME_WINDOW not in xticks:
        xticks.append(TIME_WINDOW)
    ax3.set_xticks(sorted(xticks))
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(results_dir, f'3_accel_z_60s_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig3)
    
    # ========== Figure 4: Accelerometer Correction ==========
    fig4 = plt.figure(figsize=(12, 6))
    ax4 = fig4.add_subplot(111)
    accel_diff = corrected_imu['accel'][:min_len] - raw_imu['accel'][:min_len]
    ax4.plot(raw_imu['time'][:min_len], accel_diff[:, 0], 
            label='X', color='#C44569', linewidth=1.5)
    ax4.plot(raw_imu['time'][:min_len], accel_diff[:, 1], 
            label='Y', color='#218C74', linewidth=1.5)
    ax4.plot(raw_imu['time'][:min_len], accel_diff[:, 2], 
            label='Z', color='#0652DD', linewidth=1.5)
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Correction (m/s²)', fontsize=12)
    ax4.set_title(f'Accelerometer Correction (Corrected - Raw, First {TIME_WINDOW}s)', fontsize=14, fontweight='bold')
    ax4.set_xlim([0, TIME_WINDOW])
    ax4.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    xticks = [t for t in ax4.get_xticks() if 0 <= t <= TIME_WINDOW]
    if TIME_WINDOW not in xticks:
        xticks.append(TIME_WINDOW)
    ax4.set_xticks(sorted(xticks))
    ax4.yaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
    ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(results_dir, f'4_accel_correction_60s_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig4)
    
    # ========== Figure 5: Gyroscope X-axis ==========
    fig5 = plt.figure(figsize=(12, 6))
    ax5 = fig5.add_subplot(111)
    ax5.plot(corrected_imu['time'][:min_len], corrected_imu['gyro'][:min_len, 0], 
            label='Corrected', color=colors_corrected, linewidth=1.5, alpha=0.8)
    ax5.plot(raw_imu['time'][:min_len], raw_imu['gyro'][:min_len, 0], 
            label='Raw', color=colors_raw, linewidth=2.0, linestyle='--')
    ax5.set_xlabel('Time (s)', fontsize=12)
    ax5.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax5.set_title(f'Gyroscope X-axis: Raw vs Corrected (First {TIME_WINDOW}s)', fontsize=14, fontweight='bold')
    ax5.set_xlim([0, TIME_WINDOW])
    ax5.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    xticks = [t for t in ax5.get_xticks() if 0 <= t <= TIME_WINDOW]
    if TIME_WINDOW not in xticks:
        xticks.append(TIME_WINDOW)
    ax5.set_xticks(sorted(xticks))
    ax5.yaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
    ax5.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(results_dir, f'5_gyro_x_60s_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig5)
    
    # ========== Figure 6: Gyroscope Y-axis ==========
    fig6 = plt.figure(figsize=(12, 6))
    ax6 = fig6.add_subplot(111)
    ax6.plot(corrected_imu['time'][:min_len], corrected_imu['gyro'][:min_len, 1], 
            label='Corrected', color=colors_corrected, linewidth=1.5, alpha=0.8)
    ax6.plot(raw_imu['time'][:min_len], raw_imu['gyro'][:min_len, 1], 
            label='Raw', color=colors_raw, linewidth=2.0, linestyle='--')
    ax6.set_xlabel('Time (s)', fontsize=12)
    ax6.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax6.set_title(f'Gyroscope Y-axis: Raw vs Corrected (First {TIME_WINDOW}s)', fontsize=14, fontweight='bold')
    ax6.set_xlim([0, TIME_WINDOW])
    ax6.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    xticks = [t for t in ax6.get_xticks() if 0 <= t <= TIME_WINDOW]
    if TIME_WINDOW not in xticks:
        xticks.append(TIME_WINDOW)
    ax6.set_xticks(sorted(xticks))
    ax6.yaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
    ax6.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(results_dir, f'6_gyro_y_60s_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig6)
    
    # ========== Figure 7: Gyroscope Z-axis ==========
    fig7 = plt.figure(figsize=(12, 6))
    ax7 = fig7.add_subplot(111)
    ax7.plot(corrected_imu['time'][:min_len], corrected_imu['gyro'][:min_len, 2], 
            label='Corrected', color=colors_corrected, linewidth=1.5, alpha=0.8)
    ax7.plot(raw_imu['time'][:min_len], raw_imu['gyro'][:min_len, 2], 
            label='Raw', color=colors_raw, linewidth=2.0, linestyle='--')
    ax7.set_xlabel('Time (s)', fontsize=12)
    ax7.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax7.set_title(f'Gyroscope Z-axis: Raw vs Corrected (First {TIME_WINDOW}s)', fontsize=14, fontweight='bold')
    ax7.set_xlim([0, TIME_WINDOW])
    ax7.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    xticks = [t for t in ax7.get_xticks() if 0 <= t <= TIME_WINDOW]
    if TIME_WINDOW not in xticks:
        xticks.append(TIME_WINDOW)
    ax7.set_xticks(sorted(xticks))
    ax7.yaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
    ax7.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax7.legend(fontsize=11)
    ax7.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(results_dir, f'7_gyro_z_60s_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig7)
    
    # ========== Figure 8: Gyroscope Correction ==========
    fig8 = plt.figure(figsize=(12, 6))
    ax8 = fig8.add_subplot(111)
    gyro_diff = corrected_imu['gyro'][:min_len] - raw_imu['gyro'][:min_len]
    ax8.plot(raw_imu['time'][:min_len], gyro_diff[:, 0], 
            label='X', color='#C44569', linewidth=1.5)
    ax8.plot(raw_imu['time'][:min_len], gyro_diff[:, 1], 
            label='Y', color='#218C74', linewidth=1.5)
    ax8.plot(raw_imu['time'][:min_len], gyro_diff[:, 2], 
            label='Z', color='#0652DD', linewidth=1.5)
    ax8.set_xlabel('Time (s)', fontsize=12)
    ax8.set_ylabel('Correction (rad/s)', fontsize=12)
    ax8.set_title(f'Gyroscope Correction (Corrected - Raw, First {TIME_WINDOW}s)', fontsize=14, fontweight='bold')
    ax8.set_xlim([0, TIME_WINDOW])
    ax8.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    xticks = [t for t in ax8.get_xticks() if 0 <= t <= TIME_WINDOW]
    if TIME_WINDOW not in xticks:
        xticks.append(TIME_WINDOW)
    ax8.set_xticks(sorted(xticks))
    ax8.yaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
    ax8.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax8.legend(fontsize=11)
    ax8.grid(True, alpha=0.3)
    ax8.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(results_dir, f'8_gyro_correction_60s_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig8)
    
    # ========== Position plots (if available) ==========
    if position and len(position['time']) > 0:
        # Figure 9: Local Position
        fig9 = plt.figure(figsize=(12, 6))
        ax9 = fig9.add_subplot(111)
        ax9.plot(position['time'], position['position'][:, 0], 
                label='X', color='#C44569', linewidth=1.5)
        ax9.plot(position['time'], position['position'][:, 1], 
                label='Y', color='#218C74', linewidth=1.5)
        ax9.plot(position['time'], position['position'][:, 2], 
                label='Z', color='#0652DD', linewidth=1.5)
        ax9.set_xlabel('Time (s)', fontsize=12)
        ax9.set_ylabel('Position (m)', fontsize=12)
        ax9.set_title(f'Local Position (First {TIME_WINDOW}s)', fontsize=14, fontweight='bold')
        ax9.set_xlim([0, TIME_WINDOW])
        ax9.legend(fontsize=11)
        ax9.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(results_dir, f'9_position_60s_{timestamp}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig9)
        
        # Figure 10: 3D Trajectory
        fig10 = plt.figure(figsize=(10, 8))
        ax10 = fig10.add_subplot(111, projection='3d')
        ax10.plot(position['position'][:, 0], 
                position['position'][:, 1], 
                position['position'][:, 2], 
                color='#0652DD', linewidth=2)
        ax10.scatter(position['position'][0, 0], 
                   position['position'][0, 1], 
                   position['position'][0, 2], 
                   color='green', s=100, label='Start', marker='o')
        ax10.scatter(position['position'][-1, 0], 
                   position['position'][-1, 1], 
                   position['position'][-1, 2], 
                   color='red', s=100, label='End', marker='x')
        ax10.set_xlabel('X (m)', fontsize=12)
        ax10.set_ylabel('Y (m)', fontsize=12)
        ax10.set_zlabel('Z (m)', fontsize=12)
        ax10.set_title(f'3D Trajectory (First {TIME_WINDOW}s)', fontsize=14, fontweight='bold')
        ax10.legend(fontsize=11)
        plt.tight_layout()
        output_path = os.path.join(results_dir, f'10_trajectory_3d_60s_{timestamp}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig10)
    
    print(f"\nAll plots saved to: {results_dir}")
    print(f"Timestamp: {timestamp}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 plot_imu_comparison_60s.py <bagfile>")
        sys.exit(1)
    
    bag_path = sys.argv[1]
    
    if not os.path.exists(bag_path):
        print(f"Error: Bag file not found: {bag_path}")
        sys.exit(1)
    
    try:
        plot_comparison(bag_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
