#!/usr/bin/env python3
"""Tennis form comparison analyzer for two videos"""

import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os

from pose_estimator import create_pose_estimator
from metrics_engine import create_metrics_engine
from visualizer import create_visualizer, create_video_writer
from frame_loader import create_frame_loader


@dataclass
class ComparisonMetrics:
    """Metrics for comparing two poses"""
    frame_idx: int
    angles_video1: Dict[str, Optional[float]]
    angles_video2: Dict[str, Optional[float]]
    angle_differences: Dict[str, Optional[float]]
    overall_difference: float
    highlight_joints: List[str]  # Joints with significant differences


class FormComparisonAnalyzer:
    """Analyzer for comparing tennis forms between two videos"""
    
    def __init__(self, difference_threshold: float = 20.0):
        self.difference_threshold = difference_threshold  # degrees
        self.pose_estimator = None
        self.metrics_engine1 = None
        self.metrics_engine2 = None
        self.visualizer = None
        
        # Key joints for tennis analysis
        self.key_joints = [
            'right_elbow', 'left_elbow',
            'right_shoulder', 'left_shoulder', 
            'right_knee', 'left_knee',
            'right_hip', 'left_hip'
        ]
        
        # Colors for visualization
        self.colors = {
            'video1': (0, 255, 0),      # Green
            'video2': (255, 0, 0),      # Blue  
            'difference': (0, 0, 255),   # Red
            'normal': (255, 255, 255),   # White
        }
    
    def initialize_components(self):
        """Initialize analysis components"""
        self.pose_estimator = create_pose_estimator()
        self.metrics_engine1 = create_metrics_engine()
        self.metrics_engine2 = create_metrics_engine()
        self.visualizer = create_visualizer()
    
    def cleanup(self):
        """Clean up resources"""
        if self.pose_estimator:
            self.pose_estimator.close()
    
    def extract_poses_from_video(self, video_path: str) -> List[Dict]:
        """Extract pose data from a video"""
        poses = []
        
        with create_frame_loader(video_path) as frame_loader:
            for frame_idx, frame in frame_loader.load_frames():
                try:
                    pose_result = self.pose_estimator.process_frame(frame)
                    if pose_result and pose_result['landmarks']:
                        landmarks = pose_result['landmarks']
                        visibility = pose_result['visibility']
                        angles = self.pose_estimator.get_key_angles(landmarks)
                        
                        poses.append({
                            'frame_idx': frame_idx,
                            'landmarks': landmarks,
                            'visibility': visibility,
                            'angles': angles,
                            'frame': frame
                        })
                    else:
                        # Add empty pose data for frames without detection
                        poses.append({
                            'frame_idx': frame_idx,
                            'landmarks': [],
                            'visibility': [],
                            'angles': {},
                            'frame': frame
                        })
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    poses.append({
                        'frame_idx': frame_idx,
                        'landmarks': [],
                        'visibility': [],
                        'angles': {},
                        'frame': frame
                    })
        
        return poses
    
    def synchronize_videos(self, poses1: List[Dict], poses2: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Synchronize two video sequences by matching frame counts"""
        min_frames = min(len(poses1), len(poses2))
        
        # Simple frame-by-frame synchronization
        # In a more advanced version, you could implement time-based sync or motion correlation
        sync_poses1 = poses1[:min_frames]
        sync_poses2 = poses2[:min_frames]
        
        return sync_poses1, sync_poses2
    
    def calculate_angle_differences(self, angles1: Dict[str, Optional[float]], 
                                   angles2: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """Calculate angle differences between two pose sets"""
        differences = {}
        
        for joint in self.key_joints:
            angle1 = angles1.get(joint)
            angle2 = angles2.get(joint)
            
            if angle1 is not None and angle2 is not None:
                # Calculate absolute difference
                diff = abs(angle1 - angle2)
                # Handle angle wraparound (180 degrees)
                if diff > 180:
                    diff = 360 - diff
                differences[joint] = diff
            else:
                differences[joint] = None
        
        return differences
    
    def identify_highlight_joints(self, angle_differences: Dict[str, Optional[float]]) -> List[str]:
        """Identify joints with significant differences for highlighting"""
        highlight_joints = []
        
        for joint, diff in angle_differences.items():
            if diff is not None and diff > self.difference_threshold:
                highlight_joints.append(joint)
        
        return highlight_joints
    
    def compare_poses(self, poses1: List[Dict], poses2: List[Dict]) -> List[ComparisonMetrics]:
        """Compare poses frame by frame"""
        sync_poses1, sync_poses2 = self.synchronize_videos(poses1, poses2)
        comparison_metrics = []
        
        for i, (pose1, pose2) in enumerate(zip(sync_poses1, sync_poses2)):
            angles1 = pose1['angles']
            angles2 = pose2['angles']
            
            angle_differences = self.calculate_angle_differences(angles1, angles2)
            highlight_joints = self.identify_highlight_joints(angle_differences)
            
            # Calculate overall difference score
            valid_differences = [diff for diff in angle_differences.values() if diff is not None]
            overall_difference = np.mean(valid_differences) if valid_differences else 0.0
            
            metrics = ComparisonMetrics(
                frame_idx=i,
                angles_video1=angles1,
                angles_video2=angles2,
                angle_differences=angle_differences,
                overall_difference=overall_difference,
                highlight_joints=highlight_joints
            )
            
            comparison_metrics.append(metrics)
        
        return comparison_metrics
    
    def create_comparison_visualization(self, pose1: Dict, pose2: Dict, 
                                     comparison_metric: ComparisonMetrics) -> np.ndarray:
        """Create side-by-side comparison visualization"""
        frame1 = pose1['frame']
        frame2 = pose2['frame']
        landmarks1 = pose1['landmarks']
        landmarks2 = pose2['landmarks']
        
        height, width = frame1.shape[:2]
        
        # Create side-by-side canvas
        comparison_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)
        
        # Place original frames
        comparison_frame[:, :width] = frame1
        comparison_frame[:, width:] = frame2
        
        # Draw pose overlays
        if landmarks1:
            self._draw_pose_with_highlights(
                comparison_frame[:, :width], 
                landmarks1, 
                comparison_metric.highlight_joints, 
                self.colors['video1'],
                'video1'
            )
        
        if landmarks2:
            self._draw_pose_with_highlights(
                comparison_frame[:, width:], 
                landmarks2, 
                comparison_metric.highlight_joints, 
                self.colors['video2'],
                'video2',
                x_offset=0  # No offset needed for right side
            )
        
        # Add comparison info
        self._add_comparison_info(comparison_frame, comparison_metric)
        
        return comparison_frame
    
    def _draw_pose_with_highlights(self, frame: np.ndarray, landmarks: List[Tuple[int, int]], 
                                 highlight_joints: List[str], base_color: Tuple[int, int, int],
                                 video_label: str, x_offset: int = 0):
        """Draw pose with highlighted joints for significant differences"""
        
        # Draw normal pose connections
        pose_connections = [
            (11, 13), (13, 15), (12, 14), (14, 16),  # arms
            (11, 12),  # shoulders
            (11, 23), (12, 24),  # torso
            (23, 24),  # hips
            (23, 25), (25, 27), (27, 29),  # left leg
            (24, 26), (26, 28), (28, 30),  # right leg
        ]
        
        # Map joint names to landmark indices for highlighting
        joint_landmark_map = {
            'right_elbow': [12, 14, 16],  # shoulder, elbow, wrist
            'left_elbow': [11, 13, 15],
            'right_shoulder': [12, 14],
            'left_shoulder': [11, 13],
            'right_knee': [24, 26, 28],   # hip, knee, ankle
            'left_knee': [23, 25, 27],
            'right_hip': [12, 24],
            'left_hip': [11, 23]
        }
        
        # Draw connections
        for connection in pose_connections:
            start_idx, end_idx = connection
            if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                landmarks[start_idx] and landmarks[end_idx]):
                
                start_point = (landmarks[start_idx][0] + x_offset, landmarks[start_idx][1])
                end_point = (landmarks[end_idx][0] + x_offset, landmarks[end_idx][1])
                
                cv2.line(frame, start_point, end_point, base_color, 2)
        
        # Draw landmarks
        for i, landmark in enumerate(landmarks):
            if landmark:
                point = (landmark[0] + x_offset, landmark[1])
                cv2.circle(frame, point, 3, base_color, -1)
        
        # Highlight joints with significant differences
        for joint in highlight_joints:
            if joint in joint_landmark_map:
                landmark_indices = joint_landmark_map[joint]
                for idx in landmark_indices:
                    if idx < len(landmarks) and landmarks[idx]:
                        point = (landmarks[idx][0] + x_offset, landmarks[idx][1])
                        cv2.circle(frame, point, 8, self.colors['difference'], 3)
        
        # Add video label
        cv2.putText(frame, f"{video_label.title()}", (10 + x_offset, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, base_color, 2)
    
    def _add_comparison_info(self, frame: np.ndarray, comparison_metric: ComparisonMetrics):
        """Add comparison information overlay"""
        height, width = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, height - 150), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Add comparison metrics
        y_pos = height - 130
        cv2.putText(frame, f"Frame: {comparison_metric.frame_idx}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_pos += 20
        cv2.putText(frame, f"Overall Difference: {comparison_metric.overall_difference:.1f}°", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # List highlighted joints
        if comparison_metric.highlight_joints:
            y_pos += 20
            highlighted_text = "High Diff: " + ", ".join(comparison_metric.highlight_joints)
            cv2.putText(frame, highlighted_text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['difference'], 1)
    
    def create_comparison_video(self, poses1: List[Dict], poses2: List[Dict], 
                               comparison_metrics: List[ComparisonMetrics], 
                               output_path: str, fps: float = 30.0):
        """Create comparison video with side-by-side analysis"""
        
        if not poses1 or not poses2:
            raise ValueError("Both video pose data must be provided")
        
        # Get frame dimensions
        height, width = poses1[0]['frame'].shape[:2]
        
        with create_video_writer(output_path, fps, width * 2, height) as video_writer:
            for i, (pose1, pose2, metrics) in enumerate(zip(poses1, poses2, comparison_metrics)):
                comparison_frame = self.create_comparison_visualization(pose1, pose2, metrics)
                video_writer.write_frame(comparison_frame)
    
    def export_comparison_metrics(self, comparison_metrics: List[ComparisonMetrics], 
                                 output_path: str):
        """Export comparison metrics to CSV"""
        
        data = []
        for metrics in comparison_metrics:
            row = {
                'frame_idx': metrics.frame_idx,
                'overall_difference': metrics.overall_difference,
            }
            
            # Add individual joint angles and differences
            for joint in self.key_joints:
                row[f'{joint}_video1'] = metrics.angles_video1.get(joint)
                row[f'{joint}_video2'] = metrics.angles_video2.get(joint)
                row[f'{joint}_difference'] = metrics.angle_differences.get(joint)
            
            # Add highlight status
            row['highlighted_joints'] = ','.join(metrics.highlight_joints)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    
    def create_comparison_charts(self, comparison_metrics: List[ComparisonMetrics]) -> go.Figure:
        """Create interactive comparison charts"""
        
        frames = [m.frame_idx for m in comparison_metrics]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Overall Angle Differences Over Time',
                'Joint Angle Comparison - Video 1 vs Video 2',
                'Highlighted Joints Distribution',
                'Key Joint Differences'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Overall differences over time
        overall_diffs = [m.overall_difference for m in comparison_metrics]
        fig.add_trace(
            go.Scatter(x=frames, y=overall_diffs, name="Overall Difference", 
                      line=dict(color="red", width=2)),
            row=1, col=1
        )
        
        # Plot 2: Individual joint comparisons for a sample of frames
        sample_frames = frames[::max(1, len(frames)//10)]  # Sample every 10th frame
        for joint in self.key_joints[:4]:  # Show top 4 joints to avoid clutter
            joint_diffs = []
            for frame_idx in sample_frames:
                if frame_idx < len(comparison_metrics):
                    diff = comparison_metrics[frame_idx].angle_differences.get(joint)
                    joint_diffs.append(diff if diff is not None else 0)
                else:
                    joint_diffs.append(0)
            
            fig.add_trace(
                go.Scatter(x=sample_frames, y=joint_diffs, name=joint, mode='lines+markers'),
                row=1, col=2
            )
        
        # Plot 3: Highlighted joints distribution
        highlight_counts = {}
        for metrics in comparison_metrics:
            for joint in metrics.highlight_joints:
                highlight_counts[joint] = highlight_counts.get(joint, 0) + 1
        
        if highlight_counts:
            fig.add_trace(
                go.Bar(x=list(highlight_counts.keys()), y=list(highlight_counts.values()),
                      name="Highlight Frequency"),
                row=2, col=1
            )
        
        # Plot 4: Average differences per joint
        avg_differences = {}
        for joint in self.key_joints:
            diffs = []
            for metrics in comparison_metrics:
                diff = metrics.angle_differences.get(joint)
                if diff is not None:
                    diffs.append(diff)
            avg_differences[joint] = np.mean(diffs) if diffs else 0
        
        fig.add_trace(
            go.Bar(x=list(avg_differences.keys()), y=list(avg_differences.values()),
                  name="Average Difference", marker_color='lightblue'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Tennis Form Comparison Analysis",
            showlegend=True
        )
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Frame", row=1, col=1)
        fig.update_xaxes(title_text="Frame", row=1, col=2)
        fig.update_xaxes(title_text="Joint", row=2, col=1)
        fig.update_xaxes(title_text="Joint", row=2, col=2)
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Angle Difference (degrees)", row=1, col=1)
        fig.update_yaxes(title_text="Angle Difference (degrees)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Average Difference (degrees)", row=2, col=2)
        
        return fig


def create_comparison_analyzer(difference_threshold: float = 20.0) -> FormComparisonAnalyzer:
    """Factory function to create form comparison analyzer"""
    return FormComparisonAnalyzer(difference_threshold)