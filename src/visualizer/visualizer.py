import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class DrawingConfig:
    pose_color: Tuple[int, int, int] = (0, 255, 0)
    pose_thickness: int = 2
    joint_color: Tuple[int, int, int] = (255, 0, 0)
    joint_radius: int = 3
    centroid_color: Tuple[int, int, int] = (0, 0, 255)
    centroid_radius: int = 8
    ball_color: Tuple[int, int, int] = (255, 255, 0)
    ball_thickness: int = 2
    text_color: Tuple[int, int, int] = (255, 255, 255)
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX
    text_scale: float = 0.6
    text_thickness: int = 2


class TennisVisualizer:
    def __init__(self, config: Optional[DrawingConfig] = None):
        self.config = config or DrawingConfig()
        
        # MediaPipe pose connections
        self.pose_connections = [
            (11, 13), (13, 15), (12, 14), (14, 16),  # arms
            (11, 12),  # shoulders
            (11, 23), (12, 24),  # torso
            (23, 24),  # hips
            (23, 25), (25, 27), (27, 29), (29, 31),  # left leg
            (24, 26), (26, 28), (28, 30), (30, 32),  # right leg
            (15, 17), (15, 19), (15, 21),  # left hand
            (16, 18), (16, 20), (16, 22),  # right hand
            (27, 31), (28, 32)  # feet
        ]
    
    def draw_pose(self, frame: np.ndarray, landmarks: List[Tuple[int, int]], 
                  visibility: Optional[List[float]] = None) -> np.ndarray:
        if not landmarks:
            return frame
        
        frame_copy = frame.copy()
        
        # Draw connections
        for connection in self.pose_connections:
            start_idx, end_idx = connection
            if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                landmarks[start_idx] and landmarks[end_idx]):
                
                if visibility is None or (visibility[start_idx] > 0.5 and visibility[end_idx] > 0.5):
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    cv2.line(frame_copy, start_point, end_point, 
                            self.config.pose_color, self.config.pose_thickness)
        
        # Draw joints
        for i, landmark in enumerate(landmarks):
            if landmark and (visibility is None or visibility[i] > 0.5):
                cv2.circle(frame_copy, landmark, self.config.joint_radius, 
                          self.config.joint_color, -1)
        
        return frame_copy
    
    def draw_centroid(self, frame: np.ndarray, centroid: Tuple[float, float], 
                     frame_idx: int) -> np.ndarray:
        frame_copy = frame.copy()
        
        # Convert to integer coordinates
        center_point = (int(centroid[0]), int(centroid[1]))
        
        # Draw centroid point
        cv2.circle(frame_copy, center_point, self.config.centroid_radius, 
                  self.config.centroid_color, -1)
        
        # Add label
        label = f"COG: ({center_point[0]}, {center_point[1]})"
        label_pos = (center_point[0] + 15, center_point[1] - 15)
        cv2.putText(frame_copy, label, label_pos,
                   self.config.text_font, self.config.text_scale, 
                   self.config.text_color, self.config.text_thickness)
        
        return frame_copy
    
    def draw_ball_detection(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                           confidence: float, ball_id: Optional[int] = None) -> np.ndarray:
        frame_copy = frame.copy()
        
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), 
                     self.config.ball_color, self.config.ball_thickness)
        
        # Add label
        label = f"Ball {ball_id}: {confidence:.2f}" if ball_id else f"Ball: {confidence:.2f}"
        cv2.putText(frame_copy, label, (x1, y1 - 10),
                   self.config.text_font, self.config.text_scale, 
                   self.config.text_color, self.config.text_thickness)
        
        return frame_copy
    
    def draw_metrics(self, frame: np.ndarray, metrics: Dict[str, Any], 
                    frame_idx: int) -> np.ndarray:
        frame_copy = frame.copy()
        
        y_offset = 30
        spacing = 25
        
        # Frame info
        cv2.putText(frame_copy, f"Frame: {frame_idx}", (10, y_offset),
                   self.config.text_font, self.config.text_scale, 
                   self.config.text_color, self.config.text_thickness)
        y_offset += spacing
        
        # Add metrics
        for key, value in metrics.items():
            if isinstance(value, float):
                text = f"{key}: {value:.2f}"
            else:
                text = f"{key}: {value}"
            
            cv2.putText(frame_copy, text, (10, y_offset),
                       self.config.text_font, self.config.text_scale, 
                       self.config.text_color, self.config.text_thickness)
            y_offset += spacing
        
        return frame_copy
    
    def create_composite_frame(self, frame: np.ndarray, pose_landmarks: List[Tuple[int, int]],
                              centroid: Optional[Tuple[float, float]] = None,
                              ball_bbox: Optional[Tuple[int, int, int, int]] = None,
                              ball_confidence: Optional[float] = None,
                              metrics: Optional[Dict[str, Any]] = None,
                              frame_idx: int = 0) -> np.ndarray:
        
        result_frame = frame.copy()
        
        # Draw pose
        if pose_landmarks:
            result_frame = self.draw_pose(result_frame, pose_landmarks)
        
        # Draw centroid
        if centroid:
            result_frame = self.draw_centroid(result_frame, centroid, frame_idx)
        
        # Draw ball detection
        if ball_bbox and ball_confidence:
            result_frame = self.draw_ball_detection(result_frame, ball_bbox, ball_confidence)
        
        # Draw metrics
        if metrics:
            result_frame = self.draw_metrics(result_frame, metrics, frame_idx)
        
        return result_frame


class VideoWriter:
    def __init__(self, output_path: str, fps: float, width: int, height: int,
                 fourcc: str = 'mp4v'):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = None
    
    def __enter__(self):
        self.writer = cv2.VideoWriter(self.output_path, self.fourcc, 
                                     self.fps, (self.width, self.height))
        if not self.writer.isOpened():
            raise ValueError(f"Cannot create video writer for: {self.output_path}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.release()
    
    def write_frame(self, frame: np.ndarray):
        if not self.writer:
            raise RuntimeError("VideoWriter not initialized. Use with context manager.")
        
        # Ensure frame is the correct size
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        
        self.writer.write(frame)


def create_visualizer(config: Optional[DrawingConfig] = None) -> TennisVisualizer:
    return TennisVisualizer(config)


def create_video_writer(output_path: str, fps: float, width: int, height: int,
                       fourcc: str = 'mp4v') -> VideoWriter:
    return VideoWriter(output_path, fps, width, height, fourcc)