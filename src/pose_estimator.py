import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Install with: pip install mediapipe")


@dataclass
class PoseLandmark:
    x: float
    y: float
    z: float
    visibility: float


class MediaPipePoseEstimator:
    """MediaPipe pose estimation wrapper"""
    
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, enable_segmentation: bool = False):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required but not installed. Install with: pip install mediapipe")
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=enable_segmentation
        )
        
        # Landmark names for reference
        self.landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process a single frame and extract pose landmarks"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks is None:
            return None
        
        # Extract landmarks
        landmarks = []
        visibility_scores = []
        
        height, width = frame.shape[:2]
        
        for landmark in results.pose_landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z  # Relative depth
            visibility = landmark.visibility
            
            landmarks.append((x, y))
            visibility_scores.append(visibility)
        
        return {
            'landmarks': landmarks,
            'visibility': visibility_scores,
            'raw_landmarks': results.pose_landmarks,
            'segmentation_mask': results.segmentation_mask
        }
    
    def get_landmark_by_name(self, landmarks: List[Tuple[int, int]], 
                           name: str) -> Optional[Tuple[int, int]]:
        """Get landmark by name"""
        try:
            index = self.landmark_names.index(name)
            return landmarks[index] if index < len(landmarks) else None
        except ValueError:
            return None
    
    def calculate_angle(self, landmarks: List[Tuple[int, int]], 
                       point1_name: str, point2_name: str, point3_name: str) -> Optional[float]:
        """Calculate angle between three landmarks"""
        p1 = self.get_landmark_by_name(landmarks, point1_name)
        p2 = self.get_landmark_by_name(landmarks, point2_name)
        p3 = self.get_landmark_by_name(landmarks, point3_name)
        
        if not all([p1, p2, p3]):
            return None
        
        # Calculate vectors
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Calculate angle using dot product
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude1 = np.sqrt(v1[0]**2 + v1[1]**2)
        magnitude2 = np.sqrt(v2[0]**2 + v2[1]**2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return None
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def get_key_angles(self, landmarks: List[Tuple[int, int]]) -> Dict[str, Optional[float]]:
        """Calculate key tennis-relevant angles"""
        return {
            'left_elbow': self.calculate_angle(landmarks, 'left_shoulder', 'left_elbow', 'left_wrist'),
            'right_elbow': self.calculate_angle(landmarks, 'right_shoulder', 'right_elbow', 'right_wrist'),
            'left_knee': self.calculate_angle(landmarks, 'left_hip', 'left_knee', 'left_ankle'),
            'right_knee': self.calculate_angle(landmarks, 'right_hip', 'right_knee', 'right_ankle'),
            'left_shoulder': self.calculate_angle(landmarks, 'left_elbow', 'left_shoulder', 'left_hip'),
            'right_shoulder': self.calculate_angle(landmarks, 'right_elbow', 'right_shoulder', 'right_hip'),
        }
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'pose'):
            self.pose.close()


def create_pose_estimator(**kwargs) -> MediaPipePoseEstimator:
    """Factory function to create pose estimator"""
    return MediaPipePoseEstimator(**kwargs)