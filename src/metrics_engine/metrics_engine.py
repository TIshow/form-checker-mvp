import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import csv
from pathlib import Path


@dataclass
class BodyPartMassCoefficients:
    """Body part mass distribution coefficients based on biomechanics research"""
    head: float = 0.081
    neck: float = 0.012
    torso: float = 0.497
    upper_arm_left: float = 0.028
    upper_arm_right: float = 0.028
    forearm_left: float = 0.016
    forearm_right: float = 0.016
    hand_left: float = 0.006
    hand_right: float = 0.006
    thigh_left: float = 0.1
    thigh_right: float = 0.1
    shin_left: float = 0.0465
    shin_right: float = 0.0465
    foot_left: float = 0.0145
    foot_right: float = 0.0145


@dataclass
class TennisMetrics:
    frame_idx: int
    centroid: Tuple[float, float]
    velocity: Tuple[float, float]
    acceleration: Tuple[float, float]
    ball_position: Optional[Tuple[float, float]]
    ball_velocity: Optional[Tuple[float, float]]
    impact_detected: bool
    racket_angle: Optional[float]
    body_alignment: Optional[float]
    stability_score: float


class MediaPipeLandmarkMapper:
    """Maps MediaPipe pose landmarks to body parts for mass calculations"""
    
    LANDMARK_GROUPS = {
        'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # face landmarks
        'neck': [11, 12],  # shoulder landmarks as neck approximation
        'torso': [11, 12, 23, 24],  # shoulders and hips
        'upper_arm_left': [11, 13],
        'upper_arm_right': [12, 14],
        'forearm_left': [13, 15],
        'forearm_right': [14, 16],
        'hand_left': [15, 17, 19, 21],
        'hand_right': [16, 18, 20, 22],
        'thigh_left': [23, 25],
        'thigh_right': [24, 26],
        'shin_left': [25, 27],
        'shin_right': [26, 28],
        'foot_left': [27, 29, 31],
        'foot_right': [28, 30, 32]
    }


class TennisMetricsEngine:
    def __init__(self, fps: float = 30.0, mass_coefficients: Optional[BodyPartMassCoefficients] = None):
        self.fps = fps
        self.mass_coeffs = mass_coefficients or BodyPartMassCoefficients()
        self.landmark_mapper = MediaPipeLandmarkMapper()
        
        # History for velocity/acceleration calculation
        self.centroid_history: List[Tuple[float, float]] = []
        self.ball_history: List[Optional[Tuple[float, float]]] = []
        self.metrics_history: List[TennisMetrics] = []
        
        # Impact detection parameters
        self.impact_threshold_velocity = 50.0  # pixels per frame
        self.impact_threshold_acceleration = 100.0  # pixels per frame^2
    
    def calculate_weighted_centroid(self, landmarks: List[Tuple[float, float]], 
                                   visibility: Optional[List[float]] = None) -> Tuple[float, float]:
        """Calculate center of gravity using body part mass coefficients"""
        if not landmarks or len(landmarks) < 33:
            return (0.0, 0.0)
        
        total_weighted_x = 0.0
        total_weighted_y = 0.0
        total_weight = 0.0
        
        for body_part, landmark_indices in self.landmark_mapper.LANDMARK_GROUPS.items():
            part_mass = getattr(self.mass_coeffs, body_part)
            
            # Calculate average position for this body part
            valid_landmarks = []
            for idx in landmark_indices:
                if (idx < len(landmarks) and landmarks[idx] and 
                    (visibility is None or visibility[idx] > 0.5)):
                    valid_landmarks.append(landmarks[idx])
            
            if valid_landmarks:
                part_x = np.mean([lm[0] for lm in valid_landmarks])
                part_y = np.mean([lm[1] for lm in valid_landmarks])
                
                total_weighted_x += part_x * part_mass
                total_weighted_y += part_y * part_mass
                total_weight += part_mass
        
        if total_weight > 0:
            return (total_weighted_x / total_weight, total_weighted_y / total_weight)
        else:
            # Fallback to simple average
            valid_landmarks = [lm for i, lm in enumerate(landmarks) 
                             if lm and (visibility is None or visibility[i] > 0.5)]
            if valid_landmarks:
                return (np.mean([lm[0] for lm in valid_landmarks]),
                       np.mean([lm[1] for lm in valid_landmarks]))
            return (0.0, 0.0)
    
    def calculate_velocity(self, current_pos: Tuple[float, float], 
                          previous_pos: Optional[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate velocity in pixels per second"""
        if previous_pos is None:
            return (0.0, 0.0)
        
        dx = current_pos[0] - previous_pos[0]
        dy = current_pos[1] - previous_pos[1]
        
        return (dx * self.fps, dy * self.fps)
    
    def calculate_acceleration(self, current_vel: Tuple[float, float],
                             previous_vel: Optional[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate acceleration in pixels per second^2"""
        if previous_vel is None:
            return (0.0, 0.0)
        
        dvx = current_vel[0] - previous_vel[0]
        dvy = current_vel[1] - previous_vel[1]
        
        return (dvx * self.fps, dvy * self.fps)
    
    def detect_impact(self, velocity: Tuple[float, float], 
                     acceleration: Tuple[float, float],
                     ball_velocity: Optional[Tuple[float, float]]) -> bool:
        """Detect tennis ball impact based on velocity and acceleration changes"""
        # Check for sudden velocity changes
        velocity_magnitude = np.sqrt(velocity[0]**2 + velocity[1]**2)
        acceleration_magnitude = np.sqrt(acceleration[0]**2 + acceleration[1]**2)
        
        impact_detected = (velocity_magnitude > self.impact_threshold_velocity or 
                          acceleration_magnitude > self.impact_threshold_acceleration)
        
        # Additional check: ball velocity change
        if ball_velocity:
            ball_vel_magnitude = np.sqrt(ball_velocity[0]**2 + ball_velocity[1]**2)
            if ball_vel_magnitude > self.impact_threshold_velocity * 2:
                impact_detected = True
        
        return impact_detected
    
    def calculate_stability_score(self, centroid_history: List[Tuple[float, float]], 
                                 window_size: int = 10) -> float:
        """Calculate stability score based on centroid movement consistency"""
        if len(centroid_history) < window_size:
            return 1.0
        
        recent_centroids = centroid_history[-window_size:]
        
        # Calculate standard deviation of movement
        x_coords = [c[0] for c in recent_centroids]
        y_coords = [c[1] for c in recent_centroids]
        
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        
        # Convert to stability score (lower deviation = higher stability)
        movement_variance = np.sqrt(x_std**2 + y_std**2)
        stability_score = max(0.0, 1.0 - (movement_variance / 100.0))  # Normalize to 0-1
        
        return stability_score
    
    def calculate_racket_angle(self, landmarks: List[Tuple[float, float]]) -> Optional[float]:
        """Estimate racket angle from arm position (simplified)"""
        if not landmarks or len(landmarks) < 16:
            return None
        
        # Use right arm landmarks (shoulder, elbow, wrist)
        shoulder = landmarks[12]  # right shoulder
        elbow = landmarks[14]     # right elbow
        wrist = landmarks[16]     # right wrist
        
        if not all([shoulder, elbow, wrist]):
            return None
        
        # Calculate angle between shoulder-elbow and elbow-wrist vectors
        vec1 = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
        vec2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])
        
        # Calculate angle
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        mag1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
        mag2 = np.sqrt(vec2[0]**2 + vec2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return None
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def process_frame(self, frame_idx: int, landmarks: List[Tuple[float, float]],
                     visibility: Optional[List[float]] = None,
                     ball_bbox: Optional[Tuple[int, int, int, int]] = None) -> TennisMetrics:
        """Process a single frame and calculate all metrics"""
        
        # Calculate weighted centroid
        centroid = self.calculate_weighted_centroid(landmarks, visibility)
        
        # Calculate velocities and accelerations
        previous_centroid = self.centroid_history[-1] if self.centroid_history else None
        velocity = self.calculate_velocity(centroid, previous_centroid)
        
        previous_velocity = (self.metrics_history[-1].velocity if self.metrics_history 
                           else (0.0, 0.0))
        acceleration = self.calculate_acceleration(velocity, previous_velocity)
        
        # Ball position and velocity
        ball_position = None
        ball_velocity = None
        if ball_bbox:
            ball_center_x = (ball_bbox[0] + ball_bbox[2]) / 2
            ball_center_y = (ball_bbox[1] + ball_bbox[3]) / 2
            ball_position = (ball_center_x, ball_center_y)
            
            previous_ball = self.ball_history[-1] if self.ball_history else None
            if previous_ball:
                ball_velocity = self.calculate_velocity(ball_position, previous_ball)
        
        # Impact detection
        impact_detected = self.detect_impact(velocity, acceleration, ball_velocity)
        
        # Other metrics
        racket_angle = self.calculate_racket_angle(landmarks)
        stability_score = self.calculate_stability_score(self.centroid_history + [centroid])
        
        # Create metrics object
        metrics = TennisMetrics(
            frame_idx=frame_idx,
            centroid=centroid,
            velocity=velocity,
            acceleration=acceleration,
            ball_position=ball_position,
            ball_velocity=ball_velocity,
            impact_detected=impact_detected,
            racket_angle=racket_angle,
            body_alignment=None,  # Could be calculated from pose
            stability_score=stability_score
        )
        
        # Update history
        self.centroid_history.append(centroid)
        self.ball_history.append(ball_position)
        self.metrics_history.append(metrics)
        
        # Keep history size manageable
        max_history = int(self.fps * 5)  # 5 seconds of history
        if len(self.centroid_history) > max_history:
            self.centroid_history = self.centroid_history[-max_history:]
            self.ball_history = self.ball_history[-max_history:]
            self.metrics_history = self.metrics_history[-max_history:]
        
        return metrics
    
    def export_to_csv(self, output_path: str):
        """Export metrics history to CSV file"""
        if not self.metrics_history:
            return
        
        fieldnames = [
            'frame_idx', 'centroid_x', 'centroid_y', 'velocity_x', 'velocity_y',
            'acceleration_x', 'acceleration_y', 'ball_x', 'ball_y', 
            'ball_velocity_x', 'ball_velocity_y', 'impact_detected', 
            'racket_angle', 'stability_score'
        ]
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for metrics in self.metrics_history:
                row = {
                    'frame_idx': metrics.frame_idx,
                    'centroid_x': metrics.centroid[0],
                    'centroid_y': metrics.centroid[1],
                    'velocity_x': metrics.velocity[0],
                    'velocity_y': metrics.velocity[1],
                    'acceleration_x': metrics.acceleration[0],
                    'acceleration_y': metrics.acceleration[1],
                    'ball_x': metrics.ball_position[0] if metrics.ball_position else '',
                    'ball_y': metrics.ball_position[1] if metrics.ball_position else '',
                    'ball_velocity_x': metrics.ball_velocity[0] if metrics.ball_velocity else '',
                    'ball_velocity_y': metrics.ball_velocity[1] if metrics.ball_velocity else '',
                    'impact_detected': metrics.impact_detected,
                    'racket_angle': metrics.racket_angle or '',
                    'stability_score': metrics.stability_score
                }
                writer.writerow(row)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the processed video"""
        if not self.metrics_history:
            return {}
        
        velocities = [np.sqrt(m.velocity[0]**2 + m.velocity[1]**2) for m in self.metrics_history]
        accelerations = [np.sqrt(m.acceleration[0]**2 + m.acceleration[1]**2) for m in self.metrics_history]
        stability_scores = [m.stability_score for m in self.metrics_history]
        impacts = [m.impact_detected for m in self.metrics_history]
        
        return {
            'total_frames': len(self.metrics_history),
            'avg_velocity': np.mean(velocities),
            'max_velocity': np.max(velocities),
            'avg_acceleration': np.mean(accelerations),
            'max_acceleration': np.max(accelerations),
            'avg_stability': np.mean(stability_scores),
            'total_impacts': sum(impacts),
            'impact_frames': [m.frame_idx for m in self.metrics_history if m.impact_detected]
        }


def create_metrics_engine(fps: float = 30.0, 
                         mass_coefficients: Optional[BodyPartMassCoefficients] = None) -> TennisMetricsEngine:
    return TennisMetricsEngine(fps, mass_coefficients)