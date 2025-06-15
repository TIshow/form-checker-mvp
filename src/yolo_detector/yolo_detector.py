import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import defaultdict, deque

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch torchvision ultralytics")


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    track_id: Optional[int] = None


@dataclass
class TrackState:
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    age: int
    hits: int
    time_since_update: int
    velocity: Tuple[float, float] = (0.0, 0.0)


class SimpleTracker:
    """Simple ball tracking algorithm based on IoU and distance"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100.0):
        self.max_disappeared = max_disappeared  # frames before removing track
        self.max_distance = max_distance  # maximum distance for association
        self.next_id = 0
        self.tracks: Dict[int, TrackState] = {}
        self.frame_count = 0
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_center_distance(self, box1: Tuple[int, int, int, int], 
                                 box2: Tuple[int, int, int, int]) -> float:
        """Calculate distance between centers of two bounding boxes"""
        center1_x = (box1[0] + box1[2]) / 2
        center1_y = (box1[1] + box1[3]) / 2
        center2_x = (box2[0] + box2[2]) / 2
        center2_y = (box2[1] + box2[3]) / 2
        
        return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    def predict_next_position(self, track: TrackState) -> Tuple[int, int, int, int]:
        """Predict next position based on velocity"""
        x1, y1, x2, y2 = track.bbox
        vx, vy = track.velocity
        
        # Predict center position
        center_x = (x1 + x2) / 2 + vx
        center_y = (y1 + y2) / 2 + vy
        
        # Maintain box size
        width = x2 - x1
        height = y2 - y1
        
        return (int(center_x - width/2), int(center_y - height/2),
                int(center_x + width/2), int(center_y + height/2))
    
    def update_velocity(self, track: TrackState, new_bbox: Tuple[int, int, int, int]):
        """Update track velocity based on movement"""
        old_center_x = (track.bbox[0] + track.bbox[2]) / 2
        old_center_y = (track.bbox[1] + track.bbox[3]) / 2
        new_center_x = (new_bbox[0] + new_bbox[2]) / 2
        new_center_y = (new_bbox[1] + new_bbox[3]) / 2
        
        # Calculate velocity (pixels per frame)
        vx = new_center_x - old_center_x
        vy = new_center_y - old_center_y
        
        # Apply smoothing (exponential moving average)
        alpha = 0.7
        track.velocity = (alpha * vx + (1 - alpha) * track.velocity[0],
                         alpha * vy + (1 - alpha) * track.velocity[1])
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update tracker with new detections"""
        self.frame_count += 1
        
        # Predict positions for existing tracks
        predicted_tracks = {}
        for track_id, track in self.tracks.items():
            predicted_bbox = self.predict_next_position(track)
            predicted_tracks[track_id] = predicted_bbox
        
        # Association using Hungarian algorithm (simplified version)
        matched_tracks = []
        unmatched_detections = list(detections)
        
        # Simple greedy matching based on IoU and distance
        for track_id, predicted_bbox in predicted_tracks.items():
            best_detection = None
            best_score = 0.0
            best_idx = -1
            
            for i, detection in enumerate(unmatched_detections):
                # Calculate matching score (combination of IoU and distance)
                iou = self.calculate_iou(predicted_bbox, detection.bbox)
                distance = self.calculate_center_distance(predicted_bbox, detection.bbox)
                
                # Normalize distance score
                distance_score = max(0, 1 - distance / self.max_distance)
                
                # Combined score
                score = 0.7 * iou + 0.3 * distance_score
                
                if score > best_score and score > 0.3:  # minimum threshold
                    best_score = score
                    best_detection = detection
                    best_idx = i
            
            # Update matched track
            if best_detection:
                track = self.tracks[track_id]
                self.update_velocity(track, best_detection.bbox)
                track.bbox = best_detection.bbox
                track.confidence = best_detection.confidence
                track.hits += 1
                track.time_since_update = 0
                track.age += 1
                
                # Assign track ID to detection
                best_detection.track_id = track_id
                matched_tracks.append(best_detection)
                
                # Remove from unmatched
                unmatched_detections.pop(best_idx)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            track = TrackState(
                track_id=self.next_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                age=1,
                hits=1,
                time_since_update=0
            )
            self.tracks[self.next_id] = track
            detection.track_id = self.next_id
            matched_tracks.append(detection)
            self.next_id += 1
        
        # Update unmatched tracks
        for track_id, track in list(self.tracks.items()):
            if track_id not in [d.track_id for d in matched_tracks]:
                track.time_since_update += 1
                track.age += 1
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        return matched_tracks


class YOLOBallDetector:
    """YOLO-based tennis ball detector with tracking"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu',
                 confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = None
        self.tracker = SimpleTracker()
        
        # Tennis ball class ID (usually 32 in COCO dataset, but may vary)
        self.ball_class_id = 32  # sports ball in COCO
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load YOLO model"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Cannot load YOLO model.")
            self.model = None
            return
            
        try:
            # Try to load YOLOv5/YOLOv8 model
            if 'yolov5' in model_path.lower():
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                          path=model_path, device=self.device)
            else:
                # Assume it's a standard model or custom implementation
                self.model = torch.jit.load(model_path, map_location=self.device)
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            # Fallback to pretrained YOLOv5
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
                                          device=self.device)
                print("Using pretrained YOLOv5s model")
            except Exception as e2:
                print(f"Failed to load pretrained model: {e2}")
                self.model = None
    
    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in frame using YOLO"""
        if self.model is None:
            return []
        
        try:
            # Run inference
            results = self.model(frame)
            
            # Parse results
            detections = []
            if hasattr(results, 'pandas'):
                # YOLOv5 format
                df = results.pandas().xyxy[0]
                for _, row in df.iterrows():
                    if row['confidence'] >= self.confidence_threshold:
                        detection = Detection(
                            bbox=(int(row['xmin']), int(row['ymin']), 
                                 int(row['xmax']), int(row['ymax'])),
                            confidence=float(row['confidence']),
                            class_id=int(row['class'])
                        )
                        detections.append(detection)
            else:
                # Generic format
                predictions = results.pred[0] if hasattr(results, 'pred') else results
                for pred in predictions:
                    if len(pred) >= 6 and pred[4] >= self.confidence_threshold:
                        detection = Detection(
                            bbox=(int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])),
                            confidence=float(pred[4]),
                            class_id=int(pred[5])
                        )
                        detections.append(detection)
            
            return detections
        
        except Exception as e:
            print(f"Detection failed: {e}")
            return []
    
    def filter_ball_detections(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections to keep only tennis balls"""
        return [d for d in detections if d.class_id == self.ball_class_id]
    
    def detect_and_track_balls(self, frame: np.ndarray) -> List[Detection]:
        """Detect tennis balls and update tracking"""
        # Get all detections
        all_detections = self.detect_objects(frame)
        
        # Filter for balls only
        ball_detections = self.filter_ball_detections(all_detections)
        
        # Apply Non-Maximum Suppression
        if len(ball_detections) > 1:
            ball_detections = self.apply_nms(ball_detections)
        
        # Update tracker
        tracked_balls = self.tracker.update(ball_detections)
        
        return tracked_balls
    
    def apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression"""
        if len(detections) <= 1:
            return detections
        
        # Convert to format for NMS
        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])
        
        # Apply OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), 
            self.confidence_threshold, self.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        
        return []
    
    def get_best_ball_detection(self, detections: List[Detection]) -> Optional[Detection]:
        """Get the most confident ball detection"""
        if not detections:
            return None
        
        # Prefer tracked detections, then highest confidence
        tracked_detections = [d for d in detections if d.track_id is not None]
        if tracked_detections:
            return max(tracked_detections, key=lambda x: x.confidence)
        
        return max(detections, key=lambda x: x.confidence)
    
    def reset_tracker(self):
        """Reset the tracking state"""
        self.tracker = SimpleTracker()


class MockYOLODetector(YOLOBallDetector):
    """Mock detector for testing without actual YOLO model"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_count = 0
        np.random.seed(42)  # For reproducible mock detections
    
    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """Generate mock ball detections"""
        self.frame_count += 1
        
        # Simulate ball movement across the frame
        height, width = frame.shape[:2]
        
        # Generate 0-2 ball detections per frame
        num_detections = np.random.choice([0, 1, 2], p=[0.3, 0.6, 0.1])
        
        detections = []
        for i in range(num_detections):
            # Simulate ball trajectory
            t = self.frame_count / 100.0
            x = int(width * 0.2 + (width * 0.6) * (t % 1.0))
            y = int(height * 0.5 + height * 0.2 * np.sin(t * 3.14))
            
            # Add some noise
            x += np.random.randint(-20, 21)
            y += np.random.randint(-20, 21)
            
            # Ensure bounds
            x = max(10, min(width - 30, x))
            y = max(10, min(height - 30, y))
            
            detection = Detection(
                bbox=(x, y, x + 20, y + 20),
                confidence=0.7 + 0.3 * np.random.random(),
                class_id=self.ball_class_id
            )
            detections.append(detection)
        
        return detections


def create_yolo_detector(model_path: Optional[str] = None, device: str = 'cpu',
                        use_mock: bool = False, **kwargs) -> YOLOBallDetector:
    """Factory function to create YOLO detector"""
    if use_mock:
        return MockYOLODetector(**kwargs)
    else:
        return YOLOBallDetector(model_path, device, **kwargs)