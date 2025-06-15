import cv2
from typing import Generator, Tuple, Optional
import numpy as np


class FrameLoader:
    def __init__(self, video_path: str, buffer_size: int = 1):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.cap = None
        self._fps = None
        self._total_frames = None
        self._width = None
        self._height = None
    
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def total_frames(self) -> int:
        return self._total_frames
    
    @property
    def video_info(self) -> dict:
        return {
            'fps': self._fps,
            'total_frames': self._total_frames,
            'width': self._width,
            'height': self._height,
            'duration': self._total_frames / self._fps if self._fps > 0 else 0
        }
    
    def load_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        if not self.cap:
            raise RuntimeError("FrameLoader not initialized. Use with context manager.")
        
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            yield frame_idx, frame
            frame_idx += 1
    
    def load_frames_with_progress(self) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        if not self.cap:
            raise RuntimeError("FrameLoader not initialized. Use with context manager.")
        
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            progress = frame_idx / self._total_frames if self._total_frames > 0 else 0
            yield frame_idx, frame, progress
            frame_idx += 1
    
    def seek_frame(self, frame_number: int) -> Optional[np.ndarray]:
        if not self.cap:
            raise RuntimeError("FrameLoader not initialized. Use with context manager.")
        
        if frame_number < 0 or frame_number >= self._total_frames:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None


def create_frame_loader(video_path: str, buffer_size: int = 1) -> FrameLoader:
    return FrameLoader(video_path, buffer_size)