#!/usr/bin/env python3
"""Test video processing without pose estimation"""

import sys
import os
import numpy as np
import cv2
sys.path.insert(0, 'src')

from main import process_video

def create_test_video(filename: str, duration_seconds: int = 3, fps: int = 30):
    """Create a simple test video"""
    width, height = 640, 480
    total_frames = duration_seconds * fps
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        # Create a frame with moving circle (simulating a ball)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(height):
            frame[y, :] = [0, int(255 * y / height * 0.3), 0]
        
        # Moving circle
        center_x = int(50 + (width - 100) * frame_num / total_frames)
        center_y = int(height // 2 + 50 * np.sin(frame_num * 0.2))
        cv2.circle(frame, (center_x, center_y), 15, (0, 255, 255), -1)
        
        # Add some text
        cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Created test video: {filename}")

def main():
    """Test video processing without pose estimation"""
    print("🎾 Testing Tennis Form Checker without pose estimation\n")
    
    # Create test video
    test_video = "test_video.mp4"
    create_test_video(test_video)
    
    try:
        print("🔄 Processing test video...")
        results = process_video(
            test_video,
            "test_output",
            use_mock_yolo=True,
            enable_pose=False
        )
        
        print(f"\n✅ Processing successful!")
        print(f"Frames processed: {results['processed_frames']}")
        print(f"Pose enabled: {results['pose_enabled']}")
        print(f"Output video: {results['output_video']}")
        print(f"Output CSV: {results['output_csv']}")
        
        # Check output files exist
        if os.path.exists(results['output_video']):
            print("✅ Output video created")
        else:
            print("❌ Output video not found")
            
        if os.path.exists(results['output_csv']):
            print("✅ Output CSV created")
        else:
            print("❌ Output CSV not found")
        
        print("\n🎉 Test completed successfully!")
        print("You can now try the Streamlit app with pose estimation disabled.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if os.path.exists(test_video):
            os.remove(test_video)
            print(f"🧹 Cleaned up test video")

if __name__ == "__main__":
    main()