#!/usr/bin/env python3
"""Basic functionality test for tennis form checker"""

import sys
import os
sys.path.insert(0, 'src')

def test_imports():
    """Test all module imports"""
    print("🔍 Testing imports...")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
        return False
    
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe {mp.__version__}")
    except ImportError as e:
        print(f"❌ MediaPipe: {e}")
        return False
    
    try:
        from frame_loader import create_frame_loader
        print("✅ frame_loader")
    except ImportError as e:
        print(f"❌ frame_loader: {e}")
        return False
    
    try:
        from visualizer import create_visualizer
        print("✅ visualizer")
    except ImportError as e:
        print(f"❌ visualizer: {e}")
        return False
    
    try:
        from metrics_engine import create_metrics_engine
        print("✅ metrics_engine")
    except ImportError as e:
        print(f"❌ metrics_engine: {e}")
        return False
    
    try:
        from yolo_detector import create_yolo_detector
        print("✅ yolo_detector")
    except ImportError as e:
        print(f"❌ yolo_detector: {e}")
        return False
    
    return True

def test_mock_processing():
    """Test mock video processing pipeline"""
    print("\n🔍 Testing mock processing...")
    
    try:
        from yolo_detector import create_yolo_detector
        from metrics_engine import create_metrics_engine
        from visualizer import create_visualizer
        import numpy as np
        
        # Create mock components
        detector = create_yolo_detector(use_mock=True)
        metrics_engine = create_metrics_engine(fps=30.0)
        visualizer = create_visualizer()
        
        # Create a mock frame
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test ball detection
        detections = detector.detect_and_track_balls(mock_frame)
        print(f"✅ Mock ball detection: {len(detections)} detections")
        
        # Test metrics calculation with empty landmarks
        metrics = metrics_engine.process_frame(0, [], None, None)
        print(f"✅ Metrics calculation: centroid at {metrics.centroid}")
        
        # Test visualization
        result_frame = visualizer.create_composite_frame(
            frame=mock_frame,
            pose_landmarks=[],
            centroid=metrics.centroid,
            frame_idx=0
        )
        print(f"✅ Visualization: output frame shape {result_frame.shape}")
        
        print("✅ Mock processing pipeline works!")
        return True
        
    except Exception as e:
        print(f"❌ Mock processing error: {e}")
        return False

def test_csv_export():
    """Test CSV export functionality"""
    print("\n🔍 Testing CSV export...")
    
    try:
        from metrics_engine import create_metrics_engine
        import tempfile
        import os
        
        metrics_engine = create_metrics_engine(fps=30.0)
        
        # Process some mock frames
        for i in range(5):
            metrics_engine.process_frame(i, [], None, None)
        
        # Export to temporary CSV
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            metrics_engine.export_to_csv(tmp.name)
            
            # Check if file was created and has content
            if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                print("✅ CSV export successful")
                os.unlink(tmp.name)
                return True
            else:
                print("❌ CSV export failed: empty file")
                return False
                
    except Exception as e:
        print(f"❌ CSV export error: {e}")
        return False

def main():
    """Run all tests"""
    print("🎾 Tennis Form Checker - Basic Functionality Test\n")
    
    tests = [
        test_imports,
        test_mock_processing,
        test_csv_export
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The system is ready for basic functionality.")
        print("\n📝 Next steps:")
        print("1. Run: uv run streamlit run src/main.py")
        print("2. Upload a test video")
        print("3. Use 'Mock YOLO' option for testing without real models")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()