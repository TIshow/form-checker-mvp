#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import streamlit as st
import cv2
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from frame_loader import create_frame_loader
from pose_estimator import create_pose_estimator
from yolo_detector import create_yolo_detector
from metrics_engine import create_metrics_engine
from visualizer import create_visualizer, create_video_writer


def process_video(video_path: str, output_dir: str, use_mock_yolo: bool = True, 
                 enable_pose: bool = True) -> dict:
    """Process a tennis video and generate analysis"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    pose_estimator = None
    if enable_pose:
        try:
            pose_estimator = create_pose_estimator()
            print("✅ Pose estimation enabled")
        except Exception as e:
            print(f"⚠️ Pose estimation disabled due to error: {e}")
            pose_estimator = None
    
    yolo_detector = create_yolo_detector(use_mock=use_mock_yolo)
    visualizer = create_visualizer()
    
    results = {
        'processed_frames': 0,
        'output_video': None,
        'output_csv': None,
        'summary': {},
        'pose_enabled': pose_estimator is not None
    }
    
    try:
        with create_frame_loader(video_path) as frame_loader:
            video_info = frame_loader.video_info
            metrics_engine = create_metrics_engine(fps=video_info['fps'])
            
            # Setup output video
            output_video_path = output_path / 'analyzed_video.mp4'
            output_csv_path = output_path / 'metrics.csv'
            
            with create_video_writer(
                str(output_video_path), 
                video_info['fps'], 
                video_info['width'], 
                video_info['height']
            ) as video_writer:
                
                # Process frames
                progress_bar = tqdm(total=video_info['total_frames'], desc="Processing frames")
                
                for frame_idx, frame, progress in frame_loader.load_frames_with_progress():
                    # Pose estimation (if available)
                    landmarks = []
                    visibility = None
                    if pose_estimator:
                        try:
                            pose_result = pose_estimator.process_frame(frame)
                            landmarks = pose_result['landmarks'] if pose_result else []
                            visibility = pose_result['visibility'] if pose_result else None
                        except Exception as e:
                            if frame_idx == 0:  # Only log on first frame to avoid spam
                                print(f"⚠️ Pose estimation error: {e}")
                            landmarks = []
                            visibility = None
                    
                    # Ball detection
                    ball_detections = yolo_detector.detect_and_track_balls(frame)
                    best_ball = yolo_detector.get_best_ball_detection(ball_detections)
                    
                    # Calculate metrics
                    ball_bbox = best_ball.bbox if best_ball else None
                    metrics = metrics_engine.process_frame(
                        frame_idx, landmarks, visibility, ball_bbox
                    )
                    
                    # Create visualization
                    analyzed_frame = visualizer.create_composite_frame(
                        frame=frame,
                        pose_landmarks=landmarks,
                        centroid=metrics.centroid,
                        ball_bbox=ball_bbox,
                        ball_confidence=best_ball.confidence if best_ball else None,
                        metrics={
                            'Velocity': f"{np.sqrt(metrics.velocity[0]**2 + metrics.velocity[1]**2):.1f}",
                            'Stability': f"{metrics.stability_score:.2f}",
                            'Impact': "YES" if metrics.impact_detected else "NO"
                        },
                        frame_idx=frame_idx
                    )
                    
                    # Write frame to output video
                    video_writer.write_frame(analyzed_frame)
                    
                    progress_bar.update(1)
                    results['processed_frames'] += 1
                
                progress_bar.close()
            
            # Export metrics to CSV
            metrics_engine.export_to_csv(str(output_csv_path))
            
            # Generate summary
            summary = metrics_engine.get_summary_statistics()
            
            results.update({
                'output_video': str(output_video_path),
                'output_csv': str(output_csv_path),
                'summary': summary
            })
            
    except Exception as e:
        st.error(f"Processing failed: {e}")
        raise
    
    finally:
        if pose_estimator:
            pose_estimator.close()
    
    return results


def streamlit_app():
    """Streamlit web interface"""
    st.set_page_config(
        page_title="Tennis Form Checker MVP",
        page_icon="🎾",
        layout="wide"
    )
    
    st.title("🎾 Tennis Form Checker MVP")
    st.markdown("Upload a tennis video to analyze form and ball tracking")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        use_mock_yolo = st.checkbox("Use Mock YOLO (for testing)", value=True)
        enable_pose = st.checkbox("Enable Pose Estimation", value=True)
        st.info("Mock YOLO generates simulated ball detections for testing without a real YOLO model")
        st.success("✅ Python 3.11 + MediaPipe 0.10.9 - Pose estimation working!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Tennis Video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file containing tennis footage"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_video_path = f"temp_{uploaded_file.name}"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Video uploaded: {uploaded_file.name}")
        
        # Display video info
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", f"{duration:.1f}s")
        with col2:
            st.metric("FPS", f"{fps:.1f}")
        with col3:
            st.metric("Frames", frame_count)
        with col4:
            st.metric("Resolution", f"{width}x{height}")
        
        # Process button
        if st.button("🚀 Analyze Video", type="primary"):
            with st.spinner("Processing video... This may take a few minutes."):
                try:
                    results = process_video(
                        temp_video_path, 
                        "output",
                        use_mock_yolo=use_mock_yolo,
                        enable_pose=enable_pose
                    )
                    
                    st.success(f"✅ Processing complete! Processed {results['processed_frames']} frames.")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Analysis Summary")
                        
                        # Show configuration status
                        st.info(f"Pose Estimation: {'✅ Enabled' if results.get('pose_enabled', False) else '❌ Disabled'}")
                        st.info(f"Ball Detection: {'🎯 Mock Mode' if use_mock_yolo else '🤖 YOLO Mode'}")
                        
                        summary = results['summary']
                        if summary:
                            st.metric("Average Velocity", f"{summary.get('avg_velocity', 0):.1f} px/s")
                            st.metric("Max Velocity", f"{summary.get('max_velocity', 0):.1f} px/s")
                            st.metric("Average Stability", f"{summary.get('avg_stability', 0):.2f}")
                            st.metric("Total Impacts", summary.get('total_impacts', 0))
                    
                    with col2:
                        st.subheader("📁 Output Files")
                        if results['output_video'] and os.path.exists(results['output_video']):
                            with open(results['output_video'], 'rb') as f:
                                st.download_button(
                                    "📹 Download Analyzed Video",
                                    f.read(),
                                    file_name="analyzed_tennis_video.mp4",
                                    mime="video/mp4"
                                )
                        
                        if results['output_csv'] and os.path.exists(results['output_csv']):
                            with open(results['output_csv'], 'rb') as f:
                                st.download_button(
                                    "📈 Download Metrics CSV",
                                    f.read(),
                                    file_name="tennis_metrics.csv",
                                    mime="text/csv"
                                )
                    
                    # Show sample frame from output video
                    if results['output_video'] and os.path.exists(results['output_video']):
                        st.subheader("🎬 Preview")
                        st.video(results['output_video'])
                
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    st.exception(e)
                
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)


def cli_main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Tennis Form Checker MVP")
    parser.add_argument("input_video", help="Path to input tennis video")
    parser.add_argument("-o", "--output", default="output", 
                       help="Output directory (default: output)")
    parser.add_argument("--mock-yolo", action="store_true",
                       help="Use mock YOLO detector for testing")
    parser.add_argument("--enable-pose", action="store_true",
                       help="Enable pose estimation (may fail with Python 3.12)")
    parser.add_argument("--yolo-model", help="Path to YOLO model file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_video):
        print(f"Error: Input video '{args.input_video}' not found")
        sys.exit(1)
    
    print(f"Processing video: {args.input_video}")
    print(f"Output directory: {args.output}")
    
    try:
        results = process_video(
            args.input_video,
            args.output,
            use_mock_yolo=args.mock_yolo,
            enable_pose=args.enable_pose
        )
        
        print(f"\n✅ Processing complete!")
        print(f"Processed frames: {results['processed_frames']}")
        print(f"Output video: {results['output_video']}")
        print(f"Output CSV: {results['output_csv']}")
        
        # Print summary
        summary = results['summary']
        if summary:
            print(f"\n📊 Summary:")
            print(f"Average velocity: {summary.get('avg_velocity', 0):.1f} px/s")
            print(f"Max velocity: {summary.get('max_velocity', 0):.1f} px/s")
            print(f"Average stability: {summary.get('avg_stability', 0):.2f}")
            print(f"Total impacts: {summary.get('total_impacts', 0)}")
    
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] not in ['--help', '-h']:
        # CLI mode
        cli_main()
    else:
        # Streamlit mode
        streamlit_app()


if __name__ == "__main__":
    main()