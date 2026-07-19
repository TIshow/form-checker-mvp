#!/usr/bin/env python3
"""Test comparison with more realistic human figures"""

import sys
import os
import numpy as np
import cv2
sys.path.insert(0, 'src')

from comparison_analyzer import create_comparison_analyzer


def draw_realistic_person(frame, center_x, center_y, arm_angle_deg, leg_width_factor, frame_num):
    """Draw a more realistic human figure that MediaPipe can detect"""
    
    # Convert angle to radians
    arm_angle = np.radians(arm_angle_deg)
    
    # Body proportions (more realistic)
    head_radius = 20
    shoulder_width = 50
    torso_length = 80
    arm_length = 70
    leg_length = 90
    
    # Calculate key points
    head_center = (center_x, center_y - 140)
    left_shoulder = (center_x - shoulder_width//2, center_y - 100)
    right_shoulder = (center_x + shoulder_width//2, center_y - 100)
    torso_bottom = (center_x, center_y - 20)
    left_hip = (center_x - 30, center_y - 20)
    right_hip = (center_x + 30, center_y - 20)
    
    # Leg positions with varying width
    leg_spread = int(25 * leg_width_factor)
    left_foot = (center_x - leg_spread, center_y + leg_length)
    right_foot = (center_x + leg_spread, center_y + leg_length)
    left_knee = (center_x - leg_spread//2, center_y + leg_length//2)
    right_knee = (center_x + leg_spread//2, center_y + leg_length//2)
    
    # Arm positions with varying angles
    left_elbow = (left_shoulder[0] - 30, left_shoulder[1] + 40)
    left_hand = (left_elbow[0] - 20, left_elbow[1] + 30)
    
    # Right arm with dynamic angle (tennis swing)
    right_elbow = (
        right_shoulder[0] + int(35 * np.cos(arm_angle + np.pi/4)),
        right_shoulder[1] + int(35 * np.sin(arm_angle + np.pi/4))
    )
    right_hand = (
        right_elbow[0] + int(35 * np.cos(arm_angle)),
        right_elbow[1] + int(35 * np.sin(arm_angle))
    )
    
    # Draw filled shapes for better detection
    
    # Head (filled circle)
    cv2.circle(frame, head_center, head_radius, (220, 180, 140), -1)  # Skin color
    cv2.circle(frame, head_center, head_radius, (255, 255, 255), 2)   # Outline
    
    # Face features for better detection
    eye_offset = 8
    cv2.circle(frame, (head_center[0] - eye_offset, head_center[1] - 5), 3, (50, 50, 50), -1)
    cv2.circle(frame, (head_center[0] + eye_offset, head_center[1] - 5), 3, (50, 50, 50), -1)
    cv2.circle(frame, (head_center[0], head_center[1] + 5), 2, (100, 50, 50), -1)
    
    # Torso (filled rectangle)
    torso_rect = [
        (left_shoulder[0], left_shoulder[1]),
        (right_shoulder[0], right_shoulder[1]),
        (right_hip[0], right_hip[1]),
        (left_hip[0], left_hip[1])
    ]
    cv2.fillPoly(frame, [np.array(torso_rect)], (100, 150, 200))  # Shirt color
    cv2.polylines(frame, [np.array(torso_rect)], True, (255, 255, 255), 2)
    
    # Arms (thick lines with joints)
    # Left arm
    cv2.line(frame, left_shoulder, left_elbow, (220, 180, 140), 8)
    cv2.line(frame, left_elbow, left_hand, (220, 180, 140), 8)
    cv2.circle(frame, left_shoulder, 6, (255, 255, 255), -1)
    cv2.circle(frame, left_elbow, 6, (255, 255, 255), -1)
    cv2.circle(frame, left_hand, 6, (255, 255, 255), -1)
    
    # Right arm (with tennis racket)
    cv2.line(frame, right_shoulder, right_elbow, (220, 180, 140), 8)
    cv2.line(frame, right_elbow, right_hand, (220, 180, 140), 8)
    cv2.circle(frame, right_shoulder, 6, (255, 255, 255), -1)
    cv2.circle(frame, right_elbow, 6, (255, 255, 255), -1)
    cv2.circle(frame, right_hand, 6, (255, 255, 255), -1)
    
    # Tennis racket
    racket_end = (right_hand[0] + int(20 * np.cos(arm_angle)), 
                  right_hand[1] + int(20 * np.sin(arm_angle)))
    cv2.line(frame, right_hand, racket_end, (139, 69, 19), 6)  # Racket handle
    cv2.ellipse(frame, racket_end, (15, 10), int(np.degrees(arm_angle)), 0, 360, (139, 69, 19), 2)
    
    # Legs (thick lines with joints)
    cv2.line(frame, left_hip, left_knee, (50, 100, 150), 8)  # Pants color
    cv2.line(frame, left_knee, left_foot, (50, 100, 150), 8)
    cv2.line(frame, right_hip, right_knee, (50, 100, 150), 8)
    cv2.line(frame, right_knee, right_foot, (50, 100, 150), 8)
    
    # Joint markers
    for joint in [left_hip, right_hip, left_knee, right_knee, left_foot, right_foot]:
        cv2.circle(frame, joint, 6, (255, 255, 255), -1)
    
    # Shoes
    cv2.ellipse(frame, left_foot, (20, 10), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(frame, right_foot, (20, 10), 0, 0, 360, (50, 50, 50), -1)
    
    return frame


def create_realistic_test_videos():
    """Create realistic test videos with detectable human figures"""
    
    def create_realistic_video(filename: str, arm_angle_range: float, leg_stance_factor: float, 
                             duration_seconds: int = 3, fps: int = 15):
        """Create a tennis video with realistic human figure"""
        width, height = 640, 480
        total_frames = duration_seconds * fps
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        for frame_num in range(total_frames):
            # Tennis court background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = [34, 139, 34]  # Tennis court green
            
            # Court lines
            cv2.line(frame, (50, height//2), (width-50, height//2), (255, 255, 255), 3)
            cv2.rectangle(frame, (100, 100), (width-100, height-100), (255, 255, 255), 2)
            cv2.line(frame, (width//2, 100), (width//2, height-100), (255, 255, 255), 2)
            
            # Player position
            progress = frame_num / total_frames
            player_x = int(width * 0.5)
            player_y = int(height * 0.6)
            
            # Arm swing motion (varies by arm_angle_range)
            swing_phase = np.sin(progress * 4 * np.pi)
            arm_angle_deg = swing_phase * arm_angle_range  # Different range for each video
            
            # Leg stance (varies by leg_stance_factor)
            leg_width = leg_stance_factor
            
            # Draw realistic person
            frame = draw_realistic_person(frame, player_x, player_y, arm_angle_deg, leg_width, frame_num)
            
            # Add tennis ball for context
            ball_progress = (progress * 2) % 1
            ball_x = int(100 + (width - 200) * ball_progress)
            ball_y = int(height * 0.3 + 50 * np.sin(ball_progress * np.pi))
            cv2.circle(frame, (ball_x, ball_y), 8, (0, 255, 255), -1)
            
            # Add informative text
            cv2.putText(frame, f"Style A: Swing Range {arm_angle_range:.0f}°, Stance {leg_stance_factor:.1f}x", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) if "style1" in filename else \
            cv2.putText(frame, f"Style B: Swing Range {arm_angle_range:.0f}°, Stance {leg_stance_factor:.1f}x", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Frame {frame_num}", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            out.write(frame)
        
        out.release()
        print(f"✅ Created {filename} (swing: {arm_angle_range}°, stance: {leg_stance_factor}x)")
    
    # Create two videos with distinctly different characteristics
    create_realistic_video("test_style1_conservative.mp4", arm_angle_range=45, leg_stance_factor=1.0)
    create_realistic_video("test_style2_aggressive.mp4", arm_angle_range=90, leg_stance_factor=1.6)
    
    return "test_style1_conservative.mp4", "test_style2_aggressive.mp4"


def test_realistic_comparison():
    """Test comparison with realistic human figures"""
    print("🎾 Testing Realistic Tennis Form Comparison")
    print("=" * 50)
    
    try:
        # Create realistic test videos
        print("🎬 Creating realistic test videos...")
        video1_path, video2_path = create_realistic_test_videos()
        
        # Initialize analyzer
        print("🔧 Initializing comparison analyzer...")
        analyzer = create_comparison_analyzer(difference_threshold=10.0)  # Lower threshold for testing
        analyzer.initialize_components()
        
        # Extract poses from both videos
        print("📹 Extracting poses from Style 1 (Conservative)...")
        poses1 = analyzer.extract_poses_from_video(video1_path)
        detected1 = sum(1 for p in poses1 if p['landmarks'])
        print(f"   Extracted {len(poses1)} frames, {detected1} with pose detection ({detected1/len(poses1)*100:.1f}%)")
        
        print("📹 Extracting poses from Style 2 (Aggressive)...")
        poses2 = analyzer.extract_poses_from_video(video2_path)
        detected2 = sum(1 for p in poses2 if p['landmarks'])
        print(f"   Extracted {len(poses2)} frames, {detected2} with pose detection ({detected2/len(poses2)*100:.1f}%)")
        
        if detected1 < len(poses1) * 0.5 or detected2 < len(poses2) * 0.5:
            print("⚠️  Low pose detection rate. This is expected with simplified figures.")
            print("💡 For better results, use real tennis videos with actual people.")
        
        # Compare poses
        print("⚖️ Comparing poses...")
        comparison_metrics = analyzer.compare_poses(poses1, poses2)
        valid_comparisons = sum(1 for m in comparison_metrics if m.overall_difference > 0)
        print(f"   Generated {len(comparison_metrics)} comparisons, {valid_comparisons} with valid differences")
        
        # Create comparison video
        print("🎬 Creating comparison video...")
        output_dir = "realistic_comparison_output"
        os.makedirs(output_dir, exist_ok=True)
        
        comparison_video_path = os.path.join(output_dir, "realistic_comparison.mp4")
        analyzer.create_comparison_video(
            poses1, poses2, comparison_metrics, 
            comparison_video_path, fps=15.0
        )
        print(f"   Comparison video saved: {comparison_video_path}")
        
        # Export metrics
        print("📊 Exporting comparison metrics...")
        comparison_csv_path = os.path.join(output_dir, "realistic_comparison_metrics.csv")
        analyzer.export_comparison_metrics(comparison_metrics, comparison_csv_path)
        print(f"   Metrics CSV saved: {comparison_csv_path}")
        
        # Display summary
        if comparison_metrics and valid_comparisons > 0:
            valid_diffs = [m.overall_difference for m in comparison_metrics if m.overall_difference > 0]
            total_highlights = sum(len(m.highlight_joints) for m in comparison_metrics)
            
            print(f"\n📈 Analysis Summary:")
            print(f"   Valid comparisons: {valid_comparisons}/{len(comparison_metrics)}")
            print(f"   Average difference: {np.mean(valid_diffs):.1f}°" if valid_diffs else "   No valid differences")
            print(f"   Maximum difference: {np.max(valid_diffs):.1f}°" if valid_diffs else "")
            print(f"   Total highlighted instances: {total_highlights}")
            
            # Show examples of detected differences
            significant_frames = [m for m in comparison_metrics if m.overall_difference > 5.0]
            if significant_frames:
                print(f"\n🔍 Significant differences detected in {len(significant_frames)} frames:")
                for i, metrics in enumerate(significant_frames[:5]):  # Show first 5
                    highlighted = ", ".join(metrics.highlight_joints) if metrics.highlight_joints else "None"
                    print(f"   Frame {metrics.frame_idx:2d}: {metrics.overall_difference:5.1f}° | Highlighted: {highlighted}")
        
        # Create charts
        print("📊 Creating analysis charts...")
        fig = analyzer.create_comparison_charts(comparison_metrics)
        chart_path = os.path.join(output_dir, "realistic_comparison_charts.html")
        fig.write_html(chart_path)
        print(f"   Interactive charts saved: {chart_path}")
        
        # Cleanup
        analyzer.cleanup()
        
        print(f"\n🎉 Realistic comparison test completed!")
        print(f"📁 Output files:")
        print(f"   - Video: {comparison_video_path}")
        print(f"   - Metrics: {comparison_csv_path}")
        print(f"   - Charts: {chart_path}")
        print(f"\n💡 Note: For best results with real tennis videos:")
        print(f"   - Use similar camera angles and distances")
        print(f"   - Ensure good lighting and clear player visibility")
        print(f"   - Videos should show the full body during tennis strokes")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup test videos
        for video_file in ["test_style1_conservative.mp4", "test_style2_aggressive.mp4"]:
            if os.path.exists(video_file):
                os.remove(video_file)
                print(f"🧹 Cleaned up {video_file}")


if __name__ == "__main__":
    test_realistic_comparison()