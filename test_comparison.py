#!/usr/bin/env python3
"""Test tennis form comparison functionality"""

import sys
import os
import numpy as np
import cv2
sys.path.insert(0, 'src')

from comparison_analyzer import create_comparison_analyzer


def create_test_tennis_videos():
    """Create two test tennis videos with different poses for comparison"""
    
    def create_video_with_pose(filename: str, arm_swing_amplitude: float, 
                              leg_stance_width: float, duration_seconds: int = 3, fps: int = 20):
        """Create a tennis video with specific pose characteristics"""
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
            
            # Player position
            progress = frame_num / total_frames
            player_x = int(width * 0.5)  # Center position
            player_y = int(height * 0.7)
            
            # Player body with varying pose characteristics
            head_y = player_y - 120
            torso_top = player_y - 90
            torso_bottom = player_y - 30
            
            # Head
            cv2.circle(frame, (player_x, head_y), 25, (255, 200, 150), -1)
            
            # Torso
            cv2.line(frame, (player_x, torso_top), (player_x, torso_bottom), (255, 255, 255), 8)
            
            # Arms with different swing characteristics
            swing_phase = np.sin(progress * 4 * np.pi)
            arm_length = 60
            
            # Right arm (racket arm) - varies by arm_swing_amplitude
            arm_angle = swing_phase * arm_swing_amplitude
            arm_x = int(player_x + arm_length * np.cos(arm_angle))
            arm_y = int(torso_top + 20 + arm_length * np.sin(arm_angle))
            cv2.line(frame, (player_x, torso_top + 20), (arm_x, arm_y), (255, 255, 255), 5)
            
            # Racket
            cv2.circle(frame, (arm_x, arm_y), 8, (200, 100, 50), -1)
            
            # Left arm
            left_arm_x = int(player_x - 40)
            left_arm_y = int(torso_top + 30)
            cv2.line(frame, (player_x, torso_top + 20), (left_arm_x, left_arm_y), (255, 255, 255), 5)
            
            # Legs with different stance width
            leg_spread = int(30 * leg_stance_width)
            cv2.line(frame, (player_x, torso_bottom), (player_x - leg_spread, player_y), (255, 255, 255), 6)
            cv2.line(frame, (player_x, torso_bottom), (player_x + leg_spread, player_y), (255, 255, 255), 6)
            
            # Add descriptive text
            cv2.putText(frame, f"Swing: {arm_swing_amplitude:.1f}, Stance: {leg_stance_width:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame {frame_num}", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            out.write(frame)
        
        out.release()
        print(f"✅ Created {filename} (swing: {arm_swing_amplitude}, stance: {leg_stance_width})")
    
    # Create two videos with different characteristics
    create_video_with_pose("test_video1_normal.mp4", arm_swing_amplitude=0.8, leg_stance_width=1.0)
    create_video_with_pose("test_video2_different.mp4", arm_swing_amplitude=1.4, leg_stance_width=1.5)
    
    return "test_video1_normal.mp4", "test_video2_different.mp4"


def test_comparison_analyzer():
    """Test the comparison analyzer functionality"""
    print("🎾 Testing Tennis Form Comparison Analyzer")
    print("=" * 50)
    
    try:
        # Create test videos
        print("🎬 Creating test videos...")
        video1_path, video2_path = create_test_tennis_videos()
        
        # Initialize analyzer
        print("🔧 Initializing comparison analyzer...")
        analyzer = create_comparison_analyzer(difference_threshold=15.0)
        analyzer.initialize_components()
        
        # Extract poses from both videos
        print("📹 Extracting poses from video 1...")
        poses1 = analyzer.extract_poses_from_video(video1_path)
        print(f"   Extracted {len(poses1)} frames from video 1")
        
        print("📹 Extracting poses from video 2...")
        poses2 = analyzer.extract_poses_from_video(video2_path)
        print(f"   Extracted {len(poses2)} frames from video 2")
        
        # Compare poses
        print("⚖️ Comparing poses...")
        comparison_metrics = analyzer.compare_poses(poses1, poses2)
        print(f"   Generated {len(comparison_metrics)} comparison metrics")
        
        # Create comparison video
        print("🎬 Creating comparison video...")
        output_dir = "test_comparison_output"
        os.makedirs(output_dir, exist_ok=True)
        
        comparison_video_path = os.path.join(output_dir, "test_comparison.mp4")
        analyzer.create_comparison_video(
            poses1, poses2, comparison_metrics, 
            comparison_video_path, fps=20.0
        )
        print(f"   Comparison video saved: {comparison_video_path}")
        
        # Export metrics
        print("📊 Exporting comparison metrics...")
        comparison_csv_path = os.path.join(output_dir, "test_comparison_metrics.csv")
        analyzer.export_comparison_metrics(comparison_metrics, comparison_csv_path)
        print(f"   Metrics CSV saved: {comparison_csv_path}")
        
        # Display summary statistics
        if comparison_metrics:
            overall_diffs = [m.overall_difference for m in comparison_metrics]
            total_highlights = sum(len(m.highlight_joints) for m in comparison_metrics)
            
            print(f"\n📈 Analysis Summary:")
            print(f"   Total frames compared: {len(comparison_metrics)}")
            print(f"   Average difference: {np.mean(overall_diffs):.1f}°")
            print(f"   Maximum difference: {np.max(overall_diffs):.1f}°")
            print(f"   Minimum difference: {np.min(overall_diffs):.1f}°")
            print(f"   Total highlighted instances: {total_highlights}")
            
            # Show frame-by-frame differences
            print(f"\n🔍 Frame-by-frame differences (first 10 frames):")
            for i, metrics in enumerate(comparison_metrics[:10]):
                highlighted_joints = ", ".join(metrics.highlight_joints) if metrics.highlight_joints else "None"
                print(f"   Frame {i:2d}: {metrics.overall_difference:5.1f}° | Highlighted: {highlighted_joints}")
            
            # Show joint-specific statistics
            joint_stats = {}
            for joint in analyzer.key_joints:
                diffs = []
                for metrics in comparison_metrics:
                    diff = metrics.angle_differences.get(joint)
                    if diff is not None:
                        diffs.append(diff)
                if diffs:
                    joint_stats[joint] = {
                        'avg': np.mean(diffs),
                        'max': np.max(diffs),
                        'highlighted_count': sum(1 for m in comparison_metrics if joint in m.highlight_joints)
                    }
            
            print(f"\n🎯 Joint-specific analysis:")
            for joint, stats in joint_stats.items():
                print(f"   {joint:15s}: Avg {stats['avg']:5.1f}° | Max {stats['max']:5.1f}° | Highlighted {stats['highlighted_count']:2d} times")
        
        # Create charts
        print("📊 Creating analysis charts...")
        fig = analyzer.create_comparison_charts(comparison_metrics)
        chart_path = os.path.join(output_dir, "comparison_charts.html")
        fig.write_html(chart_path)
        print(f"   Interactive charts saved: {chart_path}")
        
        # Cleanup
        analyzer.cleanup()
        
        print(f"\n🎉 Comparison analysis test completed successfully!")
        print(f"📁 Output files in: {output_dir}/")
        print(f"   - {comparison_video_path}")
        print(f"   - {comparison_csv_path}")
        print(f"   - {chart_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup test videos
        for video_file in ["test_video1_normal.mp4", "test_video2_different.mp4"]:
            if os.path.exists(video_file):
                os.remove(video_file)
                print(f"🧹 Cleaned up {video_file}")


def test_ui_integration():
    """Test UI integration"""
    print("\n🖥️ Testing UI Integration")
    print("=" * 30)
    
    try:
        from comparison_ui import comparison_ui
        print("✅ Comparison UI module imported successfully")
        print("💡 Run 'uv run streamlit run src/main.py' to test the full UI")
        return True
    except ImportError as e:
        print(f"❌ UI integration test failed: {e}")
        return False


def main():
    """Run all comparison tests"""
    print("🎾 TENNIS FORM COMPARISON - TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_comparison_analyzer,
        test_ui_integration
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
        print()
    
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Comparison functionality is ready.")
        print("\n🚀 Ready to use:")
        print("1. uv run python run_app.py")
        print("2. Select '⚖️ Video Comparison' from the sidebar")
        print("3. Upload two tennis videos")
        print("4. Enjoy side-by-side form comparison!")
    else:
        print("⚠️ Some tests failed. Check the output above.")


if __name__ == "__main__":
    main()