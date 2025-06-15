#!/usr/bin/env python3
"""Launch the Streamlit app with proper configuration"""

import subprocess
import sys
import os

def main():
    """Launch Streamlit app"""
    print("🎾 Starting Tennis Form Checker MVP")
    print("📋 Configuration:")
    print("  - Pose Estimation: Disabled by default (Python 3.12 compatibility)")
    print("  - Ball Detection: Mock YOLO enabled")
    print("  - Video Processing: ✅ Ready")
    print()
    print("🚀 Launching Streamlit app...")
    print("   Open http://localhost:8501 in your browser")
    print()
    print("💡 Usage Tips:")
    print("  1. Keep 'Enable Pose Estimation' UNCHECKED for stable operation")
    print("  2. Keep 'Use Mock YOLO' CHECKED for testing")
    print("  3. Upload any MP4 video file")
    print("  4. Click 'Analyze Video' to start processing")
    print()
    
    # Set environment variables for better Streamlit experience
    env = os.environ.copy()
    env['STREAMLIT_SERVER_HEADLESS'] = 'true'
    env['STREAMLIT_SERVER_PORT'] = '8501'
    env['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    
    try:
        # Launch Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "src/main.py"]
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Tennis Form Checker MVP")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

if __name__ == "__main__":
    main()