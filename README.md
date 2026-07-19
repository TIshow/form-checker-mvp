# Tennis Form Checker MVP 🎾

A Python-based tennis form analysis application that uses computer vision to analyze tennis videos. The app provides pose estimation using MediaPipe, ball tracking with YOLO, center of gravity calculations with body mass distribution, and generates annotated videos with CSV metrics export.

## Features

### 🎾 Single Video Analysis
- **Pose Estimation**: MediaPipe-based human pose detection with 33 keypoints
- **Ball Tracking**: YOLO-based tennis ball detection with simple tracking algorithm
- **Center of Gravity**: Biomechanically accurate CoG calculation using body part mass coefficients
- **Metrics Export**: Frame-by-frame analysis exported to CSV
- **Video Visualization**: Annotated output video with pose overlay, ball detection, and metrics
- **Memory Efficient**: Generator-based frame processing for large videos

### ⚖️ Video Comparison Analysis (NEW!)
- **Side-by-Side Comparison**: Compare two tennis videos simultaneously
- **Joint Angle Analysis**: Calculate and compare joint angles (elbows, shoulders, knees, hips)
- **Difference Highlighting**: Red markers on joints with significant differences
- **Synchronized Playback**: Both videos play in sync with pose overlays
- **Numerical Comparison**: Frame-by-frame angle differences with CSV export
- **Interactive Charts**: Plotly-based analysis charts and trends
- **Threshold Configuration**: Adjustable difference threshold for highlighting

### 🖥️ Streamlit UI
- **Easy-to-use web interface**
- **Mode Selection**: Choose between single video or comparison analysis
- **Real-time Progress**: Live updates during video processing
- **Download Results**: Get analyzed videos and CSV data

## Installation

### Requirements

- Python 3.11+ (Python 3.12+ requires manual MediaPipe installation)
- UV package manager (recommended) or pip

### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <your-repo-url>
cd tennis-form-checker-mvp

# Install with pose estimation
uv sync --extra pose

# Install full functionality (pose + YOLO)
uv sync --extra full

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
```

### Using Pip

```bash
# Core installation
pip install -e .

# With pose estimation
pip install -e .[pose]

# Full installation
pip install -e .[full]
```

### Python Version Setup

For best compatibility, use Python 3.11:

```bash
# Using asdf
asdf install python 3.11.10
asdf local python 3.11.10

# Using pyenv
pyenv install 3.11.10
pyenv local 3.11.10

# Then reinstall dependencies
rm -rf .venv
uv sync --extra pose
```

## Usage

### Launch the Application

```bash
# Recommended: Use the launcher script
uv run python run_app.py

# Or start directly
uv run streamlit run src/main.py
```

Then open your browser to `http://localhost:8501`.

### Single Video Analysis

1. Select "🎾 Single Video Analysis" from the sidebar
2. Configure settings:
   - ✅ Enable "Enable Pose Estimation"
   - ✅ Enable "Use Mock YOLO" (for testing without PyTorch models)
3. Upload a tennis video (MP4, AVI, MOV, MKV)
4. Click "🚀 Analyze Video"
5. Download results:
   - Analyzed video with pose overlay and metrics
   - CSV file with frame-by-frame data

### Video Comparison Analysis

1. Select "⚖️ Video Comparison" from the sidebar
2. Configure comparison settings:
   - Set difference threshold (5-50 degrees)
   - Enable interactive charts
   - Enable frame-by-frame analysis
3. Upload two tennis videos:
   - Video 1: Reference video
   - Video 2: Comparison video
4. Click "🚀 Start Comparison Analysis"
5. Review results:
   - Side-by-side comparison video
   - Interactive analysis charts
   - Detailed joint angle comparisons
   - CSV export with numerical differences

## Project Structure

```
src/
├── frame_loader/          # Video frame streaming and loading
│   ├── __init__.py
│   └── frame_loader.py
├── visualizer/            # Video visualization and output
│   ├── __init__.py
│   └── visualizer.py
├── metrics_engine/        # Biomechanical analysis and CoG calculation
│   ├── __init__.py
│   └── metrics_engine.py
├── yolo_detector/         # Ball detection and tracking
│   ├── __init__.py
│   └── yolo_detector.py
├── pose_estimator.py      # MediaPipe pose estimation wrapper
├── comparison_analyzer.py # NEW: Video comparison analysis engine
├── comparison_ui.py       # NEW: Streamlit UI for comparison
├── main.py               # Main application entry point
└── __init__.py
```

## Key Components

### 1. Frame Loader (`frame_loader/`)
- Generator-based video processing for memory efficiency
- Supports progress tracking and frame seeking
- Works with any video format supported by OpenCV

### 2. Pose Estimator (`pose_estimator.py`)
- MediaPipe integration for 33-point pose detection
- Angle calculations for tennis-specific metrics
- Visibility scoring for landmark quality

### 3. Metrics Engine (`metrics_engine/`)
- **Body Mass Distribution**: Scientifically accurate mass coefficients for each body part
- **Center of Gravity**: Weighted centroid calculation using biomechanical data
- **Motion Analysis**: Velocity, acceleration, and stability calculations
- **Impact Detection**: Tennis ball contact detection based on motion changes

### 4. YOLO Detector (`yolo_detector/`)
- Tennis ball detection using YOLO models
- **Simple Tracking**: IoU and distance-based ball tracking
- Mock detector for testing without GPU/models

### 5. Visualizer (`visualizer/`)
- OpenCV-based video output (no MoviePy dependency)
- Pose overlay with anatomically correct connections
- Real-time metrics display on video frames
- MP4 output with preserved frame rate

### 6. Comparison Analyzer (`comparison_analyzer.py`) **NEW!**
- **Form Comparison**: Side-by-side analysis of two tennis videos
- **Joint Angle Comparison**: Calculate differences in key tennis joints
- **Synchronization**: Frame-by-frame matching of video sequences
- **Difference Highlighting**: Visual markers for significant differences
- **Statistical Analysis**: Summary statistics and trend analysis

### 7. Comparison UI (`comparison_ui.py`) **NEW!**
- **Streamlit Interface**: User-friendly web interface for comparison
- **Interactive Charts**: Plotly-based visualization of differences
- **Real-time Analysis**: Live progress updates during processing
- **Result Export**: Download comparison videos and CSV data

## Body Part Mass Coefficients

The metrics engine uses research-based mass distribution:

- Head: 8.1%
- Torso: 49.7%
- Arms: 5.0% each
- Forearms: 1.6% each
- Hands: 0.6% each
- Thighs: 10.0% each
- Shins: 4.65% each
- Feet: 1.45% each

## Output Files

### Single Video Analysis

1. **Analyzed Video** (`analyzed_video.mp4`): Original video with overlays
   - Pose skeleton with anatomical connections
   - Center of gravity marker
   - Ball detection boxes (if enabled)
   - Real-time metrics display

2. **Metrics CSV** (`metrics.csv`): Frame-by-frame data
   - Centroid coordinates
   - Velocity and acceleration
   - Ball position and velocity
   - Impact detection flags
   - Stability scores

### Video Comparison Analysis

1. **Comparison Video** (`comparison_analysis.mp4`): Side-by-side analysis
   - Both videos with pose overlays
   - Highlighted joints with significant differences
   - Real-time difference metrics
   - Frame synchronization

2. **Comparison CSV** (`comparison_metrics.csv`): Detailed comparison data
   - Joint angles for both videos
   - Angle differences for each joint
   - Overall difference scores
   - Highlighted joint indicators

3. **Interactive Charts** (in browser): Live analysis visualization
   - Overall differences over time
   - Joint-specific comparison trends
   - Highlighted joints distribution
   - Statistical summaries

## Use Cases

### Single Video Analysis
- **Form Analysis**: Analyze individual tennis technique
- **Progress Tracking**: Monitor improvement over time
- **Coaching**: Identify areas for improvement
- **Biomechanical Study**: Research tennis movement patterns

### Video Comparison Analysis
- **Technique Comparison**: Professional vs. amateur analysis
- **Before/After Coaching**: Measure training effectiveness
- **Style Analysis**: Compare different playing techniques
- **Player Matching**: Analyze different players' styles
- **Skill Development**: Track progression between sessions

## Performance Notes

- **Memory Usage**: Generator-based processing keeps memory usage constant regardless of video length
- **Processing Speed**: ~10-30 FPS on modern hardware (depends on video resolution)
- **Comparison Speed**: ~5-15 FPS for dual video analysis
- **GPU Support**: YOLO detection can utilize CUDA if available
- **Apple Silicon**: Optimized MediaPipe version for M1/M2 Macs

## Testing

### Run Basic Tests

```bash
# Test single video functionality
uv run python test_basic_functionality.py

# Test pose estimation
uv run python test_pose_estimation.py

# Test comparison functionality
uv run python test_comparison.py

# Test with realistic figures
uv run python test_comparison_realistic.py
```

### Test Results Example

```
📈 Analysis Summary:
   Valid comparisons: 38/45
   Average difference: 17.6°
   Maximum difference: 37.7°
   Total highlighted instances: 110
```

## Troubleshooting

### Common Issues

1. **MediaPipe Installation**: 
   - Python 3.11 recommended for stability
   - For Python 3.12+: `pip install mediapipe` manually

2. **YOLO Model Loading**: 
   - Use "Mock YOLO" for testing without PyTorch
   - For real YOLO: `pip install torch torchvision ultralytics`

3. **Memory Issues**: 
   - Reduce video resolution
   - Use shorter video clips for comparison

4. **Low Pose Detection**: 
   - Ensure good lighting in videos
   - Use videos with clear full-body visibility
   - Similar camera angles work best for comparison

### Debug Mode

```bash
# Enable verbose logging
PYTHONPATH=src python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from main import process_video
process_video('test.mp4', 'output', use_mock_yolo=True, enable_pose=True)
"
```

## Development

### Setup Development Environment

```bash
# Install with development dependencies
uv sync --extra dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Code formatting
black src/
flake8 src/
mypy src/
```

### Adding New Features

1. Follow the modular structure - one responsibility per module
2. Use factory functions (`create_*`) for component initialization
3. Maintain type hints and docstrings
4. Add tests for new functionality
5. Update documentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code style
4. Add tests for new features
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- MediaPipe by Google for pose estimation
- Ultralytics for YOLO implementation
- Biomechanics research for body mass distribution data
- Plotly for interactive visualizations
- Streamlit for the web interface