# Tennis Form Checker MVP 🎾

A Python-based tennis form analysis application that uses computer vision to analyze tennis videos. The app provides pose estimation using MediaPipe, ball tracking with YOLO, center of gravity calculations with body mass distribution, and generates annotated videos with CSV metrics export.

## Features

- **Pose Estimation**: MediaPipe-based human pose detection with 33 keypoints
- **Ball Tracking**: YOLO-based tennis ball detection with simple tracking algorithm
- **Center of Gravity**: Biomechanically accurate CoG calculation using body part mass coefficients
- **Metrics Export**: Frame-by-frame analysis exported to CSV
- **Video Visualization**: Annotated output video with pose overlay, ball detection, and metrics
- **Memory Efficient**: Generator-based frame processing for large videos
- **Streamlit UI**: Easy-to-use web interface

## Installation

### Requirements

- Python 3.9+
- UV package manager (recommended) or pip

### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <your-repo-url>
cd tennis-form-checker-mvp

# Install core dependencies (UI + basic video processing)
uv sync

# Install with pose estimation (requires Python 3.9-3.11)
uv sync --extra pose

# Install with YOLO detection (requires compatible PyTorch)
uv sync --extra yolo

# Install full functionality
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

# With YOLO detection  
pip install -e .[yolo]

# Full installation
pip install -e .[full]
```

### Python 3.12+ Compatibility

If you're using Python 3.12+, install MediaPipe manually after core installation:

```bash
# Install core dependencies first
uv sync

# Then install MediaPipe manually
pip install mediapipe

# For YOLO detection, also install PyTorch manually
pip install torch torchvision ultralytics
```

## Usage

### Streamlit Web Interface

```bash
# Recommended: Use the launcher script
uv run python run_app.py

# Or start directly
uv run streamlit run src/main.py
```

Then open your browser to `http://localhost:8501` and upload a tennis video.

**⚠️ Important for Python 3.12 users:**
- Keep "Enable Pose Estimation" **UNCHECKED** (MediaPipe compatibility issue)
- Keep "Use Mock YOLO" **CHECKED** for testing without PyTorch
- The system will still provide ball tracking and motion analysis

### Command Line Interface

```bash
# Basic usage (safe for Python 3.12)
uv run python src/main.py input_video.mp4 --mock-yolo

# Specify output directory
uv run python src/main.py input_video.mp4 -o my_analysis --mock-yolo

# Enable pose estimation (may fail with Python 3.12)
uv run python src/main.py input_video.mp4 --mock-yolo --enable-pose

# Test with generated video
uv run python test_without_pose.py
```

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

1. **Analyzed Video** (`analyzed_video.mp4`): Original video with overlays
   - Pose skeleton
   - Center of gravity marker
   - Ball detection boxes
   - Real-time metrics

2. **Metrics CSV** (`metrics.csv`): Frame-by-frame data
   - Centroid coordinates
   - Velocity and acceleration
   - Ball position and velocity
   - Impact detection flags
   - Stability scores

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

## Performance Notes

- **Memory Usage**: Generator-based processing keeps memory usage constant regardless of video length
- **Processing Speed**: ~10-30 FPS on modern hardware (depends on video resolution)
- **GPU Support**: YOLO detection can utilize CUDA if available
- **Apple Silicon**: Optimized MediaPipe version for M1/M2 Macs

## Troubleshooting

### Common Issues

1. **MediaPipe Installation**: Use version 0.10.11 for Apple Silicon stability
2. **YOLO Model Loading**: Use `--mock-yolo` flag for testing without models
3. **Memory Issues**: Reduce video resolution or use batch processing
4. **Import Errors**: Ensure all dependencies are installed with `uv sync`

### Debug Mode

```bash
# Enable verbose logging
PYTHONPATH=src python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from main import process_video
process_video('test.mp4', 'output', use_mock_yolo=True)
"
```

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