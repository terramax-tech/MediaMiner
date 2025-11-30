#!/bin/bash

################################
# mm_setup-v0.1
################################


echo "==============================================="
echo "TMX MediaMiner - Adult Content Management Tool"
echo "==============================================="
echo ""

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Step 1: Create virtual environment
echo "[INFO] Creating virtual environment: ai_env/"
python3 -m venv ai_env

# Step 2: Activate virtual environment
echo "[INFO] Activating virtual environment..."
source ai_env/bin/activate

# Step 3: Upgrade pip
echo "[INFO] Upgrading pip..."
pip install --upgrade pip

# Step 4: Install core dependencies
echo "[INFO] Installing core dependencies..."
pip install Pillow imageio imageio-ffmpeg opencv-python numpy PyQt6 requests

# Step 5: Install AI/ML dependencies
echo "[INFO] Installing AI/ML libraries (this will take a few minutes)..."

# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install transformers and additional ML tools
pip install transformers
pip install scenedetect[opencv]  # For scene detection
pip install nudenet  # For adult content detection and body part recognition

# Step 6: Install video processing tools
echo "[INFO] Installing video processing tools..."
pip install python-ffmpeg

# Step 7: Check ffmpeg availability
echo "[INFO] Checking for ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "[WARNING] ffmpeg not found in system PATH."
    echo "[INFO] You may need to install ffmpeg separately for full functionality."
    echo "[INFO] Ubuntu/Debian: sudo apt install ffmpeg"
    echo "[INFO] MacOS: brew install ffmpeg"
    echo "[INFO] Using imageio-ffmpeg as fallback..."
else
    echo "[INFO] ffmpeg found: $(which ffmpeg)"
fi

# Step 8: Verify installations
echo "[INFO] Verifying installations..."

python3 -c "import PyQt6; print('[OK] PyQt6')" || { echo "[ERROR] PyQt6 failed!"; exit 1; }
python3 -c "import imageio; print('[OK] imageio')" || { echo "[ERROR] imageio failed!"; exit 1; }
python3 -c "import cv2; print('[OK] opencv-python')" || { echo "[ERROR] opencv-python failed!"; exit 1; }
python3 -c "import torch; print('[OK] PyTorch')" || { echo "[ERROR] PyTorch failed!"; exit 1; }
python3 -c "import transformers; print('[OK] transformers')" || { echo "[ERROR] transformers failed!"; exit 1; }
python3 -c "import scenedetect; print('[OK] scenedetect')" || { echo "[ERROR] scenedetect failed!"; exit 1; }

echo ""
echo "==============================================="
echo "[SUCCESS] Setup complete!"
echo "==============================================="
echo ""
echo "Note: NudeNet models will download automatically on first run."
echo ""
echo "To run the application:"
echo "  ./run.sh"
echo ""
echo "==============================================="
