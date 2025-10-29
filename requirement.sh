#!/usr/bin/bash

# Exit on error
set -e

echo "Installing FFmpeg..."
sudo pacman -S --needed ffmpeg cudnn

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python packages..."
pip install faster-whisper numpy nvidia-cublas-cu12 nvidia-cudnn-cu12 watchdog pyinstaller

echo ""
echo "✓ Setup complete!"
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo "edit path in Configuration.py"
echo "to run code, run:"
echo "  python transcript.py audio.mp3"
