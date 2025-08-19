#!/bin/bash
# Enhanced PiperTTS Training System - Runpod Setup Script
# This script sets up the complete environment for PiperTTS training on Runpod

set -e  # Exit on any error

echo "🚀 Setting up Enhanced PiperTTS Training System on Runpod..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    espeak-ng \
    espeak-ng-data \
    htop \
    nano \
    tmux \
    build-essential \
    pkg-config

# Clone the repository
echo "📥 Cloning PiperTTS repository..."
cd /workspace
if [ ! -d "piper1-gpl" ]; then
    git clone https://github.com/Royza/piper1-gpl.git
fi
cd piper1-gpl

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
python3.10 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PiperTTS core
echo "🎤 Installing PiperTTS core..."
pip install -e .

# Install additional dependencies from requirements.txt
echo "📋 Installing additional dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p datasets models checkpoints logs temp_synthesis

# Set up environment variables
export PYTHONPATH="${PYTHONPATH}:/workspace/piper1-gpl"
export CUDA_VISIBLE_DEVICES=0

# Display system information
echo "💻 System Information:"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
if [ $(python -c 'import torch; print(torch.cuda.device_count())') -gt 0 ]; then
    echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "GPU memory: $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")')"
fi

# Start the web server
echo ""
echo "🎉 Setup complete! Starting Enhanced PiperTTS Training Web Server..."
echo "📊 Access your interface at: http://localhost:5000"
echo "📈 TensorBoard available at: http://localhost:6006"
echo "💾 Workspace mounted at: /workspace"
echo "🔧 SSH access available for debugging"
echo ""

# Start the Flask web server
python training_web_app.py --host 0.0.0.0 --port 5000
