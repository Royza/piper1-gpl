#!/bin/bash
# Enhanced PiperTTS Training System - Runpod Setup Script
# This script sets up the complete environment for PiperTTS training on Runpod

set -e  # Exit on any error

echo "🚀 Setting up Enhanced PiperTTS Training System on Runpod..."

# Set default environment variables if not provided
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=${PYTHONPATH:-/workspace/piper1-gpl}
export WORKSPACE_DIR=${WORKSPACE_DIR:-/workspace}
export DEBUG=${DEBUG:-0}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

echo "🔧 Environment Configuration:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  WORKSPACE_DIR: $WORKSPACE_DIR"
echo "  DEBUG: $DEBUG"
echo "  LOG_LEVEL: $LOG_LEVEL"

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

# Clone or update the repository
echo "📥 Setting up PiperTTS repository..."
cd $WORKSPACE_DIR
if [ -d "piper1-gpl" ]; then
    echo "🔄 Repository exists, updating to latest version..."
    cd piper1-gpl
    git fetch origin
    git reset --hard origin/main
    echo "✅ Repository updated to latest version"
else
    echo "📥 Cloning new repository..."
    git clone https://github.com/Royza/piper1-gpl.git
    cd piper1-gpl
fi

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

# Set up environment variables for the session
export PYTHONPATH="${PYTHONPATH}:${WORKSPACE_DIR}/piper1-gpl"
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

# Test Flask installation
echo "🧪 Testing Flask installation..."
python -c "import flask; print(f'Flask version: {flask.__version__}')"

# Start the web server
echo ""
echo "🎉 Setup complete! Starting Enhanced PiperTTS Training Web Server..."
echo "📊 Access your interface at: http://localhost:5000"
echo "📈 TensorBoard available at: http://localhost:6006"
echo "💾 Workspace mounted at: $WORKSPACE_DIR"
echo "🔧 SSH access available for debugging"
echo ""

# Start the Flask web server with explicit host binding and correct port
echo "🚀 Starting Flask web server..."
python training_web_app.py --host 0.0.0.0 --port 5000
