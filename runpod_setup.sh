#!/bin/bash
# PiperTTS Training Setup Script for RunPod
# This script sets up the PiperTTS training environment on a RunPod instance

set -e

echo "🚀 Setting up PiperTTS Training Environment on RunPod..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update
apt-get install -y build-essential cmake ninja-build git curl wget

# Install Python dependencies
echo "🐍 Setting up Python environment..."
cd /workspace

# Clone the repository if not already present
if [ ! -d "piper1-gpl" ]; then
    echo "📥 Cloning PiperTTS repository..."
    git clone https://github.com/OHF-voice/piper1-gpl.git
fi

cd piper1-gpl

# Create and activate virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install training dependencies
echo "📚 Installing training dependencies..."
pip install --upgrade pip
pip install -e .[train]
pip install flask

# Build Cython extension
echo "⚡ Building Cython extensions..."
cd src/piper/train/vits/monotonic_align
mkdir -p monotonic_align
rm -f core.c
cythonize -i core.pyx
mv core*.so monotonic_align/
cd /workspace/piper1-gpl

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p cache checkpoints models logs

# Download default checkpoint
echo "📥 Downloading default checkpoint..."
mkdir -p checkpoints/default
cd checkpoints/default
wget -O "en_US-lessac-medium-epoch2164.ckpt" "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt"
cd /workspace/piper1-gpl

# Set up environment variables
echo "🔧 Setting up environment..."
echo 'export PYTHONPATH="/workspace/piper1-gpl/src:$PYTHONPATH"' >> ~/.bashrc
echo 'source /workspace/piper1-gpl/.venv/bin/activate' >> ~/.bashrc

# Create startup script
cat > start_training_server.sh << 'EOF'
#!/bin/bash
cd /workspace/piper1-gpl
source .venv/bin/activate
export PYTHONPATH="/workspace/piper1-gpl/src:$PYTHONPATH"
python training_web_app.py
EOF

chmod +x start_training_server.sh

# Create Jupyter notebook for easy access
cat > PiperTTS_Training.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# PiperTTS Training on RunPod\n",
    "\n",
    "This notebook provides an easy interface to start the PiperTTS training web application."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Start the training web server\n",
    "import subprocess\n",
    "import threading\n",
    "import time\n",
    "\n",
    "def start_server():\n",
    "    subprocess.run([\"/workspace/piper1-gpl/start_training_server.sh\"])\n",
    "\n",
    "# Start server in background thread\n",
    "server_thread = threading.Thread(target=start_server, daemon=True)\n",
    "server_thread.start()\n",
    "\n",
    "print(\"🚀 PiperTTS Training Server starting...\")\n",
    "print(\"📱 Access the web interface at: http://localhost:5001\")\n",
    "print(\"🔗 Or use RunPod's public URL if available\")\n",
    "\n",
    "time.sleep(3)\n",
    "print(\"✅ Server should now be running!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check server status\n",
    "import requests\n",
    "try:\n",
    "    response = requests.get(\"http://localhost:5001/status\")\n",
    "    if response.status_code == 200:\n",
    "        print(\"✅ Server is running!\")\n",
    "        print(\"📊 Status:\", response.json())\n",
    "    else:\n",
    "        print(\"❌ Server not responding properly\")\n",
    "except Exception as e:\n",
    "    print(\"❌ Server not accessible:\", str(e))\n",
    "    print(\"💡 Try running the first cell again\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Quick training example (modify paths as needed)\n",
    "import requests\n",
    "import json\n",
    "\n",
    "training_config = {\n",
    "    \"voice_name\": \"my_test_voice\",\n",
    "    \"csv_path\": \"/workspace/data/metadata.csv\",  # Update this path\n",
    "    \"audio_dir\": \"/workspace/data/audio\",        # Update this path\n",
    "    \"sample_rate\": 22050,\n",
    "    \"espeak_voice\": \"en-us\",\n",
    "    \"cache_dir\": \"/workspace/piper1-gpl/cache/my_test_voice\",\n",
    "    \"config_path\": \"/workspace/piper1-gpl/models/my_test_voice_config.json\",\n",
    "    \"batch_size\": 32,\n",
    "    \"ckpt_path\": \"/workspace/piper1-gpl/checkpoints/default/en_US-lessac-medium-epoch2164.ckpt\"\n",
    "}\n",
    "\n",
    "print(\"📋 Training Configuration:\")\n",
    "print(json.dumps(training_config, indent=2))\n",
    "print(\"\\n💡 Modify the paths above to match your data, then uncomment and run the next lines:\")\n",
    "print(\"\\n# Uncomment these lines when ready to train:\")\n",
    "print(\"# response = requests.post('http://localhost:5001/train', json=training_config)\")\n",
    "print(\"# print('Training Response:', response.json())\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Monitor GPU usage\n",
    "import subprocess\n",
    "\n",
    "try:\n",
    "    result = subprocess.run([\"nvidia-smi\"], capture_output=True, text=True)\n",
    "    print(\"🖥️ GPU Status:\")\n",
    "    print(result.stdout)\n",
    "except FileNotFoundError:\n",
    "    print(\"❌ nvidia-smi not found. GPU monitoring not available.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo "✅ Setup complete!"
echo ""
echo "🚀 To start the training web server:"
echo "   cd /workspace/piper1-gpl"
echo "   ./start_training_server.sh"
echo ""
echo "📱 The web interface will be available at:"
echo "   http://localhost:5001"
echo ""
echo "📓 Or open PiperTTS_Training.ipynb in Jupyter for guided setup"
echo ""
echo "💡 Default checkpoint downloaded to:"
echo "   /workspace/piper1-gpl/checkpoints/default/en_US-lessac-medium-epoch2164.ckpt"
