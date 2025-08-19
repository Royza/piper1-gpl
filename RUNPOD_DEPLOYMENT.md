# 🚀 RunPod Deployment Guide for PiperTTS Training

This guide walks you through setting up PiperTTS voice training on RunPod for access to powerful GPUs.

## Prerequisites

1. **RunPod Account**: Sign up at [runpod.io](https://runpod.io)
2. **Training Data**: Prepared CSV file and audio files
3. **Basic Knowledge**: Familiarity with RunPod interface

## Quick Start

### Step 1: Create RunPod Instance

1. **Login to RunPod** and go to "Pods"
2. **Select Template**: Choose "PyTorch" or "RunPod PyTorch" template
3. **GPU Selection**: Choose GPU based on your budget:
   - **RTX 4090** (24GB VRAM): Good for medium datasets
   - **A100** (40GB/80GB VRAM): Best for large datasets
   - **RTX 3090** (24GB VRAM): Budget-friendly option
4. **Storage**: Allocate at least 50GB for checkpoints and models
5. **Start Pod**

### Step 2: Automated Setup

Once your pod is running:

1. **Connect via Terminal** or **Jupyter**
2. **Download setup script**:
   ```bash
   cd /workspace
   wget https://raw.githubusercontent.com/YOUR_USERNAME/piper1-gpl/main/runpod_setup.sh
   chmod +x runpod_setup.sh
   ```
3. **Run setup**:
   ```bash
   ./runpod_setup.sh
   ```

The script will automatically:
- Install system dependencies
- Clone PiperTTS repository
- Set up Python environment
- Build Cython extensions
- Download default checkpoint
- Create startup scripts

### Step 3: Start Training Web Interface

After setup completes:

```bash
cd /workspace/piper1-gpl
./start_training_server.sh
```

Or use the provided Jupyter notebook: `PiperTTS_Training.ipynb`

### Step 4: Access Web Interface

1. **Local Access**: `http://localhost:5001`
2. **Public Access**: Use RunPod's public URL feature:
   - Go to RunPod pod settings
   - Enable "Expose HTTP Ports"
   - Access via: `https://[POD-ID]-5001.proxy.runpod.net`

## Manual Setup (Alternative)

If you prefer manual setup or need customization:

### 1. System Dependencies
```bash
apt-get update
apt-get install -y build-essential cmake ninja-build git
```

### 2. Clone Repository
```bash
cd /workspace
git clone https://github.com/OHF-voice/piper1-gpl.git
cd piper1-gpl
```

### 3. Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[train]
pip install flask
```

### 4. Build Extensions
```bash
cd src/piper/train/vits/monotonic_align
mkdir -p monotonic_align
cythonize -i core.pyx
mv core*.so monotonic_align/
cd /workspace/piper1-gpl
```

### 5. Download Checkpoint
```bash
mkdir -p checkpoints/default
cd checkpoints/default
wget -O "en_US-lessac-medium.ckpt" \
  "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt"
cd /workspace/piper1-gpl
```

### 6. Start Web Server
```bash
export PYTHONPATH="/workspace/piper1-gpl/src:$PYTHONPATH"
python training_web_app.py
```

## Preparing Your Data

### CSV Format
Create a metadata file with pipe-separated values:
```csv
audio001.wav|Hello, this is my first training sample.
audio002.wav|The quick brown fox jumps over the lazy dog.
audio003.wav|Machine learning is fascinating and powerful.
```

### Audio Requirements
- **Format**: WAV files (other formats supported via librosa)
- **Sample Rate**: 22050 Hz recommended
- **Quality**: Clean, noise-free recordings
- **Length**: 2-10 seconds per sample ideal
- **Quantity**: 1000+ samples for good results

### Directory Structure
```
/workspace/data/
├── metadata.csv
└── audio/
    ├── audio001.wav
    ├── audio002.wav
    └── audio003.wav
```

## Training Configuration

### Web Interface Fields

| Field | Description | Default | Required |
|-------|-------------|---------|----------|
| **Voice Name** | Identifier for your voice | - | ✅ |
| **CSV Path** | Path to metadata file | - | ✅ |
| **Audio Directory** | Directory containing audio files | - | ✅ |
| **Sample Rate** | Audio sample rate in Hz | 22050 | ✅ |
| **eSpeak Voice** | Language/accent for phonemization | en-us | ✅ |
| **Cache Directory** | Training cache location | `cache/{voice_name}` | ❌ |
| **Config Path** | Output config file path | `models/{voice_name}_config.json` | ❌ |
| **Batch Size** | Training batch size | 32 | ❌ |
| **Checkpoint Path** | Pre-trained model to fine-tune from | Default checkpoint URL | ❌ |

### Recommended Settings

**For RTX 4090 / A100:**
- Batch Size: 32-64
- Use default checkpoint for faster training

**For RTX 3090:**
- Batch Size: 16-32
- Monitor VRAM usage

**For smaller GPUs:**
- Batch Size: 8-16
- Consider gradient accumulation

## Monitoring Training

### Web Interface
- **Status Tab**: Real-time training status
- **Log Viewer**: Live training logs
- **Progress Tracking**: Visual progress indicators

### Command Line Monitoring
```bash
# Monitor GPU usage
nvidia-smi

# Watch training logs
tail -f logs/training_*.log

# Check disk space
df -h
```

## Export to ONNX

After training completes:

1. **Go to Export Tab** in web interface
2. **Select Checkpoint** from dropdown
3. **Enter Model Name**: e.g., `en_US-myvoice-medium.onnx`
4. **Click Export**

The exported model will be saved to the `models/` directory and can be downloaded directly from the web interface.

## Best Practices

### Data Quality
- **Consistent Speaker**: Use recordings from the same person
- **Clean Audio**: Remove background noise
- **Balanced Content**: Include diverse text samples
- **Proper Pronunciation**: Ensure clear articulation

### Training Tips
- **Start Small**: Test with 100-200 samples first
- **Use Checkpoints**: Always use a pre-trained checkpoint
- **Monitor Logs**: Watch for overfitting or errors
- **Save Frequently**: Training can take hours

### Resource Management
- **Storage**: Keep 20GB+ free space
- **Memory**: Monitor RAM usage during data loading
- **GPU**: Watch VRAM usage and adjust batch size
- **Costs**: Stop pods when not training

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce batch size
- Clear GPU cache: `torch.cuda.empty_cache()`

**"File not found" errors**
- Check file paths are correct
- Ensure files are accessible from pod

**Slow training**
- Verify GPU is being used
- Check data loading bottlenecks
- Optimize batch size

**Poor voice quality**
- Increase training data quantity
- Improve audio quality
- Check phonemization accuracy

### Getting Help

1. **Check Logs**: Training logs contain detailed error information
2. **Monitor Resources**: Use `nvidia-smi` and `htop`
3. **Community**: Join PiperTTS community forums
4. **Documentation**: Reference official PiperTTS docs

## Cost Optimization

### Tips to Reduce Costs
- **Pause Training**: Stop pods when not actively training
- **Efficient GPUs**: Choose right GPU for your dataset size
- **Batch Training**: Prepare multiple voices for single session
- **Spot Instances**: Use spot pricing when available

### Estimated Costs (USD/hour)
- RTX 3090: $0.34/hour
- RTX 4090: $0.79/hour  
- A100 40GB: $1.89/hour
- A100 80GB: $2.89/hour

*Prices subject to change - check RunPod for current rates*

## Next Steps

After successful training:
1. **Test Your Voice**: Use inference to test quality
2. **Share Models**: Export and share with community
3. **Iterate**: Improve with more data or different settings
4. **Deploy**: Use trained models in applications

---

**Happy Training! 🎤✨**

For issues or improvements to this guide, please open an issue on the GitHub repository.
