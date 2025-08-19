# Enhanced PiperTTS Training System

A comprehensive web-based interface for training custom PiperTTS voice models with advanced features for audio processing, real-time monitoring, and AI chat integration.

## 🚀 Features

### **Core Training Features**
- **Web Interface**: Beautiful Flask-based UI for easy training management
- **Real-time Monitoring**: Live progress tracking with ETA estimates
- **GPU Optimization**: Automatic hardware detection and optimization
- **Checkpoint Management**: Resume training from any checkpoint
- **ONNX Export**: Export trained models for production use

### **Audio Processing Tools**
- **Audio Conversion**: Convert audio files to optimal sample rates
- **Quality Validation**: Automatic audio quality checks
- **Batch Processing**: Process multiple audio files simultaneously
- **Smart File Management**: Automatic file organization and naming

### **AI Chat Integration**
- **LLM Support**: Load and use local language models
- **Voice Synthesis**: Real-time speech synthesis of AI responses
- **Chat History**: Persistent chat sessions with audio playback
- **Autoplay**: Automatic audio playback of AI responses

### **Advanced Monitoring**
- **TensorBoard Integration**: Professional training visualization
- **System Monitoring**: Real-time GPU, RAM, and CPU usage
- **Memory Management**: Automatic GPU memory cleanup
- **Cache Management**: LRU-based cache optimization

## 🛠️ Installation

### **Local Development**
```bash
# Clone the repository
git clone https://github.com/Royza/piper1-gpl.git
cd piper1-gpl

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Start the web server
python training_web_app.py
```

### **Runpod Deployment**
1. Create a new Pod template on Runpod
2. Use container image: `nvidia/cuda:11.8-devel-ubuntu22.04`
3. Set start command to: `bash /workspace/runpod_setup.sh`
4. Configure HTTP ports: 5000 (web interface), 6006 (TensorBoard)
5. Set volume disk to 100GB+ for persistent storage

## 📊 System Requirements

### **Minimum Requirements**
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 4060 or better)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ free space
- **OS**: Ubuntu 20.04+ or Windows 10+ with WSL2

### **Recommended for Production**
- **GPU**: RTX 5090 32GB, A100 80GB, or H200 141GB
- **RAM**: 64GB+ system memory
- **Storage**: 200GB+ SSD storage
- **Network**: High-speed internet for model downloads

## 🎯 Quick Start

1. **Start the Web Interface**
   ```bash
   python training_web_app.py
   ```

2. **Access the Interface**
   - Open browser to `http://localhost:5000`
   - Navigate through the tabs: Training, Status, Export, Files, TensorBoard, Tools, Chat

3. **Prepare Audio Data**
   - Go to **Tools** tab
   - Upload audio files
   - Convert to 22050 Hz sample rate
   - Move to training data

4. **Start Training**
   - Go to **Training** tab
   - Configure voice settings
   - Upload or specify checkpoint
   - Click "Start Training"

5. **Monitor Progress**
   - Watch real-time progress in **Status** tab
   - View TensorBoard metrics at `http://localhost:6006`
   - Monitor system resources

## 📁 Project Structure

```
piper1-gpl/
├── training_web_app.py          # Main Flask web server
├── templates/
│   └── index.html              # Web interface
├── requirements.txt            # Python dependencies
├── runpod_setup.sh            # Runpod deployment script
├── src/piper/train/           # Core PiperTTS training code
├── datasets/                  # Training datasets
├── checkpoints/               # Training checkpoints
├── models/                    # Exported ONNX models
└── logs/                      # Training logs and TensorBoard
```

## 🔧 Configuration

### **Environment Variables**
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `PYTHONPATH`: Python module path
- `WORKSPACE_DIR`: Workspace directory path

### **Training Parameters**
- **Batch Size**: Auto-optimized based on GPU memory
- **Max Epochs**: Default 3000, auto-extends for checkpoints ≥3000 epochs
- **Sample Rate**: 22050 Hz (optimal for PiperTTS)
- **eSpeak Voice**: Language-specific phonemization

## 🎵 Audio Processing

### **Supported Formats**
- **Input**: WAV, MP3, FLAC, M4A
- **Output**: WAV (22050 Hz, 16-bit)
- **Quality**: Automatic validation and optimization

### **Conversion Process**
1. Upload audio files via Tools tab
2. Select target sample rate (22050 Hz recommended)
3. Convert with quality validation
4. Move to training directory with proper naming

## 🤖 AI Chat Features

### **LLM Integration**
- Support for GGML/GGUF models via llama-cpp-python
- Local model loading and inference
- Configurable model parameters

### **Voice Synthesis**
- Real-time speech synthesis of AI responses
- Multiple synthesis methods with fallbacks
- Audio playback controls with autoplay option

## 📈 Monitoring & Analytics

### **Real-time Metrics**
- Training progress (epoch/step/ETA)
- GPU memory usage and temperature
- System resource utilization
- Audio processing statistics

### **TensorBoard Integration**
- Loss curves and learning rates
- Audio samples and spectrograms
- Model parameter histograms
- Training configuration tracking

## 🚀 Performance Optimization

### **GPU Optimization**
- Automatic batch size adjustment
- Memory-efficient data loading
- Persistent workers and pin memory
- Automatic GPU memory cleanup

### **Training Speed**
- **RTX 4060 8GB**: ~10-20 hours for 1000 epochs
- **RTX 5090 32GB**: ~2-3 hours for 1000 epochs
- **A100 80GB**: ~1-2 hours for 1000 epochs

## 🔍 Troubleshooting

### **Common Issues**
1. **CUDA OOM**: Reduce batch size or clear GPU memory
2. **Audio Quality**: Use Tools tab for audio validation
3. **Checkpoint Errors**: Ensure max_epochs > checkpoint_epoch
4. **Memory Issues**: Monitor system resources in Status tab

### **Debug Information**
- Check server logs for detailed error messages
- Use TensorBoard for training visualization
- Monitor system resources in real-time
- SSH access available on Runpod for debugging

## 📚 Documentation

- [CHANGELOG.md](CHANGELOG.md) - Version history and changes
- [ROLLBACK_NOTES.md](ROLLBACK_NOTES.md) - Rollback instructions
- [VERSION_1.1_SUMMARY.md](VERSION_1.1_SUMMARY.md) - Feature summary
- [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) - Deployment guide

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is based on PiperTTS and follows the same licensing terms.

## 🙏 Acknowledgments

- **PiperTTS Team**: Original PiperTTS implementation
- **PyTorch Lightning**: Training framework
- **Flask**: Web framework
- **Runpod**: Cloud GPU infrastructure

---

**🎤 Happy Voice Training!** ✨
