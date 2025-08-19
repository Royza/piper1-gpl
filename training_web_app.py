#!/usr/bin/env python3
"""
PiperTTS Training Web Interface
A Flask web application for training PiperTTS voices with a user-friendly interface.
"""

import os
import json
import subprocess
import threading
import platform
import psutil
import time
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import glob
import shutil
from werkzeug.utils import secure_filename
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from tensorboard import program as tb_program
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

app = Flask(__name__)

# Configure upload limits for large file batches
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB max upload size
app.config['UPLOAD_TIMEOUT'] = 3600  # 1 hour timeout

# Additional Flask configurations for large uploads
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Increase request timeout and buffer size
import werkzeug
werkzeug.serving.WSGIRequestHandler.protocol_version = "HTTP/1.1"

# Configuration
DATASETS_DIR = Path("datasets")
CHECKPOINTS_DIR = Path("checkpoints") 
LOGS_DIR = Path("logs")

# Feature flags
ENABLE_AUDIO_VALIDATION = True  # Set to False to disable audio validation

# Cache management settings
MAX_CACHE_SIZE_GB = 10.0  # Maximum cache size in GB
CACHE_CLEANUP_THRESHOLD = 0.8  # Clean when cache reaches 80% of max size

# TensorBoard settings
TENSORBOARD_LOG_DIR = Path("logs/tensorboard")
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
tensorboard_process = None

# Create directories
for dir_path in [DATASETS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Global variables to track training status
training_status = {
    "is_training": False,
    "current_job": None,
    "log_file": None,
    "progress": 0
}

# Global variables to track upload progress
upload_progress = {
    "is_uploading": False,
    "current_file": 0,
    "total_files": 0,
    "current_filename": "",
    "operation": ""  # "upload" or "convert"
}

# Add these new global variables after the existing ones
LLM_MODELS_DIR = Path.cwd() / "models" / "LLM"
LLM_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Chat session storage
chat_sessions = {}

# Loaded models storage
loaded_models = {
    "llm": None,
    "voice": None
}

def detect_system_capabilities():
    """Detect system capabilities for optimal training settings"""
    # Get fresh memory info each time
    memory_info = psutil.virtual_memory()
    
    # Debug memory detection
    print(f"DEBUG: Memory total bytes: {memory_info.total}")
    print(f"DEBUG: Memory total GB: {memory_info.total / (1024**3)}")
    print(f"DEBUG: Platform: {platform.system()}")
    print(f"DEBUG: Platform release: {platform.release()}")
    
    info = {
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_gb": round(memory_info.total / (1024**3), 1),
        "memory_used_gb": round(memory_info.used / (1024**3), 1),
        "memory_percent": round(memory_info.percent, 1),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_memory_gb": 0,
        "gpu_memory_used_gb": 0,
        "gpu_memory_percent": 0,
        "gpu_names": []
    }
    
    if TORCH_AVAILABLE:
        info["gpu_available"] = torch.cuda.is_available()
        if info["gpu_available"]:
            info["gpu_count"] = torch.cuda.device_count()
            max_gpu_memory = 0
            total_gpu_used = 0
            
            for i in range(info["gpu_count"]):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                
                # Get current GPU memory usage
                try:
                    gpu_memory_used = torch.cuda.memory_allocated(i) / (1024**3)
                    total_gpu_used += gpu_memory_used
                except:
                    gpu_memory_used = 0
                
                info["gpu_names"].append(gpu_name)
                max_gpu_memory = max(max_gpu_memory, gpu_memory)
            
            info["gpu_memory_gb"] = round(max_gpu_memory, 1)
            info["gpu_memory_used_gb"] = round(total_gpu_used, 1)
            info["gpu_memory_percent"] = round((total_gpu_used / max_gpu_memory) * 100, 1) if max_gpu_memory > 0 else 0
    
    return info

def calculate_optimal_settings(system_info=None):
    """Calculate optimal training settings based on system capabilities"""
    if system_info is None:
        system_info = detect_system_capabilities()
    
    settings = {
        "num_workers": 1,
        "batch_size": 32,
        "precision": "32",
        "log_every_n_steps": 1,
        "max_epochs": 3000,
        "devices": 1
    }
    
    # Calculate optimal num_workers
    cpu_cores = system_info["cpu_count"]
    if cpu_cores >= 16:
        settings["num_workers"] = min(12, cpu_cores - 4)  # Leave some cores for system
    elif cpu_cores >= 8:
        settings["num_workers"] = min(8, cpu_cores - 2)
    elif cpu_cores >= 4:
        settings["num_workers"] = min(4, cpu_cores - 1)
    else:
        settings["num_workers"] = 1
    
    # Adjust batch size based on GPU memory
    if system_info["gpu_available"]:
        gpu_memory_gb = system_info["gpu_memory_gb"]
        if gpu_memory_gb >= 24:  # High-end GPUs (A100, RTX 4090, etc.)
            settings["batch_size"] = 64
            settings["precision"] = "16-mixed"  # Use mixed precision for speed
        elif gpu_memory_gb >= 16:  # Mid-high end GPUs (RTX 3080Ti, A4000, etc.)
            settings["batch_size"] = 48
            settings["precision"] = "16-mixed"
        elif gpu_memory_gb >= 12:  # Mid-range GPUs (RTX 3060Ti, RTX 4060Ti, etc.)
            settings["batch_size"] = 40
        elif gpu_memory_gb >= 8:   # Lower mid-range GPUs
            settings["batch_size"] = 32
        elif gpu_memory_gb >= 6:   # Entry level GPUs
            settings["batch_size"] = 24
        else:                      # Very low VRAM
            settings["batch_size"] = 16
        
        # Use multiple GPUs if available
        if system_info["gpu_count"] > 1:
            settings["devices"] = min(system_info["gpu_count"], 4)  # Cap at 4 GPUs
    else:
        # CPU training - smaller batch sizes
        settings["batch_size"] = 16
        settings["precision"] = "32"
        settings["devices"] = 1
    
    # Adjust logging based on expected batch count
    # Estimate: assume ~200 samples average, with validation split
    estimated_training_samples = 180  # Conservative estimate
    estimated_batches = max(1, estimated_training_samples // settings["batch_size"])
    
    if estimated_batches <= 10:
        settings["log_every_n_steps"] = 1
    elif estimated_batches <= 50:
        settings["log_every_n_steps"] = 5
    else:
        settings["log_every_n_steps"] = 10
    
    return settings

@app.route('/')
def index():
    """Main training configuration page"""
    return render_template('index.html', autoplay_audio=request.args.get('autoplay', 'false').lower() == 'true')

@app.route('/create_session', methods=['POST'])
def create_session():
    """Create a new training session with organized directory structure"""
    data = request.json
    voice_name = data.get('voice_name', '').strip()
    
    if not voice_name:
        return jsonify({"error": "Voice name is required"}), 400
    
    # Sanitize voice name for filesystem
    safe_voice_name = secure_filename(voice_name)
    if not safe_voice_name:
        return jsonify({"error": "Invalid voice name"}), 400
    
    # Create session directory structure
    session_dir = DATASETS_DIR / safe_voice_name
    csv_dir = session_dir / "csv"
    audio_dir = session_dir / "audio"
    cache_dir = session_dir / "cache"
    models_dir = session_dir / "models"
    
    try:
        # Create all directories
        for dir_path in [session_dir, csv_dir, audio_dir, cache_dir, models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return jsonify({
            "message": f"Session created for voice: {voice_name}",
            "session_dir": str(session_dir),
            "paths": {
                "csv_dir": str(csv_dir),
                "audio_dir": str(audio_dir),
                "cache_dir": str(cache_dir),
                "models_dir": str(models_dir)
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to create session: {str(e)}"}), 500

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Upload and save CSV metadata file"""
    voice_name = request.form.get('voice_name', '').strip()
    
    if not voice_name:
        return jsonify({"error": "Voice name is required"}), 400
    
    if 'csv_file' not in request.files:
        return jsonify({"error": "No CSV file provided"}), 400
    
    csv_file = request.files['csv_file']
    if csv_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    safe_voice_name = secure_filename(voice_name)
    csv_dir = DATASETS_DIR / safe_voice_name / "csv"
    
    if not csv_dir.exists():
        return jsonify({"error": "Session not created. Please create session first."}), 400
    
    try:
        # Save CSV file (overwrite existing)
        csv_path = csv_dir / "metadata.csv"
        csv_file.save(csv_path)
        
        return jsonify({
            "message": "CSV file uploaded successfully",
            "path": str(csv_path)
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to upload CSV: {str(e)}"}), 500

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Upload and save audio files"""
    try:
        # Check if request is too large
        if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({"error": f"Upload too large. Maximum size is {app.config['MAX_CONTENT_LENGTH'] / (1024**3):.1f}GB. Try uploading in smaller batches of 500-1000 files."}), 413
        
        voice_name = request.form.get('voice_name', '').strip()
        
        if not voice_name:
            return jsonify({"error": "Voice name is required"}), 400
        
        if 'audio_files' not in request.files:
            return jsonify({"error": "No audio files provided"}), 400
    
    audio_files = request.files.getlist('audio_files')
    if not audio_files or all(f.filename == '' for f in audio_files):
        return jsonify({"error": "No files selected"}), 400
    
    safe_voice_name = secure_filename(voice_name)
    audio_dir = DATASETS_DIR / safe_voice_name / "audio"
    
    if not audio_dir.exists():
        return jsonify({"error": "Session not created. Please create session first."}), 400
    
    try:
        # Clear existing audio files
        for existing_file in audio_dir.glob("*"):
            if existing_file.is_file():
                existing_file.unlink()
        
        # Save and validate new audio files
        uploaded_files = []
        validation_warnings = []
        total_files = len([f for f in audio_files if f.filename])
        
        # Initialize progress tracking
        upload_progress.update({
            "is_uploading": True,
            "current_file": 0,
            "total_files": total_files,
            "current_filename": "",
            "operation": "upload"
        })
        
        for i, audio_file in enumerate(audio_files, 1):
            if audio_file.filename:
                # Update progress
                upload_progress.update({
                    "current_file": i,
                    "current_filename": audio_file.filename
                })
                
                safe_filename = secure_filename(audio_file.filename)
                if safe_filename:
                    file_path = audio_dir / safe_filename
                    audio_file.save(file_path)
                    
                    # Validate audio file (if enabled)
                    if ENABLE_AUDIO_VALIDATION:
                        validation_result = validate_audio_file(file_path)
                        if validation_result["valid"]:
                            uploaded_files.append(safe_filename)
                            if validation_result.get("warnings"):
                                validation_warnings.extend([f"{safe_filename}: {w}" for w in validation_result["warnings"]])
                        else:
                            # Remove invalid file
                            file_path.unlink()
                            validation_warnings.append(f"{safe_filename}: {validation_result['error']} (file removed)")
                    else:
                        # Skip validation, just accept the file
                        uploaded_files.append(safe_filename)
        
        # Reset progress tracking
        upload_progress.update({
            "is_uploading": False,
            "current_file": 0,
            "total_files": 0,
            "current_filename": "",
            "operation": ""
        })
        
        response_data = {
            "message": f"{len(uploaded_files)}/{total_files} audio files uploaded successfully",
            "files": uploaded_files,
            "audio_dir": str(audio_dir)
        }
        
        if validation_warnings:
            response_data["warnings"] = validation_warnings
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": f"Failed to upload audio files: {str(e)}"}), 500

def validate_audio_file(file_path):
    """Validate audio file format, sample rate, and quality"""
    if not LIBROSA_AVAILABLE:
        # Fallback: basic file validation without librosa
        return validate_audio_basic(file_path)
    
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=None)
        duration = len(audio) / sr
        
        warnings = []
        
        # Check duration (should be reasonable for TTS training)
        if duration < 0.5:
            return {"valid": False, "error": "Audio too short (< 0.5 seconds)"}
        elif duration > 30:
            warnings.append("Audio is very long (> 30 seconds), consider splitting")
        
        # Check sample rate (22050 is optimal for Piper)
        if sr not in [16000, 22050, 44100, 48000]:
            warnings.append(f"Unusual sample rate ({sr}Hz), will be resampled to 22050Hz")
        elif sr != 22050:
            warnings.append(f"Sample rate is {sr}Hz, will be resampled to 22050Hz")
        
        # Check for silence (too much silence might indicate poor quality)
        rms = librosa.feature.rms(y=audio)[0]
        silence_ratio = np.sum(rms < 0.01) / len(rms)
        if silence_ratio > 0.5:
            warnings.append("Audio contains significant silence, consider trimming")
        
        # Check for clipping
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0.99:
            warnings.append("Audio may be clipped (very high amplitude)")
        elif max_amplitude < 0.1:
            warnings.append("Audio level is very low, consider normalizing")
        
        return {
            "valid": True,
            "duration": round(duration, 2),
            "sample_rate": sr,
            "warnings": warnings
        }
        
    except Exception as e:
        return {"valid": False, "error": f"Could not process audio file: {str(e)}"}

def validate_audio_basic(file_path):
    """Basic audio validation without librosa dependency"""
    try:
        # Check file extension
        valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        if not any(str(file_path).lower().endswith(ext) for ext in valid_extensions):
            return {"valid": False, "error": "Unsupported audio format"}
        
        # Check file size (should be reasonable)
        file_size = file_path.stat().st_size
        if file_size < 1000:  # Less than 1KB
            return {"valid": False, "error": "File too small (likely empty or corrupted)"}
        elif file_size > 100_000_000:  # More than 100MB
            return {"valid": False, "error": "File too large (>100MB)"}
        
        return {
            "valid": True,
            "warnings": ["Audio validation limited (librosa not available)"]
        }
        
    except Exception as e:
        return {"valid": False, "error": f"Could not validate audio file: {str(e)}"}

def get_directory_size(directory):
    """Get total size of directory in bytes"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                try:
                    total_size += filepath.stat().st_size
                except (OSError, FileNotFoundError):
                    continue
    except Exception:
        pass
    return total_size

def cleanup_cache_lru(cache_dir, max_size_bytes):
    """Clean up cache using LRU (Least Recently Used) strategy"""
    try:
        if not cache_dir.exists():
            return 0, 0
        
        current_size = get_directory_size(cache_dir)
        if current_size <= max_size_bytes:
            return 0, current_size
        
        # Get all cache files with their access times
        cache_files = []
        for file_path in cache_dir.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    cache_files.append({
                        'path': file_path,
                        'atime': stat.st_atime,
                        'size': stat.st_size
                    })
                except (OSError, FileNotFoundError):
                    continue
        
        # Sort by access time (oldest first)
        cache_files.sort(key=lambda x: x['atime'])
        
        # Remove files until we're under the limit
        removed_count = 0
        bytes_removed = 0
        target_size = max_size_bytes * CACHE_CLEANUP_THRESHOLD  # Clean to 80% of max
        
        for file_info in cache_files:
            if current_size - bytes_removed <= target_size:
                break
            
            try:
                file_info['path'].unlink()
                bytes_removed += file_info['size']
                removed_count += 1
            except (OSError, FileNotFoundError):
                continue
        
        return removed_count, bytes_removed
        
    except Exception as e:
        print(f"Error during cache cleanup: {str(e)}")
        return 0, 0

def check_and_cleanup_caches():
    """Check all cache directories and clean up if needed"""
    max_size_bytes = int(MAX_CACHE_SIZE_GB * 1024**3)
    total_removed = 0
    total_bytes_removed = 0
    
    # Check all session cache directories
    if DATASETS_DIR.exists():
        for session_dir in DATASETS_DIR.iterdir():
            if session_dir.is_dir():
                cache_dir = session_dir / "cache"
                if cache_dir.exists():
                    removed, bytes_removed = cleanup_cache_lru(cache_dir, max_size_bytes)
                    total_removed += removed
                    total_bytes_removed += bytes_removed
                    
                    if removed > 0:
                        print(f"🧹 Cleaned {removed} files ({bytes_removed / (1024**3):.2f}GB) from {session_dir.name} cache")
    
    return total_removed, total_bytes_removed

@app.route('/list_sessions')
def list_sessions():
    """List all available training sessions"""
    sessions = []
    
    if DATASETS_DIR.exists():
        for session_dir in DATASETS_DIR.iterdir():
            if session_dir.is_dir():
                csv_file = session_dir / "csv" / "metadata.csv"
                audio_dir = session_dir / "audio"
                
                # Count audio files
                audio_count = len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0
                
                sessions.append({
                    "name": session_dir.name,
                    "path": str(session_dir),
                    "has_csv": csv_file.exists(),
                    "audio_count": audio_count,
                    "ready_for_training": csv_file.exists() and audio_count > 0
                })
    
    return jsonify(sessions)

@app.route('/train', methods=['POST'])
def start_training():
    """Start a new training job"""
    if training_status["is_training"]:
        return jsonify({"error": "Training is already in progress"}), 400
    
    # Get form data
    data = request.json
    voice_name = data.get('voice_name', '').strip()
    sample_rate = int(data.get('sample_rate', 22050))
    espeak_voice = data.get('espeak_voice', 'en-us')
    batch_size = int(data.get('batch_size', 32))
    ckpt_path = data.get('ckpt_path', '').strip()
    
    # Validation
    if not voice_name:
        return jsonify({"error": "Voice name is required"}), 400
    
    safe_voice_name = secure_filename(voice_name)
    session_dir = DATASETS_DIR / safe_voice_name
    
    # Check if session exists
    if not session_dir.exists():
        return jsonify({"error": "Training session not found. Please create session first."}), 400
    
    # Auto-generate paths based on session structure
    csv_path = session_dir / "csv" / "metadata.csv"
    audio_dir = session_dir / "audio"
    cache_dir = session_dir / "cache"
    config_path = session_dir / "models" / f"{voice_name}_config.json"
    
    # Clear cache directory to avoid stale cached data issues
    try:
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Cleared cache directory: {cache_dir}")
    except Exception as e:
        print(f"Warning: Could not clear cache directory: {str(e)}")
    
    # Check and cleanup caches if they're getting too large
    try:
        removed_files, bytes_removed = check_and_cleanup_caches()
        if removed_files > 0:
            print(f"🧹 Cache cleanup: removed {removed_files} files ({bytes_removed / (1024**3):.2f}GB)")
    except Exception as e:
        print(f"Warning: Cache cleanup failed: {str(e)}")
    
    # Validation
    if not csv_path.exists():
        return jsonify({"error": "CSV file not found. Please upload metadata.csv first."}), 400
        
    if not audio_dir.exists() or not list(audio_dir.glob("*.wav")):
        return jsonify({"error": "Audio files not found. Please upload audio files first."}), 400
    
    # Get optimal settings for this system
    system_info = detect_system_capabilities()
    optimal_settings = calculate_optimal_settings(system_info)
    
    # Use provided batch_size if specified, otherwise use optimal
    if batch_size == 32:  # Default value, use optimal
        batch_size = optimal_settings["batch_size"]
    
    # Intelligent max_epochs handling based on checkpoint
    max_epochs = optimal_settings["max_epochs"]  # Default: 3000
    if ckpt_path and ckpt_path.strip() and ckpt_path != "none":
        try:
            # Try to extract epoch from checkpoint filename (common patterns: epoch####, epoch-####, etc.)
            import re
            epoch_patterns = [
                r'epoch[_-]?(\d+)',  # epoch123, epoch_123, epoch-123
                r'step[_-]?(\d+)',   # step123 (fallback)
            ]
            
            checkpoint_epoch = None
            for pattern in epoch_patterns:
                epoch_match = re.search(pattern, ckpt_path, re.IGNORECASE)
                if epoch_match:
                    checkpoint_epoch = int(epoch_match.group(1))
                    break
            
            if checkpoint_epoch is not None and checkpoint_epoch >= 3000:
                # Only extend if checkpoint is already at 3000+ epochs
                max_epochs = checkpoint_epoch + 1000
                print(f"📊 Checkpoint at epoch {checkpoint_epoch} (≥3000), extending max_epochs to {max_epochs}")
            elif checkpoint_epoch is not None:
                print(f"📊 Checkpoint at epoch {checkpoint_epoch} (<3000), using default max_epochs: {max_epochs}")
            else:
                print(f"📊 Could not detect checkpoint epoch, using default max_epochs: {max_epochs}")
                
        except Exception as e:
            print(f"⚠️ Could not parse checkpoint epoch from filename, using default max_epochs: {e}")
    
    # Update optimal_settings with the calculated max_epochs
    optimal_settings["max_epochs"] = max_epochs
    
    # Prepare training command
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"training_{voice_name}_{timestamp}.log"
    
    cmd = [
        "python3", "-m", "piper.train", "fit",
        "--data.voice_name", voice_name,
        "--data.csv_path", csv_path,
        "--data.audio_dir", audio_dir,
        "--model.sample_rate", str(sample_rate),
        "--data.espeak_voice", espeak_voice,
        "--data.cache_dir", cache_dir,
        "--data.config_path", config_path,
        "--data.batch_size", str(batch_size),
        "--trainer.default_root_dir", str(CHECKPOINTS_DIR / voice_name),
        # Dynamic performance optimizations
        "--data.num_workers", str(optimal_settings["num_workers"]),
        "--trainer.log_every_n_steps", str(optimal_settings["log_every_n_steps"]),
        "--trainer.precision", optimal_settings["precision"],
        "--trainer.devices", str(optimal_settings["devices"]),
        "--trainer.max_epochs", str(optimal_settings["max_epochs"]),
        # TensorBoard logging
        "--trainer.logger", "TensorBoardLogger",
        "--trainer.logger.save_dir", str(TENSORBOARD_LOG_DIR),
        "--trainer.logger.name", voice_name,
    ]
    
    # Log the detected system and chosen settings
    print(f"🖥️  System detected: {system_info['cpu_count']} CPU cores, {system_info['memory_gb']}GB RAM")
    if system_info["gpu_available"]:
        print(f"🚀 GPU detected: {system_info['gpu_count']}x {', '.join(system_info['gpu_names'])}")
        print(f"💾 GPU memory: {system_info['gpu_memory_gb']}GB")
    print(f"⚙️  Optimal settings: batch_size={batch_size}, num_workers={optimal_settings['num_workers']}, precision={optimal_settings['precision']}")
    print(f"📊 Logging every {optimal_settings['log_every_n_steps']} steps, using {optimal_settings['devices']} device(s)")
    
    # Add checkpoint if provided
    if ckpt_path and ckpt_path != "none":
        if ckpt_path.startswith("http"):
            # Download checkpoint if it's a URL
            import urllib.request
            ckpt_filename = f"{voice_name}-checkpoint_{timestamp}.ckpt"
            local_ckpt_path = CHECKPOINTS_DIR / ckpt_filename
            try:
                print(f"Downloading checkpoint from: {ckpt_path}")
                urllib.request.urlretrieve(ckpt_path, local_ckpt_path)
                cmd.extend(["--ckpt_path", str(local_ckpt_path)])
                print(f"Checkpoint downloaded to: {local_ckpt_path}")
            except Exception as e:
                return jsonify({"error": f"Failed to download checkpoint: {str(e)}"}), 400
        else:
            # Handle local checkpoint path
            ckpt_file = Path(ckpt_path)
            if not ckpt_file.exists():
                return jsonify({"error": f"Checkpoint file not found: {ckpt_path}"}), 400
            if not ckpt_file.suffix.lower() == '.ckpt':
                return jsonify({"error": f"Invalid checkpoint file. Expected .ckpt file, got: {ckpt_file.suffix}"}), 400
            
            print(f"Using local checkpoint: {ckpt_file}")
            # Convert to absolute path to avoid any path resolution issues
            abs_ckpt_path = ckpt_file.resolve()
            cmd.extend(["--ckpt_path", str(abs_ckpt_path)])
    
    # Start training in background thread
    def run_training():
        training_status["is_training"] = True
        training_status["current_job"] = voice_name
        training_status["log_file"] = str(log_file)
        training_status["voice_name"] = voice_name
        training_status["start_time"] = datetime.now().isoformat()
        training_status["error"] = None
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd, 
                    stdout=f, 
                    stderr=subprocess.STDOUT,
                    cwd=Path.cwd(),
                    env=dict(os.environ, PYTHONPATH=str(Path.cwd() / "src"))
                )
                training_status["process"] = process
                return_code = process.wait()
                
                # Check if training failed
                if return_code != 0:
                    error_msg = analyze_training_error(log_file)
                    training_status["error"] = error_msg
                    with open(log_file, 'a') as f:
                        f.write(f"\n=== TRAINING FAILED (exit code: {return_code}) ===\n")
                        f.write(f"Error analysis: {error_msg}\n")
                        
        except Exception as e:
            error_msg = f"Training process failed to start: {str(e)}"
            training_status["error"] = error_msg
            with open(log_file, 'a') as f:
                f.write(f"\nTraining failed with error: {error_msg}\n")
        finally:
            training_status["is_training"] = False
            training_status["current_job"] = None
            # Clear GPU memory to prevent leaks between training runs
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("🧹 GPU memory cleared after training")

    # Start training thread
    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "message": f"Training started for voice: {voice_name}",
        "log_file": str(log_file)
    })

def analyze_training_error(log_file_path):
    """Analyze training log to provide helpful error messages"""
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read().lower()
        
        # Common error patterns and user-friendly messages
        error_patterns = [
            ("misconfigurationexception", "Checkpoint configuration mismatch. The checkpoint has more epochs than max_epochs setting. Increase max_epochs or use a different checkpoint."),
            ("current_epoch=", "Checkpoint epoch exceeds max_epochs limit. Either increase max_epochs in training settings or choose a checkpoint with fewer epochs."),
            ("cuda out of memory", "GPU ran out of memory. Try reducing batch size or closing other GPU applications."),
            ("filenotfounderror", "Required files are missing. Check that all audio files and CSV metadata exist."),
            ("permission denied", "Permission error. Check file permissions and that files aren't being used by other programs."),
            ("module not found", "Missing Python dependencies. Ensure all required packages are installed."),
            ("no such file or directory", "File path error. Verify all file paths are correct and accessible."),
            ("connection refused", "Network connection failed. Check internet connection if downloading checkpoints."),
            ("disk space", "Insufficient disk space. Free up storage and try again."),
            ("nan", "Training instability detected. Try reducing learning rate or checking data quality."),
            ("assertion error", "Data format error. Verify CSV format and audio file compatibility."),
            ("timeout", "Training process timed out. This might indicate system resource issues.")
        ]
        
        for pattern, message in error_patterns:
            if pattern in log_content:
                return message
        
        return "Training failed with unknown error. Check the training logs for details."
        
    except Exception as e:
        return f"Could not analyze error log: {str(e)}"

@app.route('/status')
def get_status():
    """Get current training status"""
    # Create a copy without the non-serializable process object
    status_copy = {
        "is_training": training_status["is_training"],
        "current_job": training_status["current_job"],
        "log_file": training_status["log_file"],
        "voice_name": training_status.get("voice_name"),
        "start_time": training_status.get("start_time"),
        "progress": extract_training_progress(),
        "error": training_status.get("error")
    }
    return jsonify(status_copy)

@app.route('/upload_progress')
def get_upload_progress():
    """Get current upload progress"""
    return jsonify(upload_progress)

def extract_training_progress():
    """Extract training progress from log file with ETA estimation"""
    if not training_status.get("log_file") or not training_status["is_training"]:
        return {"epoch": 0, "step": 0, "progress_percent": 0, "eta_minutes": None, "steps_per_second": 0}
    
    try:
        log_path = Path(training_status["log_file"])
        if not log_path.exists():
            return {"epoch": 0, "step": 0, "progress_percent": 0, "eta_minutes": None, "steps_per_second": 0}
        
        # Read last few lines of log file for progress info
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_epoch = 0
        current_step = 0
        max_epochs = 1000  # Default from optimal settings
        
        # Track timing for ETA calculation
        import re
        from datetime import datetime, timedelta
        
        step_timestamps = []
        recent_steps = []
        
        # Parse log lines for training progress
        for line in reversed(lines[-100:]):  # Check last 100 lines for better timing data
            line = line.strip()
            
            # Look for PyTorch Lightning progress patterns: "Epoch 2173:  17%|█▋        | 1/6 [00:51<04:17,  0.02it/s]"
            if "Epoch" in line and "%" in line and "|" in line:
                epoch_match = re.search(r'Epoch (\d+):', line)
                step_match = re.search(r'\| (\d+)/(\d+) \[', line)
                
                if epoch_match and step_match:
                    epoch = int(epoch_match.group(1))
                    current_step_in_epoch = int(step_match.group(1))
                    total_steps_in_epoch = int(step_match.group(2))
                    
                    # Update current progress
                    if epoch > current_epoch or (epoch == current_epoch and current_step_in_epoch > current_step):
                        current_epoch = epoch
                        current_step = current_step_in_epoch
                    
                    # Extract timestamp for ETA calculation
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp_match:
                        try:
                            timestamp_str = timestamp_match.group(1)
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                            step_timestamps.append((step, timestamp))
                            recent_steps.append(step)
                        except ValueError:
                            pass
        
        # Calculate steps per second and ETA
        steps_per_second = 0
        eta_minutes = None
        
        if len(step_timestamps) >= 2:
            # Sort by step number
            step_timestamps.sort(key=lambda x: x[0])
            
            # Calculate average steps per second from recent data
            recent_timestamps = step_timestamps[-10:]  # Last 10 data points
            if len(recent_timestamps) >= 2:
                time_diff = (recent_timestamps[-1][1] - recent_timestamps[0][1]).total_seconds()
                step_diff = recent_timestamps[-1][0] - recent_timestamps[0][0]
                
                if time_diff > 0:
                    steps_per_second = step_diff / time_diff
                    
                    # Estimate remaining steps (rough calculation)
                    # Assume each epoch has similar number of steps as current
                    if current_epoch > 0 and current_step > 0:
                        estimated_steps_per_epoch = current_step // max(1, current_epoch)
                        remaining_epochs = max(0, max_epochs - current_epoch)
                        remaining_steps_this_epoch = max(0, estimated_steps_per_epoch - (current_step % estimated_steps_per_epoch))
                        total_remaining_steps = remaining_steps_this_epoch + (remaining_epochs * estimated_steps_per_epoch)
                        
                        if steps_per_second > 0:
                            eta_seconds = total_remaining_steps / steps_per_second
                            eta_minutes = eta_seconds / 60
        
        # Calculate progress percentage (more accurate with step info)
        if current_epoch > 0 and current_step > 0:
            # Better progress calculation using steps
            estimated_total_steps = (current_step / max(1, current_epoch)) * max_epochs
            progress_percent = min(100, (current_step / estimated_total_steps) * 100) if estimated_total_steps > 0 else 0
        else:
            # Fallback to epoch-based calculation
            progress_percent = min(100, (current_epoch / max_epochs) * 100) if max_epochs > 0 else 0
        
        return {
            "epoch": current_epoch,
            "step": current_step,
            "progress_percent": round(progress_percent, 1),
            "eta_minutes": round(eta_minutes, 1) if eta_minutes else None,
            "steps_per_second": round(steps_per_second, 2),
            "max_epochs": max_epochs
        }
        
    except Exception as e:
        print(f"Error extracting progress: {str(e)}")
        return {"epoch": 0, "step": 0, "progress_percent": 0, "eta_minutes": None, "steps_per_second": 0}

@app.route('/system_info')
def get_system_info():
    """Get system capabilities and optimal settings"""
    system_info = detect_system_capabilities()
    optimal_settings = calculate_optimal_settings(system_info)
    
    # Generate memory warnings
    warnings = []
    if system_info["gpu_available"]:
        if system_info["gpu_memory_percent"] > 80:
            warnings.append("⚠️ GPU memory usage is high (>80%). Consider reducing batch size or clearing GPU cache.")
        elif system_info["gpu_memory_percent"] > 60:
            warnings.append("⚠️ GPU memory usage is moderate (>60%). Monitor for potential OOM errors.")
    
    if system_info["memory_percent"] > 85:
        warnings.append("⚠️ System RAM usage is high (>85%). Close other applications for better performance.")
    
    return jsonify({
        "system": system_info,
        "optimal_settings": optimal_settings,
        "recommendations": {
            "performance_level": "high" if system_info["gpu_memory_gb"] >= 16 else 
                               "medium" if system_info["gpu_memory_gb"] >= 8 else "basic",
            "expected_training_speed": "fast" if system_info["gpu_available"] and system_info["gpu_memory_gb"] >= 12 else
                                     "medium" if system_info["gpu_available"] else "slow",
            "multi_gpu_ready": system_info["gpu_count"] > 1,
            "memory_sufficient": system_info["memory_gb"] >= 16
        },
        "warnings": warnings
    })

@app.route('/stop_training', methods=['POST'])
def stop_training():
    """Stop the current training process"""
    if not training_status["is_training"]:
        return jsonify({"error": "No training is currently running"}), 400
    
    try:
        # Get the process from training_status if available
        if "process" in training_status and training_status["process"]:
            process = training_status["process"]
            process.terminate()
            
            # Wait a bit for graceful shutdown
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                process.kill()
                process.wait()
        
        # Reset training status
        training_status["is_training"] = False
        training_status["current_job"] = None
        training_status["error"] = "Training stopped by user"
        training_status["process"] = None
        
        # Clear GPU memory after stopping
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return jsonify({"message": "Training stopped successfully"})
        
    except Exception as e:
        return jsonify({"error": f"Failed to stop training: {str(e)}"}), 500

@app.route('/clear_gpu_memory', methods=['POST'])
def clear_gpu_memory():
    """Manually clear GPU memory cache"""
    if not TORCH_AVAILABLE:
        return jsonify({"error": "PyTorch not available"}), 400
    
    if not torch.cuda.is_available():
        return jsonify({"error": "CUDA not available"}), 400
    
    try:
        # Get memory stats before clearing
        memory_before = torch.cuda.memory_allocated() / (1024**3)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Clear any loaded models from memory
        global chat_sessions, loaded_models
        
        # Clear chat sessions (this will unload any loaded LLM models)
        chat_sessions.clear()
        
        # Clear loaded models
        loaded_models["llm"] = None
        loaded_models["voice"] = None
        
        # Clear any loaded voice models by forcing garbage collection
        import gc
        gc.collect()
        
        # Additional cleanup for any remaining references
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Get memory stats after clearing
        memory_after = torch.cuda.memory_allocated() / (1024**3)
        freed_memory = memory_before - memory_after
        
        return jsonify({
            "message": "GPU memory cleared and all models unloaded successfully",
            "memory_freed_gb": round(freed_memory, 2),
            "memory_before_gb": round(memory_before, 2),
            "memory_after_gb": round(memory_after, 2)
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to clear GPU memory: {str(e)}"}), 500

@app.route('/cleanup_cache', methods=['POST'])
def cleanup_cache():
    """Manually clean up cache directories"""
    try:
        removed_files, bytes_removed = check_and_cleanup_caches()
        
        return jsonify({
            "message": f"Cache cleanup completed",
            "files_removed": removed_files,
            "bytes_removed": bytes_removed,
            "gb_removed": round(bytes_removed / (1024**3), 2),
            "max_cache_size_gb": MAX_CACHE_SIZE_GB
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to cleanup cache: {str(e)}"}), 500

@app.route('/start_tensorboard', methods=['POST'])
def start_tensorboard():
    """Start TensorBoard server"""
    global tensorboard_process
    
    if not TENSORBOARD_AVAILABLE:
        return jsonify({"error": "TensorBoard not available. Install with: pip install tensorboard"}), 400
    
    if tensorboard_process and tensorboard_process.poll() is None:
        return jsonify({"error": "TensorBoard is already running"}), 400
    
    try:
        import subprocess
        
        # Start TensorBoard on port 6006
        tensorboard_process = subprocess.Popen([
            "tensorboard", 
            "--logdir", str(TENSORBOARD_LOG_DIR),
            "--port", "6006",
            "--host", "0.0.0.0"
        ])
        
        return jsonify({
            "message": "TensorBoard started successfully",
            "url": "http://localhost:6006",
            "log_dir": str(TENSORBOARD_LOG_DIR)
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to start TensorBoard: {str(e)}"}), 500

@app.route('/stop_tensorboard', methods=['POST'])
def stop_tensorboard():
    """Stop TensorBoard server"""
    global tensorboard_process
    
    if not tensorboard_process or tensorboard_process.poll() is not None:
        return jsonify({"error": "TensorBoard is not running"}), 400
    
    try:
        tensorboard_process.terminate()
        tensorboard_process.wait(timeout=5)
        tensorboard_process = None
        
        return jsonify({"message": "TensorBoard stopped successfully"})
        
    except Exception as e:
        return jsonify({"error": f"Failed to stop TensorBoard: {str(e)}"}), 500

@app.route('/tensorboard_status')
def tensorboard_status():
    """Get TensorBoard status"""
    global tensorboard_process
    
    is_running = tensorboard_process is not None and tensorboard_process.poll() is None
    
    return jsonify({
        "is_running": is_running,
        "url": "http://localhost:6006" if is_running else None,
        "log_dir": str(TENSORBOARD_LOG_DIR),
        "available": TENSORBOARD_AVAILABLE
    })

@app.route('/logs/<path:log_file>')
def get_log(log_file):
    """Stream training logs"""
    log_path = LOGS_DIR / log_file
    if not log_path.exists():
        return "Log file not found", 404
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/plain'}
    except Exception as e:
        return f"Error reading log: {str(e)}", 500

@app.route('/checkpoints')
def list_checkpoints():
    """List all available checkpoints"""
    checkpoints = []
    for ckpt_file in CHECKPOINTS_DIR.rglob("*.ckpt"):
        stat = ckpt_file.stat()
        checkpoints.append({
            "name": ckpt_file.name,
            "path": str(ckpt_file),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })
    
    return jsonify(checkpoints)

@app.route('/voice_names')
def list_voice_names():
    """List all voice name directories under datasets"""
    voice_names = []
    
    if DATASETS_DIR.exists():
        for session_dir in DATASETS_DIR.iterdir():
            if session_dir.is_dir():
                voice_names.append(session_dir.name)
    
    # Sort alphabetically
    voice_names.sort()
    return jsonify(voice_names)

@app.route('/models')
def list_models():
    """List all exported ONNX models from all sessions"""
    models = []
    
    # Search in all session model directories
    if DATASETS_DIR.exists():
        for session_dir in DATASETS_DIR.iterdir():
            if session_dir.is_dir():
                models_dir = session_dir / "models"
                if models_dir.exists():
                    for model_file in models_dir.glob("*.onnx"):
                        stat = model_file.stat()
                        config_file = model_file.with_suffix('.onnx.json')
                        models.append({
                            "name": model_file.name,
                            "session": session_dir.name,
                            "path": str(model_file),
                            "config_path": str(config_file) if config_file.exists() else None,
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
    
    return jsonify(models)

@app.route('/voice_configs')
def list_voice_configs():
    """List all voice JSON config files from all sessions"""
    configs = []
    
    # Search in all session model directories
    if DATASETS_DIR.exists():
        for session_dir in DATASETS_DIR.iterdir():
            if session_dir.is_dir():
                models_dir = session_dir / "models"
                if models_dir.exists():
                    for config_file in models_dir.glob("*.onnx.json"):
                        stat = config_file.stat()
                        configs.append({
                            "name": config_file.name,
                            "session": session_dir.name,
                            "path": str(config_file),
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
    
    return jsonify(configs)

@app.route('/logs')
def list_logs():
    """List all available log files"""
    logs = []
    for log_file in LOGS_DIR.glob("*.log"):
        stat = log_file.stat()
        logs.append({
            "name": log_file.name,
            "path": str(log_file),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })
    
    # Sort by modification time (newest first)
    logs.sort(key=lambda x: x["modified"], reverse=True)
    return jsonify(logs)

@app.route('/export', methods=['POST'])
def export_onnx():
    """Export a checkpoint to ONNX format"""
    data = request.json
    checkpoint_path = data.get('checkpoint_path', '').strip()
    output_name = data.get('output_name', '').strip()
    voice_name = data.get('voice_name', '').strip()
    
    if not checkpoint_path or not output_name:
        return jsonify({"error": "Checkpoint path and output name are required"}), 400
    
    if not Path(checkpoint_path).exists():
        return jsonify({"error": f"Checkpoint not found: {checkpoint_path}"}), 400
    
    # Prepare output path - always save to session models directory if voice_name provided
    if not output_name.endswith('.onnx'):
        output_name += '.onnx'
    
    if voice_name:
        safe_voice_name = secure_filename(voice_name)
        session_dir = DATASETS_DIR / safe_voice_name
        models_dir = session_dir / "models"
        models_dir.mkdir(exist_ok=True)
        output_path = models_dir / output_name
        
        # Check if file already exists and rename with suffix
        if output_path.exists():
            base_name = output_name[:-5]  # Remove .onnx
            counter = 1
            while output_path.exists():
                new_name = f"{base_name}_v{counter}.onnx"
                output_path = models_dir / new_name
                counter += 1
            
            # Also rename the corresponding JSON file if it exists
            json_path = models_dir / f"{base_name}.onnx.json"
            if json_path.exists():
                new_json_name = f"{base_name}_v{counter-1}.onnx.json"
                new_json_path = models_dir / new_json_name
                try:
                    import shutil
                    shutil.move(str(json_path), str(new_json_path))
                    print(f"📄 Renamed existing JSON: {json_path} -> {new_json_path}")
                except Exception as e:
                    print(f"⚠️ Could not rename JSON file: {e}")
            
            print(f"📁 Renamed existing ONNX file to: {output_path.name}")
    else:
        # Fallback to checkpoints directory if no voice_name
        output_path = CHECKPOINTS_DIR / output_name
    
    # Export command
    cmd = [
        "python3", "-m", "piper.train.export_onnx",
        "--checkpoint", checkpoint_path,
        "--output-file", str(output_path)
    ]
    
    try:
        print(f"🔄 Starting ONNX export: {checkpoint_path} -> {output_path}")
        print(f"📋 Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            cwd=Path.cwd(),
            env=dict(os.environ, PYTHONPATH=str(Path.cwd() / "src"))
        )
        
        if result.returncode == 0:
            print(f"✅ ONNX export completed successfully: {output_path}")
            
            # Create the JSON config file with the same name as the ONNX file
            json_config_created = False
            if voice_name:
                safe_voice_name = secure_filename(voice_name)
                session_dir = DATASETS_DIR / safe_voice_name
                
                # Look for the config file in the session models directory
                config_source = session_dir / "models" / f"{voice_name}_config.json"
                json_dest = output_path.with_suffix('.onnx.json')
                
                if config_source.exists():
                    try:
                        import shutil
                        shutil.copy2(config_source, json_dest)
                        json_config_created = True
                        print(f"📄 Created JSON config: {config_source} -> {json_dest}")
                    except Exception as e:
                        print(f"⚠️ Could not create JSON config: {e}")
                else:
                    print(f"⚠️ JSON config source not found: {config_source}")
                    # Create a minimal JSON config file
                    try:
                        minimal_config = {
                            "model_type": "vits",
                            "model_path": str(output_path),
                            "voice_name": voice_name,
                            "created_from_checkpoint": checkpoint_path,
                            "export_date": datetime.now().isoformat(),
                            # Required fields for PiperTTS
                            "num_symbols": 256,
                            "num_speakers": 1,
                            "audio": {
                                "sample_rate": 22050,
                                "quality": "medium"
                            },
                            "espeak": {
                                "voice": "en-us"
                            },
                            "inference": {
                                "noise_scale": 0.667,
                                "length_scale": 1.0,
                                "noise_w": 0.8
                            },
                            "phoneme_id_map": {
                                "_": [0], "^": [1], "$": [2], " ": [3], "!": [4], "'": [5],
                                "(": [6], ")": [7], ",": [8], "-": [9], ".": [10], ":": [11],
                                ";": [12], "?": [13], "a": [14], "b": [15], "c": [16], "d": [17],
                                "e": [18], "f": [19], "h": [20], "i": [21], "j": [22], "k": [23],
                                "l": [24], "m": [25], "n": [26], "o": [27], "p": [28], "q": [29],
                                "r": [30], "s": [31], "t": [32], "u": [33], "v": [34], "w": [35],
                                "x": [36], "y": [37], "z": [38], "æ": [39], "ç": [40], "ð": [41],
                                "ø": [42], "ħ": [43], "ŋ": [44], "œ": [45], "ǀ": [46], "ǁ": [47],
                                "ǂ": [48], "ǃ": [49], "ɐ": [50], "ɑ": [51], "ɒ": [52], "ɓ": [53],
                                "ɔ": [54], "ɕ": [55], "ɖ": [56], "ɗ": [57], "ɘ": [58], "ə": [59],
                                "ɚ": [60], "ɛ": [61], "ɜ": [62], "ɞ": [63], "ɟ": [64], "ɠ": [65],
                                "ɡ": [66], "ɢ": [67], "ɣ": [68], "ɤ": [69], "ɥ": [70], "ɦ": [71],
                                "ɧ": [72], "ɨ": [73], "ɪ": [74], "ɫ": [75], "ɬ": [76], "ɭ": [77],
                                "ɮ": [78], "ɯ": [79], "ɰ": [80], "ɱ": [81], "ɲ": [82], "ɳ": [83],
                                "ɴ": [84], "ɵ": [85], "ɶ": [86], "ɸ": [87], "ɹ": [88], "ɺ": [89],
                                "ɻ": [90], "ɽ": [91], "ɾ": [92], "ʀ": [93], "ʁ": [94], "ʂ": [95],
                                "ʃ": [96], "ʄ": [97], "ʈ": [98], "ʉ": [99], "ʊ": [100], "ʋ": [101],
                                "ʌ": [102], "ʍ": [103], "ʎ": [104], "ʏ": [105], "ʐ": [106], "ʑ": [107],
                                "ʒ": [108], "ʔ": [109], "ʕ": [110], "ʘ": [111], "ʙ": [112], "ʛ": [113],
                                "ʜ": [114], "ʝ": [115], "ʟ": [116], "ʡ": [117], "ʢ": [118], "ʲ": [119],
                                "ˈ": [120], "ˌ": [121], "ː": [122], "ˑ": [123], "˞": [124], "β": [125],
                                "θ": [126], "χ": [127], "ᵻ": [128], "ⱱ": [129], "0": [130], "1": [131],
                                "2": [132], "3": [133], "4": [134], "5": [135], "6": [136], "7": [137],
                                "8": [138], "9": [139], "̧": [140], "̃": [141], "̪": [142], "̯": [143],
                                "̩": [144], "ʰ": [145], "ˤ": [146], "ε": [147], "↓": [148], "#": [149],
                                "\"": [150], "↑": [151], "̺": [152], "̻": [153]
                            },
                            "speaker_id_map": {},
                            "phoneme_type": "espeak",
                            "phoneme_map": {},
                            "language": {
                                "code": "en_US",
                                "family": "en",
                                "region": "US",
                                "name_native": "English",
                                "name_english": "English",
                                "country_english": "United States"
                            },
                            "piper_version": "1.0.0"
                        }
                        with open(json_dest, 'w', encoding='utf-8') as f:
                            json.dump(minimal_config, f, indent=2, ensure_ascii=False)
                        json_config_created = True
                        print(f"📄 Created minimal JSON config: {json_dest}")
                    except Exception as e:
                        print(f"⚠️ Could not create minimal JSON config: {e}")
            
            message = f"Successfully exported to {output_path}"
            if json_config_created:
                message += f" and {output_path.with_suffix('.onnx.json')}"
            
            return jsonify({
                "message": message,
                "output_path": str(output_path),
                "json_config_created": json_config_created,
                "json_config_path": str(output_path.with_suffix('.onnx.json')) if json_config_created else None
            })
        else:
            error_msg = f"Export failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr}"
            if result.stdout:
                error_msg += f"\nOutput: {result.stdout}"
            
            print(f"❌ {error_msg}")
            return jsonify({
                "error": error_msg
            }), 500
            
    except Exception as e:
        error_msg = f"Export failed: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500

@app.route('/delete_file', methods=['POST'])
def delete_file():
    """Delete a file from checkpoints, models, or logs directories"""
    data = request.json
    file_path = data.get('file_path', '').strip()
    
    if not file_path:
        return jsonify({"error": "File path is required"}), 400
    
    try:
        file_path = Path(file_path)
        # Security: only allow deletion from our designated directories
        allowed_dirs = [CHECKPOINTS_DIR, LOGS_DIR]
        
        # Also allow deletion from session model directories
        for session_dir in DATASETS_DIR.glob("*/models"):
            allowed_dirs.append(session_dir)
        
        # Check if file is in an allowed directory
        resolved_path = file_path.resolve()
        allowed = False
        for allowed_dir in allowed_dirs:
            try:
                resolved_path.relative_to(allowed_dir.resolve())
                allowed = True
                break
            except ValueError:
                continue
        
        if not allowed:
            return jsonify({"error": "File deletion not allowed from this directory"}), 403
        
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            return jsonify({"message": f"File deleted successfully: {file_path.name}"})
        else:
            return jsonify({"error": "File not found"}), 404
            
    except Exception as e:
        return jsonify({"error": f"Failed to delete file: {str(e)}"}), 500

@app.route('/download/<path:file_path>')
def download_file(file_path):
    """Download checkpoint or model files"""
    try:
        # Handle different path formats
        file_path_obj = Path(file_path)
        
        # Security: only allow downloads from our designated directories
        allowed_dirs = [CHECKPOINTS_DIR, LOGS_DIR, Path.cwd()]
        
        # Add session model directories
        if DATASETS_DIR.exists():
            for session_dir in DATASETS_DIR.iterdir():
                if session_dir.is_dir():
                    models_dir = session_dir / "models"
                    if models_dir.exists():
                        allowed_dirs.append(models_dir)
        
        # Try to find the file
        for allowed_dir in allowed_dirs:
            # Try direct path
            full_path = allowed_dir / file_path_obj
            if full_path.exists() and full_path.is_file():
                return send_file(str(full_path.resolve()), as_attachment=True, download_name=full_path.name)
            
            # Try just the filename in the directory
            filename_only = file_path_obj.name
            full_path = allowed_dir / filename_only
            if full_path.exists() and full_path.is_file():
                return send_file(str(full_path.resolve()), as_attachment=True, download_name=full_path.name)
        
        # Try absolute path if it exists and is in allowed locations
        if file_path_obj.is_absolute() and file_path_obj.exists():
            # Check if the absolute path is within our allowed directories
            resolved_path = file_path_obj.resolve()
            for allowed_dir in allowed_dirs:
                try:
                    resolved_path.relative_to(allowed_dir.resolve())
                    return send_file(str(resolved_path), as_attachment=True, download_name=resolved_path.name)
                except ValueError:
                    continue
        
        return "File not found", 404
        
    except Exception as e:
        return f"Download error: {str(e)}", 500

def normalize_path(path_str):
    """Normalize path for cross-platform compatibility"""
    if not path_str:
        return ""
    
    # Convert Windows paths to WSL format if running in WSL
    if platform.system() == "Linux" and path_str.startswith(("C:", "D:", "E:", "F:")):
        # Convert C:\path\to\file to /mnt/c/path/to/file
        drive = path_str[0].lower()
        rest_path = path_str[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest_path}"
    
    # Convert WSL paths to Windows format if needed
    if path_str.startswith("/mnt/") and platform.system() == "Windows":
        parts = path_str.split("/")
        if len(parts) >= 3:
            drive = parts[2].upper()
            rest_path = "/".join(parts[3:]).replace("/", "\\")
            return f"{drive}:\\{rest_path}"
    
    return str(Path(path_str).resolve())

@app.route('/browse/file', methods=['POST'])
def browse_file():
    """Browse for a file using system file dialog or fallback"""
    data = request.json
    current_path = data.get('current_path', '')
    
    # Normalize the current path
    current_path = normalize_path(current_path)
    initial_dir = str(Path(current_path).parent) if current_path and Path(current_path).exists() else str(Path.cwd())
    
    if TK_AVAILABLE:
        try:
            # Create a hidden root window
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            # Open file dialog
            file_path = filedialog.askopenfilename(
                initialdir=initial_dir,
                title="Select File",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Checkpoint files", "*.ckpt"),
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ]
            )
            
            root.destroy()
            
            if file_path:
                return jsonify({"path": normalize_path(file_path)})
            else:
                return jsonify({"error": "No file selected"}), 400
                
        except Exception as e:
            return jsonify({"error": f"File dialog failed: {str(e)}"}), 500
    
    return jsonify({"error": "File dialog not available"}), 500

@app.route('/browse/directory', methods=['POST'])
def browse_directory():
    """Browse for a directory using system file dialog or fallback"""
    data = request.json
    current_path = data.get('current_path', '')
    
    # Normalize the current path
    current_path = normalize_path(current_path)
    initial_dir = current_path if current_path and Path(current_path).exists() else str(Path.cwd())
    
    if TK_AVAILABLE:
        try:
            # Create a hidden root window
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            # Open directory dialog
            dir_path = filedialog.askdirectory(
                initialdir=initial_dir,
                title="Select Directory"
            )
            
            root.destroy()
            
            if dir_path:
                return jsonify({"path": normalize_path(dir_path)})
            else:
                return jsonify({"error": "No directory selected"}), 400
                
        except Exception as e:
            return jsonify({"error": f"Directory dialog failed: {str(e)}"}), 500
    
    return jsonify({"error": "Directory dialog not available"}), 500



@app.route('/upload_checkpoint', methods=['POST'])
def upload_checkpoint():
    """Upload a checkpoint file and save it to the checkpoints directory"""
    try:
        if 'checkpoint' not in request.files:
            return jsonify({"error": "No checkpoint file provided"}), 400
        
        file = request.files['checkpoint']
        voice_name = request.form.get('voice_name', 'unknown').strip()
        
        if file.filename == '':
            return jsonify({"error": "No checkpoint file selected"}), 400
        
        if not file.filename.lower().endswith('.ckpt'):
            return jsonify({"error": "File must be a .ckpt checkpoint file"}), 400
        
        # Create safe filename with voice name prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_voice_name = secure_filename(voice_name) if voice_name != 'unknown' else 'unknown'
        safe_filename = secure_filename(file.filename)
        
        # Remove .ckpt extension from original name and add our naming convention
        original_name = safe_filename.replace('.ckpt', '')
        dest_filename = f"{safe_voice_name}-{original_name}-{timestamp}.ckpt"
        dest_path = CHECKPOINTS_DIR / dest_filename
        
        # Save the uploaded file
        file.save(dest_path)
        
        return jsonify({
            "path": str(dest_path),
            "message": f"Checkpoint uploaded successfully as {dest_filename}"
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to upload checkpoint: {str(e)}"}), 500

@app.route('/validate_path', methods=['POST'])
def validate_path():
    """Validate if a path exists and return information about it"""
    data = request.json
    path_str = data.get('path', '')
    
    if not path_str:
        return jsonify({"valid": False, "error": "Empty path"})
    
    try:
        # Normalize the path
        normalized_path = normalize_path(path_str)
        path_obj = Path(normalized_path)
        
        if path_obj.exists():
            return jsonify({
                "valid": True,
                "exists": True,
                "is_file": path_obj.is_file(),
                "is_directory": path_obj.is_dir(),
                "normalized_path": str(path_obj.resolve())
            })
        else:
            return jsonify({
                "valid": True,
                "exists": False,
                "normalized_path": str(path_obj.resolve())
            })
            
    except Exception as e:
        return jsonify({
            "valid": False,
            "error": f"Invalid path: {str(e)}"
        })

@app.route('/convert_audio', methods=['POST'])
def convert_audio():
    """Convert audio files to specified sample rate"""
    try:
        if not LIBROSA_AVAILABLE:
            return jsonify({"error": "Audio conversion requires librosa. Install with: pip install librosa"}), 400
        
        if 'audio_files' not in request.files:
            return jsonify({"error": "No audio files provided"}), 400
        
        audio_files = request.files.getlist('audio_files')
        target_sample_rate = int(request.form.get('target_sample_rate', 22050))
        
        if not audio_files:
            return jsonify({"error": "No audio files selected"}), 400
        
        # Create temporary conversion directory
        conversion_dir = Path("temp_audio_conversion")
        conversion_dir.mkdir(exist_ok=True)
        
        converted_files = []
        total_files = len([f for f in audio_files if f.filename])
        
        # Initialize progress tracking
        upload_progress.update({
            "is_uploading": True,
            "current_file": 0,
            "total_files": total_files,
            "current_filename": "",
            "operation": "convert"
        })
        
        for i, audio_file in enumerate(audio_files, 1):
            if audio_file.filename == '':
                continue
            
            # Update progress
            upload_progress.update({
                "current_file": i,
                "current_filename": audio_file.filename
            })
                
            # Save uploaded file temporarily
            temp_input_path = conversion_dir / f"input_{audio_file.filename}"
            audio_file.save(temp_input_path)
            
            try:
                # Load audio with librosa
                audio_data, original_sr = librosa.load(temp_input_path, sr=None)
                
                # Resample if needed
                if original_sr != target_sample_rate:
                    audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sample_rate)
                
                # Generate output filename with conversion info
                name, ext = os.path.splitext(audio_file.filename)
                output_filename = f"{name}_converted_{target_sample_rate}Hz.wav"
                output_path = conversion_dir / output_filename
                
                # Save converted audio
                import soundfile as sf
                sf.write(output_path, audio_data, target_sample_rate)
                
                # Get file info
                file_stats = output_path.stat()
                duration = len(audio_data) / target_sample_rate
                
                converted_files.append({
                    "filename": output_filename,
                    "path": str(output_path),
                    "size": file_stats.st_size,
                    "sample_rate": target_sample_rate,
                    "duration": round(duration, 2),
                    "original_filename": audio_file.filename
                })
                
            except Exception as e:
                print(f"Failed to convert {audio_file.filename}: {e}")
                continue
            finally:
                # Clean up temporary input file
                if temp_input_path.exists():
                    temp_input_path.unlink()
        
        # Reset progress tracking
        upload_progress.update({
            "is_uploading": False,
            "current_file": 0,
            "total_files": 0,
            "current_filename": "",
            "operation": ""
        })
        
        if not converted_files:
            return jsonify({"error": "No audio files could be converted"}), 400
        
        return jsonify({
            "message": f"{len(converted_files)}/{total_files} audio files converted successfully",
            "converted_files": converted_files,
            "target_sample_rate": target_sample_rate
        })
        
    except Exception as e:
        return jsonify({"error": f"Audio conversion failed: {str(e)}"}), 500

@app.route('/move_converted_audio', methods=['POST'])
def move_converted_audio():
    """Move converted audio files to training data directory"""
    try:
        data = request.get_json()
        voice_name = data.get('voice_name', '').strip()
        converted_files = data.get('converted_files', [])
        
        if not voice_name:
            return jsonify({"error": "Voice name is required"}), 400
        
        if not converted_files:
            return jsonify({"error": "No converted files to move"}), 400
        
        # Sanitize voice name
        safe_voice_name = secure_filename(voice_name)
        if not safe_voice_name:
            return jsonify({"error": "Invalid voice name"}), 400
        
        # Create target directory
        target_dir = DATASETS_DIR / safe_voice_name / "audio"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear target directory first
        for existing_file in target_dir.glob("*"):
            if existing_file.is_file():
                existing_file.unlink()
        
        files_moved = 0
        for file_info in converted_files:
            source_path = Path(file_info['path'])
            if source_path.exists():
                # Extract original filename and rename to match CSV expectations
                original_filename = file_info['original_filename']
                name, ext = os.path.splitext(original_filename)
                proper_filename = f"{name}.wav"  # Use original name with .wav extension
                target_path = target_dir / proper_filename
                
                # Move and rename file
                import shutil
                shutil.move(str(source_path), str(target_path))
                files_moved += 1
        
        # Clean up conversion directory
        conversion_dir = Path("temp_audio_conversion")
        if conversion_dir.exists():
            import shutil
            shutil.rmtree(conversion_dir, ignore_errors=True)
        
        return jsonify({
            "message": f"Successfully moved {files_moved} audio files to training data",
            "files_moved": files_moved,
            "target_directory": str(target_dir)
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to move audio files: {str(e)}"}), 500

@app.route('/synthesize_text', methods=['POST'])
def synthesize_text():
    """Synthesize text using uploaded ONNX model"""
    try:
        # Check if training is running
        if training_status["is_training"]:
            return jsonify({"error": "Cannot synthesize while training is running (VRAM conflict)"}), 400
        
        if 'onnx_model' not in request.files or 'text' not in request.form:
            return jsonify({"error": "ONNX model file and text are required"}), 400
        
        onnx_file = request.files['onnx_model']
        text = request.form['text'].strip()
        
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        if onnx_file.filename == '':
            return jsonify({"error": "No ONNX model file selected"}), 400
        
        # Save ONNX model temporarily
        temp_dir = Path("temp_synthesis")
        temp_dir.mkdir(exist_ok=True)
        
        onnx_path = temp_dir / "model.onnx"
        onnx_file.save(onnx_path)
        
        # Check for JSON config file
        json_path = temp_dir / "model.onnx.json"
        
        # Try to get JSON file from request
        if 'json_config' in request.files and request.files['json_config'].filename != '':
            json_file = request.files['json_config']
            json_file.save(json_path)
            print(f"📄 Using uploaded JSON config: {json_file.filename}")
        else:
            # JSON file not uploaded, return error asking for it
            return jsonify({
                "error": "Missing JSON configuration file", 
                "details": "PiperTTS requires a JSON config file alongside the ONNX model. Please upload both files.",
                "missing_file": "model.onnx.json",
                "requires_json": True
            }), 400
        
        # Create output path for audio
        output_path = temp_dir / "output.wav"
        
        try:
            # Use piper for synthesis
            cmd = [
                "python", "-c", f"""
import sys
import time
sys.path.insert(0, 'src')
from piper import PiperVoice
import wave
import numpy as np

# Load model
voice = PiperVoice.load('{onnx_path}')

# Synthesize - this returns an iterable of AudioChunk objects
audio_chunks = voice.synthesize('{text}')

# Save as WAV using the built-in synthesize_wav method
with wave.open('{output_path}', 'wb') as wav_file:
    voice.synthesize_wav('{text}', wav_file)
"""
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=Path.cwd(),
                timeout=30
            )
            
            if result.returncode != 0:
                return jsonify({"error": f"Synthesis failed: {result.stderr}"}), 500
            
            if not output_path.exists():
                return jsonify({"error": "Audio file was not generated"}), 500
            
            # Return the audio file
            def cleanup():
                try:
                    if onnx_path.exists():
                        onnx_path.unlink()
                    if output_path.exists():
                        output_path.unlink()
                    if temp_dir.exists():
                        temp_dir.rmdir()
                except:
                    pass
            
            # Schedule cleanup after response
            import atexit
            atexit.register(cleanup)
            
            return send_file(
                output_path,
                as_attachment=True,
                download_name=f"synthesized_{int(time.time())}.wav",
                mimetype="audio/wav"
            )
            
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Synthesis timed out"}), 500
        except Exception as e:
            return jsonify({"error": f"Synthesis error: {str(e)}"}), 500
        finally:
            # Clean up temporary files
            try:
                if onnx_path.exists():
                    onnx_path.unlink()
                if output_path.exists() and not output_path.is_file():  # Don't delete if we're sending it
                    output_path.unlink()
            except:
                pass
        
    except Exception as e:
        return jsonify({"error": f"Synthesis failed: {str(e)}"}), 500

@app.route('/llm_models', methods=['GET'])
def list_llm_models():
    """List available LLM models in the LLM directory"""
    try:
        models = []
        if LLM_MODELS_DIR.exists():
            for model_file in LLM_MODELS_DIR.glob("*"):
                if model_file.is_file():
                    models.append({
                        "name": model_file.name,
                        "path": str(model_file),
                        "size_mb": round(model_file.stat().st_size / (1024 * 1024), 1),
                        "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                    })
        
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": f"Failed to list LLM models: {str(e)}"}), 500

@app.route('/upload_llm_model', methods=['POST'])
def upload_llm_model():
    """Upload an LLM model file"""
    try:
        if 'llm_model' not in request.files:
            return jsonify({"error": "No LLM model file provided"}), 400
        
        file = request.files['llm_model']
        if file.filename == '':
            return jsonify({"error": "No LLM model file selected"}), 400
        
        # Create safe filename
        safe_filename = secure_filename(file.filename)
        dest_path = LLM_MODELS_DIR / safe_filename
        
        # Save the file
        file.save(dest_path)
        
        return jsonify({
            "message": f"LLM model uploaded successfully: {safe_filename}",
            "model_path": str(dest_path)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to upload LLM model: {str(e)}"}), 500

@app.route('/download_llm_model', methods=['POST'])
def download_llm_model():
    """Download an LLM model from URL"""
    try:
        data = request.json
        url = data.get('url', '').strip()
        model_name = data.get('model_name', '').strip()
        
        if not url:
            return jsonify({"error": "URL is required"}), 400
        
        if not model_name:
            # Extract filename from URL
            model_name = url.split('/')[-1]
        
        # Create safe filename
        safe_filename = secure_filename(model_name)
        dest_path = LLM_MODELS_DIR / safe_filename
        
        # Download the file
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return jsonify({
            "message": f"LLM model downloaded successfully: {safe_filename}",
            "model_path": str(dest_path)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to download LLM model: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat_with_llm():
    """Chat with the loaded LLM model"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        user_message = data.get('message', '').strip()
        llm_model_path = data.get('llm_model_path', '').strip()
        voice_model_path = data.get('voice_model_path', '').strip()
        voice_json_path = data.get('voice_json_path', '').strip()
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        if not llm_model_path:
            return jsonify({"error": "LLM model path is required"}), 400
        
        # Initialize chat session if it doesn't exist
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                "messages": [],
                "llm_model_path": llm_model_path,
                "voice_model_path": voice_model_path,
                "voice_json_path": voice_json_path
            }
        
        # Add user message to chat history
        chat_sessions[session_id]["messages"].append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate AI response using loaded LLM model
        ai_response = ""
        if loaded_models["llm"] and loaded_models["llm"].get("model"):
            try:
                # Use the loaded LLM model for inference
                llm_model = loaded_models["llm"]["model"]
                prompt = f"User: {user_message}\nAssistant:"
                
                response = llm_model(prompt, max_tokens=150, stop=["\n", "User:", "Human:"])
                ai_response = response.get('choices', [{}])[0].get('text', '').strip()
                
                print(f"LLM Response: {ai_response}")  # Debug log
                
                if not ai_response:
                    ai_response = f"I understand you said: '{user_message}'. This is a response from the loaded LLM model."
            except Exception as llm_error:
                print(f"LLM inference failed: {llm_error}")
                ai_response = f"I understand you said: '{user_message}'. (LLM inference failed: {str(llm_error)})"
        elif loaded_models["llm"] and loaded_models["llm"].get("note"):
            # Model was validated but not loaded (llama-cpp-python not available)
            ai_response = f"I understand you said: '{user_message}'. (LLM model validated but not loaded - llama-cpp-python not available)"
        else:
            ai_response = f"This is a placeholder response. You said: '{user_message}'. (LLM model not properly loaded)"
        
        # Add AI response to chat history
        chat_sessions[session_id]["messages"].append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate speech if voice model is loaded
        audio_url = None
        if loaded_models["voice"] and loaded_models["voice"].get("model"):
            try:
                # Create temporary audio file
                temp_audio_path = Path.cwd() / f"temp_chat_audio_{int(time.time())}.wav"
                
                # Use the loaded voice model for synthesis
                voice_model = loaded_models["voice"]["model"]
                
                # Debug: Check available methods
                print(f"Voice model type: {type(voice_model)}")
                print(f"Available methods: {[m for m in dir(voice_model) if not m.startswith('_')]}")
                
                # Try different synthesis methods
                try:
                    # Method 1: Try synthesize_wav with proper path handling
                    voice_model.synthesize_wav(ai_response, str(temp_audio_path))
                    print(f"Speech synthesis successful using synthesize_wav: {temp_audio_path}")
                except Exception as wav_error:
                    print(f"synthesize_wav failed: {wav_error}")
                    try:
                        # Method 2: Try synthesize with proper AudioChunk handling
                        audio_data = voice_model.synthesize(ai_response)
                        
                        # Convert to numpy array and save as WAV
                        import numpy as np
                        import soundfile as sf
                        
                        # Handle different AudioChunk types
                        audio_list = []
                        for chunk in audio_data:
                            if hasattr(chunk, 'audio_float_array'):
                                # AudioChunk with audio_float_array attribute (most common)
                                audio_list.extend(chunk.audio_float_array)
                            elif hasattr(chunk, 'audio_int16_array'):
                                # AudioChunk with audio_int16_array attribute
                                audio_list.extend(chunk.audio_int16_array)
                            elif hasattr(chunk, 'audio'):
                                # AudioChunk object with .audio attribute
                                audio_list.extend(chunk.audio)
                            elif hasattr(chunk, 'samples'):
                                # AudioChunk with samples attribute
                                audio_list.extend(chunk.samples)
                            elif hasattr(chunk, 'data'):
                                # AudioChunk with data attribute
                                audio_list.extend(chunk.data)
                            elif hasattr(chunk, 'get_audio'):
                                # AudioChunk with get_audio method
                                audio_list.extend(chunk.get_audio())
                            elif hasattr(chunk, '__iter__') and not isinstance(chunk, (str, bytes)):
                                # Iterable but not string/bytes
                                audio_list.extend(chunk)
                            elif isinstance(chunk, (int, float)):
                                # Direct numeric value
                                audio_list.append(chunk)
                            else:
                                # Try to convert to int
                                try:
                                    audio_list.append(int(chunk))
                                except:
                                    print(f"Warning: Skipping unhandled chunk type: {type(chunk)}")
                                    # Try to inspect the chunk more deeply
                                    print(f"Chunk attributes: {[attr for attr in dir(chunk) if not attr.startswith('_')]}")
                                    if hasattr(chunk, '__dict__'):
                                        print(f"Chunk dict: {chunk.__dict__}")
                        
                        if audio_list:
                            # Convert to numpy array and handle float32 to int16 conversion
                            audio_array = np.array(audio_list)
                            
                            # If it's float32, convert to int16
                            if audio_array.dtype == np.float32:
                                # Normalize to [-1, 1] range and convert to int16
                                audio_array = np.clip(audio_array, -1.0, 1.0)
                                audio_array = (audio_array * 32767).astype(np.int16)
                            elif audio_array.dtype != np.int16:
                                audio_array = audio_array.astype(np.int16)
                            
                            sf.write(str(temp_audio_path), audio_array, 22050)
                            print(f"Speech synthesis successful using manual AudioChunk handling: {temp_audio_path}")
                        else:
                            raise Exception("No audio data extracted from chunks")
                            
                    except Exception as manual_error:
                        print(f"Manual AudioChunk handling failed: {manual_error}")
                        # Method 3: Try using synthesize_to_file if available
                        try:
                            if hasattr(voice_model, 'synthesize_to_file'):
                                voice_model.synthesize_to_file(ai_response, str(temp_audio_path))
                                print(f"Speech synthesis successful using synthesize_to_file: {temp_audio_path}")
                            else:
                                raise Exception("synthesize_to_file not available")
                        except Exception as file_error:
                            print(f"synthesize_to_file failed: {file_error}")
                            # Method 4: Try phoneme-based synthesis
                            try:
                                # Convert text to phonemes first, then to audio
                                phonemes = voice_model.phonemize(ai_response)
                                phoneme_ids = voice_model.phonemes_to_ids(phonemes)
                                audio_data = voice_model.phoneme_ids_to_audio(phoneme_ids)
                                
                                # Handle the audio data
                                import numpy as np
                                import soundfile as sf
                                
                                audio_list = []
                                for chunk in audio_data:
                                    if hasattr(chunk, 'audio'):
                                        audio_list.extend(chunk.audio)
                                    elif hasattr(chunk, 'samples'):
                                        audio_list.extend(chunk.samples)
                                    elif hasattr(chunk, 'data'):
                                        audio_list.extend(chunk.data)
                                    elif hasattr(chunk, '__iter__') and not isinstance(chunk, (str, bytes)):
                                        audio_list.extend(chunk)
                                    elif isinstance(chunk, (int, float)):
                                        audio_list.append(chunk)
                                    else:
                                        try:
                                            audio_list.append(int(chunk))
                                        except:
                                            print(f"Warning: Skipping unhandled chunk type: {type(chunk)}")
                                
                                if audio_list:
                                    # Convert to numpy array and handle float32 to int16 conversion
                                    audio_array = np.array(audio_list)
                                    
                                    # If it's float32, convert to int16
                                    if audio_array.dtype == np.float32:
                                        # Normalize to [-1, 1] range and convert to int16
                                        audio_array = np.clip(audio_array, -1.0, 1.0)
                                        audio_array = (audio_array * 32767).astype(np.int16)
                                    elif audio_array.dtype != np.int16:
                                        audio_array = audio_array.astype(np.int16)
                                    
                                    sf.write(str(temp_audio_path), audio_array, 22050)
                                    print(f"Speech synthesis successful using phoneme-based method: {temp_audio_path}")
                                else:
                                    raise Exception("No audio data extracted from phoneme-based synthesis")
                                    
                            except Exception as phoneme_error:
                                print(f"Phoneme-based synthesis failed: {phoneme_error}")
                                raise Exception(f"All speech synthesis methods failed: {wav_error}, {manual_error}, {file_error}, {phoneme_error}")
                
                if temp_audio_path.exists():
                    audio_url = f"/download/{temp_audio_path.name}"
                    print(f"Speech synthesis successful: {temp_audio_path}")
                else:
                    print(f"Speech synthesis failed: file not created")
                    
            except Exception as e:
                print(f"Speech synthesis failed: {e}")
                # Fallback to subprocess method
                try:
                    temp_audio_path = Path.cwd() / f"temp_chat_audio_fallback_{int(time.time())}.wav"
                    synthesis_result = subprocess.run([
                        "python3", "-c", f"""
import sys
sys.path.append('/mnt/c/cursor/piper1-gpl/src')
from piper.voice import PiperVoice
import soundfile as sf
import numpy as np

voice = PiperVoice.load('{voice_model_path}')
audio_data = voice.synthesize('{ai_response}')
audio_array = np.array(list(audio_data), dtype=np.int16)
sf.write('{temp_audio_path}', audio_array, 22050)
"""], capture_output=True, text=True, cwd=Path.cwd())
                    
                    if temp_audio_path.exists():
                        audio_url = f"/download/{temp_audio_path.name}"
                except Exception as fallback_error:
                    print(f"Fallback speech synthesis also failed: {fallback_error}")
        else:
            print("Voice model not properly loaded for speech synthesis")
        
        return jsonify({
            "response": ai_response,
            "audio_url": audio_url,
            "session_id": session_id,
            "message_count": len(chat_sessions[session_id]["messages"])
        })
    except Exception as e:
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500

@app.route('/chat_history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    """Get chat history for a session"""
    try:
        if session_id in chat_sessions:
            return jsonify({
                "messages": chat_sessions[session_id]["messages"],
                "session_id": session_id
            })
        else:
            return jsonify({"messages": [], "session_id": session_id})
    except Exception as e:
        return jsonify({"error": f"Failed to get chat history: {str(e)}"}), 500

@app.route('/clear_chat/<session_id>', methods=['POST'])
def clear_chat(session_id):
    """Clear chat history for a session"""
    try:
        if session_id in chat_sessions:
            chat_sessions[session_id]["messages"] = []
        return jsonify({"message": "Chat history cleared"})
    except Exception as e:
        return jsonify({"error": f"Failed to clear chat: {str(e)}"}), 500

@app.route('/load_llm_model', methods=['POST'])
def load_llm_model():
    """Load an LLM model into memory"""
    try:
        data = request.json
        model_path = data.get('model_path', '').strip()
        
        if not model_path:
            return jsonify({"error": "Model path is required"}), 400
        
        if not Path(model_path).exists():
            return jsonify({"error": f"Model file not found: {model_path}"}), 400
        
        # Try to load the LLM model using llama-cpp-python or similar
        try:
            from llama_cpp import Llama
            
            # Load the model with appropriate parameters
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,  # Context window
                n_threads=4,  # Number of CPU threads
                n_gpu_layers=0  # Use CPU for now, can be adjusted for GPU
            )
            
            # Test the model with a simple prompt
            test_response = llm("Hello", max_tokens=10, stop=["\n"])
            
            loaded_models["llm"] = {
                "path": model_path,
                "loaded_at": datetime.now().isoformat(),
                "size": Path(model_path).stat().st_size,
                "model": llm,
                "test_response": test_response
            }
            
            return jsonify({
                "message": "LLM model loaded and tested successfully",
                "model_info": {
                    "path": model_path,
                    "loaded_at": loaded_models["llm"]["loaded_at"],
                    "size": loaded_models["llm"]["size"],
                    "test_response": str(test_response)
                }
            })
            
        except ImportError:
            # Fallback: just validate the file exists
            loaded_models["llm"] = {
                "path": model_path,
                "loaded_at": datetime.now().isoformat(),
                "size": Path(model_path).stat().st_size,
                "model": None,
                "note": "llama-cpp-python not available, model not loaded into memory"
            }
            
            return jsonify({
                "message": "LLM model file validated (llama-cpp-python not available for inference)",
                "model_info": loaded_models["llm"]
            })
        
    except Exception as e:
        return jsonify({"error": f"Failed to load LLM model: {str(e)}"}), 500

@app.route('/load_voice_model', methods=['POST'])
def load_voice_model():
    """Load a voice model into memory"""
    try:
        data = request.json
        model_path = data.get('model_path', '').strip()
        json_path = data.get('json_path', '').strip()
        
        if not model_path or not json_path:
            return jsonify({"error": "Both model path and JSON path are required"}), 400
        
        if not Path(model_path).exists():
            return jsonify({"error": f"Model file not found: {model_path}"}), 400
        
        if not Path(json_path).exists():
            return jsonify({"error": f"JSON config file not found: {json_path}"}), 400
        
        # Load the actual PiperTTS voice model
        try:
            import sys
            sys.path.append('/mnt/c/cursor/piper1-gpl/src')
            from piper.voice import PiperVoice
            
            # Load the voice model
            voice = PiperVoice.load(model_path)
            
            # Test the voice model with a simple synthesis
            test_audio = list(voice.synthesize("Test"))
            
            loaded_models["voice"] = {
                "model_path": model_path,
                "json_path": json_path,
                "loaded_at": datetime.now().isoformat(),
                "model_size": Path(model_path).stat().st_size,
                "json_size": Path(json_path).stat().st_size,
                "model": voice,
                "test_audio": test_audio
            }
            
            return jsonify({
                "message": "Voice model loaded and tested successfully",
                "model_info": {
                    "model_path": model_path,
                    "json_path": json_path,
                    "loaded_at": loaded_models["voice"]["loaded_at"],
                    "model_size": loaded_models["voice"]["model_size"],
                    "json_size": loaded_models["voice"]["json_size"],
                    "test_audio_length": len(test_audio)
                }
            })
            
        except Exception as voice_error:
            # Fallback: just validate the files exist
            loaded_models["voice"] = {
                "model_path": model_path,
                "json_path": json_path,
                "loaded_at": datetime.now().isoformat(),
                "model_size": Path(model_path).stat().st_size,
                "json_size": Path(json_path).stat().st_size,
                "model": None,
                "note": f"Voice model validation failed: {str(voice_error)}"
            }
            
            return jsonify({
                "message": "Voice model files validated (loading failed)",
                "model_info": loaded_models["voice"]
            })
        
    except Exception as e:
        return jsonify({"error": f"Failed to load voice model: {str(e)}"}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle request entity too large errors"""
    return jsonify({"error": "Upload too large. Please try uploading fewer files or use a smaller batch size."}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({"error": "Internal server error. Please try again with a smaller batch."}), 500

if __name__ == '__main__':
    print("Starting PiperTTS Training Web Interface...")
    print(f"Datasets directory: {DATASETS_DIR.absolute()}")
    print(f"Checkpoints directory: {CHECKPOINTS_DIR.absolute()}")
    print(f"Logs directory: {LOGS_DIR.absolute()}")
    print("Session-based training structure enabled!")
    print(f"Max upload size: {app.config['MAX_CONTENT_LENGTH'] / (1024**3):.1f}GB")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
