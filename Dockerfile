# Use an official NVIDIA CUDA image with Python support
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV RUNPOD_HANDLER_ENTRYPOINT=runpod_handler

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    ninja-build \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /workspace

# Clone your repo (if not already mounted during build)
# Alternatively, comment this out if Dockerfile is in the repo already
# RUN git clone https://github.com/Royza/piper1-gpl.git /workspace

# Install training dependencies
COPY . /workspace
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -e .[train]

# Build Cython extension
RUN chmod +x ./build_monotonic_align.sh && ./build_monotonic_align.sh
RUN python3 setup.py build_ext --inplace

# Default RunPod handler entrypoint
CMD ["python3", "-m", "runpod"]
