# RangeLDM Dockerfile
# Multi-stage build for optimized image size

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/RangeLDM:/workspace/RangeLDM/ldm:/workspace/RangeLDM/vae:$PYTHONPATH \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    RANGELDM_CACHE=/workspace/cache/rangeldm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    build-essential \
    cmake \
    ninja-build \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/RangeLDM/

# Install Python dependencies for VAE
WORKDIR /workspace/RangeLDM/vae
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir \
    pytorch-lightning==2.0.7 \
    einops==0.7.0 \
    omegaconf==2.3.0 \
    pillow==10.0.0 \
    opencv-python==4.8.0.76 \
    numpy==1.24.3 \
    scipy==1.11.2 \
    matplotlib==3.7.2 \
    scikit-learn==1.3.0 \
    tqdm==4.66.1 \
    wandb==0.15.8 \
    tensorboard==2.14.0 \
    lpips==0.1.4 \
    kornia==0.7.0

# Install Python dependencies for LDM
WORKDIR /workspace/RangeLDM/ldm
RUN pip install --no-cache-dir \
    accelerate==0.27.2 \
    diffusers==0.26.3 \
    transformers==4.38.1 \
    datasets==2.18.0 \
    huggingface-hub==0.21.4 \
    peft==0.9.0 \
    safetensors==0.4.2 \
    xformers==0.0.20 \
    bitsandbytes==0.42.0

# Install Python dependencies for metrics
WORKDIR /workspace/RangeLDM/metrics
RUN pip install --no-cache-dir \
    open3d==0.17.0 \
    pandas==2.0.3 \
    pyyaml==6.0.1 \
    pyemd==0.5.1 \
    seaborn==0.12.2

# Install additional utilities
RUN pip install --no-cache-dir \
    ipython==8.14.0 \
    jupyter==1.0.0 \
    ipywidgets==8.1.0 \
    black==23.7.0 \
    flake8==6.1.0 \
    pytest==7.4.0

# Set working directory to project root
WORKDIR /workspace/RangeLDM

# Create directories for data, outputs, checkpoints, cache, and datasets
RUN mkdir -p \
    /workspace/data \
    /workspace/outputs \
    /workspace/checkpoints \
    /workspace/cache \
    /workspace/cache/rangeldm \
    /workspace/cache/torch \
    /workspace/cache/huggingface \
    /workspace/datasets \
    /datasets \
    /datasets/SemanticKITTI \
    /datasets/KITTI360 \
    /datasets/nuscenes

# Set default command
CMD ["/bin/bash"]

# Expose Jupyter port (optional)
EXPOSE 8888

# Expose TensorBoard port (optional)
EXPOSE 6006
