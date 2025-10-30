# Dockerfile for OpenPI with NVIDIA CUDA support

# Use NVIDIA CUDA base image with CUDA 12.8 and cuDNN
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    git-lfs \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Set environment variables for optimal CUDA usage
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set uv environment variables
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv
ENV GIT_LFS_SKIP_SMUDGE=1

# Set Python path (will be available after .venv is created via mount)
ENV PYTHONPATH=/workspace:/workspace/packages/openpi-client/src
ENV PATH="/workspace/.venv/bin:$PATH"

# Default command
CMD ["/bin/bash"]
