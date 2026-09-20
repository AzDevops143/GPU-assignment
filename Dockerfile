# Latest Official NVIDIA CUDA Development Image with Ubuntu 22.04
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install essential build tools, Python, and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install scientific Python packages + Jupyter
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    pandas \
    jupyterlab \
    notebook \
    nbconvert

# Set working directory inside the container
WORKDIR /workspace

# Copy application files
COPY heat_diffusion.cu /workspace/
COPY heat_diffusion_cuda.ipynb /workspace/
COPY Makefile /workspace/

# Compile targeting all latest & modern NVIDIA GPU architectures:
# - sm_75: Turing (Tesla T4, RTX 2080)
# - sm_80: Ampere (A100, A30)
# - sm_86: Ampere (RTX 3080/3090)
# - sm_89: Ada Lovelace (RTX 4090, L40S, L4)
# - sm_90: Hopper (H100, H200)
# - compute_90: Forward-compatible PTX for Blackwell (B100, B200, GB200, RTX 50-series)
RUN nvcc -O3 -std=c++17 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_90,code=compute_90 \
    heat_diffusion.cu -o heat_diffusion

# Default command
CMD ["./heat_diffusion", "256", "1e-4", "2000000"]
