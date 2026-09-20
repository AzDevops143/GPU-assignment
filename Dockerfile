# Official NVIDIA CUDA Development Image with Ubuntu 22.04
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

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

# Compile the CUDA executable
# -arch=all compiles for supported NVIDIA architectures (sm_50 to sm_90)
RUN nvcc -O3 -arch=all heat_diffusion.cu -o heat_diffusion

# Default command
CMD ["./heat_diffusion", "256", "1e-4", "2000000"]
