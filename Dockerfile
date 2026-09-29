# Now here we took  CUDA 12.8+ includes native Blackwell architecture and Nsight tools support for problem 1 implementation
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# I Install dev tools, Python, and Nsight Systems CLI
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
    nsight-systems-cli \
    && rm -rf /var/lib/apt/lists/*

# I Install Python packages for Jupyter and data analysis
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    pandas \
    jupyterlab \
    notebook \
    nbconvert

WORKDIR /workspace

# Here Copy project files
COPY heat_diffusion.cu /workspace/
COPY heat_diffusion_cuda.ipynb /workspace/
COPY Makefile /workspace/

# Now Compile with line-info (-lineinfo) for Nsight source correlation,
# and include Blackwell targets (sm_100, sm_120) alongside Hopper and Ada
RUN nvcc -O3 -lineinfo -std=c++17 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_100,code=sm_100 \
    -gencode arch=compute_120,code=sm_120 \
    -gencode arch=compute_100,code=compute_100 \
    heat_diffusion.cu -o heat_diffusion

CMD ["./heat_diffusion", "256", "1e-4", "2000000"]
