FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

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
    cuda-nsight-systems-12-8 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    pandas \
    jupyterlab \
    notebook \
    nbconvert

WORKDIR /workspace

COPY JOR.cu /workspace/
COPY Makefile /workspace/
COPY heat_diffusion_JOR.ipynb /workspace/

RUN nvcc -O3 -lineinfo -std=c++17 \
    -gencode arch=compute_100,code=sm_100 \
    JOR.cu -o JOR && \
    ln -sf JOR heat_diffusion

CMD ["./JOR", "256", "1e-4", "2000000"]
