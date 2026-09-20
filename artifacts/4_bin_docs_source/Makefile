# CUDA Compiler and Multi-Architecture Flags for Modern NVIDIA GPUs
NVCC := nvcc
NVCC_FLAGS := -O3 -std=c++17 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_90,code=compute_90

# Target binary name
TARGET := heat_diffusion
SRC := heat_diffusion.cu

# Docker configuration
IMAGE_NAME := gpu-assignment
TAG := latest

.PHONY: all clean run docker-build docker-run

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET) 256 1e-4 2000000

docker-build:
	docker build -t $(IMAGE_NAME):$(TAG) .

docker-run:
	docker run --gpus all --rm -it $(IMAGE_NAME):$(TAG)

clean:
	rm -f $(TARGET) *.csv
