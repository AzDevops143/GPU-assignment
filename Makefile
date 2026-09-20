# CUDA Compiler and Flags
NVCC := nvcc
NVCC_FLAGS := -O3 -arch=all

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
