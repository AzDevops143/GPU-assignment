NVCC := nvcc
NVCC_FLAGS := -O3 -std=c++17 -lineinfo \
    -gencode arch=compute_100,code=sm_100

TARGET := JOR
SRC := JOR.cu

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
