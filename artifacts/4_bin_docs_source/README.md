# GPU-assignment: 2D Heat Diffusion with CUDA & Docker

Containerized CUDA application and automated CI/CD pipeline using **Docker** and **GitHub Actions**.

---

## 🚀 Overview

This repository implements the 2D Heat Diffusion 5-point Jacobi stencil in CUDA C++ comparing:
- **Global Memory Implementation**
- **Shared Memory Tiled Implementation**
- In-kernel convergence detection using atomic reductions without intermediate CPU-GPU grid copies.

The project is fully containerized using the latest official NVIDIA CUDA development base image (`nvidia/cuda:12.6.2-devel-ubuntu22.04`) and configured for automated continuous integration (CI) via GitHub Actions with multi-architecture compilation.

### ⚡ Supported NVIDIA GPU Architectures
The compiled binary and container image natively target all modern & latest NVIDIA GPUs:
| Architecture | Compute Capability | Example GPUs |
| :--- | :---: | :--- |
| **Hopper** | `sm_90` | **NVIDIA H100, H200, GH200 Grace Hopper** |
| **Ada Lovelace** | `sm_89` | **NVIDIA RTX 4090, RTX 4080, L40S, L4** |
| **Ampere** | `sm_80`, `sm_86` | **NVIDIA A100, A30, RTX 3090, RTX 3080** |
| **Turing** | `sm_75` | **NVIDIA Tesla T4, RTX 2080** |
| **Blackwell (PTX)** | `compute_90` | **NVIDIA B100, B200, GB200, RTX 50-series** |

---

## 🛠 Project Structure

```text
docker implement/
├── .github/
│   └── workflows/
│       └── docker-ci.yml           # GitHub Actions workflow for Docker build & test
├── Dockerfile                      # Multi-stage CUDA 12.4 Docker environment
├── docker-compose.yml              # Compose configuration with GPU device reservation
├── Makefile                        # Build targets for nvcc and Docker
├── heat_diffusion.cu               # Standalone CUDA C++ source code
├── heat_diffusion_cuda.ipynb       # Jupyter notebook with analysis & profiling
├── GPU_Programming_Problems.pdf    # Assignment reference problem set
├── .gitignore                      # Git ignore rules
└── README.md                       # Documentation
```

---

## ⚙️ How GitHub Actions Works

The included workflow [`.github/workflows/docker-ci.yml`](.github/workflows/docker-ci.yml) triggers on every `push` or `pull_request` to `main`:

1. **Automated Docker Build:** Builds the Docker image from `Dockerfile` with full CUDA 12.6 toolkit and latest NVIDIA GPU multi-architecture flags.
2. **CUDA Compilation Verification:** Runs `nvcc -O3` inside the container to ensure zero compilation or syntax errors.
3. **Complete Downloadable Pipeline Artifacts (Points 1 to 4):**
   Automatically packages all simulation, benchmark, and analytical outputs into a single downloadable ZIP (`gpu-assignment-complete-artifacts.zip`):
   - **Point 1 — High-Resolution PNG Visualizations (`1_plots/`):**
     - `execution_time.png`: Execution time vs. grid size (Global vs. Shared memory).
     - `speedup.png`: Shared memory speedup ratio curve.
     - `iterations.png`: Jacobi stencil iterations required for convergence.
     - `temperature_field.png`: 2D steady-state thermal distribution heatmap.
   - **Point 2 — Benchmark Spreadsheet (`2_excel/`):**
     - `heat_diffusion_results.xlsx`: Excel workbook with `raw_results`, `summary`, and `correctness` sheets.
   - **Point 3 — Full 2D Grid Temperature CSV Datasets (`3_csv_grids/`):**
     - `grid_global_*.csv` and `grid_shared_*.csv` for $N \in \{128, 256, 512, 1024\}$.
   - **Point 4 — Binaries, Documentation & Source (`4_bin_docs_source/`):**
     - `heat_diffusion_linux_x86_64`: Compiled Linux CUDA binary (Hopper/Ada/Ampere/Turing/Blackwell PTX).
     - `answers.pdf`: Complete 11-page pedagogical solution and architecture guide.
     - `GPU_Programming_Problems.pdf`: Original assignment problem set.
     - `heat_diffusion.cu`: Standalone CUDA C++ source code.
     - `heat_diffusion_cuda.ipynb`: Interactive Jupyter notebook.
     - `Dockerfile`, `Makefile`, `docker-compose.yml`, `README.md`.
4. **Container Registry Publishing:** Automatically logs in to **GitHub Container Registry (GHCR)** using `${{ secrets.GITHUB_TOKEN }}` and publishes the tagged image:
   ```bash
   docker pull ghcr.io/azdevops143/gpu-assignment:latest
   ```

> [!NOTE]
> Standard GitHub-hosted runners (`ubuntu-latest`) do not contain physical GPUs, but they **can fully compile, build, and verify** CUDA code in Docker. To run real kernel executions on GPU hardware directly inside GitHub Actions, connect a **Self-Hosted Runner** with an NVIDIA GPU and NVIDIA Container Toolkit.

---

## 📦 Local Usage with Docker

### 1. Build the Docker Image
```bash
docker build -t gpu-assignment:latest .
```
Or via Makefile:
```bash
make docker-build
```

### 2. Run with NVIDIA GPU Passthrough
Ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed on your host:
```bash
docker run --gpus all --rm -it gpu-assignment:latest
```
Or with Docker Compose:
```bash
docker compose up
```

---

## 📤 Pushing to Your GitHub Repository

To push this repository to `https://github.com/AzDevops143/GPU-assignment.git`:

```bash
# 1. Initialize git repository
git init -b main

# 2. Add all files
git add .

# 3. Commit
git commit -m "feat: setup Dockerfile, CUDA source, and GitHub Actions CI workflow"

# 4. Link remote repository
git remote add origin https://github.com/AzDevops143/GPU-assignment.git

# 5. Push to GitHub
git push -u origin main
```
Once pushed, click the **Actions** tab on your GitHub repository to watch the Docker container build and publish automatically!
