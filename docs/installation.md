Installation Guide
This guide provides detailed instructions for setting up the Intelligent Mining Systems environment for autonomous operations research.
Table of Contents

System Requirements
Environment Setup
MMSegmentation Installation
Verification
Troubleshooting
Docker Installation

System Requirements
Hardware Requirements
Minimum Requirements:

CPU: Intel i5 or AMD Ryzen 5 (4+ cores)
RAM: 16 GB
Storage: 50 GB free space
GPU: NVIDIA GPU with 6+ GB VRAM (optional but recommended)

Recommended for Full Experiments:

CPU: Intel i7/i9 or AMD Ryzen 7/9 (8+ cores)
RAM: 32 GB or more
Storage: 100+ GB SSD
GPU: NVIDIA RTX 3080/A100 with 12+ GB VRAM

Software Requirements

Operating System: Linux (Ubuntu 18.04+), macOS 10.15+, or Windows 10+
Python: 3.8, 3.9, or 3.10
CUDA: 11.0+ (for GPU support)
Git: Latest version
Conda/Miniconda: Latest version

Environment Setup
Option 1: Conda Environment (Recommended)

Install Miniconda
# Linux/macOS
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc


Clone Repository
git clone https://github.com/ClaudioUrrea/segmentation.git
cd segmentation


Create Environment
conda env create -f environment.yml
conda activate intelligent-mining


Verify Installation
python --version  # Should show Python 3.8.x
which python      # Should point to conda environment



Option 2: Virtual Environment

Create Virtual Environment
python3 -m venv intelligent-mining
source intelligent-mining/bin/activate  # Linux/macOS


Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt



MMSegmentation Installation

Install MMSegmentation
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install mmsegmentation


Development Installation
cd mmsegmentation
pip install -v -e .



Verification
import torch
import torchvision
import mmcv
import mmseg

print(f"PyTorch version: {torch.__version__}")
print(f"MMCV version: {mmcv.__version__}")
print(f"MMSeg version: {mmseg.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

Troubleshooting
CUDA Version Mismatch
nvidia-smi
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

MMSegmentation Import Error
pip uninstall mmsegmentation
pip install mmsegmentation
cd mmsegmentation
pip install -v -e .

Docker Installation
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel
WORKDIR /workspace
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 git
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
RUN pip install mmsegmentation
COPY . .
RUN pip install -e .
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]

Build and run:
docker build -t intelligent-mining .
docker run --gpus all -p 8888:8888 -v $(pwd):/workspace intelligent-mining

Next Steps

Dataset Setup: Follow dataset_preparation.md
Training: See training_guide.md
Evaluation: Check evaluation_guide.md



\*\*Installation Complete!\*\*



You're now ready to work with intelligent systems for autonomous mining operations.







