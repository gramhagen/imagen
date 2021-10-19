#!/bin/bash

# Install CUDA drivers
sudo apt-get update
sudo apt-get install -y build-essential linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install -y cuda

# Setup Docker with NVIDIA drivers
curl https://get.docker.com | sh
sudo systemctl --now enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Build Docker image
mkdir /app
cd /app
git clone https://github.com/gramhagen/imagen
cd imagen
sudo docker build -t imagen .

# Start Docker image
sudo docker run --gpus all -v /app/imagen/src:/app -p 8501:8501 -d --restart unless-stopped -t imagen
