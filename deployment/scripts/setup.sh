#!/bin/bash

# Build Docker image
git clone https://github.com/gramhagen/imagen
cd imagen
docker build -t imagen .
cd src

# Start Docker image
sudo docker run --name imagen_st --gpus all -v `pwd`:/app -p 8501:8501 -d --restart unless-stopped -t imagen
