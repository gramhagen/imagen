# Use NVIDIA PyTorch as base image
FROM nvcr.io/nvidia/pytorch:21.09-py3

RUN apt-get update && \
    apt-get install -y ffmpeg

# Copy and install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Start Streamlit
WORKDIR /app

CMD ["streamlit", "run", "--server.headless=true", "imagen.py"]
