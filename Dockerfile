# Use NVIDIA PyTorch as base image
FROM nvcr.io/nvidia/pytorch:21.09-py3

ENV DEBIAN_FRONTEND noninteractive
ENV LANG utf-8
ENV LANG_ALL utf-8

RUN apt-get update && \
    apt-get install -y ffmpeg

# Copy and install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Start Streamlit
WORKDIR /app

CMD ["streamlit", "run", "--server.headless=true", "--server.address=0.0.0.0", "imagen.py"]
