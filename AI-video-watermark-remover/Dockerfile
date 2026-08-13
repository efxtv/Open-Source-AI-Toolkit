FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=7861
ENV GRADIO_SERVER_PORT=7861

# System dependencies required by OpenCV, FFmpeg and LaMa
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Application directory
RUN mkdir -p /root/WaterMarkRemover

WORKDIR /root/WaterMarkRemover

# Download application
RUN wget -O app.py \
    "https://raw.githubusercontent.com/efxtv/Open-Source-AI-Toolkit/refs/heads/main/AI-video-watermark-remover/app.py"

# Download requirements
RUN wget -O requirements.txt \
    "https://raw.githubusercontent.com/efxtv/Open-Source-AI-Toolkit/refs/heads/main/AI-video-watermark-remover/requirements.txt"

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies during image build
RUN pip install --no-cache-dir -r requirements.txt

# Gradio port
EXPOSE 7861

# Start application
CMD ["python", "app.py"]
