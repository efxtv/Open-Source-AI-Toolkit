# Use a lightweight Python base image
FROM python:3.10-slim

# 1. Install system dependencies (FFmpeg is required for Whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Pre-download the Whisper model (optional, but saves time later)
# This downloads the 'base' model during build so the container starts faster
RUN python -c "import whisper; whisper.load_model('base')"

# 6. Copy your application files
COPY app.py cli.py ./

# 7. Expose the Gradio port
EXPOSE 7860

# 8. Command to run the Gradio app
# We use 0.0.0.0 so it is accessible outside the container
CMD ["python", "app.py"]
