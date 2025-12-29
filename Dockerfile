FROM python:3.9-slim

WORKDIR /usr/src/app

# Install system dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model directory
RUN mkdir -p /usr/src/app/model

# Environment variables for GGUF model
ENV GGUF_MODEL_PATH=/usr/src/app/model/qwen2.5-1.5b-tour-assistant-q4.gguf
ENV N_CTX=2048
ENV N_THREADS=4
ENV N_GPU_LAYERS=0

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
