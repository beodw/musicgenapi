# Use RunPod's optimized PyTorch base
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# 1. Install SYSTEM tools first (fixes the Git warning and compilation errors)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup workspace
WORKDIR /

# 3. Install Python requirements
# We upgrade pip to ensure it can handle modern wheel metadata
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy handler code
COPY handler.py .

# 5. Run the handler
CMD [ "python", "-u", "/handler.py" ]