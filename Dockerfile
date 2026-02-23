FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Conclusive Fix: Install the tools that 'pip' uses to build packages from scratch
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Use a clean pip and install with verbose logging so we can see the 'Real' error if it fails
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --verbose --no-cache-dir -r requirements.txt

COPY handler.py .
CMD [ "python", "-u", "/handler.py" ]