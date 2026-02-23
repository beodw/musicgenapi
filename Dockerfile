# Use RunPod's optimized PyTorch base
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# 1. Install git (Required to clone the library)
RUN apt-get update && apt-get install -y git ffmpeg build-essential && rm -rf /var/lib/apt/lists/*

# 2. Clone the official HeartMuLa repository
# This is the actual 'heartlib' source code
RUN git clone https://github.com/HeartMuLa/heartlib.git /heartlib_src

# 3. Install the library in 'editable' mode or via its setup.py
# This registers 'heartlib' in your Python environment
WORKDIR /heartlib_src
RUN pip install -e .

# 4. Install your other specific requirements
WORKDIR /
COPY requirements.txt .
# Remove 'heartlib' from your requirements.txt file before building!
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your handler and run
COPY handler.py .
CMD [ "python", "-u", "/handler.py" ]