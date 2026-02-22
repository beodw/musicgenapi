import runpod
import torch
import base64
import os
import tempfile
from heartlib import HeartMuLaGenPipeline

# 1. Initialize Pipeline Globally (Efficient: stays in VRAM between requests)
HF_TOKEN = os.environ.get("HF_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Load the model directly from Hugging Face Repo
print("🚀 Loading HeartMuLa 3B into VRAM...")
pipe = HeartMuLaGenPipeline.from_pretrained(
    "HeartMuLa/HeartMuLa-oss-3B-happy-new-year",
    device=device,
    dtype=dtype,
    version="3B",
    token=HF_TOKEN
)

def handler(job):
    """
    Takes 'input' from your Next.js fetch call.
    Returns the audio file as a Base64 string.
    """
    job_input = job["input"]
    
    # Extract params (Mapping to your Next.js payload keys)
    prompt = job_input.get("prompt")
    duration = job_input.get("duration", 8)
    genre = job_input.get("genre", "afrobeats")
    
    # Validation
    if not prompt:
        return {"error": "Missing musicPrompt/lyrics"}

    # Temporary file for the audio output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_path = f.name

    try:
        with torch.no_grad():
            pipe(
                {"lyrics": prompt, "tags": genre},
                max_audio_length_ms=duration * 1000,
                save_path=output_path,
                topk=50,
                temperature=1.0,
                cfg_scale=1.5,
                lazy_load=True
            )
        
        # Read the generated file and encode to Base64
        with open(output_path, "rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
        
        return encoded_string # Next.js will receive this in result.output

    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)

# Start the RunPod Serverless worker
runpod.serverless.start({"handler": handler})