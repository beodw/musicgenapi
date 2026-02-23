import os
import sys
import tempfile
import torch
import base64
import runpod
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import BitsAndBytesConfig

# --- AUTHENTICATION ---
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- MOCKING SPACES ---
# This ensures heartlib doesn't crash looking for the HuggingFace Spaces environment
from types import ModuleType
mock_spaces = ModuleType("spaces")
mock_spaces.GPU = lambda func=None, **kwargs: (lambda f: f) if func is None else func
sys.modules["spaces"] = mock_spaces

# Now we can safely import heartlib
from heartlib import HeartMuLaGenPipeline

# Global variable to store the model in memory
pipe = None

def download_models():
    """Download all required model files with authentication (Colab Logic)."""
    # Use /tmp for serverless as it is the most reliable writable directory
    cache_dir = os.environ.get("HF_HOME", "/tmp")
    model_dir = os.path.join(cache_dir, "heartmula_models")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # Download HeartMuLaGen
    for filename in ["tokenizer.json", "gen_config.json"]:
        hf_hub_download(
            repo_id="HeartMuLa/HeartMuLaGen",
            filename=filename,
            local_dir=model_dir,
            token=HF_TOKEN
        )

    # Download HeartMuLa-oss-3B
    snapshot_download(
        repo_id="HeartMuLa/HeartMuLa-oss-3B-happy-new-year",
        local_dir=os.path.join(model_dir, "HeartMuLa-oss-3B"),
        token=HF_TOKEN
    )

    # Download HeartCodec-oss
    snapshot_download(
        repo_id="HeartMuLa/HeartCodec-oss-20260123",
        local_dir=os.path.join(model_dir, "HeartCodec-oss"),
        token=HF_TOKEN
    )
    return model_dir

def init_pipeline():
    """Initializes the pipeline once when the container starts."""
    global pipe
    model_dir = download_models()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    # Load pipeline with exact Colab settings
    pipe = HeartMuLaGenPipeline.from_pretrained(
        model_dir,
        device=device,
        dtype=dtype,
        version="3B",
        # use_4bit=True, # Keeping this commented out as per your snippet
    )
    print("✅ Pipeline Initialized")

def handler(job):
    """The function RunPod calls for every request."""
    job_input = job["input"]
    
    # Extract inputs (with Colab examples as defaults)
    lyrics = job_input.get("lyrics", "[Intro]\nFreetown magic.")
    tags = job_input.get("tags", "afrobeats, energetic")
    max_duration_seconds = job_input.get("max_duration_seconds", 30)
    temperature = job_input.get("temperature", 1.0)
    topk = job_input.get("topk", 5)
    cfg_scale = job_input.get("cfg_scale", 1.0)

    output_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            output_path = f.name

        max_audio_length_ms = max_duration_seconds * 1000

        # Exact generation logic from your Colab
        with torch.no_grad():
            pipe(
                {"lyrics": lyrics, "tags": tags},
                max_audio_length_ms=max_audio_length_ms,
                save_path=output_path,
                topk=topk,
                temperature=temperature,
                cfg_scale=cfg_scale,
                lazy_load=True,
            )

        # Encode to Base64 for the API response
        with open(output_path, "rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
        
        return {"audio_base64": encoded_string}

    except Exception as e:
        return {"error": str(e)}
    finally:
        # Cleanup temp files to keep the container lean
        if output_path and os.path.exists(output_path):
            os.remove(output_path)

# Start the initialization
init_pipeline()

# Start the RunPod Serverless loop
runpod.serverless.start({"handler": handler})