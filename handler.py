import os
import sys
import tempfile
import torch
import base64
import runpod
import requests  # Added for downloading reference audio
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import BitsAndBytesConfig

# --- AUTHENTICATION ---
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- MOCKING SPACES ---
from types import ModuleType
mock_spaces = ModuleType("spaces")
mock_spaces.GPU = lambda func=None, **kwargs: (lambda f: f) if func is None else func
sys.modules["spaces"] = mock_spaces

from heartlib import HeartMuLaGenPipeline

pipe = None

def download_models():
    cache_dir = os.environ.get("HF_HOME", "/tmp")
    model_dir = os.path.join(cache_dir, "heartmula_models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    for filename in ["tokenizer.json", "gen_config.json"]:
        hf_hub_download(repo_id="HeartMuLa/HeartMuLaGen", filename=filename, local_dir=model_dir, token=HF_TOKEN)

    snapshot_download(repo_id="HeartMuLa/HeartMuLa-oss-3B-happy-new-year", local_dir=os.path.join(model_dir, "HeartMuLa-oss-3B"), token=HF_TOKEN)
    snapshot_download(repo_id="HeartMuLa/HeartCodec-oss-20260123", local_dir=os.path.join(model_dir, "HeartCodec-oss"), token=HF_TOKEN)
    return model_dir

def init_pipeline():
    global pipe
    model_dir = download_models()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = HeartMuLaGenPipeline.from_pretrained(model_dir, device=device, dtype=dtype, version="3B")
    print("✅ Pipeline Initialized")

def download_temp_audio(url):
    """Downloads an audio file from a URL to a temporary local file."""
    suffix = os.path.splitext(url)[1] or ".mp3"
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                temp_file.write(chunk)
        return temp_file.name
    except Exception as e:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        raise Exception(f"Failed to download reference audio: {str(e)}")

def handler(job):
    job_input = job["input"]
    
    lyrics = job_input.get("lyrics", "")
    tags = job_input.get("tags", "afrobeats, energetic")
    max_duration_seconds = job_input.get("max_duration_seconds", 30)
    temperature = job_input.get("temperature", 1.0)
    topk = job_input.get("topk", 5)
    cfg_scale = job_input.get("cfg_scale", 1.0)
    
    # New parameter for audio-to-audio
    ref_audio_url = job_input.get("ref_audio_url")

    output_path = ""
    ref_path = None
    try:
        # Handle the reference audio download if URL exists
        model_input = {"lyrics": lyrics, "tags": tags}
        if ref_audio_url:
            ref_path = download_temp_audio(ref_audio_url)
            model_input["ref_audio_path"] = ref_path

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            output_path = f.name

        max_audio_length_ms = max_duration_seconds * 1000

        with torch.no_grad():
            pipe(
                model_input, # Injected ref_audio_path here if provided
                max_audio_length_ms=max_audio_length_ms,
                save_path=output_path,
                topk=topk,
                temperature=temperature,
                cfg_scale=cfg_scale,
                lazy_load=True,
            )

        with open(output_path, "rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
        
        return {"audio_base64": encoded_string}

    except Exception as e:
        return {"error": str(e)}
    finally:
        # Cleanup both the output and the reference file
        for path in [output_path, ref_path]:
            if path and os.path.exists(path):
                os.remove(path)

init_pipeline()
runpod.serverless.start({"handler": handler})