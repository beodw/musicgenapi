import os
import sys
import tempfile
import torch
import base64
import runpod
import requests
import gc
import io
import soundfile as sf
from huggingface_hub import hf_hub_download, snapshot_download

# --- AUTHENTICATION & CONFIG ---
HF_TOKEN = os.environ.get("HF_TOKEN")
SUPABASE_REF_URL = os.environ.get("SUPABASE_REF_URL")

# --- MOCKING SPACES ---
from types import ModuleType
mock_spaces = ModuleType("spaces")
mock_spaces.GPU = lambda func=None, **kwargs: (lambda f: f) if func is None else func
sys.modules["spaces"] = mock_spaces

from heartlib import HeartMuLaGenPipeline

pipe = None

def download_models():
    """Download models explicitly if they do not exist."""
    cache_dir = os.environ.get("HF_HOME", "/tmp")
    model_dir = os.path.join(cache_dir, "heartmula_models")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        # Download Configs
        for filename in ["tokenizer.json", "gen_config.json"]:
            hf_hub_download(repo_id="HeartMuLa/HeartMuLaGen", filename=filename, local_dir=model_dir, token=HF_TOKEN)
        
        # Download 3B Model weights
        snapshot_download(repo_id="HeartMuLa/HeartMuLa-oss-3B-happy-new-year", local_dir=os.path.join(model_dir, "HeartMuLa-oss-3B"), token=HF_TOKEN)
        
        # Download Codec
        snapshot_download(repo_id="HeartMuLa/HeartCodec-oss-20260123", local_dir=os.path.join(model_dir, "HeartCodec-oss"), token=HF_TOKEN)
    
    return model_dir

def init_pipeline():
    """Initializes the model pipeline once."""
    global pipe
    model_dir = download_models()
    
    if not torch.cuda.is_available(): 
        print("❌ Error: CUDA not available.")
        return
    
    pipe = HeartMuLaGenPipeline.from_pretrained(
        model_dir, 
        device=torch.device("cuda"), 
        dtype=torch.bfloat16, 
        version="3B"
    )
    print("✅ Pipeline Initialized and Ready")

def download_temp_audio(url):
    """Downloads reference audio from Supabase/URL."""
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
    """RunPod Generator Handler for Streaming."""
    torch.cuda.empty_cache()
    gc.collect()
    
    job_input = job["input"]
    lyrics = job_input.get("lyrics", "")
    tags = job_input.get("tags", "afrobeats, energetic")
    max_duration_seconds = job_input.get("max_duration_seconds", 30)
    ref_audio_url = job_input.get("ref_audio_url", SUPABASE_REF_URL)

    ref_path = None
    try:
        model_input = {"lyrics": lyrics, "tags": tags}
        if ref_audio_url:
            ref_path = download_temp_audio(ref_audio_url)
            model_input["ref_audio_path"] = ref_path

        with torch.no_grad():
            # generate_audio is the internal iterator for HeartMuLa
            gen_iter = pipe.generate_audio(
                model_input,
                max_audio_length_ms=max_duration_seconds * 1000,
                streaming=True, 
                chunk_size=20 # Tokens per chunk for streaming speed
            )

            for i, audio_tensor in enumerate(gen_iter):
                if audio_tensor is None:
                    continue
                
                # Tensor to Numpy to MP3
                audio_data = audio_tensor.cpu().numpy()
                buffer = io.BytesIO()
                sf.write(buffer, audio_data, samplerate=44100, format='mp3')
                buffer.seek(0)
                
                # RunPod yields chunks to the /stream endpoint
                yield {
                    "audio_base64": base64.b64encode(buffer.read()).decode('utf-8'),
                    "chunk_index": i,
                    "is_final": False
                }

        yield {"is_final": True, "refresh_worker": True}

    except Exception as e:
        yield {"error": f"Streaming Error: {str(e)}", "refresh_worker": True}
    finally:
        if ref_path and os.path.exists(ref_path):
            os.remove(ref_path)

if __name__ == "__main__":
    init_pipeline()
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True })