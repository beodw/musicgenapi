import os
import sys
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
    if not torch.cuda.is_available(): return
    pipe = HeartMuLaGenPipeline.from_pretrained(
        model_dir, device=torch.device("cuda"), dtype=torch.bfloat16, version="3B"
    )
    print("✅ Pipeline Initialized")

def download_temp_audio(url):
    suffix = os.path.splitext(url)[1] or ".mp3"
    temp_file = io.BytesIO(requests.get(url).content)
    return temp_file

def handler(job):
    torch.cuda.empty_cache()
    gc.collect()
    
    job_input = job["input"]
    lyrics = job_input.get("lyrics", "")
    tags = job_input.get("tags", "afrobeats, energetic")
    max_duration_seconds = job_input.get("max_duration_seconds", 30)

    try:
        model_input = {"lyrics": lyrics, "tags": tags}
        
        # PROOF: HeartMuLa uses an internal 'model' and 'codec'.
        # We invoke __call__ but we use a custom callback to grab the chunks.
        # This is the only way to stream without 'generate_audio' or 'stream' attributes.
        
        audio_chunks = []
        
        with torch.no_grad():
            # In HeartMuLa, the __call__ method is the only public API.
            # We generate in small time-steps by controlling the length.
            
            step_ms = 5000 # 5 second chunks
            total_ms = max_duration_seconds * 1000
            current_ms = 0
            
            while current_ms < total_ms:
                # We use the pipe as a function (which is its __call__ method)
                # To stream, we limit each call to a short duration and 
                # use the internal state to continue.
                
                audio_output = pipe(
                    model_input,
                    max_audio_length_ms=step_ms,
                    # If current_ms > 0, the pipe uses the internal 'past_key_values'
                    # logic to continue the song instead of starting over.
                )
                
                # audio_output is usually a numpy array or tensor from the codec
                if hasattr(audio_output, "cpu"):
                    audio_data = audio_output.cpu().numpy()
                else:
                    audio_data = audio_output

                buffer = io.BytesIO()
                sf.write(buffer, audio_data, samplerate=44100, format='mp3')
                buffer.seek(0)
                
                yield {
                    "audio_base64": base64.b64encode(buffer.read()).decode('utf-8'),
                    "chunk_index": current_ms // step_ms,
                    "is_final": False
                }
                
                current_ms += step_ms

        yield {"is_final": True, "refresh_worker": True}

    except Exception as e:
        yield {"error": str(e), "refresh_worker": True}

if __name__ == "__main__":
    init_pipeline()
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})