import os
import sys
import torch
import base64
import runpod
import gc
import io
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download, snapshot_download

# --- AUTHENTICATION & CONFIG ---
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
    if not torch.cuda.is_available(): return
    pipe = HeartMuLaGenPipeline.from_pretrained(
        model_dir, 
        device=torch.device("cuda"), 
        dtype=torch.bfloat16, 
        version="3B"
    )
    print("✅ Pipeline Initialized")

def handler(job):
    torch.cuda.empty_cache()
    gc.collect()
    
    job_input = job["input"]
    lyrics = job_input.get("lyrics", "No lyrics provided")
    tags = job_input.get("tags", "afrobeats")
    max_duration_seconds = job_input.get("max_duration_seconds", 10) # Start small to test

    try:
        model_input = {"lyrics": lyrics, "tags": tags}
        current_ms = 0
        step_ms = 5000 
        total_ms = max_duration_seconds * 1000
        
        while current_ms < total_ms:
            # FIX: We use a more direct call and check the type carefully
            # Some versions of heartlib require return_type="tensor"
            audio_output = pipe(
                model_input,
                max_audio_length_ms=step_ms,
                continue_sequence=(current_ms > 0)
            )

            if audio_output is None:
                # If the model yields nothing, we shouldn't just exit silently
                yield {"error": f"Model returned None at {current_ms}ms", "is_final": True}
                return

            # Robust conversion to Numpy
            if hasattr(audio_output, 'cpu'):
                audio_data = audio_output.detach().cpu().float().numpy()
            elif isinstance(audio_output, dict) and 'audio' in audio_output:
                audio_data = audio_output['audio'].detach().cpu().float().numpy()
            else:
                audio_data = np.array(audio_output)

            # Ensure 1D array for soundfile
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()

            # Encode
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, samplerate=44100, format='mp3')
            buffer.seek(0)
            
            yield {
                "audio_base64": base64.b64encode(buffer.read()).decode('utf-8'),
                "chunk_index": current_ms // step_ms,
                "is_final": False
            }
            
            current_ms += step_ms

        yield {"is_final": True}

    except Exception as e:
        yield {"error": str(e), "refresh_worker": True}

if __name__ == "__main__":
    init_pipeline()
    # return_aggregate_stream=True ensures that even if the stream fails, 
    # you see the yielded error in the final output.
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})