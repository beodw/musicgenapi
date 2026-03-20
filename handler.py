import os
import sys
import torch
import base64
import runpod
import gc
import io
import soundfile as sf
from huggingface_hub import hf_hub_download, snapshot_download

# --- AUTHENTICATION & CONFIG ---
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- MOCKING SPACES (Required for heartlib internal imports) ---
from types import ModuleType
mock_spaces = ModuleType("spaces")
mock_spaces.GPU = lambda func=None, **kwargs: (lambda f: f) if func is None else func
sys.modules["spaces"] = mock_spaces

from heartlib import HeartMuLaGenPipeline

pipe = None

def download_models():
    """Download models explicitly to /tmp/heartmula_models."""
    cache_dir = os.environ.get("HF_HOME", "/tmp")
    model_dir = os.path.join(cache_dir, "heartmula_models")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        # 1. Download basic configs
        for filename in ["tokenizer.json", "gen_config.json"]:
            hf_hub_download(repo_id="HeartMuLa/HeartMuLaGen", filename=filename, local_dir=model_dir, token=HF_TOKEN)
        
        # 2. Download the 3B LLM weights
        snapshot_download(repo_id="HeartMuLa/HeartMuLa-oss-3B-happy-new-year", 
                          local_dir=os.path.join(model_dir, "HeartMuLa-oss-3B"), 
                          token=HF_TOKEN)
        
        # 3. Download the Audio Codec (VQ)
        snapshot_download(repo_id="HeartMuLa/HeartCodec-oss-20260123", 
                          local_dir=os.path.join(model_dir, "HeartCodec-oss"), 
                          token=HF_TOKEN)
    return model_dir

def init_pipeline():
    """Initializes the model pipeline once on worker start."""
    global pipe
    model_dir = download_models()
    
    if not torch.cuda.is_available(): 
        return
    
    # Load model to GPU in bfloat16 to save VRAM
    pipe = HeartMuLaGenPipeline.from_pretrained(
        model_dir, 
        device=torch.device("cuda"), 
        dtype=torch.bfloat16, 
        version="3B"
    )
    print("✅ Pipeline Initialized")

def handler(job):
    """RunPod Generator Handler - This yields chunks to the /stream endpoint."""
    # Critical for RunPod: Clean memory before every request
    torch.cuda.empty_cache()
    gc.collect()
    
    job_input = job["input"]
    lyrics = job_input.get("lyrics", "")
    tags = job_input.get("tags", "afrobeats, energetic")
    max_duration_seconds = job_input.get("max_duration_seconds", 30)

    try:
        model_input = {"lyrics": lyrics, "tags": tags}
        
        with torch.no_grad():
            # We split the generation into 5-second segments to enable streaming
            step_ms = 5000 
            total_ms = max_duration_seconds * 1000
            current_ms = 0
            
            while current_ms < total_ms:
                # IMPORTANT: We call pipe() directly (the __call__ method)
                # We do NOT call .save() or .export() which require torchcodec
                audio_output = pipe(
                    model_input,
                    max_audio_length_ms=step_ms,
                    continue_sequence=(current_ms > 0)
                )
                
                if audio_output is None:
                    break

                # Convert the raw GPU tensor to a CPU Numpy array immediately.
                # This breaks the link to any internal heartlib encoding logic.
                if torch.is_tensor(audio_output):
                    audio_data = audio_output.detach().cpu().numpy()
                else:
                    audio_data = audio_output

                # Manual MP3 encoding using soundfile + BytesIO
                buffer = io.BytesIO()
                # HeartMuLa generates at 44.1kHz
                sf.write(buffer, audio_data, samplerate=44100, format='mp3')
                buffer.seek(0)
                
                # Push the base64 chunk to the RunPod stream
                yield {
                    "audio_base64": base64.b64encode(buffer.read()).decode('utf-8'),
                    "chunk_index": current_ms // step_ms,
                    "is_final": False
                }
                
                current_ms += step_ms

        # Final signal for the frontend
        yield {"is_final": True, "refresh_worker": False}

    except Exception as e:
        yield {"error": f"Inference Error: {str(e)}", "refresh_worker": True}

if __name__ == "__main__":
    init_pipeline()
    # RunPod serverless entry point
    runpod.serverless.start({"handler": handler})