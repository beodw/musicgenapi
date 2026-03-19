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
from heartlib import HeartMuLaGenPipeline

# --- AUTHENTICATION & CONFIG ---
HF_TOKEN = os.environ.get("HF_TOKEN")
SUPABASE_REF_URL = os.environ.get("SUPABASE_REF_URL")

# --- MOCKING SPACES (For HeartMuLa compatibility) ---
from types import ModuleType
mock_spaces = ModuleType("spaces")
mock_spaces.GPU = lambda func=None, **kwargs: (lambda f: f) if func is None else func
sys.modules["spaces"] = mock_spaces

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
    if not torch.cuda.is_available(): 
        return
    # Initialize with streaming/lazy optimization
    pipe = HeartMuLaGenPipeline.from_pretrained(
        model_dir, 
        device=torch.device("cuda"), 
        dtype=torch.bfloat16, 
        version="3B"
    )
    print("✅ Streaming Pipeline Initialized")

def download_temp_audio(url):
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

def cleanup():
    torch.cuda.empty_cache() 
    gc.collect()

# --- THE GENERATOR HANDLER ---
def handler(job):
    """
    Generator handler for RunPod. 
    Yields chunks of base64 encoded audio as they are generated.
    """
    cleanup()
    job_input = job["input"]
    
    lyrics = job_input.get("lyrics", "")
    tags = job_input.get("tags", "afrobeats, energetic")
    max_duration_seconds = job_input.get("max_duration_seconds", 30)
    temperature = job_input.get("temperature", 1.0)
    topk = job_input.get("topk", 5)
    cfg_scale = job_input.get("cfg_scale", 1.0)
    ref_audio_url = job_input.get("ref_audio_url", SUPABASE_REF_URL)

    ref_path = None
    try:
        model_input = {"lyrics": lyrics, "tags": tags}
        if ref_audio_url:
            ref_path = download_temp_audio(ref_audio_url)
            model_input["ref_audio_path"] = ref_path

        max_audio_length_ms = max_duration_seconds * 1000

        # HeartMuLa streaming inference
        # We use pipe.stream() or equivalent iterator provided by HeartMuLa
        # This typically yields audio tensors (samples)
        with torch.no_grad():
            audio_generator = pipe.stream(
                model_input,
                max_audio_length_ms=max_audio_length_ms,
                topk=topk,
                temperature=temperature,
                cfg_scale=cfg_scale,
            )

            for i, audio_chunk_tensor in enumerate(audio_generator):
                # Convert GPU tensor to CPU numpy array
                audio_data = audio_chunk_tensor.cpu().numpy()
                
                # Convert to MP3/WAV in-memory using BytesIO
                buffer = io.BytesIO()
                # HeartMuLa usually outputs at 44100Hz or 32000Hz
                sf.write(buffer, audio_data, samplerate=44100, format='mp3')
                buffer.seek(0)
                
                chunk_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                
                # Yield the chunk to RunPod
                yield {
                    "chunk_index": i,
                    "audio_base64": chunk_base64,
                    "is_final": False
                }

        # Signal completion
        yield {"is_final": True, "refresh_worker": True}

    except Exception as e:
        yield {"error": str(e), "refresh_worker": True}
    finally:
        if ref_path and os.path.exists(ref_path):
            os.remove(ref_path)
        cleanup()

init_pipeline()
# Use 'runpod.serverless.start' with the generator handler
runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})