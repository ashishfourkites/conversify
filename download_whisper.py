from faster_whisper import WhisperModel
import os

print("Downloading pre-converted Whisper tiny model...")
cache_dir = "conversify/data/models_cache"
os.makedirs(cache_dir, exist_ok=True)

try:
    model = WhisperModel("guillaumekln/faster-whisper-tiny", 
                         device="cpu", 
                         compute_type="int8",
                         download_root=cache_dir)
    print("Model downloaded successfully!")
    
    # Test the model
    print("Testing model with a simple transcription...")
    segments, info = model.transcribe("conversify/data/warmup_audio.wav", language="en")
    print(f"Detected language: {info.language}")
except Exception as e:
    print(f"Error: {e}")
