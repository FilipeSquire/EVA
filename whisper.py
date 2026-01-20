import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

# 1. Setup Model (Download happens automatically on first run)
# Use "tiny" for speed, "base" for accuracy.
# device="cuda" for your GTX 1060, or "cpu" for Mac (CTranslate2 has specific Mac support too)
model_size = "tiny" 
model = WhisperModel(model_size, device="cpu", compute_type="int8")

print("Recording for 5 seconds...")

# 2. Record Audio
fs = 16000  # Whisper expects 16kHz audio
duration = 5  # seconds
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()  # Wait until recording is finished

print("Transcribing...")

# 3. Transcribe
# Whisper expects raw audio data
segments, info = model.transcribe(recording.flatten(), beam_size=5)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")