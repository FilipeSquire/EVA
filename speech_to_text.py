import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model
import torch
import collections
from faster_whisper import WhisperModel
import os
import json
from datetime import datetime

# --- CONFIGURATION ---
# Both models work best at 16000 Hz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
# OpenWakeWord prefers chunks of 1280 samples (80ms) for efficiency, but works with smaller multiples
CHUNK = 512 
WAKE_WORD_THRESHOLD = 0.5

# Initialize Whisper model
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# --- INITIALIZE MODELS ---
print("Loading Wake Word Model...")
# Using the pre-trained 'hey_jarvis' model. You can swap this later.
oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

print("Loading Silero VAD...")
# Load Silero VAD from Torch Hub (downloads automatically)
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
vad_iterator = VADIterator(vad_model, threshold=0.3, min_silence_duration_ms=1000)

# --- AUDIO STREAM SETUP ---
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("\n--- SYSTEM READY ---")
print("Say 'Hey Jarvis' to trigger me.")

# --- STATE MANAGEMENT ---
# Buffer to store audio when user is speaking
audio_buffer = collections.deque() 
is_awake = False 
should_exit = False 

try:
    while True:
        # 1. Get Audio Chunk
        data = stream.read(CHUNK, exception_on_overflow=False)
        # Convert raw bytes to numpy array (int16)
        audio_int16 = np.frombuffer(data, dtype=np.int16)

        # ---------------------------------------------------------
        # STATE 1: PASSIVE LISTENING (Waiting for Wake Word)
        # ---------------------------------------------------------
        if not is_awake:
            # Feed audio to OpenWakeWord
            prediction = oww_model.predict(audio_int16)
            
            # Check if "hey_jarvis" score is high enough
            if prediction["hey_jarvis"] > WAKE_WORD_THRESHOLD:
                print("\n🤖 WAKE WORD DETECTED! (Listening for command...)")
                is_awake = True
                vad_iterator.reset_states() # Reset VAD logic
                audio_buffer.clear() # Clear old audio

        # ---------------------------------------------------------
        # STATE 2: ACTIVE LISTENING (VAD / Recording)
        # ---------------------------------------------------------
        else:
            # Silero expects float32 tensor between -1 and 1
            audio_float32 = torch.from_numpy(audio_int16.astype(np.float32) / 32768.0)
            
            # Feed to VAD Iterator
            # This function returns a dict if speech starts or ends
            speech_dict = vad_iterator(audio_float32, return_seconds=True)
            
            # Always save audio while awake (so we don't miss words)
            audio_buffer.append(data)

            # Check if Silero thinks you stopped talking
            if speech_dict:
                if "end" in speech_dict:
                    print("✅ End of speech detected. Processing...")
                    
                    # Transcribe the captured audio using Whisper
                    full_audio_data = b''.join(audio_buffer)
                    audio_int16 = np.frombuffer(full_audio_data, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    segments, info = whisper_model.transcribe(audio_float32.flatten(), beam_size=5)
                    transcription = " ".join([segment.text for segment in segments])
                    
                    # Check for exit command
                    if "exit program" in transcription.lower():
                        should_exit = True
                        print("Exit command detected. Shutting down...")
                    
                    # Save to file with metadata
                    timestamp = datetime.now().isoformat()
                    audio_length_seconds = len(full_audio_data) / (RATE * 2)  # 16-bit audio
                    transcription_data = {
                        "transcription_model": "faster-whisper base",
                        "vad_model": "silero_vad",
                        "wake_word_model": "hey_jarvis",
                        "timestamp": timestamp,
                        "audio_length_seconds": audio_length_seconds,
                        "transcription": transcription
                    }
                    os.makedirs("memory/trans", exist_ok=True)
                    filename = f"transcription_{timestamp.replace(':', '-').replace('.', '-')}.txt"
                    with open(f"memory/trans/{filename}", "w") as f:
                        json.dump(transcription_data, f, indent=4)
                    
                    print(f"Transcribed: {transcription}")
                    print(f"Saved to memory/trans/{filename}")
                    print("Returning to sleep...")
                    
                    is_awake = False
                    audio_buffer.clear()

                    if should_exit:
                        break

except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop_stream()
    stream.close()
    audio.terminate()

# Cleanup if exited via voice command
if should_exit:
    print("Exiting due to voice command.")
    stream.stop_stream()
    stream.close()
    audio.terminate()