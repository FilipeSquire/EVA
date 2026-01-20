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
import time
import asyncio

# Import AI brain
from brain1 import ask_ai, confirm_websearch, ask_ai_stream_with_metadata, confirm_websearch_stream, save_session
from voice.voice_01 import speak

# Import utils
from utils.voice_helpers import tts_worker, stream_and_speak_response

# Voice enhancements
import threading



# Start the TTS worker thread immediately
threading.Thread(target=tts_worker, daemon=True).start()


# --- CONFIGURATION ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512
WAKE_WORD_THRESHOLD = 0.7  # Higher threshold to reduce false positives
SESSION_TIMEOUT_SECONDS = 10  # Exit session after 10 seconds of silence
WAKE_WORD_COOLDOWN_SECONDS = 2  # Ignore wake word for 2 seconds after session ends

# Initialize Whisper model
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# --- INITIALIZE MODELS ---
print("Loading Wake Word Model...")
oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

print("Loading Silero VAD...")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
vad_iterator = VADIterator(vad_model, threshold=0.3, min_silence_duration_ms=2000)

# --- AUDIO STREAM SETUP ---
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("\n--- SYSTEM READY ---")
print("Say 'Hey Jarvis' to start a session.")
print("Session ends with 'close program' or 10 seconds of silence.")

# --- STATE MANAGEMENT ---
class SessionState:
    PASSIVE = "passive"      # Waiting for wake word
    IN_SESSION = "in_session"  # Active session, listening for commands
    RECORDING = "recording"   # Currently recording speech

audio_buffer = collections.deque()
state = SessionState.PASSIVE
should_exit = False
last_interaction_time = time.time()
last_session_end_time = 0  # Track when session ended for cooldown


def transcribe_audio(audio_data: bytes) -> str:
    """Transcribe audio bytes to text using Whisper."""
    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    segments, info = whisper_model.transcribe(audio_float32.flatten(), beam_size=5)
    return " ".join([segment.text for segment in segments]).strip()


def build_conversation_context(session_conversations: list) -> str:
    """
    Build a context string from previous conversations in the session.

    Args:
        session_conversations: List of conversation dicts with prompt/response

    Returns:
        Formatted context string for the AI
    """
    if not session_conversations:
        return None

    context_parts = ["Previous conversation in this session:"]
    for conv in session_conversations:
        context_parts.append(f"User: {conv['prompt']}")
        context_parts.append(f"Assistant: {conv['response']}")

    return "\n".join(context_parts)


async def process_command_streaming(transcription: str, session_conversations: list = None):
    """
    Process the transcribed command and stream AI response with TTS.
    Returns (should_end_session, ai_response_object or None)

    Args:
        transcription: The user's spoken command
        session_conversations: Previous conversations in this session for context
    """
    # Check for end session command
    if "close program" in transcription.lower() or "close session" in transcription.lower():
        speak("Ending session. Goodbye!")
        return True, None

    # Build context from previous conversations
    context = build_conversation_context(session_conversations)

    # Get streaming AI response with metadata callback
    stream_gen, get_response = await ask_ai_stream_with_metadata(
        prompt=transcription,
        provider="openai",
        context=context
    )

    # Stream and speak the response
    await stream_and_speak_response(stream_gen)

    # Get the full response metadata after streaming completes
    ai_response = get_response()

    return False, ai_response


async def handle_websearch_confirmation_streaming(previous_response, transcription: str):
    """Handle user confirmation for web search with streaming TTS."""
    text = transcription.lower().strip()

    # Check if any confirmation word is present in the response
    confirm_words = ["yes", "yeah", "sure", "go ahead", "search", "please", "do it", "okay", "ok", "yep", "yup"]
    decline_words = ["no", "nope", "don't", "skip", "cancel", "never mind"]

    # Check for decline first (more specific)
    if any(word in text for word in decline_words):
        speak("Okay, skipping web search.")
        return

    # Check for confirmation
    if any(word in text for word in confirm_words):
        print("🔍 Searching the web...")
        stream = confirm_websearch_stream(previous_response)
        await stream_and_speak_response(stream)
        return

    # Default to skipping if unclear
    speak("I didn't catch that. Skipping web search.")


def end_session_and_save(session_conversations: list, session_start_time: str):
    """Helper to save session and clear conversation list."""
    if session_conversations:
        save_session(session_conversations, session_start_time)
    return [], None  # Return empty list and None for session_start


async def main_loop():
    global state, should_exit, last_interaction_time, audio_buffer, last_session_end_time

    pending_websearch_response = None  # Store response waiting for confirmation
    session_conversations = []  # Accumulate conversations during session
    session_start_time = None  # Track when session started

    try:
        while True:
            # Get audio chunk
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_int16 = np.frombuffer(data, dtype=np.int16)

            # ---------------------------------------------------------
            # STATE: PASSIVE (Waiting for Wake Word)
            # ---------------------------------------------------------
            if state == SessionState.PASSIVE:
                # Skip wake word detection during cooldown period
                if time.time() - last_session_end_time < WAKE_WORD_COOLDOWN_SECONDS:
                    continue

                prediction = oww_model.predict(audio_int16)
                score = prediction["hey_jarvis"]

                # Debug: uncomment to see scores
                # if score > 0.3:
                #     print(f"Wake word score: {score:.2f}")

                if score > WAKE_WORD_THRESHOLD:
                    print("\n🤖 WAKE WORD DETECTED! Starting session...")
                    print("(Say 'close program' or stay quiet for 10 seconds to exit)")
                    state = SessionState.IN_SESSION
                    last_interaction_time = time.time()
                    vad_iterator.reset_states()
                    audio_buffer.clear()
                    pending_websearch_response = None
                    session_conversations = []  # Start fresh conversation list
                    session_start_time = datetime.now().isoformat()

            # ---------------------------------------------------------
            # STATE: IN_SESSION (Active session, listening)
            # ---------------------------------------------------------
            elif state == SessionState.IN_SESSION:
                # Check for session timeout
                if time.time() - last_interaction_time > SESSION_TIMEOUT_SECONDS:
                    print("\n⏰ Session timeout (10 seconds of silence). Exiting program...")
                    # Save all conversations from this session
                    end_session_and_save(session_conversations, session_start_time)
                    return  # Exit the main loop and terminate program

                # Convert audio for VAD
                audio_float32 = torch.from_numpy(audio_int16.astype(np.float32) / 32768.0)
                speech_dict = vad_iterator(audio_float32, return_seconds=True)

                # Check if speech started
                if speech_dict and "start" in speech_dict:
                    print("🎤 Listening...")
                    state = SessionState.RECORDING
                    audio_buffer.clear()
                    audio_buffer.append(data)

            # ---------------------------------------------------------
            # STATE: RECORDING (Capturing speech)
            # ---------------------------------------------------------
            elif state == SessionState.RECORDING:
                audio_float32 = torch.from_numpy(audio_int16.astype(np.float32) / 32768.0)
                speech_dict = vad_iterator(audio_float32, return_seconds=True)
                audio_buffer.append(data)

                # Check if speech ended
                if speech_dict and "end" in speech_dict:
                    print("✅ Processing...")

                    # Transcribe
                    full_audio_data = b''.join(audio_buffer)
                    transcription = transcribe_audio(full_audio_data)
                    audio_length = len(full_audio_data) / (RATE * 2)

                    print(f"📝 You said: {transcription}")

                    # Check if this is a websearch confirmation
                    if pending_websearch_response is not None:
                        await handle_websearch_confirmation_streaming(
                            pending_websearch_response, transcription
                        )
                        pending_websearch_response = None
                    else:
                        # Process as normal command with streaming (pass conversation history for context)
                        end_session, ai_response = await process_command_streaming(
                            transcription, session_conversations
                        )

                        # Add conversation to session list
                        if ai_response:
                            session_conversations.append({
                                "timestamp": datetime.now().isoformat(),
                                "prompt": transcription,
                                "response": ai_response.text,
                                "provider": ai_response.provider,
                                "used_websearch": False
                            })

                        if end_session:
                            print("\n👋 Session ended. Exiting program...")
                            # Save all conversations from this session
                            end_session_and_save(session_conversations, session_start_time)
                            return  # Exit the main loop and terminate program

                        # Check if AI is asking for websearch confirmation
                        if ai_response and ai_response.needs_websearch:
                            pending_websearch_response = ai_response

                    # Reset for next command in session
                    last_interaction_time = time.time()
                    state = SessionState.IN_SESSION
                    audio_buffer.clear()
                    vad_iterator.reset_states()

    except KeyboardInterrupt:
        print("\n\nStopping...")
        # Save any pending conversations before exit
        if session_conversations:
            end_session_and_save(session_conversations, session_start_time)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("Audio resources cleaned up.")


if __name__ == "__main__":
    asyncio.run(main_loop())
