import subprocess
import threading
import queue
import re

# Create a queue to hold sentences to be spoken
tts_queue = queue.Queue()

# TTS Configuration - match voice_01.py settings
TTS_VOICE = "Zoe (Premium)"
TTS_RATE = "150"


def tts_worker():
    """
    Background thread that constantly checks for new sentences
    and speaks them using macOS 'say'.
    """
    while True:
        text = tts_queue.get()
        if text is None:  # Signal to stop
            break

        try:
            print(f"🗣️ Speaking: {text}")
            subprocess.run(["say", "-v", TTS_VOICE, "-r", TTS_RATE, text])
        except Exception as e:
            print(f"TTS Error: {e}")

        tts_queue.task_done()


async def stream_and_speak_response(ai_stream_iterator):
    """
    Consumes the AI stream, buffers text, splits into sentences,
    and feeds the TTS worker. Waits for all speech to complete before returning.

    Flow:
    1. Receive text chunks from the AI stream
    2. Buffer chunks until a sentence boundary is detected (. ! ?)
    3. Send complete sentences to TTS queue immediately
    4. Background worker speaks sentences while stream continues
    5. After stream ends, speak any remaining text and wait for completion
    """
    buffer = ""
    # Match sentence endings followed by space or end of string
    sentence_pattern = re.compile(r'([.!?])(?:\s+|$)')

    async for chunk in ai_stream_iterator:
        if chunk:
            buffer += chunk

            # Find all sentence boundaries in the buffer
            while True:
                match = sentence_pattern.search(buffer)
                if not match:
                    break

                # Extract the complete sentence (including punctuation)
                end_pos = match.end()
                sentence = buffer[:end_pos].strip()

                if sentence:
                    tts_queue.put(sentence)

                # Keep the rest in the buffer
                buffer = buffer[end_pos:]

    # Stream finished - speak any remaining text
    if buffer.strip():
        tts_queue.put(buffer.strip())

    # Wait for all sentences to be spoken before returning
    tts_queue.join()