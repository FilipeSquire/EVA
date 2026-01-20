import subprocess

def speak(text):
    """
    Uses macOS native 'say' command for local TTS.
    """
    print(f"🗣️ Speaking: {text}")
    try:
        # Changed "Samantha" to "Zoe"
        # -r 175 is a good conversational speed for Zoe
        subprocess.run(["say", "-v", "Zoe (Premium)", "-r", "150", text])
    except Exception as e:
        print(f"TTS Error: {e}")