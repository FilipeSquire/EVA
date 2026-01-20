# EVA Project

## Update 1.0

A voice-activated AI assistant with wake word detection, speech-to-text, conversational AI, and text-to-speech capabilities.

---

## How It Works

### Overview

EVA is a hands-free voice assistant that listens for a wake word ("Hey Jarvis"), processes your voice commands through AI, and responds with natural speech. The system maintains conversation context within a session, allowing for follow-up questions.

### Architecture Flow

```
[Microphone] → [Wake Word Detection] → [Voice Activity Detection] → [Speech-to-Text]
                                                                          ↓
[Speaker] ← [Text-to-Speech] ← [Streaming Response] ← [AI Brain (OpenAI/Gemini)]
```

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Wake Word | OpenWakeWord | Detects "Hey Jarvis" to start session |
| VAD | Silero VAD | Detects when you start/stop speaking |
| Speech-to-Text | Faster Whisper | Transcribes your voice to text |
| AI Brain | LangGraph + OpenAI/Gemini | Processes commands and generates responses |
| Text-to-Speech | macOS `say` (Zoe Premium) | Speaks responses naturally |

---

## Major Features

### Wake Word Activation
- Say "Hey Jarvis" to start a session
- 0.7 confidence threshold to reduce false positives
- 2-second cooldown after session ends

### Session Management
- Session stays active until you say "close program" or stay quiet for 10 seconds
- Conversation context is maintained throughout the session
- All conversations are saved to `memory/trans/` when session ends

### Streaming TTS
- AI responses are streamed in real-time
- Sentences are spoken as soon as they're complete (no waiting for full response)
- Reduces perceived latency significantly

### Web Search Integration
- If EVA doesn't know something, she'll ask: "Want me to search the web for that?"
- Say "yes" to confirm and get real-time web search results
- Say "no" to skip

### Casual Personality
- EVA speaks naturally with casual phrases and contractions
- Responses are short and punchy (1-2 sentences)
- Uses friendly language like "gotcha", "for sure", "no worries"

---

## Project Structure

```
EVA/
├── speech_to_text.py      # Main entry point - voice loop
├── brain1.py              # LangGraph AI orchestration
├── apis/
│   ├── openai.py          # OpenAI Responses API integration
│   └── gemini.py          # Google Gemini API integration
├── voice/
│   └── voice_01.py        # Text-to-speech (macOS say)
├── utils/
│   └── voice_helpers.py   # Streaming TTS worker
├── memory/
│   └── trans/             # Saved session transcripts
└── requirements.txt       # Python dependencies
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/EVA.git
cd EVA

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here (optional, for Gemini)

# Run EVA
python speech_to_text.py
```

---

## Usage

1. **Start the program**: `python speech_to_text.py`
2. **Wait for**: `--- SYSTEM READY ---`
3. **Say**: "Hey Jarvis"
4. **Ask anything**: "What's the capital of France?"
5. **End session**: Say "close program" or stay quiet for 10 seconds

---

## Configuration

Key settings in `speech_to_text.py`:

```python
WAKE_WORD_THRESHOLD = 0.7      # Wake word sensitivity (0-1)
SESSION_TIMEOUT_SECONDS = 10    # Silence before session ends
WAKE_WORD_COOLDOWN_SECONDS = 2  # Cooldown after session ends
```

Voice settings in `utils/voice_helpers.py`:

```python
TTS_VOICE = "Zoe (Premium)"     # macOS voice
TTS_RATE = "150"                # Speech rate
```

---

## Requirements

- Python 3.11+
- macOS (for native TTS)
- Microphone
- OpenAI API key (required)
- Google API key (optional, for Gemini)

---

## Dependencies

- `faster-whisper` - Speech-to-text
- `pyaudio` - Audio input
- `openwakeword` - Wake word detection
- `torch` - Silero VAD
- `openai` - OpenAI API
- `langgraph` - AI orchestration
- `google-genai` - Gemini API (optional)
