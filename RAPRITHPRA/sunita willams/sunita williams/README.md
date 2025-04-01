# AI Chatbot Voice Functionality

## Voice Feature Usage

The voice functionality in the AI Chatbot has been enhanced with better debug messages and usability improvements.

### How to use Voice Commands

1. **Toggle Voice Feature:**
   - Type `!voice` to toggle voice mode on/off
   - When enabled, you'll see: `[VOICE] ✓ Voice input enabled`
   - When disabled, you'll see: `[VOICE] ✗ Voice input disabled`

2. **Use Voice Input:**
   - Type `voice` (without the ! prefix) to activate voice recording
   - The system will show detailed status messages:
     ```
     [VOICE] Activating voice input mode...
     [VOICE] Initializing microphone and speech recognition...
     [VOICE] 🎤 Listening for voice input in en-US...
     [VOICE] Please speak clearly into your microphone...
     [VOICE] Adjusting for ambient noise...
     [VOICE] Ready to capture speech (timeout: 5s)
     ```
   - Speak your query clearly into the microphone
   - After processing, you'll see: `[VOICE] ✓ Successfully captured: "your speech text"`

3. **Wake Word Feature:**
   - Set a wake word with: `!wake your-wake-word`
   - In continuous listening mode, say the wake word followed by your command

### Troubleshooting

If voice recognition fails, you'll see detailed error messages:
- `[VOICE] ⚠ No speech detected or recognition failed`
- `[VOICE] Please check your microphone and try again`

Make sure your system has a working microphone and the required Python packages:
```
pip install SpeechRecognition==3.10.0 gtts==2.3.2 pygame==2.5.2
```

### Command-line Options

Start the application with voice enabled by default:
```
python iti_app.py --voice-enabled
```

For continuous voice interaction mode:
```
python iti_app.py --continuous-voice
``` 