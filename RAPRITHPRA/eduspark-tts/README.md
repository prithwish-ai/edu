# EduSpark TTS (Text-to-Speech) Module

This repository contains the TTS (Text-to-Speech) module from the EduSpark educational chatbot platform. This component provides text-to-speech functionality with support for multiple languages and comprehensive debugging tools.

## Features

- Text-to-Speech functionality with gTTS integration
- Multi-language support
- Comprehensive debugging tools
- Filesystem permission verification
- Browser compatibility testing
- Performance evaluation

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the server:
   ```
   python server.py
   ```

3. Access the debug tools:
   - Basic TTS Test: http://localhost:5000/tts-test
   - Advanced Debugging: http://localhost:5000/tts-debug
   - Test Index: http://localhost:5000/test

## Files

- `server.py`: Main server file with TTS endpoints and routing
- `templates/tts_test.html`: Basic TTS testing interface
- `templates/tts_debug.html`: Advanced debugging tools
- `templates/test_index.html`: Index page for all test tools

## Debugging Features

The TTS debugging tools allow you to:
- Verify gTTS installation and functionality
- Test filesystem permissions for temporary audio files
- Check browser audio compatibility
- Analyze network requests
- Evaluate TTS performance with different text lengths 