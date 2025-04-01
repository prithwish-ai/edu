"""
Voice interaction manager for the AI Chatbot.

This module provides voice interaction capabilities, enabling the chatbot
to process speech input and generate speech output.
"""

import os
import json
import time
from datetime import datetime
import pygame
from typing import Dict, List, Optional, Any, Tuple
from colorama import Fore, Style

class VoiceInteractionManager:
    """Manages voice interaction capabilities for the chatbot."""
    
    def __init__(self, voice_settings_path="voice_settings.json"):
        """Initialize the voice interaction manager.
        
        Args:
            voice_settings_path: Path to voice settings file
        """
        self.settings = {
            "enabled": False,
            "volume": 0.9,
            "rate": 150,
            "voice_id": "default",
            "language": "en",
            "auto_detect_language": True,
            "wake_word": "iti",
            "continuous_listening": False
        }
        
        self.voice_settings_path = voice_settings_path
        self.is_available = self._check_availability()
        self.is_speaking = False
        self.is_listening = False
        self.recognizer = None
        self.tts_engine = None
        
        # Load settings
        self._load_settings()
        
        # Initialize pygame for audio playback
        try:
            pygame.mixer.init()
            print(f"{Fore.GREEN}✓ Audio playback initialized{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Audio playback initialization failed: {e}{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}✓ Voice interaction manager initialized{Style.RESET_ALL}")
    
    def _check_availability(self):
        """Check if required packages are available."""
        try:
            import speech_recognition as sr
            import gtts
            self.recognizer = sr.Recognizer()
            return True
        except ImportError:
            print(f"{Fore.YELLOW}Voice interaction requires additional packages.{Style.RESET_ALL}")
            print("Install with: pip install SpeechRecognition==3.10.0 gtts==2.3.2 pygame==2.5.2")
            return False
    
    def _load_settings(self):
        """Load voice settings from file."""
        try:
            if os.path.exists(self.voice_settings_path):
                with open(self.voice_settings_path, "r", encoding="utf-8") as f:
                    saved_settings = json.load(f)
                    self.settings.update(saved_settings)
                print(f"{Fore.GREEN}✓ Loaded voice settings{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load voice settings: {e}{Style.RESET_ALL}")
    
    def _save_settings(self):
        """Save voice settings to file."""
        try:
            with open(self.voice_settings_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save voice settings: {e}{Style.RESET_ALL}")
    
    def speak(self, text, language=None):
        """Convert text to speech and play it.
        
        Args:
            text: Text to convert to speech
            language: Language code (overrides default if provided)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available or not text:
            return False
        
        try:
            import gtts
            from gtts import gTTS
            import tempfile
            
            # Set language
            lang = language or self.settings["language"]
            
            # Create TTS
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_filename = fp.name
            
            tts.save(temp_filename)
            
            # Play audio
            self.is_speaking = True
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.set_volume(self.settings["volume"])
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up
            self.is_speaking = False
            try:
                os.unlink(temp_filename)
            except:
                pass
            
            return True
            
        except Exception as e:
            print(f"{Fore.YELLOW}Speech generation error: {e}{Style.RESET_ALL}")
            self.is_speaking = False
            return False
    
    def listen(self, timeout=5, phrase_time_limit=None, language=None):
        """Listen for speech and convert to text.
        
        Args:
            timeout: Maximum time to wait for speech
            phrase_time_limit: Maximum time to listen for a phrase
            language: Language code for speech recognition (e.g., 'en-US')
        
        Returns:
            Recognized text or None if not recognized
        """
        if not self.is_available:
            return None
        
        try:
            import speech_recognition as sr
            
            # Initialize recognizer if needed
            if self.recognizer is None:
                self.recognizer = sr.Recognizer()
            
            # Use provided language or default from settings
            recognition_language = language or self.settings["language"]
            print(f"{Fore.CYAN}Listening in language: {recognition_language}{Style.RESET_ALL}")
            
            # Adjust for ambient noise
            with sr.Microphone() as source:
                print(f"{Fore.CYAN}Adjusting for ambient noise...{Style.RESET_ALL}")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print(f"{Fore.CYAN}Listening...{Style.RESET_ALL}")
                self.is_listening = True
                
                try:
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                except sr.WaitTimeoutError:
                    self.is_listening = False
                    return None
            
            self.is_listening = False
            print(f"{Fore.CYAN}Processing speech...{Style.RESET_ALL}")
            
            # Try to recognize
            try:
                # Use Google's speech recognition with specified language
                text = self.recognizer.recognize_google(audio, language=recognition_language)
                print(f"{Fore.GREEN}Recognized: {text}{Style.RESET_ALL}")
                return text
            except sr.UnknownValueError:
                print(f"{Fore.YELLOW}Could not understand audio{Style.RESET_ALL}")
                return None
            except sr.RequestError as e:
                print(f"{Fore.YELLOW}Recognition service error: {e}{Style.RESET_ALL}")
                return None
                
        except Exception as e:
            print(f"{Fore.YELLOW}Speech recognition error: {e}{Style.RESET_ALL}")
            self.is_listening = False
            return None
    
    def toggle_voice(self):
        """Toggle voice interaction on/off.
        
        Returns:
            New state (True for enabled, False for disabled)
        """
        self.settings["enabled"] = not self.settings["enabled"]
        self._save_settings()
        return self.settings["enabled"]
    
    def set_language(self, language_code):
        """Set voice language.
        
        Args:
            language_code: Language code (e.g., 'en', 'hi', 'fr')
        
        Returns:
            True if successful, False otherwise
        """
        # Validate language code
        try:
            import gtts
            from gtts.lang import tts_langs
            
            available_langs = tts_langs()
            if language_code not in available_langs:
                print(f"{Fore.YELLOW}Unsupported language code: {language_code}{Style.RESET_ALL}")
                return False
            
            self.settings["language"] = language_code
            self._save_settings()
            return True
            
        except Exception as e:
            print(f"{Fore.YELLOW}Error setting language: {e}{Style.RESET_ALL}")
            return False
    
    def set_volume(self, volume):
        """Set voice volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        
        Returns:
            True if successful, False otherwise
        """
        if not 0.0 <= volume <= 1.0:
            print(f"{Fore.YELLOW}Volume must be between 0.0 and 1.0{Style.RESET_ALL}")
            return False
        
        self.settings["volume"] = volume
        self._save_settings()
        return True
    
    def set_wake_word(self, wake_word=None):
        """Set wake word for voice activation.
        
        Args:
            wake_word: Wake word or phrase (None to disable)
        
        Returns:
            True if successful, False otherwise
        """
        self.settings["wake_word"] = wake_word
        self._save_settings()
        return True
    
    def listen_for_wake_word(self, timeout=5):
        """Listen specifically for the wake word.
        
        Args:
            timeout: Maximum time to wait for wake word
        
        Returns:
            True if wake word detected, False otherwise
        """
        if not self.is_available or not self.settings["wake_word"]:
            return False
        
        text = self.listen(timeout=timeout)
        if not text:
            return False
        
        # Check if wake word is in the recognized text
        return self.settings["wake_word"].lower() in text.lower()
    
    def continuous_listen(self, callback, stop_event=None):
        """Continuously listen for speech in a separate thread.
        
        Args:
            callback: Function to call with recognized text
            stop_event: Event to signal stopping the listening thread
        """
        if not self.is_available:
            return
        
        import threading
        
        if stop_event is None:
            stop_event = threading.Event()
        
        def listen_thread():
            while not stop_event.is_set():
                if self.settings["wake_word"]:
                    # Wait for wake word
                    if self.listen_for_wake_word(timeout=3):
                        # Wake word detected, listen for command
                        text = self.listen(timeout=5)
                        if text:
                            callback(text)
                else:
                    # No wake word, just listen
                    text = self.listen(timeout=5)
                    if text:
                        callback(text)
                
                # Small delay to prevent CPU overuse
                time.sleep(0.1)
        
        # Start listening thread
        thread = threading.Thread(target=listen_thread)
        thread.daemon = True
        thread.start()
        
        return stop_event 