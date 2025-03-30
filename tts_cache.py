import os
import hashlib
import time
from collections import OrderedDict
from config import get_config

class TTSCache:
    """Cache manager for TTS generated audio files.
    Uses an LRU (Least Recently Used) cache strategy.
    """
    def __init__(self):
        self.config = get_config()
        self.cache_enabled = self.config.TTS_CACHE_ENABLED
        self.cache_size = self.config.TTS_CACHE_SIZE
        self.temp_dir = self.config.TEMP_AUDIO_DIR
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        # LRU cache for filename mappings
        self.cache = OrderedDict()
        
        # Load existing files into cache
        self._load_existing_files()
    
    def _load_existing_files(self):
        """Load existing audio files into the cache."""
        if not self.cache_enabled:
            return
            
        try:
            for filename in os.listdir(self.temp_dir):
                if filename.startswith('speech_') and filename.endswith('.mp3'):
                    # Add file to cache with current time as timestamp
                    file_path = os.path.join(self.temp_dir, filename)
                    file_age = os.path.getmtime(file_path)
                    self.cache[filename] = file_age
            
            # Sort cache by file age (oldest first)
            self.cache = OrderedDict(sorted(self.cache.items(), key=lambda x: x[1]))
            
            # Trim cache if needed
            self._trim_cache()
        except Exception as e:
            print(f"Error loading existing files into cache: {str(e)}")
    
    def _generate_filename(self, text, language):
        """Generate a unique filename for the TTS audio based on text and language."""
        # Create a unique hash based on text and language
        text_hash = hashlib.md5(f"{text}_{language}".encode()).hexdigest()
        filename = f"speech_{text_hash}.mp3"
        return filename
    
    def _trim_cache(self):
        """Remove oldest files when cache exceeds size limit."""
        if not self.cache_enabled:
            return
            
        try:
            # Remove oldest files until we're under the cache size limit
            while len(self.cache) > self.cache_size:
                oldest_file, _ = self.cache.popitem(last=False)  # Remove from the beginning (oldest)
                oldest_path = os.path.join(self.temp_dir, oldest_file)
                if os.path.exists(oldest_path):
                    os.remove(oldest_path)
        except Exception as e:
            print(f"Error trimming cache: {str(e)}")
    
    def get(self, text, language):
        """Get audio file from cache if it exists.
        
        Args:
            text (str): The text to convert to speech
            language (str): The language code
            
        Returns:
            str or None: Full path to cached file if exists, None otherwise
        """
        if not self.cache_enabled:
            return None
            
        try:
            filename = self._generate_filename(text, language)
            file_path = os.path.join(self.temp_dir, filename)
            
            if filename in self.cache and os.path.exists(file_path):
                # Update access time
                self.cache.pop(filename)
                self.cache[filename] = time.time()
                return file_path
                
            return None
        except Exception as e:
            print(f"Error getting file from cache: {str(e)}")
            return None
    
    def put(self, text, language, file_path):
        """Add file to cache.
        
        Args:
            text (str): The text that was converted to speech
            language (str): The language code
            file_path (str): The full path to the audio file
        
        Returns:
            str: The full path to the cached file
        """
        if not self.cache_enabled:
            return file_path
            
        try:
            filename = self._generate_filename(text, language)
            cache_path = os.path.join(self.temp_dir, filename)
            
            # Only copy if it's not already at the cache path
            if file_path != cache_path:
                import shutil
                shutil.copy2(file_path, cache_path)
                
                # Remove the original temporary file
                os.remove(file_path)
            
            # Add/update in cache
            if filename in self.cache:
                self.cache.pop(filename)
            self.cache[filename] = time.time()
            
            # Trim cache if needed
            self._trim_cache()
            
            return cache_path
        except Exception as e:
            print(f"Error putting file in cache: {str(e)}")
            return file_path
    
    def cleanup(self, max_age=None):
        """Clean up old temporary files.
        
        Args:
            max_age (int, optional): Maximum age of files in seconds. 
                If None, uses config.CLEANUP_INTERVAL. Defaults to None.
        """
        if max_age is None:
            max_age = self.config.CLEANUP_INTERVAL
            
        try:
            current_time = time.time()
            
            # Check all files in temp directory
            for filename in os.listdir(self.temp_dir):
                if not filename.startswith('speech_') or not filename.endswith('.mp3'):
                    continue
                    
                file_path = os.path.join(self.temp_dir, filename)
                file_age = os.path.getmtime(file_path)
                
                # Remove file if older than max_age
                if current_time - file_age > max_age:
                    if filename in self.cache:
                        self.cache.pop(filename)
                    os.remove(file_path)
        except Exception as e:
            print(f"Error during cache cleanup: {str(e)}") 