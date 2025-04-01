import os
import secrets

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(16))
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'temp/uploads'
    JWT_ACCESS_TOKEN_EXPIRES = 86400  # 1 day in seconds
    API_PREFIX = '/api'
    LOG_LEVEL = 'INFO'
    TTS_CACHE_ENABLED = True
    TTS_CACHE_SIZE = 100  # Number of audio files to cache
    TEMP_AUDIO_DIR = 'temp_audio'
    CLEANUP_INTERVAL = 3600  # 1 hour in seconds
    
class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///agro_app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///agro_app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Use more secure settings for production
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    # Enable HTTPS
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_HTTPONLY = True
    # Rate limiting
    RATELIMIT_DEFAULT = "300 per day;100 per hour;5 per minute"
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'WARNING')

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_ENABLED = False
    PRESERVE_CONTEXT_ON_EXCEPTION = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get the current configuration."""
    env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default']) 