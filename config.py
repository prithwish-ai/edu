import os
import secrets
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    
    if env == 'production':
        # Production configuration
        return {
            'DATABASE_URL': os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/edu_spark'),
            'SECRET_KEY': os.getenv('SECRET_KEY', os.urandom(24)),
            'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB
            'UPLOAD_FOLDER': '/tmp/uploads',
            'TTS_CACHE_DIR': '/tmp/tts_cache',
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'AWS_BUCKET_NAME': os.getenv('AWS_BUCKET_NAME'),
            'AWS_REGION': os.getenv('AWS_REGION', 'us-east-1')
        }
    else:
        # Development configuration
        return {
            'DATABASE_URL': 'sqlite:///agro_app.db',
            'SECRET_KEY': 'dev-secret-key',
            'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB
            'UPLOAD_FOLDER': 'temp/uploads',
            'TTS_CACHE_DIR': 'temp/tts_cache',
            'AWS_ACCESS_KEY_ID': None,
            'AWS_SECRET_ACCESS_KEY': None,
            'AWS_BUCKET_NAME': None,
            'AWS_REGION': 'us-east-1'
        } 