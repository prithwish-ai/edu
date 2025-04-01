# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file, send_from_directory, after_this_request
from flask_session import Session
from flask_cors import CORS
import os
import jwt
import json
import time
import datetime
import requests
import threading
import hashlib
import secrets
import sqlite3
import base64
import tempfile
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image
import uuid
from dotenv import load_dotenv
from iti_app import (
    ChatBot, 
    ContextualModel, 
    YouTubeResourceManager, 
    MultilingualSupport, 
    ProgressTracker, 
    StudyMaterialGenerator, 
    QuizManager
)
import sys
# Add new imports for API backend
import jwt
from datetime import datetime, timedelta
import secrets
import hashlib
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
import time
from flask import after_this_request
import re

# Database setup
Base = declarative_base()

# User model
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(100), nullable=False)  # Will store hashed password
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    contacts = relationship("Contact", back_populates="user")
    uploads = relationship("Upload", back_populates="user")
    
    def __repr__(self):
        return f"<User {self.username}>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat()
        }

# Contact model
class Contact(Base):
    __tablename__ = 'contacts'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="contacts")
    
    def __repr__(self):
        return f"<Contact {self.name} - {self.email}>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'message': self.message,
            'created_at': self.created_at.isoformat()
        }

# Upload model for plant images
class Upload(Base):
    __tablename__ = 'uploads'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(255), nullable=False)
    filetype = Column(String(50), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Disease detection results
    disease = Column(String(255), nullable=True)
    confidence = Column(String(50), nullable=True)
    treatment = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="uploads")
    
    def __repr__(self):
        return f"<Upload {self.filename}>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'filepath': self.filepath,
            'filetype': self.filetype,
            'disease': self.disease,
            'confidence': self.confidence,
            'treatment': self.treatment,
            'created_at': self.created_at.isoformat()
        }

# Only try to load the TensorFlow model
model = None
model_type = None

# Look for TensorFlow model in models directory
tensorflow_model_path = os.path.join(os.path.dirname(__file__), "models", "plant_disease_prediction_model.h5")
try:
    if os.path.exists(tensorflow_model_path):
        print(f"Found TensorFlow model at {tensorflow_model_path}")
        # Use TensorFlow to load the model
        try:
            # Import here to avoid circular imports and only when needed
            from tensorflow.keras.models import load_model
            
            # Load the model
            model = load_model(tensorflow_model_path, compile=False)
            model_type = 'tensorflow'
            print(f"Successfully loaded TensorFlow plant disease model")
        except Exception as tf_err:
            print(f"Error loading TensorFlow model: {str(tf_err)}")
            model = None
    else:
        print(f"Warning: TensorFlow model not found at {tensorflow_model_path}. Will attempt to use PyTorch model instead.", file=sys.stderr)
except Exception as e:
    print(f"Warning: Failed to load plant disease model: {str(e)}. Will attempt to use other models if available.", file=sys.stderr)

# Import additional modules
from core import (
    admission_manager,
    context_manager,
    course_finder,
    exam_preparation,
    industry_connection,
    mentorship,
    plant_disease_detector,
    practical_assessment,
    progress_bar,
    progress_tracker,
    scholarship_manager,
    task_progress,
    trade_comparison,
    voice_manager,
    multimodal_manager  # Add the new multimodal manager
)
from core.market_predictor import MarketPredictor
import os
import json
from datetime import datetime, timedelta
import secrets
import tempfile
from werkzeug.utils import secure_filename
import random
import math

# Setup
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'ITI-EduSpark-SecretKey'

# Load environment variables
load_dotenv()

# Configure app settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Setup Flask Session
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)  # Initialize the Session
app.config['JWT_SECRET_KEY'] = secrets.token_hex(32)
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)
app.config['API_PREFIX'] = '/api'
app.config['YOUTUBE_API_KEY'] = os.getenv('YOUTUBE_API_KEY')

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agro_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
Base.metadata.create_all(engine)
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

# Close database sessions after each request
@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()

# Initialize the components
chatbot = ChatBot(continuous_learning=True)
context_model = ContextualModel()
youtube_manager = YouTubeResourceManager(api_key=app.config.get('YOUTUBE_API_KEY'))
multilingual = MultilingualSupport()
study_materials = StudyMaterialGenerator()
quiz_manager = QuizManager()

# Initialize additional components
admission_mgr = admission_manager.AdmissionManager()
context_mgr = context_manager.ContextManager()
course_finder_mgr = course_finder.CourseFinderManager()
exam_prep_mgr = exam_preparation.ExamPreparationManager()
industry_conn_mgr = industry_connection.IndustryConnectionManager()
market_pred_mgr = MarketPredictor()
mentorship_mgr = mentorship.MentorshipManager()
plant_disease_mgr = plant_disease_detector.PlantDiseaseDetector()
practical_assessment_mgr = practical_assessment.PracticalAssessmentManager()
progress_bar_mgr = progress_bar.MultiProgressBarManager()
progress_track_mgr = progress_tracker.MultiProgressTracker()
scholarship_mgr = scholarship_manager.ScholarshipManager()
trade_comparison_mgr = trade_comparison.TradeComparisonManager()
voice_mgr = voice_manager.VoiceInteractionManager()
multimodal_mgr = multimodal_manager.MultimodalManager()  # Initialize the multimodal manager

# Store active sessions
active_sessions = {}

# Helper functions for the API backend
def hash_password(password):
    """Hash a password for storing."""
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), 
                                salt, 100000)
    pwdhash = pwdhash.hex()
    return (salt + pwdhash).decode('ascii')

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    salt = stored_password[:64]
    stored_password = stored_password[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha512', 
                                  provided_password.encode('utf-8'), 
                                  salt.encode('ascii'), 
                                  100000)
    pwdhash = pwdhash.hex()
    return pwdhash == stored_password

def generate_token(user_id):
    """Generate JWT token for a user"""
    payload = {
        'exp': datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES'],
        'iat': datetime.utcnow(),
        'sub': user_id
    }
    return jwt.encode(
        payload,
        app.config['JWT_SECRET_KEY'],
        algorithm='HS256'
    )

def decode_token(auth_token):
    """Decode the JWT token"""
    try:
        payload = jwt.decode(
            auth_token,
            app.config['JWT_SECRET_KEY'],
            algorithms=['HS256']
        )
        return payload['sub']
    except jwt.ExpiredSignatureError:
        return 'Signature expired. Please log in again.'
    except jwt.InvalidTokenError:
        return 'Invalid token. Please log in again.'

def token_required(f):
    """Decorator to protect routes with JWT token"""
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Token is missing!'}), 401
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            user_id = decode_token(token)
            if isinstance(user_id, str) and user_id.startswith('Signature expired'):
                return jsonify({'message': user_id}), 401
            if isinstance(user_id, str) and user_id.startswith('Invalid token'):
                return jsonify({'message': user_id}), 401
            
            current_user = db_session.query(User).filter_by(id=user_id).first()
            if not current_user:
                return jsonify({'message': 'User not found!'}), 401
            
        except Exception as e:
            return jsonify({'message': str(e)}), 401
        
        return f(current_user, *args, **kwargs)
    
    # Rename the function to avoid Flask routing confusion
    decorated.__name__ = f.__name__
    return decorated

# Clean inactive sessions periodically
@app.before_request
def clean_inactive_sessions():
    """Remove sessions that have been inactive for more than 30 minutes."""
    current_time = datetime.now()
    sessions_to_remove = []
    
    for user_id, session_data in active_sessions.items():
        if (current_time - session_data['last_activity']).total_seconds() > 1800:  # 30 minutes
            sessions_to_remove.append(user_id)
    
    for user_id in sessions_to_remove:
        del active_sessions[user_id]

# Routes for main pages
@app.route('/')
def index():
    """Render the main landing page."""
    # Check for Hindi language preference
    user_language = session.get('app_language', 'en')
    if user_language == 'hi':
        return render_template('api_index_hindi.html')
    
    # Use api_index.html as the default English version
    return render_template('api_index.html')

@app.route('/home')
def home():
    """Render the home page."""
    # Simply redirect to index to avoid duplication
    return redirect(url_for('index'))

@app.route('/about')
def about():
    """Render the about page."""
    # For simplicity, we're keeping the about page in English regardless of language preference
    return render_template('api_index.html')

@app.route('/features')
def features():
    """Render the features page."""
    # For simplicity, we're keeping the features page in English regardless of language preference
    return render_template('api_index.html')

@app.route('/contact')
def contact():
    """Render the contact page."""
    return render_template('api_contact.html')

# Routes for user features
@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        user_message = request.json.get('message', '')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
    
        print(f"Received chat message: '{user_message}'")
        
        # Get the user's language preference from session
        user_language = session.get('app_language', 'en')
        print(f"User language preference: {user_language}")
        
        # Check if this is a request for videos/tutorials
        video_keywords = ['video', 'tutorial', 'watch', 'youtube', 'show me', 'how to']
        is_video_request = any(keyword in user_message.lower() for keyword in video_keywords)
        
        if is_video_request:
            print("Detected video request, searching for relevant videos...")
            # Extract the main topic from the message - use the message without the video keywords
            # This is a simple approach, could be improved with NLP
            topic = user_message
            for keyword in video_keywords:
                topic = topic.lower().replace(keyword, '').strip()
            
            # Get video recommendations
            try:
                max_results = 3
                videos = youtube_manager.search_videos(topic, max_results, user_language)
                
                if videos:
                    # Create response templates based on language
                    if user_language == 'hi':
                        intro_text = f"{topic} के बारे में कुछ वीडियो यहां हैं:"
                    elif user_language == 'bn':
                        intro_text = f"{topic} সম্পর্কে কিছু ভিডিও এখানে রয়েছে:"
                    elif user_language == 'te':
                        intro_text = f"{topic} గురించి కొన్ని వీడియోలు ఇక్కడ ఉన్నాయి:"
                    elif user_language == 'mr':
                        intro_text = f"{topic} बद्दल काही व्हिडिओ येथे आहेत:"
                    elif user_language == 'ta':
                        intro_text = f"{topic} பற்றிய சில வீடியோக்கள் இங்கே உள்ளன:"
                    else:
                        intro_text = f"Here are some videos about {topic}:"
                    
                    video_response = f"{intro_text}\n\n"
                    for i, video in enumerate(videos, 1):
                        video_response += f"{i}. {video.get('title', 'Video')} - {video.get('url', '#')}\n"
                    
                    # Translate response if not in English and translation is available
                    if user_language != 'en':
                        try:
                            # See if we need to translate titles
                            for video in videos:
                                if 'title' in video and multilingual.is_available:
                                    video['original_title'] = video['title']
                                    translated_title = multilingual.translate(video['title'], target_language=user_language)
                                    if translated_title:
                                        video['title'] = translated_title
                        except Exception as trans_err:
                            print(f"Error translating video titles: {str(trans_err)}")
                    
                    return jsonify({
                        'response': video_response,
                        'videos': videos,
                        'timestamp': datetime.now().isoformat(),
                        'language': user_language,
                        'is_video_response': True
                    })
                else:
                    # No videos found, continue with normal chat response
                    print("No videos found, continuing with normal response")
            except Exception as video_err:
                print(f"Error finding videos: {str(video_err)}")
                # Continue with normal chat response
        
        # First try to use the more sophisticated ChatBot instance
        try:
            print("Attempting to use the main ChatBot instance...")
            # Add context if the context model is available
            context = None
            extracted_entities = {}
            
            try:
                if 'context_model' in globals() and context_model is not None:
                    print("Using context model for additional context...")
                    context = context_model.add_context(user_message)
                    extracted_entities = context_model.extract_entities(user_message)
                    print(f"Extracted entities: {extracted_entities}")
            except Exception as context_err:
                print(f"Context processing error: {str(context_err)}")
            
            # Set language for this response if needed
            try:
                if 'chatbot' in globals() and chatbot is not None and hasattr(chatbot, 'set_language_preference'):
                    chatbot.set_language_preference(user_language)
                    print(f"Set chatbot language to: {user_language}")
            except Exception as lang_err:
                print(f"Error setting chatbot language: {str(lang_err)}")
            
            # Process the message with the chatbot - use the global chatbot instance
            print("Generating response with main ChatBot...")
            response = chatbot.generate_response(
                user_message,
                additional_context=[context] if context else None,
                extracted_entities=extracted_entities
            )
            
            print(f"Response from main ChatBot: '{response}'")
            
            # Return the response with timestamp and language
            return jsonify({
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'language': user_language
            })
        except Exception as chatbot_err:
            print(f"Main chatbot error: {str(chatbot_err)}")
            print("Falling back to simple ChatbotManager...")
            
            # Check if chatbot_manager is accessible
            global chatbot_manager
            if 'chatbot_manager' not in globals() or chatbot_manager is None:
                print("ChatbotManager was not initialized, creating a new instance")
                chatbot_manager = ChatbotManager()
                
            # Use the simple chatbot manager as fallback
            try:
                response = chatbot_manager.generate_response(user_message, user_language)
                print(f"Response from ChatbotManager: '{response}'")
                return jsonify({
                    'response': response,
                    'timestamp': datetime.now().isoformat(),
                    'language': user_language,
                    'note': 'Used fallback system'
                })
            except Exception as cm_err:
                print(f"ChatbotManager error: {str(cm_err)}")
                
                # Final fallback responses if everything else fails
                fallback_responses_en = [
                    "I understand your question about agriculture. Could you provide more details?",
                    "That's an interesting farming question. Let me help you with that.",
                    "I'm here to assist with agricultural education. Could you elaborate?",
                    "As your AgroEdu assistant, I can help with crops, farming techniques, and more."
                ]
                
                fallback_responses_hi = [
                    "मैं कृषि के बारे में आपका प्रश्न समझता हूं। क्या आप अधिक विवरण दे सकते हैं?",
                    "यह एक दिलचस्प खेती का प्रश्न है। मैं आपकी मदद करूंगा।",
                    "मैं कृषि शिक्षा में सहायता के लिए यहां हूं। क्या आप विस्तार कर सकते हैं?",
                    "आपके एग्रोएजु सहायक के रूप में, मैं फसलों, खेती तकनीकों और अधिक में मदद कर सकता हूं।"
                ]
                
                responses = fallback_responses_hi if user_language == 'hi' else fallback_responses_en
                fallback = random.choice(responses)
                print(f"Using emergency fallback response: '{fallback}'")
                
                return jsonify({
                    'response': fallback,
                    'timestamp': datetime.now().isoformat(),
                    'language': user_language,
                    'note': 'Used emergency fallback response'
                })
    except Exception as e:
        print(f"Error in chat route: {str(e)}")
        return jsonify({
            'response': "I apologize for the inconvenience. Our system is experiencing some issues. Please try again later.",
            'error': str(e),
            'language': request.json.get('language', session.get('app_language', 'en'))
        }), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Chat API endpoint for frontend."""
    try:
        print("Received request to /api/chat endpoint")
        data = request.json
        user_message = data.get('message', '')
        user_language = data.get('language', session.get('app_language', 'en'))
        
        print(f"User message: '{user_message}'")
        print(f"User language preference: {user_language}")
        
        # Validate language - if not supported, fall back to English
        supported_languages = ['en', 'hi', 'bn', 'ta', 'te', 'mr']
        if user_language not in supported_languages:
            print(f"Unsupported language: {user_language}, falling back to 'en'")
            user_language = 'en'
        
        # Check if this is a request for videos/tutorials
        video_keywords = ['video', 'tutorial', 'watch', 'youtube', 'show me', 'how to']
        is_video_request = any(keyword in user_message.lower() for keyword in video_keywords)
        
        if is_video_request:
            print("Detected video request, searching for relevant videos...")
            # Extract the main topic from the message - use the message without the video keywords
            # This is a simple approach, could be improved with NLP
            topic = user_message
            for keyword in video_keywords:
                topic = topic.lower().replace(keyword, '').strip()
            
            # Get video recommendations
            try:
                max_results = 3
                videos = youtube_manager.search_videos(topic, max_results, user_language)
                
                if videos:
                    # Create response templates based on language
                    if user_language == 'hi':
                        intro_text = f"{topic} के बारे में कुछ वीडियो यहां हैं:"
                    elif user_language == 'bn':
                        intro_text = f"{topic} সম্পর্কে কিছু ভিডিও এখানে রয়েছে:"
                    elif user_language == 'te':
                        intro_text = f"{topic} గురించి కొన్ని వీడియోలు ఇక్కడ ఉన్నాయి:"
                    elif user_language == 'mr':
                        intro_text = f"{topic} बद्दल काही व्हिडिओ येथे आहेत:"
                    elif user_language == 'ta':
                        intro_text = f"{topic} பற்றிய சில வீடியோக்கள் இங்கே உள்ளன:"
                    else:
                        intro_text = f"Here are some videos about {topic}:"
                    
                    video_response = f"{intro_text}\n\n"
                    for i, video in enumerate(videos, 1):
                        video_response += f"{i}. {video.get('title', 'Video')} - {video.get('url', '#')}\n"
                    
                    # Translate response if not in English and translation is available
                    if user_language != 'en':
                        try:
                            # See if we need to translate titles
                            for video in videos:
                                if 'title' in video and multilingual.is_available:
                                    video['original_title'] = video['title']
                                    translated_title = multilingual.translate(video['title'], target_language=user_language)
                                    if translated_title:
                                        video['title'] = translated_title
                        except Exception as trans_err:
                            print(f"Error translating video titles: {str(trans_err)}")
                    
                    return jsonify({
                        'response': video_response,
                        'videos': videos,
                        'timestamp': datetime.now().isoformat(),
                        'language': user_language,
                        'is_video_response': True
                    })
                else:
                    # No videos found, continue with normal chat response
                    print("No videos found, continuing with normal response")
            except Exception as video_err:
                print(f"Error finding videos: {str(video_err)}")
                # Continue with normal chat response
        
        try:
            # Prepare context and entities
            context = None
            extracted_entities = []
            
            try:
                if 'nlp_mgr' in globals() and nlp_mgr is not None:
                    entities = nlp_mgr.extract_entities(user_message)
                    if entities:
                        extracted_entities = entities
                        print(f"Extracted entities: {entities}")
                        
                    # Generate context based on message
                    context = nlp_mgr.generate_context(user_message)
                    if context:
                        print(f"Generated context: {context}")
            except Exception as context_err:
                print(f"Context processing error: {str(context_err)}")
            
            # Set language for this response if needed
            try:
                if 'chatbot' in globals() and chatbot is not None and hasattr(chatbot, 'set_language_preference'):
                    chatbot.set_language_preference(user_language)
                    print(f"Set chatbot language to: {user_language}")
            except Exception as lang_err:
                print(f"Error setting chatbot language: {str(lang_err)}")
            
            # Process the message with the chatbot - use the global chatbot instance
            print("Generating response with main ChatBot via API...")
            response = chatbot.generate_response(
                user_message,
                additional_context=[context] if context else None,
                extracted_entities=extracted_entities
            )
            
            print(f"Response from main ChatBot API: '{response}'")
            
            # Special handling for Bengali and other non-English, non-Hindi languages
            if user_language in ['bn', 'ta', 'te', 'mr'] and not is_translation_available(response, user_language):
                print(f"Translation needed for {user_language}")
                try:
                    # Try to use multilingual translation first if available
                    translated_response = None
                    try:
                        if 'multilingual' in globals() and multilingual is not None and hasattr(multilingual, 'translate'):
                            print("Attempting to use multilingual translation service...")
                            translated_response = multilingual.translate(response, target_language=user_language)
                            print(f"Multilingual translation result: '{translated_response[:50]}...'")
                    except Exception as multi_err:
                        print(f"Multilingual translation error: {str(multi_err)}")
                    
                    # Fallback to simple translation if multilingual failed or returned empty
                    if not translated_response:
                        print("Using fallback simple translation")
                        translated_response = translate_text(response, target_language=user_language)
                        print(f"Simple translation result: '{translated_response[:50]}...'")
                    
                    # Only update if we got a valid translation
                    if translated_response and translated_response != response:
                        response = translated_response
                        print(f"Using translated response: '{response[:100]}...'")
                    else:
                        print("Translation produced no changes, using original response")
                except Exception as trans_err:
                    print(f"All translation methods failed: {str(trans_err)}")
                    # Continue with original response if translation fails
            
            # Return the response with timestamp and language
            return jsonify({
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'language': user_language
            })
        except Exception as chatbot_err:
            print(f"Main chatbot API error: {str(chatbot_err)}")
            print("Falling back to simple ChatbotManager...")
            
            # Check if chatbot_manager is accessible
            global chatbot_manager
            if 'chatbot_manager' not in globals() or chatbot_manager is None:
                print("ChatbotManager was not initialized, creating a new instance")
                chatbot_manager = ChatbotManager()
            
            # Use the simple chatbot manager as fallback
            try:
                print(f"Generating response using ChatbotManager as fallback with language: {user_language}")
                response = chatbot_manager.generate_response(user_message, user_language)
                print(f"Response from ChatbotManager API fallback: '{response}'")
                
                # Special handling for Bengali and other non-English, non-Hindi languages
                if user_language in ['bn', 'ta', 'te', 'mr'] and not is_translation_available(response, user_language):
                    print(f"Translation needed for fallback response to {user_language}")
                    try:
                        # Try to use multilingual translation first if available
                        translated_response = None
                        try:
                            if 'multilingual' in globals() and multilingual is not None and hasattr(multilingual, 'translate'):
                                print("Attempting to use multilingual translation for fallback...")
                                translated_response = multilingual.translate(response, target_language=user_language)
                                print(f"Multilingual fallback result: '{translated_response[:50]}...'")
                        except Exception as multi_err:
                            print(f"Multilingual fallback translation error: {str(multi_err)}")
                        
                        # Fallback to simple translation if multilingual failed or returned empty
                        if not translated_response:
                            print("Using fallback simple translation for fallback response")
                            translated_response = translate_text(response, target_language=user_language)
                            print(f"Simple fallback translation result: '{translated_response[:50]}...'")
                        
                        # Only update if we got a valid translation
                        if translated_response and translated_response != response:
                            response = translated_response
                            print(f"Using translated fallback response: '{response[:100]}...'")
                        else:
                            print("Fallback translation produced no changes, using original response")
                    except Exception as trans_err:
                        print(f"All fallback translation methods failed: {str(trans_err)}")
                        # Continue with original response if translation fails
                
                return jsonify({
                    'response': response,
                    'timestamp': datetime.now().isoformat(),
                    'language': user_language,
                    'note': 'Used fallback system'
                })
            except Exception as cm_err:
                print(f"ChatbotManager API error: {str(cm_err)}")
                
                # Provide a fallback response if everything else fails
                fallback_responses = get_fallback_responses(user_language)
                fallback = random.choice(fallback_responses)
                print(f"Using fallback response in {user_language}: '{fallback}'")
            
                return jsonify({
                    'response': fallback,
                    'timestamp': datetime.now().isoformat(),
                    'language': user_language,
                    'note': 'Used emergency fallback response'
                })
    except Exception as e:
        print(f"Error in API chat route: {str(e)}")
        return jsonify({
            'response': "I apologize for the inconvenience. Our system is experiencing some issues. Please try again later.",
            'error': str(e),
            'language': request.json.get('language', session.get('app_language', 'en'))
        }), 500


def is_translation_available(text, language):
    """Check if the response already has a translation for the given language."""
    # Simple heuristic: Check if there are any characters from the target language script
    script_chars = {
        'bn': {'অ', 'আ', 'ই', 'ঈ', 'ক', 'খ', 'গ', 'ঙ'},  # Bengali characters
        'ta': {'அ', 'ஆ', 'இ', 'ஈ', 'க', 'ங', 'ச', 'ஞ'},  # Tamil characters
        'te': {'అ', 'ఆ', 'ఇ', 'ఈ', 'క', 'ఖ', 'గ', 'ఘ'},  # Telugu characters
        'mr': {'अ', 'आ', 'इ', 'ई', 'क', 'ख', 'ग', 'घ'}    # Marathi characters
    }
    
    if language not in script_chars:
        return False
    
    # Count characters from the target script
    char_count = sum(1 for char in text if char in script_chars[language])
    
    # If we have more than 5 characters in the target script, 
    # assume the text is already in that language
    return char_count > 5


def translate_text(text, target_language):
    """Simple function to translate text to target language."""
    try:
        import re  # Make sure re is imported

        # If we have a proper translation service, use it here
        # For now, we'll use a simple mapping for common phrases
        translations = {
            'bn': {  # Bengali
                'Hello': 'হ্যালো',
                'Hi': 'হাই',
                'Welcome': 'স্বাগতম',
                'Thank you': 'ধন্যবাদ',
                'Yes': 'হ্যাঁ',
                'No': 'না',
                'Help': 'সাহায্য',
                'Please': 'দয়া করে',
                'What': 'কি',
                'How': 'কিভাবে',
                'When': 'কখন',
                'Where': 'কোথায়',
                'Why': 'কেন',
                'Who': 'কে',
                'farming': 'চাষবাস',
                'agriculture': 'কৃষি',
                'crop': 'ফসল',
                'plant': 'গাছ',
                'disease': 'রোগ',
                'water': 'জল',
                'soil': 'মাটি',
                'seeds': 'বীজ',
                'fish': 'মাছ',
                'fin': 'ডানা'
            },
            'ta': {  # Tamil
                'Hello': 'வணக்கம்',
                'Hi': 'ஹாய்',
                'Welcome': 'வரவேற்கிறோம்',
                'Thank you': 'நன்றி',
                'Yes': 'ஆம்',
                'No': 'இல்லை',
                'Help': 'உதவி',
                'Please': 'தயவுசெய்து',
                'What': 'என்ன',
                'How': 'எப்படி',
                'When': 'எப்போது',
                'Where': 'எங்கே',
                'Why': 'ஏன்',
                'Who': 'யார்',
                'farming': 'விவசாயம்',
                'agriculture': 'வேளாண்மை',
                'crop': 'பயிர்',
                'plant': 'தாவரம்',
                'disease': 'நோய்',
                'water': 'நீர்',
                'soil': 'மண்',
                'seeds': 'விதைகள்',
                'fish': 'மீன்',
                'fin': 'துடுப்பு'
            },
            'te': {  # Telugu
                'Hello': 'హలో',
                'Hi': 'హాయ్',
                'Welcome': 'స్వాగతం',
                'Thank you': 'ధన్యవాదాలు',
                'Yes': 'అవును',
                'No': 'కాదు',
                'Help': 'సహాయం',
                'Please': 'దయచేసి',
                'What': 'ఏమిటి',
                'How': 'ఎలా',
                'When': 'ఎప్పుడు',
                'Where': 'ఎక్కడ',
                'Why': 'ఎందుకు',
                'Who': 'ఎవరు',
                'farming': 'వ్యవసాయం',
                'agriculture': 'వ్యవసాయం',
                'crop': 'పంట',
                'plant': 'మొక్క',
                'disease': 'వ్యాధి',
                'water': 'నీరు',
                'soil': 'నేల',
                'seeds': 'విత్తనాలు',
                'fish': 'చేప',
                'fin': 'రెక్క'
            },
            'mr': {  # Marathi
                'Hello': 'नमस्कार',
                'Hi': 'हाय',
                'Welcome': 'स्वागत आहे',
                'Thank you': 'धन्यवाद',
                'Yes': 'होय',
                'No': 'नाही',
                'Help': 'मदत',
                'Please': 'कृपया',
                'What': 'काय',
                'How': 'कसे',
                'When': 'कधी',
                'Where': 'कुठे',
                'Why': 'का',
                'Who': 'कोण',
                'farming': 'शेती',
                'agriculture': 'कृषी',
                'crop': 'पिक',
                'plant': 'रोपे',
                'disease': 'रोग',
                'water': 'पाणी',
                'soil': 'माती',
                'seeds': 'बिया',
                'fish': 'मासा',
                'fin': 'पंख'
            }
        }
        
        if target_language not in translations:
            print(f"Target language '{target_language}' not supported for translation")
            return text  # Return original if language not supported
            
        # Simple word replacement
        result = text
        for eng, trans in translations[target_language].items():
            # Replace whole words only (with word boundaries)
            pattern = r'\b' + re.escape(eng) + r'\b'
            result = re.sub(pattern, trans, result, flags=re.IGNORECASE)
            
        print(f"Original: '{text}'\nTranslated: '{result}'")
        return result
        
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text  # Return original on error


def get_fallback_responses(language):
    """Get language-specific fallback responses."""
    fallback_responses = {
        'en': [
            "I understand your question about agricultural topics. Could you provide more details so I can help better?",
            "That's an interesting agricultural question. Let me think about how best to help you with that.",
            "I'm here to assist with farming and agricultural education. Could you elaborate on your question?",
            "As your AgroEdu assistant, I can provide information on crops, farming techniques, and more. What specific information are you looking for?"
        ],
        'hi': [
            "मैं कृषि विषयों के बारे में आपके प्रश्न को समझता हूं। क्या आप बेहतर मदद करने के लिए अधिक विवरण प्रदान कर सकते हैं?",
            "यह एक दिलचस्प कृषि प्रश्न है। मुझे सोचने दें कि आपकी मदद कैसे की जाए।",
            "मैं खेती और कृषि शिक्षा के साथ सहायता करने के लिए यहां हूं। क्या आप अपने प्रश्न पर विस्तार कर सकते हैं?",
            "आपके एग्रोएजु सहायक के रूप में, मैं फसलों, खेती तकनीकों और अधिक पर जानकारी प्रदान कर सकता हूं। आप किस विशिष्ट जानकारी की तलाश कर रहे हैं?"
        ],
        'bn': [
            "আমি কৃষি বিষয়ে আপনার প্রশ্ন বুঝতে পারি। আপনি কি আরও বিস্তারিত তথ্য দিতে পারেন যাতে আমি আরও ভালভাবে সাহায্য করতে পারি?",
            "এটি একটি আকর্ষণীয় কৃষি প্রশ্ন। আমি চিন্তা করি কিভাবে আপনাকে সাহায্য করা যায়।",
            "আমি কৃষি এবং কৃষি শিক্ষা সম্পর্কে সাহায্য করতে এখানে আছি। আপনি কি আপনার প্রশ্ন সম্পর্কে আরও বিস্তারিত বলতে পারেন?",
            "আপনার এগ্রোএডু সহায়ক হিসাবে, আমি ফসল, কৃষি কৌশল, এবং আরও অনেক কিছু সম্পর্কে তথ্য প্রদান করতে পারি। আপনি কোন নির্দিষ্ট তথ্য খুঁজছেন?"
        ],
        'ta': [
            "விவசாய தலைப்புகள் பற்றிய உங்கள் கேள்வியை நான் புரிந்துகொள்கிறேன். நான் சிறப்பாக உதவ முடியும் என்பதற்கு நீங்கள் மேலும் விவரங்களை வழங்க முடியுமா?",
            "அது ஒரு சுவாரஸ்யமான விவசாய கேள்வி. அதற்கு எப்படி சிறப்பாக உதவுவது என்பதை நான் யோசிக்கிறேன்.",
            "நான் விவசாயம் மற்றும் விவசாய கல்வியுடன் உதவ இங்கே இருக்கிறேன். உங்கள் கேள்வியை விரிவுபடுத்த முடியுமா?",
            "உங்கள் AgroEdu உதவியாளராக, நான் பயிர்கள், விவசாய நுட்பங்கள் மற்றும் பலவற்றைப் பற்றிய தகவல்களை வழங்க முடியும். எந்த குறிப்பிட்ட தகவலை நீங்கள் தேடுகிறீர்கள்?"
        ],
        'te': [
            "వ్యవసాయ అంశాల గురించి మీ ప్రశ్నను నేను అర్థం చేసుకున్నాను. నేను మెరుగ్గా సహాయం చేయగలిగేలా మీరు మరిన్ని వివరాలను అందించగలరా?",
            "అది ఒక ఆసక్తికరమైన వ్యవసాయ ప్రశ్న. దానితో మీకు ఎలా సహాయం చేయాలో ఆలోచిస్తున్నాను.",
            "నేను వ్యవసాయం మరియు వ్యవసాయ విద్యతో సహాయం చేయడానికి ఇక్కడ ఉన్నాను. మీ ప్రశ్నను వివరించగలరా?",
            "మీ AgroEdu సహాయకుడిగా, నేను పంటలు, వ్యవసాయ పద్ధతులు మరియు మరిన్ని వాటి గురించి సమాచారాన్ని అందించగలను. మీరు ఏ నిర్దిష్ట సమాచారం కోసం చూస్తున్నారు?"
        ],
        'mr': [
            "मी शेतीविषयक प्रश्नांबद्दल आपला प्रश्न समजू शकतो. मी अधिक चांगल्या प्रकारे मदत करू शकेन अशी अधिक माहिती आपण देऊ शकता का?",
            "हा एक रंजक कृषी प्रश्न आहे. त्यासाठी कशी मदत करावी याचा मी विचार करतो.",
            "मी शेती आणि कृषी शिक्षणासाठी मदत करण्यासाठी येथे आहे. तुम्ही तुमच्या प्रश्नाबद्दल अधिक माहिती देऊ शकता का?",
            "तुमचा AgroEdu सहाय्यक म्हणून, मी पिके, शेती तंत्रे आणि अधिक माहिती देऊ शकतो. आपण कोणती विशिष्ट माहिती शोधत आहात?"
        ]
    }
    
    # Return appropriate language responses or English as fallback
    return fallback_responses.get(language, fallback_responses['en'])

@app.route('/api/multimodal-chat', methods=['POST'])
def multimodal_chat():
    """Endpoint for multimodal chat (text + images)."""
    try:
        print("Received request to /api/multimodal-chat endpoint")
        data = request.json
        user_message = data.get('message', '')
        image_data = data.get('image')
        
        print(f"Message: '{user_message}'")
        print(f"Image data received: {'Yes' if image_data else 'No'}")
        
        # Track if we processed an image
        image_processed = False
        image_path = None
        image_description = None
        
        # Process image if provided
        if image_data:
            print("Processing image data...")
            try:
                image_result = multimodal_mgr.process_image(image_data)
                print(f"Image processing result: {image_result}")
                
                if "error" not in image_result:
                    image_processed = True
                    image_path = image_result["filepath"]
                    print(f"Successfully processed image to path: {image_path}")
                    
                    # Generate a description of the image
                    try:
                        image_description = multimodal_mgr.generate_image_description(image_path)
                        print(f"Generated image description: {image_description}")
                    except Exception as desc_err:
                        print(f"Error generating image description: {str(desc_err)}")
                        image_description = "Image uploaded by user"
                else:
                    error_msg = image_result["error"]
                    print(f"Error processing image: {error_msg}")
                    # Return the error to the client
                    return jsonify({
                        'type': 'text',
                        'response': f"I encountered an error processing your image: {error_msg}. Please try again with a different image.",
                        'timestamp': datetime.now().isoformat(),
                        'error': error_msg
                    }), 200  # Return 200 to avoid breaking UI
            except Exception as img_err:
                print(f"Exception during image processing: {str(img_err)}")
                return jsonify({
                    'type': 'text',
                    'response': f"I encountered an error processing your image: {str(img_err)}. Please try again with a different image.",
                    'timestamp': datetime.now().isoformat(),
                    'error': str(img_err)
                }), 200
        
        # Process the message with additional context from the image if available
        try:
            # For the main chatbot, add the image context
            additional_context = [f"[Image description: {image_description}]"] if image_description else None
            
            # Prepare for response generation
            print(f"Sending to chatbot - Text: '{user_message}', Image path: {image_path if image_processed else 'None'}")
            
            if not user_message and image_path:
                # If no message but there is an image, create a default query
                user_message = "What can you tell me about this image?"
                print(f"No text query provided, using default: '{user_message}'")
            
            # Pass both the text and image to the generate_response method
            response = chatbot.generate_response(
                user_message, 
                additional_context=additional_context,
                image_path=image_path if image_processed else None
            )
                
            print(f"Response from multimodal chat: '{response}'")
            
            # Create the response - this could include text and/or images
            response_data = {
                'type': 'text',
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
            
            # If we processed an image, include it in the response
            if image_processed:
                response_data['image_processed'] = True
                response_data['image_description'] = image_description
                
                # Include the uploaded image path for reference
                response_data['image_path'] = image_path
            
            return jsonify(response_data)
            
        except Exception as mm_err:
            print(f"Multimodal chat error: {str(mm_err)}")
            import traceback
            traceback.print_exc()  # Print full stack trace for debugging
            
            # Fall back to the simple chatbot manager if the main one fails
            try:
                response = chatbot_manager.generate_response(user_message, user_language)
                
                if image_processed:
                    response = f"I processed your image, but encountered an error when analyzing it: {str(mm_err)}. {response}"
                
                return jsonify({
                    'type': 'text',
                    'response': response,
                    'timestamp': datetime.now().isoformat(),
                    'note': 'Used fallback system',
                    'error': str(mm_err)
                })
            except Exception as fallback_err:
                print(f"Fallback chatbot error: {str(fallback_err)}")
                return jsonify({
                    'type': 'text',
                    'response': f"I'm sorry, but I'm having trouble processing your request. Please try again later.",
                    'timestamp': datetime.now().isoformat(),
                    'error': f"Main error: {str(mm_err)}. Fallback error: {str(fallback_err)}"
                })
    except Exception as e:
        print(f"Error in multimodal chat route: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging
        
        return jsonify({
            'response': "I apologize for the inconvenience. Our system is experiencing some issues processing your multimodal message. Please try again later.",
            'error': str(e)
        }), 500

# Initialize the market data manager
market_data_manager = MarketPredictor()

# Update market_data endpoint to use the new manager
@app.route('/market_data', methods=['POST'])
def get_market_data():
    """Handle market data requests."""
    try:
        data = request.get_json()
        crop = data.get('crop', '')
        location = data.get('location', None)
        
        if not crop:
            return jsonify({
                'error': 'Crop name is required',
                'status': 'error'
            }), 400
            
        # Initialize market predictor if not already done
        if not hasattr(app, 'market_predictor'):
            app.market_predictor = MarketPredictor()
            
        # Try to fetch market data
        market_data = app.market_predictor.fetch_market_data(crop, location)
        
        # If API call fails, provide fallback data
        if market_data.get('status') == 'error':
            # Generate fallback data with reasonable values
            fallback_data = {
                'crop': crop,
                'location': location or 'Multiple',
                'price': {
                    'Rice': 40, 'Wheat': 30, 'Maize': 25, 'Cotton': 80,
                    'Potato': 20, 'Onion': 35, 'Tomato': 30, 'Soybean': 45,
                    'Sugarcane': 3, 'Groundnut': 75, 'Pulses': 70
                }.get(crop, 50),
                'unit': 'Kg',
                'timestamp': datetime.now().isoformat(),
                'source': 'fallback_data',
                'status': 'success',
                'note': 'Using fallback data - real-time data unavailable'
            }
            return jsonify(fallback_data)
            
        return jsonify(market_data)
        
    except Exception as e:
        print(f"Error in market_data route: {str(e)}")
        return jsonify({
            'error': 'Could not retrieve market data at this time. The service might be temporarily unavailable.',
            'status': 'error'
        }), 500

# Add a rule to handle incorrect URL case sensitivity
@app.before_request
def lowercase_request_path():
    """Convert all incoming request paths to lowercase to avoid case sensitivity issues."""
    if request.path != request.path.lower():
        return redirect(request.path.lower(), code=301)

# Define aliases for common routes to handle potential typos or case sensitivity
@app.route('/predictcropprices', methods=['POST'])
@app.route('/predict-crop-prices', methods=['POST'])
@app.route('/Predict_Crop_Prices', methods=['POST'])
def predict_crop_prices_alias():
    """Alias routes for predict_crop_prices."""
    return predict_crop_prices()

# Add fallback data for when the market data is unavailable
FALLBACK_CROP_DATA = {
    "rice": {
        "price": 45.23,
        "demand": "High",
        "trend": "rising",
        "location": "National Average"
    },
    "wheat": {
        "price": 28.75,
        "demand": "Medium",
        "trend": "stable",
        "location": "National Average"
    },
    "corn": {
        "price": 22.50,
        "demand": "Medium",
        "trend": "rising",
        "location": "National Average"
    },
    "potato": {
        "price": 18.40,
        "demand": "High",
        "trend": "rising",
        "location": "National Average"
    },
    "tomato": {
        "price": 35.25,
        "demand": "Medium",
        "trend": "falling",
        "location": "National Average"
    },
    "onion": {
        "price": 28.90,
        "demand": "High",
        "trend": "stable",
        "location": "National Average"
    }
}

@app.route('/predict_crop_prices', methods=['POST'])
def predict_crop_prices():
    """Predict future crop prices."""
    data = request.json
    crop = data.get('crop')
    days_ahead = data.get('days_ahead', 7)
    
    if not crop:
        return jsonify({'error': 'Crop name required'}), 400
    
    try:
        # Try to get price predictions from the manager
        try:
            prediction = market_data_manager.predict_price(crop, days_ahead)
            
            # Check if the response contains an error
            if prediction.get('status') == 'error' or 'error' in prediction:
                error_msg = prediction.get('error', 'Could not generate price prediction')
                print(f"Error predicting prices: {error_msg}, falling back to synthetic data")
                raise Exception(error_msg)  # Trigger fallback
            
            # Ensure data is in the format expected by frontend
            if 'forecast' not in prediction and 'predictions' in prediction:
                # Convert predictions format if needed
                forecast = []
                for pred in prediction['predictions']:
                    forecast_item = {
                        'date': pred.get('date', ''),
                        'price': pred.get('predicted_price', 0),
                        'price_change': 0,
                        'confidence': 'Medium'
                    }
                    # Map confidence value if available
                    conf = pred.get('confidence', 0)
                    if isinstance(conf, (int, float)):
                        if conf > 0.7:
                            forecast_item['confidence'] = 'High'
                        elif conf < 0.4:
                            forecast_item['confidence'] = 'Low'
                        else:
                            forecast_item['confidence'] = 'Medium'
                    forecast.append(forecast_item)
                
                # Calculate price changes
                if forecast and len(forecast) > 1 and 'current_price' in prediction:
                    current_price = prediction['current_price']
                    for i, item in enumerate(forecast):
                        if i == 0:
                            # First day compared to current price
                            item['price_change'] = ((item['price'] - current_price) / current_price) * 100 if current_price else 0
                        else:
                            # Other days compared to previous day
                            prev_price = forecast[i-1]['price']
                            item['price_change'] = ((item['price'] - prev_price) / prev_price) * 100 if prev_price else 0
                
                prediction['forecast'] = forecast
            
            # Add recommendation if not present
            if 'recommendation' not in prediction and 'price_trend' in prediction:
                trend = prediction.get('price_trend')
                if trend == 'increasing':
                    prediction['recommendation'] = "Prices are trending upward. Consider holding your crop for a few more days for better returns."
                elif trend == 'decreasing':
                    prediction['recommendation'] = "Prices are trending downward. It may be advisable to sell soon to avoid further price decreases."
                else:
                    prediction['recommendation'] = "Prices are relatively stable. Monitor the market closely for any significant changes."
                    
            return jsonify(prediction)
        
        except Exception as inner_e:
            print(f"Error from market_predictor: {str(inner_e)}, generating synthetic forecast")
            
            # Generate synthetic forecast data
            crop_lower = crop.lower()
            
            # Get base price from fallback data or use a default
            base_price = 50.0
            trend = "stable"
            if crop_lower in FALLBACK_CROP_DATA:
                base_price = FALLBACK_CROP_DATA[crop_lower]["price"]
                trend = FALLBACK_CROP_DATA[crop_lower]["trend"]
            
            # Create synthetic forecast
            forecast = []
            current_date = datetime.now()
            
            # Set trend factor based on trend direction
            if trend == "rising":
                trend_factor = random.uniform(0.005, 0.02)  # 0.5% to 2% daily increase
            elif trend == "falling":
                trend_factor = random.uniform(-0.02, -0.005)  # 0.5% to 2% daily decrease
            else:
                trend_factor = random.uniform(-0.005, 0.005)  # -0.5% to 0.5% daily fluctuation
            
            # Generate daily forecasts
            current_price = base_price
            for i in range(int(days_ahead)):
                # Calculate date
                forecast_date = current_date + timedelta(days=i)
                date_str = forecast_date.strftime("%d %b")
                
                # Add random fluctuation to trend
                daily_change = trend_factor + random.uniform(-0.01, 0.01)
                
                # Calculate new price
                new_price = current_price * (1 + daily_change)
                price_change = ((new_price - current_price) / current_price) * 100
                
                # Set confidence based on how far in the future
                if i < 3:
                    confidence = "High"
                elif i < 7:
                    confidence = "Medium"
                else:
                    confidence = "Low"
                
                # Add to forecast
                forecast.append({
                    "date": date_str,
                    "price": new_price,
                    "price_change": price_change,
                    "confidence": confidence
                })
                
                # Update current price for next iteration
                current_price = new_price
            
            # Generate recommendation based on trend
            if trend == "rising":
                recommendation = "Prices are trending upward. Consider holding your crop for a few more days for better returns. (Note: This is simulated data)"
            elif trend == "falling":
                recommendation = "Prices are trending downward. It may be advisable to sell soon to avoid further price decreases. (Note: This is simulated data)"
            else:
                recommendation = "Prices are relatively stable. Monitor the market closely for any significant changes. (Note: This is simulated data)"
            
            synthetic_prediction = {
                "status": "success",
                "crop": crop,
                "current_price": base_price,
                "forecast": forecast,
                "recommendation": recommendation,
                "note": "This is synthesized data as real-time market data is temporarily unavailable."
            }
            
            return jsonify(synthetic_prediction)
    
    except Exception as e:
        print(f"Error in predict_crop_prices route: {str(e)}")
        return jsonify({
            'error': 'Could not generate price prediction at this time. The service might be temporarily unavailable.',
            'message': str(e),
            'crop': crop
        }), 200  # Return 200 to avoid breaking UI

@app.route('/setup_price_alerts', methods=['POST'])
def setup_price_alerts():
    """Set up price alerts for crops."""
    if 'user_id' not in session:
        return jsonify({'error': 'No active session'}), 401
    
    data = request.json
    crop = data.get('crop')
    farmer_id = data.get('farmer_id', session['user_id'])
    contact_info = data.get('contact_info', {})
    
    if not crop:
        return jsonify({'error': 'Crop name required'}), 400
    
    user_session = active_sessions.get(session['user_id'])
    if not user_session:
        return jsonify({'error': 'Session expired'}), 401
    
    user_chatbot = user_session['chatbot']
    user_session['last_activity'] = datetime.now()
    
    try:
        alert_setup = user_chatbot.setup_price_alerts(crop, farmer_id, contact_info)
        return jsonify(alert_setup)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_youtube_recommendations', methods=['POST'])
def get_youtube_recommendations():
    """Get YouTube video recommendations."""
    data = request.json
    query = data.get('query')
    max_results = data.get('max_results', 5)
    language = data.get('language', 'en')
    
    if not query:
        return jsonify({'error': 'Search query required'}), 400
    
    # Update session activity if user is logged in
    if 'user_id' in session:
        user_session = active_sessions.get(session['user_id'])
        if user_session:
            user_session['last_activity'] = datetime.now()
    
    try:
        results = youtube_manager.search_videos(query, max_results, language)
        return jsonify({'videos': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate_text_route():
    """Translate text to different languages."""
    if 'user_id' not in session:
        return jsonify({'error': 'No active session'}), 401
    
    text = request.json.get('text')
    target_language = request.json.get('target_language', 'en')
    source_language = request.json.get('source_language')
    
    if not text:
        return jsonify({'error': 'Text required'}), 400
    
    user_session = active_sessions.get(session['user_id'])
    if not user_session:
        user_session['last_activity'] = datetime.now()
    
    try:
        # First try using the multilingual system
        try:
            if multilingual and hasattr(multilingual, 'translate'):
                translated = multilingual.translate(text, target_language, source_language)
                print(f"Multilingual translated: '{text}' to '{translated}'")
                return jsonify({
                    'translated_text': translated,
                    'source_language': source_language or multilingual.detect_language(text),
                    'target_language': target_language
                })
        except Exception as multi_err:
            print(f"Multilingual translation failed: {str(multi_err)}, falling back to basic translation")
        
        # Fallback to the basic translation function
        translated = translate_text(text, target_language)
        print(f"Basic translation: '{text}' to '{translated}'")
        return jsonify({
            'translated_text': translated,
            'source_language': source_language or 'en',
            'target_language': target_language
        })
    except Exception as e:
        print(f"All translation methods failed: {str(e)}")
        return jsonify({'error': str(e), 'original_text': text}), 500

@app.route('/detect_language', methods=['POST'])
def detect_language():
    """Detect the language of text."""
    text = request.json.get('text')
    
    if not text:
        return jsonify({'error': 'Text required'}), 400
    
    try:
        language = multilingual.detect_language(text)
        return jsonify({'detected_language': language})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/supported_languages')
def get_supported_languages():
    """Get a list of supported languages."""
    try:
        languages = multilingual.get_supported_languages()
        return jsonify({'languages': languages})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_learning_stats')
def get_learning_stats():
    """Get user's learning statistics."""
    if 'user_id' not in session:
        return jsonify({'error': 'No active session'}), 401
    
    user_session = active_sessions.get(session['user_id'])
    if not user_session:
        return jsonify({'error': 'Session expired'}), 401
    
    user_chatbot = user_session['chatbot']
    user_session['last_activity'] = datetime.now()
    
    try:
        stats = user_chatbot.get_learning_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logout')
def logout():
    """Logout the user and clean up their session."""
    if 'user_id' in session:
        user_id = session['user_id']
        if user_id in active_sessions:
            del active_sessions[user_id]
        session.pop('user_id', None)
    return redirect(url_for('index'))

# Additional routes for new modules
@app.route('/admissions', methods=['GET', 'POST'])
def admissions():
    """Handle admission applications and information."""
    if request.method == 'POST':
        application_data = request.json
        result = admission_mgr.process_application(application_data)
        return jsonify(result)
    else:
        programs = admission_mgr.get_available_programs()
        return render_template('api_dashboard.html')

@app.route('/course_finder', methods=['POST'])
def find_courses():
    """Find suitable courses based on criteria."""
    criteria = request.json
    courses = course_finder_mgr.find_courses(criteria)
    return jsonify({'courses': courses})

@app.route('/exam_preparation', methods=['GET', 'POST'])
def exam_preparation():
    """Get exam preparation materials."""
    if request.method == 'POST':
        exam_details = request.json
        materials = exam_prep_mgr.generate_preparation_materials(exam_details)
        return jsonify(materials)
    else:
        exams = exam_prep_mgr.get_available_exams()
        return render_template('api_dashboard.html')

@app.route('/industry_connections', methods=['POST'])
def get_industry_connections():
    """Get industry connection opportunities."""
    if 'user_id' not in session:
        return jsonify({'error': 'No active session'}), 401
    
    data = request.json
    category = data.get('category')  # apprenticeships, job_openings, etc.
    trade = data.get('trade')
    location = data.get('location')
    
    user_session = active_sessions.get(session['user_id'])
    if not user_session:
        return jsonify({'error': 'Session expired'}), 401
    
    user_chatbot = user_session['chatbot']
    user_session['last_activity'] = datetime.now()
    
    try:
        try:
            # Try using the industry connections manager if available
            if industry_conn_mgr and hasattr(industry_conn_mgr, 'get_opportunities'):
                opportunities = industry_conn_mgr.get_opportunities(category, trade, location)
                return jsonify({'opportunities': opportunities})
            
            # Try using chatbot if it has the method
            if hasattr(user_chatbot, 'get_industry_opportunities'):
                opportunities = user_chatbot.get_industry_opportunities(category, trade, location)
                return jsonify({'opportunities': opportunities})
        except Exception as conn_err:
            print(f"Industry connections error: {str(conn_err)}")
        
        # Fallback: Return sample industry opportunities
        fallback_opportunities = [
            {
                'title': 'Apprenticeship Program in Sustainable Farming',
                'organization': 'Green Earth Farms',
                'category': category or 'apprenticeships',
                'location': location or 'Various Locations',
                'type': trade or 'General Agriculture',
                'description': 'Learn sustainable farming techniques through hands-on experience.',
                'requirements': ['Basic knowledge of agriculture', 'Willingness to learn'],
                'duration': '6 months',
                'compensation': 'Stipend + Accommodations',
                'application_deadline': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
            },
            {
                'title': 'Agricultural Technician',
                'organization': 'AgriTech Solutions',
                'category': category or 'job_openings',
                'location': location or 'Remote/Various',
                'type': trade or 'Technical',
                'description': 'Apply modern technology to improve agricultural productivity.',
                'requirements': ['Experience with agricultural equipment', 'Technical aptitude'],
                'salary_range': '$30,000 - $45,000 per year',
                'application_deadline': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
            },
            {
                'title': 'Modern Irrigation Workshop',
                'organization': 'Water Conservation Institute',
                'category': category or 'training_programs',
                'location': location or 'Online',
                'type': trade or 'Irrigation',
                'description': 'Learn about the latest irrigation technologies and water conservation techniques.',
                'duration': '2 weeks',
                'cost': 'Free',
                'start_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
            }
        ]
        
        return jsonify({
            'opportunities': fallback_opportunities,
            'note': 'These are sample opportunities for demonstration purposes.'
        })
    except Exception as e:
        print(f"General error in industry_connections route: {str(e)}")
        return jsonify({
            'error': 'Could not retrieve industry opportunities at this time',
            'message': str(e)
        }), 200  # Return 200 to avoid breaking UI

@app.route('/mentorship', methods=['POST'])
def get_mentorship():
    """Get mentorship information or request a mentor."""
    if 'user_id' not in session:
        return jsonify({'error': 'No active session'}), 401
    
    data = request.json
    action = data.get('action', 'information')  # information, request, etc.
    specialization = data.get('specialization')
    experience_level = data.get('experience_level', 'beginner')
    goals = data.get('goals', [])
    
    user_session = active_sessions.get(session['user_id'])
    if not user_session:
        return jsonify({'error': 'Session expired'}), 401
    
    user_chatbot = user_session['chatbot']
    user_session['last_activity'] = datetime.now()
    
    try:
        try:
            # Try using the mentorship manager if available
            if mentorship_mgr:
                if action == 'request' and hasattr(mentorship_mgr, 'request_mentor'):
                    result = mentorship_mgr.request_mentor(specialization, experience_level, goals)
                    return jsonify(result)
                elif action == 'information' and hasattr(mentorship_mgr, 'get_mentorship_info'):
                    info = mentorship_mgr.get_mentorship_info(specialization)
                    return jsonify(info)
                elif hasattr(mentorship_mgr, 'get_mentors'):
                    mentors = mentorship_mgr.get_mentors(specialization, experience_level)
                    return jsonify({'mentors': mentors})
            
            # Try using chatbot if it has the method
            if hasattr(user_chatbot, 'get_mentorship_information'):
                info = user_chatbot.get_mentorship_information(action, specialization, experience_level, goals)
                return jsonify(info)
        except Exception as mentorship_err:
            print(f"Mentorship error: {str(mentorship_err)}")
        
        # Fallback: Return sample mentorship information
        if action == 'request':
            fallback_request = {
                'request_id': f"MENT-{random.randint(1000, 9999)}",
                'status': 'pending',
                'specialization': specialization or 'General Agriculture',
                'message': 'Your mentorship request has been received. We will match you with a suitable mentor shortly.',
                'next_steps': [
                    'You will receive a notification when a mentor is assigned',
                    'Prepare specific questions for your first session',
                    'Consider your learning goals and areas for growth'
                ],
                'note': 'This is a sample response for demonstration purposes.'
            }
            return jsonify(fallback_request)
        else:  # information or default
            fallback_mentors = [
                {
                    'name': 'Dr. Sarah Johnson',
                    'specialization': specialization or 'Sustainable Farming',
                    'experience': '15 years',
                    'bio': 'Expert in organic farming techniques with a PhD in Agricultural Sciences.',
                    'availability': 'Weekdays, 10AM-2PM',
                    'rating': 4.9,
                    'languages': ['English', 'Spanish']
                },
                {
                    'name': 'Robert Chen',
                    'specialization': specialization or 'Crop Management',
                    'experience': '8 years',
                    'bio': 'Agricultural engineer specializing in efficient crop management systems.',
                    'availability': 'Evenings and weekends',
                    'rating': 4.7,
                    'languages': ['English', 'Chinese']
                },
                {
                    'name': 'Maria Garcia',
                    'specialization': specialization or 'Livestock Management',
                    'experience': '12 years',
                    'bio': 'Experienced rancher with expertise in sustainable livestock practices.',
                    'availability': 'Weekends only',
                    'rating': 4.8,
                    'languages': ['English', 'Portuguese']
                }
            ]
            
            fallback_info = {
                'mentors': fallback_mentors,
                'program_details': {
                    'description': 'Our mentorship program connects novice farmers with experienced professionals for personalized guidance.',
                    'benefits': [
                        'One-on-one guidance from industry experts',
                        'Customized learning plans based on your goals',
                        'Access to exclusive resources and networks',
                        'Practical advice for immediate implementation'
                    ],
                    'duration': '3 months standard program',
                    'process': [
                        'Submit request with your areas of interest',
                        'Get matched with an appropriate mentor',
                        'Schedule your first virtual meeting',
                        'Set goals and create a learning plan',
                        'Regular check-ins and progress tracking'
                    ]
                },
                'note': 'This is sample information for demonstration purposes.'
            }
            return jsonify(fallback_info)
    except Exception as e:
        print(f"General error in mentorship route: {str(e)}")
        return jsonify({
            'error': 'Could not process mentorship request at this time',
            'message': str(e)
        }), 200  # Return 200 to avoid breaking UI

@app.route('/trade_comparison', methods=['POST'])
def compare_trades():
    """Compare multiple trades side by side."""
    if 'user_id' not in session:
        return jsonify({'error': 'No active session'}), 401
    
    data = request.json
    trade_names = data.get('trade_names', [])
    
    if not trade_names or not isinstance(trade_names, list) or len(trade_names) < 1:
        return jsonify({'error': 'At least one trade name is required'}), 400
    
    user_session = active_sessions.get(session['user_id'])
    if not user_session:
        return jsonify({'error': 'Session expired'}), 401
    
    user_chatbot = user_session['chatbot']
    user_session['last_activity'] = datetime.now()
    
    try:
        try:
            # Try using the trade comparison manager if available
            if trade_comparison_mgr and hasattr(trade_comparison_mgr, 'compare_trades'):
                comparison = trade_comparison_mgr.compare_trades(trade_names)
                return jsonify(comparison)
            
            # Try using chatbot if it has the method
            if hasattr(user_chatbot, 'compare_trades'):
                comparison = user_chatbot.compare_trades(trade_names)
                return jsonify(comparison)
        except Exception as comp_err:
            print(f"Trade comparison error: {str(comp_err)}")
        
        # Fallback: Return sample trade comparison
        agricultural_trades = {
            "Crop Management": {
                "duration": "6 months",
                "certification": "Agricultural Operations Certificate",
                "skill_level": "Intermediate",
                "job_prospects": "High demand in farming regions",
                "salary": {"entry_level": "$30,000", "experienced": "$45,000", "highly_skilled": "$60,000"},
                "skills_gained": [
                    "Crop rotation planning",
                    "Pest identification and management",
                    "Irrigation system operation",
                    "Soil testing and amendment"
                ]
            },
            "Livestock Management": {
                "duration": "8 months",
                "certification": "Animal Husbandry Certificate",
                "skill_level": "Intermediate",
                "job_prospects": "Steady demand in rural areas",
                "salary": {"entry_level": "$28,000", "experienced": "$42,000", "highly_skilled": "$55,000"},
                "skills_gained": [
                    "Animal health monitoring",
                    "Feed ratio calculation",
                    "Breeding program management",
                    "Housing system maintenance"
                ]
            },
            "Agricultural Mechanics": {
                "duration": "10 months",
                "certification": "Farm Equipment Technician Certificate",
                "skill_level": "Advanced",
                "job_prospects": "High demand nationwide",
                "salary": {"entry_level": "$35,000", "experienced": "$50,000", "highly_skilled": "$70,000"},
                "skills_gained": [
                    "Tractor and implement repair",
                    "Hydraulic system troubleshooting",
                    "Electrical system diagnostics",
                    "Preventive maintenance planning"
                ]
            }
        }
        
        comparison = {
            "trades": [],
            "comparison_points": [
                "duration", "certification", "skill_level", "job_prospects", 
                "salary", "skills_gained"
            ]
        }
        
        for trade_name in trade_names:
            if trade_name in agricultural_trades:
                trade_data = agricultural_trades[trade_name]
                trade_data["name"] = trade_name
                comparison["trades"].append(trade_data)
            else:
                # Create a default trade entry if not found
                comparison["trades"].append({
                    "name": trade_name,
                    "duration": "Variable",
                    "certification": "Industry Standard Certification",
                    "skill_level": "Varies",
                    "job_prospects": "Depends on local market",
                    "salary": {"entry_level": "Varies", "experienced": "Varies", "highly_skilled": "Varies"},
                    "skills_gained": ["Various technical skills", "Industry-specific knowledge"]
                })
        
        return jsonify({
            **comparison,
            "note": "This is sample comparison data for demonstration purposes."
        })
    except Exception as e:
        print(f"General error in trade_comparison route: {str(e)}")
        return jsonify({
            'error': 'Could not generate trade comparison at this time',
            'message': str(e)
        }), 200  # Return 200 to avoid breaking UI

@app.route('/practical_assessment', methods=['POST'])
def handle_practical_assessment():
    """Handle practical assessment submission or retrieval."""
    if 'user_id' not in session:
        return jsonify({'error': 'No active session'}), 401
    
    data = request.json
    action = data.get('action', 'get')  # get, submit, or details
    trade = data.get('trade')
    assessment_title = data.get('title')
    submission = data.get('submission')
    
    if not trade:
        return jsonify({'error': 'Trade name is required'}), 400
    
    # For actions that require an assessment title
    if action in ['submit', 'details'] and not assessment_title:
        return jsonify({'error': 'Assessment title is required'}), 400
    
    # For submit action, require a submission
    if action == 'submit' and not submission:
        return jsonify({'error': 'Submission data is required'}), 400
    
    user_session = active_sessions.get(session['user_id'])
    if not user_session:
        return jsonify({'error': 'Session expired'}), 401
    
    user_chatbot = user_session['chatbot']
    user_session['last_activity'] = datetime.now()
    
    try:
        try:
            # Try using the practical assessment manager if available
            if practical_assessment_mgr:
                if action == 'get' and hasattr(practical_assessment_mgr, 'get_available_practicals'):
                    practicals = practical_assessment_mgr.get_available_practicals(trade)
                    return jsonify({'practicals': practicals})
                elif action == 'details' and hasattr(practical_assessment_mgr, 'get_practical_details'):
                    details = practical_assessment_mgr.get_practical_details(trade, assessment_title)
                    return jsonify({'details': details})
                elif action == 'submit' and hasattr(practical_assessment_mgr, 'save_user_attempt'):
                    result = practical_assessment_mgr.save_user_attempt(
                        session['user_id'], trade, assessment_title, submission
                    )
                    return jsonify({'success': result})
            
            # Try using chatbot if it has the method
            if hasattr(user_chatbot, 'handle_practical_assessment'):
                result = user_chatbot.handle_practical_assessment(action, trade, assessment_title, submission)
                return jsonify(result)
        except Exception as assess_err:
            print(f"Practical assessment error: {str(assess_err)}")
        
        # Fallback: Return sample practical assessment data
        if action == 'get':
            fallback_practicals = {
                trade: [
                    {
                        "title": "Basic Soil Testing and Analysis",
                        "difficulty": "Beginner",
                        "duration": "2 hours",
                        "overview": "Conduct basic soil tests to determine pH, nutrient levels, and texture."
                    },
                    {
                        "title": "Irrigation System Setup",
                        "difficulty": "Intermediate",
                        "duration": "4 hours",
                        "overview": "Design and set up a small-scale drip irrigation system."
                    },
                    {
                        "title": "Crop Disease Identification",
                        "difficulty": "Advanced",
                        "duration": "3 hours",
                        "overview": "Identify and diagnose common crop diseases and recommend treatment options."
                    }
                ]
            }
            return jsonify({'practicals': fallback_practicals})
            
        elif action == 'details':
            fallback_details = {
                "title": assessment_title or "Basic Soil Testing and Analysis",
                "difficulty": "Intermediate",
                "duration": "3 hours",
                "tools_required": ["Soil testing kit", "pH meter", "Sampling tools"],
                "materials_required": ["Soil samples", "Distilled water", "Recording sheet"],
                "safety_equipment": ["Gloves", "Safety goggles"],
                "steps": [
                    {
                        "step_number": 1,
                        "description": "Sample Collection",
                        "details": "Collect soil samples from different parts of the field at a depth of 15-20 cm.",
                        "common_mistakes": ["Taking samples from only one location", "Insufficient depth"]
                    },
                    {
                        "step_number": 2,
                        "description": "Sample Preparation",
                        "details": "Remove debris and stones, then mix samples thoroughly. Allow to air dry if needed.",
                        "common_mistakes": ["Not removing non-soil materials", "Using wet soil for testing"]
                    },
                    {
                        "step_number": 3,
                        "description": "pH Testing",
                        "details": "Use pH meter or test kit according to manufacturer instructions. Record readings.",
                        "common_mistakes": ["Not calibrating equipment", "Contaminating samples"]
                    },
                    {
                        "step_number": 4,
                        "description": "Nutrient Testing",
                        "details": "Test for major nutrients (N, P, K) using appropriate test kits.",
                        "common_mistakes": ["Misreading color changes", "Not following timing instructions"]
                    },
                    {
                        "step_number": 5,
                        "description": "Analysis and Recommendation",
                        "details": "Interpret results and provide recommendations for soil amendments.",
                        "common_mistakes": ["Misinterpreting test readings", "Not considering crop-specific needs"]
                    }
                ],
                "evaluation_criteria": [
                    "Accuracy of sampling procedure",
                    "Proper handling of testing equipment",
                    "Precision in measurements",
                    "Logical interpretation of results",
                    "Quality of recommendations"
                ]
            }
            return jsonify({'details': fallback_details})
            
        elif action == 'submit':
            fallback_result = {
                'success': True,
                'submission_id': f"SUB-{random.randint(1000, 9999)}",
                'message': "Your practical assessment submission has been received and will be evaluated.",
                'next_steps': "You will receive feedback within 2-3 business days.",
                'note': 'This is a sample response for demonstration purposes.'
            }
            return jsonify(fallback_result)
    except Exception as e:
        print(f"General error in practical_assessment route: {str(e)}")
        return jsonify({
            'error': 'Could not process practical assessment request at this time',
            'message': str(e)
        }), 200  # Return 200 to avoid breaking UI

@app.route('/scholarships', methods=['POST'])
def handle_scholarships():
    """Search for or apply to scholarships."""
    if 'user_id' not in session:
        return jsonify({'error': 'No active session'}), 401
    
    data = request.json
    action = data.get('action', 'search')  # search, details, apply, eligibility
    category = data.get('category')
    name = data.get('name')
    student_profile = data.get('profile')
    application = data.get('application')
    
    user_session = active_sessions.get(session['user_id'])
    if not user_session:
        return jsonify({'error': 'Session expired'}), 401
    
    user_chatbot = user_session['chatbot']
    user_session['last_activity'] = datetime.now()
    
    try:
        try:
            # Try using the scholarship manager if available
            if scholarship_mgr:
                if action == 'search' and hasattr(scholarship_mgr, 'get_all_scholarships'):
                    if category and hasattr(scholarship_mgr, 'get_scholarships_by_category'):
                        scholarships = scholarship_mgr.get_scholarships_by_category(category)
                        return jsonify({'scholarships': scholarships})
                    else:
                        scholarships = scholarship_mgr.get_all_scholarships()
                        return jsonify({'scholarships': scholarships})
                elif action == 'details' and name and hasattr(scholarship_mgr, 'get_scholarship_by_name'):
                    details = scholarship_mgr.get_scholarship_by_name(name)
                    return jsonify({'details': details})
                elif action == 'eligibility' and student_profile and hasattr(scholarship_mgr, 'get_eligible_scholarships'):
                    eligible = scholarship_mgr.get_eligible_scholarships(student_profile)
                    return jsonify({'eligible_scholarships': eligible})
                elif action == 'apply' and application and hasattr(scholarship_mgr, 'process_application'):
                    result = scholarship_mgr.process_application(application)
                    return jsonify(result)
            
            # Try using chatbot if it has the method
            if hasattr(user_chatbot, 'handle_scholarship_request'):
                result = user_chatbot.handle_scholarship_request(action, category, name, student_profile, application)
                return jsonify(result)
        except Exception as scholarship_err:
            print(f"Scholarship error: {str(scholarship_err)}")
        
        # Fallback: Return sample scholarship data
        if action == 'search':
            fallback_scholarships = {
                "government_scholarships": [
                    {
                        "name": "National Agriculture Scholarship",
                        "provider": "Ministry of Agriculture",
                        "description": "Scholarships for students pursuing agricultural studies.",
                        "eligibility": ["10th pass with minimum 60%", "Family income below Rs. 3 lakh/annum"],
                        "benefits": ["Rs. 10,000 per month", "Book allowance", "Research stipend"],
                        "deadline": (datetime.now() + timedelta(days=45)).strftime('%Y-%m-%d')
                    },
                    {
                        "name": "Farm Innovation Grant",
                        "provider": "Agricultural Research Foundation",
                        "description": "Support for students with innovative ideas in farming technology.",
                        "eligibility": ["Currently enrolled in agricultural program", "Project proposal required"],
                        "benefits": ["Project funding up to Rs. 50,000", "Mentorship", "Incubation support"],
                        "deadline": (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
                    }
                ],
                "private_scholarships": [
                    {
                        "name": "Future Farmers Scholarship",
                        "provider": "AgriCorp Industries",
                        "description": "Supporting the next generation of agricultural professionals.",
                        "eligibility": ["Merit-based selection", "Interest in modern farming techniques"],
                        "benefits": ["Rs. 25,000 per semester", "Internship opportunity"],
                        "deadline": (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
                    }
                ],
                "international_scholarships": [
                    {
                        "name": "Global Agriculture Exchange Program",
                        "provider": "International Farming Association",
                        "description": "Study abroad program for agricultural students.",
                        "eligibility": ["Minimum 2 years of agricultural studies", "English proficiency"],
                        "benefits": ["Full tuition coverage", "Monthly stipend", "Travel allowance"],
                        "deadline": (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')
                    }
                ]
            }
            
            if category and category in fallback_scholarships:
                return jsonify({'scholarships': {category: fallback_scholarships[category]}})
            return jsonify({'scholarships': fallback_scholarships})
            
        elif action == 'details' and name:
            fallback_details = {
                "name": name,
                "provider": "Ministry of Agriculture",
                "description": "Comprehensive scholarship for agricultural studies.",
                "eligibility": [
                    "10th pass with minimum 60%",
                    "Family income below Rs. 3 lakh/annum",
                    "Currently enrolled in agricultural program"
                ],
                "benefits": [
                    "Rs. 10,000 per month stipend",
                    "Book allowance of Rs. 5,000 per semester",
                    "Research project funding up to Rs. 20,000"
                ],
                "application_process": [
                    "Complete online application form",
                    "Submit academic transcripts",
                    "Provide income certificate",
                    "Two letters of recommendation",
                    "Statement of purpose"
                ],
                "important_dates": {
                    "opening_date": (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d'),
                    "deadline": (datetime.now() + timedelta(days=45)).strftime('%Y-%m-%d'),
                    "results": (datetime.now() + timedelta(days=75)).strftime('%Y-%m-%d')
                },
                "contact": {
                    "email": "scholarships@agriculture.gov.in",
                    "phone": "+91-11-1234567",
                    "website": "https://agriculture.gov.in/scholarships"
                },
                "note": "This is sample data for demonstration purposes."
            }
            return jsonify({'details': fallback_details})
            
        elif action == 'eligibility' and student_profile:
            # Simple eligibility logic for demo purposes
            income = student_profile.get('income', 0)
            marks = student_profile.get('marks_10th', 0)
            
            eligible_scholarships = []
            
            if income < 300000:  # 3 lakh
                eligible_scholarships.append({
                    "name": "National Agriculture Scholarship",
                    "match_percentage": 90,
                    "requirements_met": ["Income requirement", "Educational background"],
                    "requirements_missing": [] if marks >= 60 else ["Minimum marks requirement"]
                })
            
            if marks >= 70:
                eligible_scholarships.append({
                    "name": "Merit Agricultural Scholarship",
                    "match_percentage": 85,
                    "requirements_met": ["Academic requirement"],
                    "requirements_missing": [] if income < 500000 else ["Income requirement"]
                })
            
            fallback_eligibility = {
                "eligible_scholarships": eligible_scholarships,
                "total_scholarships_checked": 10,
                "recommended_documents": [
                    "Income certificate",
                    "Academic transcripts",
                    "ID proof",
                    "Residence certificate"
                ],
                "note": "This is sample data for demonstration purposes."
            }
            return jsonify(fallback_eligibility)
            
        elif action == 'apply':
            fallback_application = {
                "application_id": f"APP-{random.randint(10000, 99999)}",
                "status": "submitted",
                "submission_date": datetime.now().strftime('%Y-%m-%d'),
                "scholarship_name": application.get('scholarship_name', 'Agricultural Scholarship'),
                "next_steps": [
                    "Your application has been submitted successfully.",
                    "You will receive email confirmation within 24 hours.",
                    "Application review typically takes 2-3 weeks."
                ],
                "note": "This is sample data for demonstration purposes."
            }
            return jsonify(fallback_application)
    except Exception as e:
        print(f"General error in scholarships route: {str(e)}")
        return jsonify({
            'error': 'Could not process scholarship request at this time',
            'message': str(e)
        }), 200  # Return 200 to avoid breaking UI

@app.route('/voice_settings', methods=['POST', 'GET'])
def handle_voice_settings():
    """Manage voice settings."""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    try:
        if request.method == 'POST':
            try:
                settings = request.json
                
                # Store settings in session for simple implementation
                if 'user_voice_settings' not in session:
                    session['user_voice_settings'] = {}
                
                # Update voice settings in session
                session['user_voice_settings'].update(settings)
                
                # Try to update settings in voice manager if available
                try:
                    if voice_mgr:
                        if hasattr(voice_mgr, 'update_settings'):
                            voice_mgr.update_settings(session['user_id'], settings)
                        elif hasattr(voice_mgr, 'set_voice_settings'):
                            voice_mgr.set_voice_settings(session['user_id'], settings)
                except Exception as voice_err:
                    print(f"Voice manager update error: {str(voice_err)}")
                
                return jsonify({
                    'status': 'success', 
                    'settings': session['user_voice_settings']
                })
            except Exception as post_err:
                print(f"Error in voice settings POST: {str(post_err)}")
                return jsonify({
                    'status': 'error',
                    'message': 'Could not update voice settings',
                    'error': str(post_err)
                }), 200  # Return 200 to avoid breaking UI
        else:  # GET
            try:
                # Get voice settings from session as fallback
                user_settings = session.get('user_voice_settings', {
                    'voice_type': 'default',
                    'speed': 1.0,
                    'pitch': 1.0,
                    'language': 'en-US',
                    'wake_word': 'Hey Assistant',
                    'continuous_listening': False
                })
                
                # Try to get settings from voice manager if available
                try:
                    if voice_mgr:
                        if hasattr(voice_mgr, 'get_user_settings'):
                            manager_settings = voice_mgr.get_user_settings(session['user_id'])
                            if manager_settings:
                                user_settings = manager_settings
                        elif hasattr(voice_mgr, 'get_voice_settings'):
                            manager_settings = voice_mgr.get_voice_settings(session['user_id'])
                            if manager_settings:
                                user_settings = manager_settings
                except Exception as voice_get_err:
                    print(f"Voice manager get settings error: {str(voice_get_err)}")
                
                # Get available voices as fallback
                available_voices = [
                    {'id': 'male_en', 'name': 'Male (English)', 'gender': 'male', 'language': 'en-US'},
                    {'id': 'female_en', 'name': 'Female (English)', 'gender': 'female', 'language': 'en-US'},
                    {'id': 'male_hi', 'name': 'Male (Hindi)', 'gender': 'male', 'language': 'hi-IN'},
                    {'id': 'female_hi', 'name': 'Female (Hindi)', 'gender': 'female', 'language': 'hi-IN'}
                ]
                
                # Try to get voices from voice manager if available
                try:
                    if voice_mgr:
                        if hasattr(voice_mgr, 'get_supported_voices'):
                            manager_voices = voice_mgr.get_supported_voices()
                            if manager_voices:
                                available_voices = manager_voices
                        elif hasattr(voice_mgr, 'get_available_voices'):
                            manager_voices = voice_mgr.get_available_voices()
                            if manager_voices:
                                available_voices = manager_voices
                except Exception as voice_list_err:
                    print(f"Voice manager get voices error: {str(voice_list_err)}")
                
                # For API response format
                return jsonify({
                    'settings': user_settings,
                    'available_voices': available_voices
                })
            except Exception as get_err:
                print(f"Error in voice settings GET: {str(get_err)}")
                return jsonify({
                    'status': 'error',
                    'message': 'Could not retrieve voice settings',
                    'settings': {
                        'voice_type': 'default',
                        'speed': 1.0,
                        'pitch': 1.0,
                        'language': 'en-US',
                        'wake_word': 'Hey Assistant',
                        'continuous_listening': False
                    },
                    'available_voices': [
                        {'id': 'male_en', 'name': 'Male (English)', 'language': 'en-US'},
                        {'id': 'female_en', 'name': 'Female (English)', 'language': 'en-US'}
                    ]
                }), 200  # Return 200 to avoid breaking UI
    except Exception as e:
        print(f"General error in voice_settings route: {str(e)}")
        return jsonify({
            'error': 'Could not process voice settings request',
            'message': str(e)
        }), 500

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    """Convert speech audio to text using VoiceInteractionManager."""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if not audio_file.filename:
            return jsonify({'error': 'Invalid audio file'}), 400
        
        # Save the audio file temporarily
        temp_filename = f"temp_audio_{int(time.time())}.wav"
        audio_file.save(temp_filename)
        
        print(f"Audio file saved temporarily as: {temp_filename}")
        
        # Get language from request, default to English
        language = request.form.get('language', 'en-US')
        
        text = None
        
        # Use voice_mgr's built-in speech recognition
        if voice_mgr and voice_mgr.is_available:
            try:
                import speech_recognition as sr
                
                # Use voice_mgr's recognizer
                recognizer = voice_mgr.recognizer
                
                # Process the audio file using SpeechRecognition directly from voice_mgr
                with sr.AudioFile(temp_filename) as source:
                    print("Reading audio file with voice_mgr's recognizer...")
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data, language=language)
                    print(f"Recognized text: {text}")
            except Exception as e:
                print(f"Error in speech recognition: {str(e)}")
        
        # Clean up temp file
        try:
            os.remove(temp_filename)
            print(f"Temporary audio file removed: {temp_filename}")
        except Exception as cleanup_err:
            print(f"Error cleaning up temp file: {str(cleanup_err)}")
        
        if text:
            return jsonify({'text': text})
        else:
            return jsonify({'error': 'Could not recognize speech'}), 422
        
    except Exception as e:
        print(f"Speech-to-text error: {str(e)}")
        return jsonify({'error': f'Error processing speech: {str(e)}'}), 500

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    """Convert text to speech using gTTS or VoiceInteractionManager."""
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data.get('text')
        language = data.get('language', 'en')
        
        # Map language codes to gTTS compatible codes if needed
        lang_map = {
            'en': 'en',
            'hi': 'hi',
            'bn': 'bn',
            'ta': 'ta',
            'te': 'te',
            'mr': 'mr',
            'kn': 'kn'  # Add Kannada support
        }
        
        gtts_lang = lang_map.get(language, 'en')
        
        print(f"Text-to-speech request for: '{text[:30]}...' in language {language} (gTTS: {gtts_lang})")
        
        # Create a temporary file for the speech output with a unique name
        import uuid
        temp_dir = os.path.join(os.getcwd(), 'temp_audio')
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = os.path.join(temp_dir, f"speech_{uuid.uuid4().hex}.mp3")
        
        try:
            # Always use gTTS directly for reliability
            from gtts import gTTS
            
            # Create TTS object
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            
            # Save to file
            tts.save(temp_filename)
            
            print(f"Speech generated and saved to {temp_filename}")
            
            # Return the audio file
            response = send_file(
                temp_filename,
                mimetype="audio/mpeg",
                as_attachment=True,
                download_name="speech.mp3",
                max_age=0
            )
            
            # Set response headers to prevent caching
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            
            # Schedule the temp file for deletion after sending
            @after_this_request
            def remove_file(response):
                try:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
                        print(f"Temporary speech file removed: {temp_filename}")
                except Exception as e:
                    print(f"Error removing temporary file: {str(e)}")
                return response
            
            return response
        
        except Exception as tts_err:
            print(f"gTTS error: {str(tts_err)}")
            
            # Try using VoiceInteractionManager as fallback if available
            if voice_mgr and voice_mgr.is_available and hasattr(voice_mgr, 'speak'):
                try:
                    print("Trying VoiceInteractionManager as fallback")
                    # Use VoiceInteractionManager's speak method
                    voice_mgr.speak(text, language)
                    
                    # If the speak method doesn't save to a file we can access, return an error
                    return jsonify({
                        'error': 'Voice played through system but could not be returned to browser',
                        'fallback': 'system_audio'
                    }), 500
                except Exception as vm_err:
                    print(f"VoiceInteractionManager error: {str(vm_err)}")
                    return jsonify({'error': f'Could not generate speech: {str(vm_err)}'}), 500
            else:
                # No fallback available
                raise
    
    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")
        return jsonify({'error': f'Error generating speech: {str(e)}'}), 500

@app.route('/task_progress', methods=['POST', 'GET'])
def handle_task_progress():
    """Manage user's task progress."""
    if 'user_id' not in session:
        return jsonify({'error': 'No active session'}), 401
    
    user_session = active_sessions.get(session['user_id'])
    if not user_session:
        return jsonify({'error': 'Session expired'}), 401
    
    user_session['last_activity'] = datetime.now()
    
    try:
        if request.method == 'POST':
            data = request.json
            action = data.get('action')
            task_id = data.get('task_id')
            steps = data.get('steps', 1)
            message = data.get('message')
            
            if not action:
                return jsonify({'error': 'Action is required'}), 400
            
            try:
                # Try to use the progress tracker manager
                if progress_track_mgr:
                    if action == 'create' and hasattr(progress_track_mgr, 'create_task'):
                        total_steps = data.get('total_steps', 1)
                        description = data.get('description', 'Unnamed Task')
                        
                        task_id = progress_track_mgr.create_task(
                            session['user_id'], 
                            total_steps, 
                            description
                        )
                        return jsonify({'success': True, 'task_id': task_id})
                    
                    elif action == 'update' and hasattr(progress_track_mgr, 'update_task'):
                        if not task_id:
                            return jsonify({'error': 'Task ID is required'}), 400
                        
                        progress_track_mgr.update_task(
                            session['user_id'], 
                            task_id, 
                            steps, 
                            message
                        )
                        return jsonify({'success': True})
                    
                    elif action == 'complete' and hasattr(progress_track_mgr, 'complete_task'):
                        if not task_id:
                            return jsonify({'error': 'Task ID is required'}), 400
                        
                        progress_track_mgr.complete_task(session['user_id'], task_id)
                        return jsonify({'success': True})
                    
                    elif action == 'delete' and hasattr(progress_track_mgr, 'delete_task'):
                        if not task_id:
                            return jsonify({'error': 'Task ID is required'}), 400
                        
                        progress_track_mgr.delete_task(session['user_id'], task_id)
                        return jsonify({'success': True})
            except Exception as tracker_err:
                print(f"Progress tracker error: {str(tracker_err)}")
            
            # Fallback responses
            if action == 'create':
                return jsonify({
                    'success': True,
                    'task_id': f"task_{random.randint(1000, 9999)}",
                    'note': 'This is sample data for demonstration purposes.'
                })
            elif action in ['update', 'complete', 'delete']:
                return jsonify({
                    'success': True,
                    'note': 'This is sample data for demonstration purposes.'
                })
            else:
                return jsonify({'error': 'Invalid action'}), 400
                
        else:  # GET request
            try:
                # Try to use the progress tracker manager
                if progress_track_mgr:
                    if hasattr(progress_track_mgr, 'get_all_tasks'):
                        tasks = progress_track_mgr.get_all_tasks(session['user_id'])
                        return jsonify({'tasks': tasks})
                    elif hasattr(progress_track_mgr, 'get_user_tasks'):
                        tasks = progress_track_mgr.get_user_tasks(session['user_id'])
                        return jsonify({'tasks': tasks})
                    elif hasattr(progress_track_mgr, 'list_tasks'):
                        tasks = progress_track_mgr.list_tasks(session['user_id'])
                        return jsonify({'tasks': tasks})
            except Exception as tracker_err:
                print(f"Progress tracker error: {str(tracker_err)}")
            
            # Fallback: Return sample task data
            fallback_tasks = [
                {
                    'id': 'task_1001',
                    'description': 'Complete soil testing module',
                    'total_steps': 5,
                    'current_steps': 3,
                    'percentage': 60,
                    'created_at': (datetime.now() - timedelta(days=2)).isoformat(),
                    'last_updated': (datetime.now() - timedelta(hours=4)).isoformat(),
                    'completed': False,
                    'category': 'learning'
                },
                {
                    'id': 'task_1002',
                    'description': 'Submit crop rotation plan',
                    'total_steps': 3,
                    'current_steps': 3,
                    'percentage': 100,
                    'created_at': (datetime.now() - timedelta(days=5)).isoformat(),
                    'last_updated': (datetime.now() - timedelta(days=1)).isoformat(),
                    'completed': True,
                    'category': 'assignment'
                },
                {
                    'id': 'task_1003',
                    'description': 'Research irrigation techniques',
                    'total_steps': 4,
                    'current_steps': 1,
                    'percentage': 25,
                    'created_at': (datetime.now() - timedelta(days=1)).isoformat(),
                    'last_updated': (datetime.now() - timedelta(hours=20)).isoformat(),
                    'completed': False,
                    'category': 'research'
                }
            ]
            return jsonify({
                'tasks': fallback_tasks,
                'note': 'This is sample data for demonstration purposes.'
            })
    except Exception as e:
        print(f"General error in task_progress route: {str(e)}")
        return jsonify({
            'error': 'Could not process task progress request at this time',
            'message': str(e)
        }), 200  # Return 200 to avoid breaking UI

# API Endpoints for the Agro & Vocational Web Application

# 1. User Authentication (Signup & Login)
@app.route(app.config['API_PREFIX'] + '/signup', methods=['POST'])
def signup():
    """User Signup API."""
    try:
        data = request.json
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # Validate data
        if not username or not email or not password:
            return jsonify({'message': 'Missing required fields!'}), 400
        
        # Check if user already exists
        existing_user = db_session.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            return jsonify({'message': 'User already exists!'}), 409
        
        # Create new user
        hashed_password = hash_password(password)
        new_user = User(
            username=username,
            email=email,
            password=hashed_password
        )
        
        db_session.add(new_user)
        db_session.commit()
        
        # Generate token
        token = generate_token(new_user.id)
        
        return jsonify({
            'message': 'User registered successfully!',
            'token': token,
            'user': new_user.to_dict()
        }), 201
        
    except Exception as e:
        db_session.rollback()
        return jsonify({'message': f'Error during signup: {str(e)}'}), 500

@app.route(app.config['API_PREFIX'] + '/login', methods=['POST'])
def login():
    """User Login API."""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        # Validate data
        if not email or not password:
            return jsonify({'message': 'Missing email or password!'}), 400
        
        # Find user
        user = db_session.query(User).filter_by(email=email).first()
        
        if not user or not verify_password(user.password, password):
            return jsonify({'message': 'Invalid email or password!'}), 401
        
        # Generate token
        token = generate_token(user.id)
        
        return jsonify({
            'message': 'Login successful!',
            'token': token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Error during login: {str(e)}'}), 500

@app.route(app.config['API_PREFIX'] + '/logout', methods=['POST'])
@token_required
def api_logout(current_user):
    """User Logout API."""
    # With JWT, we don't need to do anything on the server
    # The client will simply remove the token
    return jsonify({'message': 'Logout successful!'}), 200

@app.route(app.config['API_PREFIX'] + '/dashboard', methods=['GET'])
def dashboard_api():
    """User Dashboard API."""
    try:
        # Get user ID from token if available
        user_id = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
                user_id = decode_token(token)
                if isinstance(user_id, str) and (user_id.startswith('Signature expired') or user_id.startswith('Invalid token')):
                    user_id = None
            except (IndexError, Exception):
                user_id = None
        
        # Default guest response
        response = {
            'user': {
                'id': 0,
                'username': 'Guest User',
                'email': 'guest@example.com',
                'created_at': datetime.now().isoformat()
            },
            'recent_uploads': []
        }
        
        # If we have a valid user, get their data
        if user_id and not isinstance(user_id, str):
            user = db_session.query(User).filter_by(id=user_id).first()
            if user:
                # Get user's recent uploads
                recent_uploads = db_session.query(Upload).filter_by(
                    user_id=user.id
                ).order_by(Upload.created_at.desc()).limit(5).all()
                
                # Format the response
                response = {
                    'user': user.to_dict(),
                    'recent_uploads': [upload.to_dict() for upload in recent_uploads]
                }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'message': f'Error fetching dashboard data: {str(e)}'}), 500

# 2. Contact Form Submission
@app.route(app.config['API_PREFIX'] + '/contact', methods=['POST'])
def contact_form():
    """Contact Form API."""
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        message = data.get('message')
        
        # Validate data
        if not name or not email or not message:
            return jsonify({'message': 'Missing required fields!'}), 400
        
        # Get user if authenticated
        user_id = None
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
                user_id = decode_token(token)
                if isinstance(user_id, str) and (user_id.startswith('Signature expired') or user_id.startswith('Invalid token')):
                    user_id = None
            except (IndexError, Exception):
                user_id = None
        
        # Create new contact message
        new_contact = Contact(
            name=name,
            email=email,
            message=message,
            user_id=user_id
        )
        
        db_session.add(new_contact)
        db_session.commit()
        
        # Send email confirmation (optional, implementation not shown)
        
        return jsonify({
            'message': 'Contact form submitted successfully!',
            'contact': new_contact.to_dict()
        }), 201
        
    except Exception as e:
        db_session.rollback()
        return jsonify({'message': f'Error during form submission: {str(e)}'}), 500

# 3. File Uploads (Plant Disease Detection)
@app.route(app.config['API_PREFIX'] + '/upload-image', methods=['POST'])
@token_required
def upload_image(current_user):
    """Plant Image Upload API."""
    try:
        if 'image' not in request.files:
            return jsonify({'message': 'No image file provided!'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'message': 'No selected file!'}), 400
        
        if image_file:
            # Secure the filename
            filename = secure_filename(image_file.filename)
            
            # Create unique filename
            unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            image_file.save(filepath)
            
            # Store in database
            new_upload = Upload(
                filename=unique_filename,
                filepath=filepath,
                filetype=image_file.content_type,
                user_id=current_user.id
            )
            
            db_session.add(new_upload)
            db_session.commit()
            
            # Return the upload info
            return jsonify({
                'message': 'Image uploaded successfully!',
                'upload': new_upload.to_dict()
            }), 201
        
        return jsonify({'message': 'File upload failed!'}), 400
        
    except Exception as e:
        db_session.rollback()
        return jsonify({'message': f'Error during file upload: {str(e)}'}), 500

@app.route(app.config['API_PREFIX'] + '/detect-disease', methods=['POST'])
def detect_disease_api():
    """Plant Disease Detection API."""
    try:
        if 'image' not in request.files:
            return jsonify({'message': 'No image file provided!'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'message': 'No selected file!'}), 400
        
        if image_file:
            # Secure the filename
            filename = secure_filename(image_file.filename)
            
            # Create unique filename
            unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            image_file.save(filepath)
            
            # Check if model is available
            if model is None:
                return jsonify({'message': 'Plant disease model is not available!'}), 500
            
            # Process the image using TensorFlow directly to avoid framework conflicts
            try:
                import numpy as np
                from tensorflow.keras.preprocessing import image as keras_image
                
                # Load and preprocess the image using TensorFlow/Keras preprocessing
                img = keras_image.load_img(filepath, target_size=(224, 224))
                img_array = keras_image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # Simple normalization
                
                # Perform prediction
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions[0])
                confidence_score = float(predictions[0][predicted_class_index] * 100)
                
                # Map to disease class names - these should match your model's actual classes
                disease_classes = [
                    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
                    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
                    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
                    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
                    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 
                    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", 
                    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", 
                    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
                    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", 
                    "Strawberry___Leaf_scorch", "Strawberry___healthy",
                    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
                    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
                    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
                    "Tomato___healthy"
                ]
                
                # Check if the predicted class index is within range
                if predicted_class_index < len(disease_classes):
                    prediction = disease_classes[predicted_class_index]
                    # Format the prediction for display
                    formatted_prediction = prediction.replace('___', ' - ').replace('_', ' ')
                    
                    # Check if the plant is healthy
                    is_healthy = "healthy" in prediction.lower()
                    
                    # Get detailed disease information
                    disease_details = get_disease_details(prediction)
                    
                    # Store the detection result in database if user is authenticated
                    user_id = None
                    token = None
                    if 'Authorization' in request.headers:
                        auth_header = request.headers['Authorization']
                        try:
                            token = auth_header.split(" ")[1]
                            user_id = decode_token(token)
                            if isinstance(user_id, str) and (user_id.startswith('Signature expired') or user_id.startswith('Invalid token')):
                                user_id = None
                        except (IndexError, Exception):
                            user_id = None
                    
                    # Only store in database if we have a valid user
                    if user_id and not isinstance(user_id, str):
                        user = db_session.query(User).filter_by(id=user_id).first()
                        if user:
                            new_upload = Upload(
                                filename=unique_filename,
                                filepath=filepath,
                                filetype=image_file.content_type,
                                user_id=user.id,
                                disease=formatted_prediction,
                                confidence=f"{confidence_score:.2f}%",
                                treatment=disease_details["treatment"]
                            )
                            
                            db_session.add(new_upload)
                            db_session.commit()
                            image_id = new_upload.id
                        else:
                            image_id = None
                    else:
                        image_id = None
                    
                    # Return the detection result with detailed information
                    return jsonify({
                        'message': 'Disease detection completed!',
                        'result': {
                            'disease': formatted_prediction,
                            'confidence': f"{confidence_score:.2f}%",
                            'is_healthy': is_healthy,
                            'name': disease_details["name"],
                            'description': disease_details["description"],
                            'symptoms': disease_details["symptoms"],
                            'causes': disease_details["causes"],
                            'treatment': disease_details["treatment"],
                            'prevention': disease_details["prevention"],
                            'image_id': image_id
                        }
                    }), 200
                else:
                    # Handle out of range index
                    return jsonify({
                        'message': 'Disease detection completed, but could not identify disease!',
                        'result': {
                            'disease': 'Unknown',
                            'confidence': '0.00%',
                            'is_healthy': False,
                            'name': 'Unknown Disease',
                            'description': 'Could not determine the disease type.',
                            'symptoms': 'Not available',
                            'causes': 'Not available',
                            'treatment': 'Could not determine the disease. Please consult with an agricultural expert.',
                            'prevention': 'Regular monitoring and crop management practices are recommended.',
                            'image_id': None
                        }
                    }), 200
                
            except Exception as detect_err:
                # Clean up the uploaded file
                try:
                    os.remove(filepath)
                except:
                    pass
                
                return jsonify({
                    'message': f'Error during disease detection: {str(detect_err)}',
                    'result': {
                        'status': 'error',
                        'disease': 'Error',
                        'name': 'Analysis Error',
                        'description': 'Unable to analyze the image.',
                        'symptoms': 'Not available',
                        'causes': 'Technical error during analysis.',
                        'treatment': 'Please ensure you\'ve uploaded a clear image of the plant leaf or affected area.',
                        'prevention': 'Try again with a well-lit, clear photo of the affected plant part.'
                    }
                }), 500
        
        return jsonify({'message': 'File processing failed!'}), 400
        
    except Exception as e:
        db_session.rollback()
        return jsonify({'message': f'Error processing image: {str(e)}'}), 500

# API Frontend Routes
@app.route('/api_index.html')
def api_index():
    """Render the API frontend landing page."""
    user_language = session.get('app_language', 'en')
    if user_language == 'hi':
        return render_template('api_index_hindi.html')
    return render_template('api_index.html')

@app.route('/api_index_hindi.html')
def api_index_hindi():
    """Render the Hindi version of the API frontend landing page."""
    return render_template('api_index_hindi.html')

@app.route('/api_login.html')
def api_login():
    """Render the API frontend login page."""
    return render_template('api_login.html')

@app.route('/api_signup.html')
def api_signup():
    """Render the API frontend signup page."""
    return render_template('api_signup.html')

@app.route('/api_dashboard.html')
def api_dashboard():
    """Render the API frontend dashboard page."""
    return render_template('api_dashboard.html')

@app.route('/api_contact.html')
def api_contact():
    """Render the API frontend contact page."""
    return render_template('api_contact.html')

@app.route('/api_plant_disease.html')
def api_plant_disease():
    """Render the API frontend plant disease detection page."""
    return render_template('api_plant_disease.html')

@app.route('/api_base.html')
def api_base():
    """Render the API frontend base template."""
    return render_template('api_base.html')

@app.route('/api_chat.html')
def api_chat_page():
    """Render the chat interface template."""
    return render_template('api_chat.html')

@app.route('/api_market_prediction.html')
def api_market_prediction():
    """Render market prediction page."""
    return render_template('api_market_prediction.html')

@app.route('/api_course_finder.html')
def api_course_finder():
    """Render course finder page."""
    return render_template('api_course_finder.html')

@app.route('/api_industry_connections.html')
def api_industry_connections():
    """Render industry connections page."""
    return render_template('api_industry_connections.html')

@app.route('/api_progress_tracker.html')
def api_progress_tracker():
    """Render progress tracker page."""
    return render_template('api_progress_tracker.html')

@app.route('/api_vocational_tools.html')
def api_vocational_tools():
    """Render vocational tools page."""
    return render_template('api_vocational_tools.html')

@app.route('/api_scholarships.html')
def api_scholarships():
    """Render scholarships page."""
    return render_template('api_scholarships.html')

@app.route('/api_soil_analyzer.html')
def api_soil_analyzer():
    """Render soil analyzer page."""
    return render_template('api_soil_analyzer.html')

@app.route('/api_weather_forecast.html')
def api_weather_forecast():
    """Render weather forecast page."""
    return render_template('api_weather_forecast.html')

# Create a comprehensive disease information database
def get_disease_details(disease_name):
    """Get detailed information about a plant disease."""
    # Comprehensive database of plant diseases with detailed information
    disease_details = {
        "Apple___Apple_scab": {
            "name": "Apple Scab",
            "description": "Apple scab is a common fungal disease that affects apple trees, causing dark, scabby lesions on leaves and fruit.",
            "symptoms": "Dark olive-green spots on leaves that later turn black and scabby. Infected fruits develop similar scabby spots and may become deformed.",
            "causes": "Caused by the fungus Venturia inaequalis which overwinters in fallen leaves and infects new growth in spring, especially during wet conditions.",
            "treatment": "Apply fungicides containing captan, myclobutanil, or sulfur at 7-10 day intervals from bud break until rainy season ends. Remove and destroy infected leaves and fruit.",
            "prevention": "Plant resistant apple varieties. Ensure good air circulation by proper pruning. Clean up and destroy fallen leaves in autumn. Apply dormant sprays in late winter before buds open."
        },
        "Apple___Black_rot": {
            "name": "Apple Black Rot",
            "description": "Black rot is a fungal disease that affects apple trees, causing leaf spots, fruit rot, and cankers on branches.",
            "symptoms": "Circular purple spots on leaves that develop brown centers. Infected fruits show dark, spreading lesions with concentric rings. Branches may develop sunken cankers.",
            "causes": "Caused by the fungus Botryosphaeria obtusa which survives in cankers on branches and in mummified fruits.",
            "treatment": "Prune out diseased branches at least 8 inches below visible cankers. Apply fungicides containing captan, thiophanate-methyl, or myclobutanil during the growing season.",
            "prevention": "Remove mummified fruit and cankers from trees. Maintain tree vigor with proper nutrition and watering. Control insects that create wounds. Practice good sanitation by removing infected plant material."
        },
        "Apple___Cedar_apple_rust": {
            "name": "Cedar Apple Rust",
            "description": "Cedar apple rust is a fungal disease that requires both apple trees and juniper/cedar trees to complete its life cycle.",
            "symptoms": "Bright orange-yellow spots on leaves and sometimes fruits. Spots may have a red border and develop small black dots in the center.",
            "causes": "Caused by the fungus Gymnosporangium juniperi-virginianae which alternates between apple trees and cedar/juniper trees.",
            "treatment": "Apply fungicides containing myclobutanil or propiconazole starting at pink bud stage and continuing for several weeks. No cure exists for infected juniper galls.",
            "prevention": "Plant resistant apple varieties. Remove nearby cedar/juniper trees if possible (within 1/4 mile). Apply preventative fungicides before symptoms appear."
        },
        "Apple___healthy": {
            "name": "Healthy Apple Plant",
            "description": "This apple plant appears healthy with no signs of disease.",
            "symptoms": "Leaves are uniformly green without spots or abnormal discoloration. Fruits develop normally without lesions or rot.",
            "causes": "N/A - Plant is healthy",
            "treatment": "No treatment needed.",
            "prevention": "Continue good orchard management practices: proper watering, fertilization, pruning, and pest monitoring."
        },
        "Corn_(maize)___Common_rust_": {
            "name": "Corn Common Rust",
            "description": "Common rust is a fungal disease that affects corn plants, appearing as small rusty spots on leaves.",
            "symptoms": "Small, round to elongated, powdery, reddish-brown pustules on both leaf surfaces. Pustules later turn black as they mature.",
            "causes": "Caused by the fungus Puccinia sorghi which is spread by airborne spores that can travel long distances.",
            "treatment": "Apply fungicides containing azoxystrobin, pyraclostrobin, or propiconazole when symptoms first appear if the infection is severe and economic thresholds are reached.",
            "prevention": "Plant resistant corn hybrids. Early planting can help avoid the disease. Rotate crops to reduce inoculum. Monitor fields regularly for early detection."
        },
        "Potato___Early_blight": {
            "name": "Potato Early Blight",
            "description": "Early blight is a fungal disease that affects potato plants, creating dark, target-like spots on lower leaves.",
            "symptoms": "Dark brown to black lesions with concentric rings (target-like appearance), usually on older leaves first. Affected leaves may yellow and drop prematurely.",
            "causes": "Caused by the fungus Alternaria solani which can survive in soil, plant debris, and on related weed species.",
            "treatment": "Apply fungicides containing chlorothalonil, mancozeb, or copper-based products every 7-10 days. Remove and destroy infected leaves to slow disease spread.",
            "prevention": "Use crop rotation with non-solanaceous crops. Plant resistant varieties. Space plants for good air circulation. Mulch soil to prevent pathogen splash. Avoid overhead irrigation."
        },
        "Potato___Late_blight": {
            "name": "Potato Late Blight",
            "description": "Late blight is a serious, rapidly spreading disease that affects potato plants, causing water-soaked spots on leaves that quickly turn dark and spread.",
            "symptoms": "Water-soaked, gray-green lesions on leaves that rapidly expand and turn brown or black. White, fuzzy growth may appear on leaf undersides in humid conditions. Tubers develop reddish-brown granular rot.",
            "causes": "Caused by the oomycete Phytophthora infestans, the same pathogen that caused the Irish potato famine. It thrives in cool, wet weather.",
            "treatment": "Apply fungicides containing chlorothalonil, mancozeb, or copper-based products preventatively. In severe cases, remove and destroy infected plants entirely to prevent spread.",
            "prevention": "Plant resistant varieties. Use certified disease-free seed potatoes. Provide good drainage and air circulation. Remove volunteers and nightshade weeds. Apply preventative fungicides during cool, wet periods."
        },
        "Potato___healthy": {
            "name": "Healthy Potato Plant",
            "description": "This potato plant appears healthy with no signs of disease.",
            "symptoms": "Leaves are uniformly green without spots or abnormal discoloration. Plants are vigorous with normal growth.",
            "causes": "N/A - Plant is healthy",
            "treatment": "No treatment needed.",
            "prevention": "Continue good crop management practices: proper watering, fertilization, and pest monitoring. Rotate crops to prevent soil-borne disease buildup."
        },
        "Tomato___Bacterial_spot": {
            "name": "Tomato Bacterial Spot",
            "description": "Bacterial spot is a bacterial disease that affects tomato plants, causing small, dark, water-soaked spots on leaves, stems, and fruit.",
            "symptoms": "Small, irregular, dark lesions on leaves, stems, and fruits. Leaf spots may have yellow halos. Spots on fruits are slightly raised, scabby, and do not extend deep into the flesh.",
            "causes": "Caused by several Xanthomonas species bacteria which can be spread by water splash, tools, and infected seed.",
            "treatment": "Apply copper-based bactericides early in the season before symptoms appear. There is no cure once plants are infected, so focus on slowing spread and prevention.",
            "prevention": "Use disease-free seeds and transplants. Practice crop rotation (3-4 years). Avoid overhead irrigation. Remove and destroy infected plant debris. Sanitize garden tools."
        },
        "Tomato___Early_blight": {
            "name": "Tomato Early Blight",
            "description": "Early blight is a fungal disease that affects tomato plants, creating dark, target-like spots on lower leaves that can lead to significant defoliation.",
            "symptoms": "Dark brown lesions with concentric rings (target-like appearance) on older leaves first. Infected leaves turn yellow around lesions, then brown and drop. Stem lesions may occur at the soil line.",
            "causes": "Caused by the fungus Alternaria solani which can survive in soil and plant debris for at least a year.",
            "treatment": "Apply fungicides containing chlorothalonil, mancozeb, or copper-based products every 7-10 days. Remove and destroy infected leaves. Stake plants to improve air circulation.",
            "prevention": "Use crop rotation (3-4 years). Plant resistant varieties. Mulch to prevent soil splash. Provide adequate plant spacing. Remove lower leaves that touch the soil. Water at the base of plants."
        },
        "Tomato___Late_blight": {
            "name": "Tomato Late Blight",
            "description": "Late blight is a devastating disease that affects tomato plants, causing rapidly spreading water-soaked spots that can destroy entire plants within days.",
            "symptoms": "Large, water-soaked, gray-green lesions on leaves that quickly turn brown or black. White, fuzzy growth often appears on undersides of leaves in humid conditions. Stems develop dark brown lesions. Fruits develop firm, dark, greasy spots.",
            "causes": "Caused by the oomycete Phytophthora infestans which thrives in cool, wet weather and can spread very rapidly.",
            "treatment": "Apply fungicides containing chlorothalonil, mancozeb, or copper compounds preventatively. Once infected, plants may need to be removed and destroyed to prevent spread.",
            "prevention": "Plant resistant varieties. Improve drainage and air circulation. Remove volunteer tomato and potato plants. Apply preventative fungicides during cool, wet periods. Space plants adequately."
        },
        "Tomato___healthy": {
            "name": "Healthy Tomato Plant",
            "description": "This tomato plant appears healthy with no signs of disease.",
            "symptoms": "Leaves are uniformly green without spots or abnormal discoloration. Plant is vigorous with normal growth and fruit development.",
            "causes": "N/A - Plant is healthy",
            "treatment": "No treatment needed.",
            "prevention": "Continue good garden practices: proper watering at the base of plants, adequate spacing, staking/caging for support, balanced fertilization, and regular monitoring for pests and diseases."
        }
    }
    
    # If we have detailed information for this disease, return it
    if disease_name in disease_details:
        return disease_details[disease_name]
    
    # Generic information for diseases not in our detailed database
    # Extract crop and condition from disease name format
    parts = disease_name.split('___')
    crop = parts[0].replace('_', ' ') if len(parts) > 0 else "Unknown"
    condition = parts[1].replace('_', ' ') if len(parts) > 1 else "Unknown"
    
    if "healthy" in disease_name.lower():
        return {
            "name": f"Healthy {crop}",
            "description": f"This {crop} plant appears to be healthy.",
            "symptoms": "No symptoms of disease. Plant displays normal, healthy growth patterns.",
            "causes": "N/A - Plant is healthy",
            "treatment": "No treatment needed.",
            "prevention": f"Continue good {crop} care practices: proper watering, fertilization, and pest monitoring."
        }
    
    return {
        "name": condition,
        "description": f"A disease affecting {crop} plants.",
        "symptoms": "Symptoms may include leaf discoloration, spots, wilting, or abnormal growth.",
        "causes": f"This disease may be caused by fungal, bacterial, viral pathogens, or environmental stress.",
        "treatment": "Consult with a local agricultural extension service for specific treatment recommendations based on your location and growing conditions.",
        "prevention": "Practice crop rotation, ensure good air circulation, use disease-free seeds, and maintain plant vigor with proper nutrition and watering."
    }

class ChatbotManager:
    """Simple chatbot manager for providing agricultural information."""
    
    def __init__(self):
        """Initialize with predefined responses for agriculture topics."""
        # Define responses for different topics in English
        self.responses_en = {
            "greetings": [
                "Hello! I'm your AgroEdu assistant. How can I help you with agriculture or education today?",
                "Hi there! I'm here to assist with your farming and educational queries. What would you like to know?",
                "Welcome to AgroEdu! I'm your AI assistant for agriculture and vocational education."
            ],
            "farming_techniques": [
                "Sustainable farming techniques include crop rotation, cover cropping, and integrated pest management. These practices help maintain soil health and reduce environmental impact.",
                "Modern farming techniques focus on precision agriculture, which uses technology like GPS, sensors, and IoT devices to optimize inputs and maximize yields.",
                "Traditional farming wisdom combined with modern techniques can lead to the best results. Consider combining companion planting with data-driven irrigation systems."
            ],
            "crop_management": [
                "Effective crop management involves proper planning, timely planting, adequate irrigation, and regular monitoring for pests and diseases.",
                "For optimal crop yields, consider factors like soil preparation, seed selection, proper spacing, and regular fertilization based on soil tests.",
                "Integrated crop management combines biological, cultural, and chemical methods to produce healthy crops while minimizing environmental impact."
            ],
            "pest_control": [
                "Integrated Pest Management (IPM) combines biological controls, habitat manipulation, and resistant crop varieties with judicious pesticide use.",
                "Natural pest control methods include introducing beneficial insects, using companion planting, and applying neem oil or other organic solutions.",
                "Early detection is crucial for pest control. Regularly inspect your crops and consider using pheromone traps to monitor pest populations."
            ],
            "soil_health": [
                "Maintaining soil health requires regular testing, proper pH balance, adequate organic matter, and minimizing soil disturbance.",
                "Cover crops like legumes, grasses, and brassicas help improve soil structure, prevent erosion, and add nutrients when incorporated.",
                "Composting farm waste creates valuable organic matter that improves soil structure, drainage, and nutrient content when applied to fields."
            ],
            "water_management": [
                "Efficient water management includes drip irrigation, rainwater harvesting, and scheduling irrigation based on crop water requirements.",
                "Conservation techniques like mulching, contour farming, and building swales can help retain moisture in the soil and reduce water usage.",
                "Smart irrigation systems with soil moisture sensors can optimize water use by delivering precise amounts only when needed."
            ],
            "market_trends": [
                "Current agricultural market trends show increasing demand for organic and sustainably grown products, offering premium pricing opportunities.",
                "Direct-to-consumer marketing through farmers markets, CSAs, and online platforms is growing, allowing farmers to capture more value.",
                "Value-added products like preserves, dried herbs, or specialty items can increase farm revenue and provide income outside the growing season."
            ],
            "farm_equipment": [
                "Modern farm equipment incorporates GPS guidance, variable rate technology, and automation to increase efficiency and precision.",
                "For small farms, multipurpose equipment like walk-behind tractors with various attachments often provides the best value and versatility.",
                "Regular maintenance of farm equipment extends its lifespan and prevents costly breakdowns during critical farming operations."
            ],
            "education": [
                "Agricultural education provides hands-on learning opportunities for students interested in farming, forestry, and natural resources management.",
                "Vocational training in agriculture includes skills like equipment operation, crop production, livestock management, and farm business planning.",
                "Continuing education for farmers covers new technologies, sustainable practices, market trends, and regulatory compliance."
            ],
            "plant_disease": [
                "Plant disease management starts with prevention through proper spacing, resistant varieties, and maintaining good air circulation.",
                "Early identification of plant diseases is crucial. Look for symptoms like leaf spots, wilting, yellowing, powdery coatings, or abnormal growth.",
                "Treatment options for plant diseases include cultural practices, biological controls, and chemical interventions when necessary."
            ],
            "about": [
                "I'm an AI assistant specializing in agricultural and vocational education. I can provide information on farming practices, plant diseases, market trends, and more.",
                "AgroEdu is a platform designed to connect farmers with educational resources, tech tools, and expert knowledge to improve agricultural productivity.",
                "Our mission is to make agricultural education and information accessible to everyone, from small-scale farmers to agricultural students and professionals."
            ],
            "help": [
                "I can help with topics like farming techniques, crop management, pest control, soil health, water management, market trends, and farm equipment. Just ask me anything about agriculture or vocational education!",
                "You can ask me specific questions about farming practices, plant diseases, market predictions, or click on one of the topic buttons to explore different agricultural subjects.",
                "To get the most helpful information, try being specific about your question. For example, ask 'How do I improve soil fertility?' rather than just 'Tell me about soil.'"
            ],
            "default": [
                "That's an interesting agricultural topic. Could you provide more details so I can give you a more specific answer?",
                "I'm here to help with farming and educational questions. Could you elaborate on what you'd like to know?",
                "I'd be happy to discuss this further. Can you tell me more about your specific farming situation or question?"
            ]
        }
        
        # Define responses for different topics in Hindi
        self.responses_hi = {
            "greetings": [
                "नमस्ते! मैं आपका एग्रोएजु सहायक हूँ। मैं आज आपकी कृषि या शिक्षा से संबंधित किस प्रकार से मदद कर सकता हूँ?",
                "नमस्कार! मैं आपके कृषि और शैक्षिक प्रश्नों में सहायता करने के लिए यहां हूं। आप क्या जानना चाहेंगे?",
                "एग्रोएजु में आपका स्वागत है! मैं कृषि और व्यावसायिक शिक्षा के लिए आपका एआई सहायक हूं।"
            ],
            "farming_techniques": [
                "स्थायी कृषि तकनीकों में फसल चक्र, कवर क्रॉपिंग और एकीकृत कीट प्रबंधन शामिल हैं। ये प्रथाएं मिट्टी के स्वास्थ्य को बनाए रखने और पर्यावरणीय प्रभाव को कम करने में मदद करती हैं।",
                "आधुनिक कृषि तकनीकें सटीक कृषि पर केंद्रित हैं, जो इनपुट को अनुकूलित करने और उपज को अधिकतम करने के लिए जीपीएस, सेंसर और IoT उपकरणों जैसी तकनीक का उपयोग करती हैं।",
                "पारंपरिक कृषि ज्ञान के साथ आधुनिक तकनीकों का संयोजन सर्वोत्तम परिणाम दे सकता है। डेटा-संचालित सिंचाई प्रणालियों के साथ सहयोगी रोपण पर विचार करें।"
            ],
            "crop_management": [
                "प्रभावी फसल प्रबंधन में उचित योजना, समय पर रोपण, पर्याप्त सिंचाई और कीटों और रोगों के लिए नियमित निगरानी शामिल है।",
                "इष्टतम फसल उपज के लिए, मिट्टी की तैयारी, बीज चयन, उचित अंतरिक्ष और मिट्टी परीक्षणों के आधार पर नियमित उर्वरक जैसे कारकों पर विचार करें।",
                "एकीकृत फसल प्रबंधन पर्यावरणीय प्रभाव को कम करते हुए स्वस्थ फसलों का उत्पादन करने के लिए जैविक, सांस्कृतिक और रासायनिक विधियों को जोड़ती है।"
            ],
            "pest_control": [
                "एकीकृत कीट प्रबंधन (IPM) समझदार कीटनाशक उपयोग के साथ जैविक नियंत्रण, आवास हेरफेर और प्रतिरोधी फसल किस्मों को जोड़ता है।",
                "प्राकृतिक कीट नियंत्रण विधियों में लाभकारी कीड़ों को पेश करना, सहयोगी रोपण का उपयोग करना और नीम के तेल या अन्य जैविक समाधानों का प्रयोग करना शामिल है।",
                "कीट नियंत्रण के लिए प्रारंभिक पता लगाना महत्वपूर्ण है। नियमित रूप से अपनी फसलों का निरीक्षण करें और कीट आबादी की निगरानी के लिए फेरोमोन ट्रैप का उपयोग करने पर विचार करें।"
            ],
            "soil_health": [
                "मिट्टी के स्वास्थ्य को बनाए रखने के लिए नियमित परीक्षण, उचित पीएच संतुलन, पर्याप्त जैविक पदार्थ और मिट्टी में गड़बड़ी को कम करना आवश्यक है।",
                "लेग्यूम, घास और ब्रासिका जैसी कवर फसलें मिट्टी की संरचना में सुधार करने, कटाव को रोकने और शामिल होने पर पोषक तत्व जोड़ने में मदद करती हैं।",
                "खेत के कचरे का कंपोस्टिंग मूल्यवान जैविक पदार्थ बनाता है जो खेतों में लागू होने पर मिट्टी की संरचना, जल निकासी और पोषक तत्व सामग्री में सुधार करता है।"
            ],
            "water_management": [
                "कुशल जल प्रबंधन में ड्रिप सिंचाई, वर्षा जल संचयन और फसल जल आवश्यकताओं के आधार पर सिंचाई शेड्यूलिंग शामिल है।",
                "मल्चिंग, कंटूर खेती और स्वेल्स बनाने जैसी संरक्षण तकनीकें मिट्टी में नमी को बनाए रखने और पानी के उपयोग को कम करने में मदद कर सकती हैं।",
                "मिट्टी की नमी सेंसर वाले स्मार्ट सिंचाई सिस्टम केवल जरूरत पड़ने पर सटीक मात्रा देकर पानी के उपयोग को अनुकूलित कर सकते हैं।"
            ],
            "market_trends": [
                "वर्तमान कृषि बाजार रुझानों में जैविक और स्थायी रूप से उगाए गए उत्पादों की बढ़ती मांग दिखाई देती है, जो प्रीमियम मूल्य निर्धारण के अवसर प्रदान करती है।",
                "किसान बाजारों, सीएसए और ऑनलाइन प्लेटफॉर्म के माध्यम से प्रत्यक्ष-से-उपभोक्ता विपणन बढ़ रहा है, जिससे किसानों को अधिक मूल्य प्राप्त करने की अनुमति मिलती है।",
                "परिरक्षित, सूखे जड़ी-बूटियों या विशेष वस्तुओं जैसे मूल्य-वर्धित उत्पाद खेत राजस्व बढ़ा सकते हैं और उगाने के मौसम के बाहर आय प्रदान कर सकते हैं।"
            ],
            "farm_equipment": [
                "आधुनिक कृषि उपकरण दक्षता और सटीकता बढ़ाने के लिए जीपीएस मार्गदर्शन, परिवर्तनीय दर प्रौद्योगिकी और स्वचालन को शामिल करते हैं।",
                "छोटे खेतों के लिए, विभिन्न अटैचमेंट के साथ वॉक-बिहाइंड ट्रैक्टर जैसे बहुउद्देश्यीय उपकरण अक्सर सबसे अच्छा मूल्य और बहुमुखी प्रतिभा प्रदान करते हैं।",
                "कृषि उपकरणों का नियमित रखरखाव उनके जीवनकाल को बढ़ाता है और महत्वपूर्ण कृषि संचालन के दौरान महंगे ब्रेकडाउन को रोकता है।"
            ],
            "education": [
                "कृषि शिक्षा खेती, वानिकी और प्राकृतिक संसाधन प्रबंधन में रुचि रखने वाले छात्रों के लिए हाथों में सीखने के अवसर प्रदान करती है।",
                "कृषि में व्यावसायिक प्रशिक्षण में उपकरण संचालन, फसल उत्पादन, पशुधन प्रबंधन और खेत व्यापार योजना जैसे कौशल शामिल हैं।",
                "किसानों के लिए निरंतर शिक्षा में नई तकनीकें, स्थायी प्रथाएं, बाजार के रुझान और नियामक अनुपालन शामिल हैं।"
            ],
            "plant_disease": [
                "पौधों के रोग प्रबंधन की शुरुआत उचित अंतरिक्ष, प्रतिरोधी किस्मों और अच्छे वायु परिसंचरण को बनाए रखने के माध्यम से रोकथाम से होती है।",
                "पौधों के रोगों की प्रारंभिक पहचान महत्वपूर्ण है। पत्ती के धब्बे, मुरझाने, पीले होने, पाउडरी कोटिंग या असामान्य विकास जैसे लक्षणों की तलाश करें।",
                "पौधों के रोगों के लिए उपचार विकल्पों में सांस्कृतिक प्रथाएं, जैविक नियंत्रण और आवश्यकतानुसार रासायनिक हस्तक्षेप शामिल हैं।"
            ],
            "about": [
                "मैं कृषि और व्यावसायिक शिक्षा में विशेषज्ञता रखने वाला एक एआई सहायक हूं। मैं खेती प्रथाओं, पौधों के रोगों, बाजार के रुझानों और अधिक पर जानकारी प्रदान कर सकता हूं।",
                "एग्रोएजु एक ऐसा प्लेटफॉर्म है जिसे किसानों को शैक्षिक संसाधनों, तकनीकी उपकरणों और कृषि उत्पादकता में सुधार के लिए विशेषज्ञ ज्ञान से जोड़ने के लिए डिज़ाइन किया गया है।",
                "हमारा मिशन कृषि शिक्षा और जानकारी को सभी के लिए सुलभ बनाना है, छोटे पैमाने के किसानों से लेकर कृषि छात्रों और पेशेवरों तक।"
            ],
            "help": [
                "मैं खेती तकनीकों, फसल प्रबंधन, कीट नियंत्रण, मिट्टी के स्वास्थ्य, जल प्रबंधन, बाजार के रुझान और कृषि उपकरण जैसे विषयों के साथ मदद कर सकता हूं। बस मुझसे कृषि या व्यावसायिक शिक्षा के बारे में कुछ भी पूछें!",
                "आप मुझसे खेती प्रथाओं, पौधों के रोगों, बाजार भविष्यवाणियों के बारे में विशिष्ट प्रश्न पूछ सकते हैं या विभिन्न कृषि विषयों का पता लगाने के लिए विषय बटनों में से एक पर क्लिक कर सकते हैं।",
                "सबसे अधिक सहायक जानकारी प्राप्त करने के लिए, अपने प्रश्न के बारे में विशिष्ट होने का प्रयास करें। उदाहरण के लिए, केवल 'मिट्टी के बारे में बताओ' के बजाय 'मैं मिट्टी की उर्वरता कैसे सुधार सकता हूं?' पूछें।"
            ],
            "default": [
                "यह एक दिलचस्प कृषि विषय है। क्या आप अधिक विवरण प्रदान कर सकते हैं ताकि मैं आपको अधिक विशिष्ट उत्तर दे सकूं?",
                "मैं खेती और शैक्षिक प्रश्नों के साथ मदद करने के लिए यहां हूं। क्या आप विस्तार से बता सकते हैं कि आप क्या जानना चाहते हैं?",
                "मैं इस पर आगे चर्चा करने के लिए खुश हूंगा। क्या आप मुझे अपनी विशिष्ट खेती स्थिति या प्रश्न के बारे में अधिक बता सकते हैं?"
            ]
        }
        
        # Topic mapping for better keyword matching - only define once
        self.topic_keywords = {
            "greetings": ["hello", "hi", "hey", "greetings", "howdy", "good morning", "good afternoon", "good evening", "welcome", "नमस्ते", "नमस्कार", "सुप्रभात", "शुभ दिन", "स्वागत"],
            "farming_techniques": ["farming technique", "how to farm", "cultivation method", "agricultural practice", "sustainable", "organic farming", "permaculture", "no-till", "farming method", "खेती तकनीक", "कैसे खेती करें", "खेती विधि", "कृषि अभ्यास", "टिकाऊ", "जैविक खेती"],
            "crop_management": ["crop management", "grow crops", "crop production", "yield", "planting", "harvesting", "crop rotation", "seed", "seedling", "transplant", "फसल प्रबंधन", "फसल उगाना", "फसल उत्पादन", "उपज", "रोपण", "कटाई", "फसल चक्र", "बीज", "पौध", "प्रत्यारोपण"],
            "pest_control": ["pest", "insect", "disease control", "plant protection", "pesticide", "herbicide", "fungicide", "weeds", "natural pest control", "predator", "कीट", "कीड़े", "रोग नियंत्रण", "पौधों की सुरक्षा", "कीटनाशक", "खरपतवारनाशक", "फफूंदनाशक", "खरपतवार", "प्राकृतिक कीट नियंत्रण"],
            "soil_health": ["soil", "compost", "fertilizer", "nutrients", "organic matter", "humus", "topsoil", "soil structure", "ph level", "minerals", "मिट्टी", "खाद", "उर्वरक", "पोषक तत्व", "जैविक पदार्थ", "ह्यूमस", "उपरी मिट्टी", "मिट्टी की संरचना", "पीएच स्तर", "खनिज"],
            "water_management": ["water", "irrigation", "moisture", "drought", "watering", "drip", "sprinkler", "rainwater", "conservation", "runoff", "पानी", "सिंचाई", "नमी", "सूखा", "पानी देना", "ड्रिप", "स्प्रिंकलर", "बारिश का पानी", "संरक्षण", "अपवाह"],
            "market_trends": ["market", "price", "sell", "demand", "trend", "marketing", "consumer", "supply chain", "wholesale", "retail", "export", "बाजार", "कीमत", "बेचना", "मांग", "प्रवृत्ति", "विपणन", "उपभोक्ता", "आपूर्ति श्रृंखला", "थोक", "खुदरा", "निर्यात"],
            "farm_equipment": ["equipment", "tool", "machinery", "tractor", "implement", "harvester", "plow", "seeder", "sprayer", "machine", "उपकरण", "औजार", "मशीनरी", "ट्रैक्टर", "कार्यान्वयन", "हार्वेस्टर", "हल", "सीडर", "स्प्रेयर", "मशीन"],
            "education": ["education", "learn", "study", "course", "training", "certificate", "degree", "workshop", "skill", "knowledge", "vocational", "शिक्षा", "सीखना", "अध्ययन", "पाठ्यक्रम", "प्रशिक्षण", "प्रमाणपत्र", "डिग्री", "कार्यशाला", "कौशल", "ज्ञान", "व्यावसायिक"],
            "plant_disease": ["disease", "fungus", "bacteria", "virus", "infection", "pathogen", "symptom", "treatment", "prevention", "diagnosis", "रोग", "फफूंदी", "बैक्टीरिया", "वायरस", "संक्रमण", "रोगाणु", "लक्षण", "उपचार", "रोकथाम", "निदान"],
            "about": ["about", "who are you", "what are you", "what can you do", "your purpose", "your function", "tell me about yourself", "agrobot", "agroedu", "बारे में", "आप कौन हैं", "आप क्या हैं", "आप क्या कर सकते हैं", "आपका उद्देश्य", "अपने बारे में बताओ"],
            "help": ["help", "assist", "guide", "support", "advice", "suggestion", "how to use", "tutorial", "explain", "मदद", "सहायता", "मार्गदर्शन", "समर्थन", "सलाह", "सुझाव", "उपयोग कैसे करें", "ट्यूटोरियल", "समझाएं"]
        }
        
    def generate_response(self, user_message, user_language="en"):
        """Generate a response based on the user's message and language preference"""
        if not user_message:
            # Select responses based on language
            responses = self.responses_hi if user_language == "hi" else self.responses_en
            return random.choice(responses["greetings"])
            
        message = user_message.lower()
        
        # Score each topic based on keyword matches
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in message:
                    # Give more weight to multi-word matches
                    if " " in keyword:
                        score += len(keyword.split()) * 2
                    else:
                        score += 1
            topic_scores[topic] = score
        
        # Find the highest scoring topic
        max_score = 0
        best_topic = "default"
        
        for topic, score in topic_scores.items():
            if score > max_score:
                max_score = score
                best_topic = topic
                
        # If no keywords matched or very low score, use default responses
        if max_score < 1:
            best_topic = "default"
        
        # Select responses based on language
        responses = self.responses_hi if user_language == "hi" else self.responses_en
            
        # Return a random response from the best matching topic
        return random.choice(responses[best_topic])

# Initialize the chatbot manager
chatbot_manager = ChatbotManager()

@app.route('/api/crop_recommendations', methods=['POST'])
def get_crop_recommendations():
    """Get personalized crop recommendations based on location and season."""
    try:
        data = request.get_json()
        location = data.get('location', '')
        season = data.get('season', '')
        
        if not location or not season:
            return jsonify({
                'error': 'Location and season are required',
                'status': 'error'
            }), 400
            
        # Initialize market predictor if not already done
        if not hasattr(app, 'market_predictor'):
            app.market_predictor = MarketPredictor()
            
        recommendations = app.market_predictor.get_crop_recommendations(location, season)
        return jsonify(recommendations)
        
    except Exception as e:
        print(f"Error in crop_recommendations route: {str(e)}")
        return jsonify({
            'error': 'Could not generate crop recommendations at this time.',
            'status': 'error'
        }), 500

@app.route('/api/market_alerts', methods=['POST'])
def setup_market_alerts():
    """Set up market price alerts for farmers."""
    try:
        data = request.get_json()
        crop = data.get('crop', '')
        farmer_id = data.get('farmer_id', '')
        contact_info = data.get('contact_info', {})
        
        if not all([crop, farmer_id, contact_info]):
            return jsonify({
                'error': 'Crop, farmer ID, and contact information are required',
                'status': 'error'
            }), 400
            
        # Initialize market predictor if not already done
        if not hasattr(app, 'market_predictor'):
            app.market_predictor = MarketPredictor()
            
        alert = app.market_predictor.generate_farmer_alert(crop, farmer_id, contact_info)
        return jsonify(alert)
        
    except Exception as e:
        print(f"Error in market_alerts route: {str(e)}")
        return jsonify({
            'error': 'Could not set up market alerts at this time.',
            'status': 'error'
        }), 500

@app.route('/api/market_trends', methods=['GET'])
def get_market_trends():
    try:
        # Return structured market trends data with all required fields
        trends = {
            "Rice": {
                "direction": "up",
                "percentage": 4.5,
                "time_period": "Last 30 days",
                "description": "Rice prices have increased due to strong export demand and lower production estimates.",
                "current_price": 2200,
                "unit": "quintal"
            },
            "Wheat": {
                "direction": "down",
                "percentage": 2.3,
                "time_period": "Last 30 days",
                "description": "Wheat prices have decreased slightly due to favorable weather conditions and increased sowing area.",
                "current_price": 1980,
                "unit": "quintal"
            },
            "Maize": {
                "direction": "stable",
                "percentage": 0.5,
                "time_period": "Last 30 days",
                "description": "Maize prices remain stable with balanced supply and demand.",
                "current_price": 1650,
                "unit": "quintal"
            },
            "Onion": {
                "direction": "up",
                "percentage": 15.8,
                "time_period": "Last 30 days",
                "description": "Onion prices have surged due to unseasonal rains affecting harvest in major producing regions.",
                "current_price": 3500,
                "unit": "quintal"
            },
            "Potato": {
                "direction": "down",
                "percentage": 5.2,
                "time_period": "Last 30 days",
                "description": "Potato prices have dropped following fresh harvest arrivals in the market.",
                "current_price": 1580,
                "unit": "quintal"
            },
            "Cotton": {
                "direction": "up",
                "percentage": 3.7,
                "time_period": "Last 30 days",
                "description": "Cotton prices have increased due to high international demand and lower global stockpiles.",
                "current_price": 6200,
                "unit": "quintal"
            }
        }
        return jsonify({'status': 'success', 'trends': trends})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# New Soil Health Analyzer API endpoint
@app.route('/api/soil_health_analysis', methods=['POST'])
def analyze_soil_health():
    try:
        data = request.json
        
        # Extract soil parameters
        ph_level = float(data.get('ph_level', 0))
        nitrogen = float(data.get('nitrogen', 0))
        phosphorus = float(data.get('phosphorus', 0))
        potassium = float(data.get('potassium', 0))
        organic_matter = float(data.get('organic_matter', 0))
        moisture = float(data.get('moisture', 0))
        
        # Validate input values
        if ph_level < 0 or ph_level > 14:
            return jsonify({'status': 'error', 'message': 'pH level must be between 0 and 14'})
        
        if any(param < 0 for param in [nitrogen, phosphorus, potassium, organic_matter, moisture]):
            return jsonify({'status': 'error', 'message': 'All nutrient values must be positive'})
        
        # Perform soil health analysis
        health_score = calculate_soil_health_score(ph_level, nitrogen, phosphorus, potassium, organic_matter, moisture)
        soil_type = determine_soil_type(ph_level, organic_matter)
        
        # Generate crop recommendations
        recommendations = get_crop_recommendations_for_soil(ph_level, nitrogen, phosphorus, potassium, soil_type)
        
        # Generate improvement tips
        improvement_tips = generate_soil_improvement_tips(ph_level, nitrogen, phosphorus, potassium, organic_matter, moisture)
        
        # Return analysis results
        analysis_result = {
            'status': 'success',
            'soil_health_score': health_score,
            'soil_type': soil_type,
            'analysis': {
                'ph_level': {
                    'value': ph_level,
                    'status': get_ph_status(ph_level),
                    'ideal_range': '6.0-7.5'
                },
                'nitrogen': {
                    'value': nitrogen,
                    'status': get_nutrient_status(nitrogen, 'nitrogen'),
                    'ideal_range': '100-200 mg/kg'
                },
                'phosphorus': {
                    'value': phosphorus,
                    'status': get_nutrient_status(phosphorus, 'phosphorus'),
                    'ideal_range': '20-40 mg/kg'
                },
                'potassium': {
                    'value': potassium,
                    'status': get_nutrient_status(potassium, 'potassium'),
                    'ideal_range': '100-300 mg/kg'
                },
                'organic_matter': {
                    'value': organic_matter,
                    'status': get_organic_matter_status(organic_matter),
                    'ideal_range': '3-6%'
                },
                'moisture': {
                    'value': moisture,
                    'status': get_moisture_status(moisture),
                    'ideal_range': '20-60%'
                }
            },
            'recommendations': recommendations,
            'improvement_tips': improvement_tips
        }
        
        return jsonify(analysis_result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def calculate_soil_health_score(ph, nitrogen, phosphorus, potassium, organic_matter, moisture):
    """Calculate an overall soil health score based on soil parameters."""
    # pH score (ideal range: 6.0-7.5)
    if 6.0 <= ph <= 7.5:
        ph_score = 20
    elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
        ph_score = 15
    elif 5.0 <= ph < 5.5 or 8.0 < ph <= 8.5:
        ph_score = 10
    else:
        ph_score = 5
    
    # Nitrogen score (ideal range: 100-200 mg/kg)
    if 100 <= nitrogen <= 200:
        n_score = 20
    elif 50 <= nitrogen < 100 or 200 < nitrogen <= 250:
        n_score = 15
    elif 20 <= nitrogen < 50 or 250 < nitrogen <= 300:
        n_score = 10
    else:
        n_score = 5
    
    # Phosphorus score (ideal range: 20-40 mg/kg)
    if 20 <= phosphorus <= 40:
        p_score = 15
    elif 10 <= phosphorus < 20 or 40 < phosphorus <= 60:
        p_score = 10
    elif 5 <= phosphorus < 10 or 60 < phosphorus <= 80:
        p_score = 8
    else:
        p_score = 5
    
    # Potassium score (ideal range: 100-300 mg/kg)
    if 100 <= potassium <= 300:
        k_score = 15
    elif 50 <= potassium < 100 or 300 < potassium <= 400:
        k_score = 10
    elif 20 <= potassium < 50 or 400 < potassium <= 500:
        k_score = 8
    else:
        k_score = 5
    
    # Organic matter score (ideal range: 3-6%)
    if 3 <= organic_matter <= 6:
        om_score = 15
    elif 1.5 <= organic_matter < 3 or 6 < organic_matter <= 8:
        om_score = 10
    elif 0.5 <= organic_matter < 1.5 or 8 < organic_matter <= 10:
        om_score = 8
    else:
        om_score = 5
    
    # Moisture score (ideal range: 20-60%)
    if 20 <= moisture <= 60:
        moisture_score = 15
    elif 10 <= moisture < 20 or 60 < moisture <= 70:
        moisture_score = 10
    elif 5 <= moisture < 10 or 70 < moisture <= 80:
        moisture_score = 8
    else:
        moisture_score = 5
    
    # Calculate total score (out of 100)
    total_score = ph_score + n_score + p_score + k_score + om_score + moisture_score
    
    return total_score

def determine_soil_type(ph, organic_matter):
    """Determine soil type based on pH and organic matter content."""
    if organic_matter < 2:
        if ph < 6.0:
            return "Sandy Acidic Soil"
        elif ph > 7.5:
            return "Sandy Alkaline Soil"
        else:
            return "Sandy Loam"
    elif 2 <= organic_matter <= 5:
        if ph < 6.0:
            return "Loamy Acidic Soil"
        elif ph > 7.5:
            return "Loamy Alkaline Soil"
        else:
            return "Loam Soil"
    else:
        if ph < 6.0:
            return "Clay Acidic Soil"
        elif ph > 7.5:
            return "Clay Alkaline Soil"
        else:
            return "Clay Loam"

def get_ph_status(ph):
    """Determine pH status based on value."""
    if ph < 5.5:
        return "Acidic"
    elif 5.5 <= ph <= 6.5:
        return "Slightly Acidic"
    elif 6.5 < ph <= 7.5:
        return "Neutral (Optimal)"
    elif 7.5 < ph <= 8.5:
        return "Slightly Alkaline"
    else:
        return "Alkaline"

def get_nutrient_status(value, nutrient_type):
    """Determine nutrient status based on value and type."""
    if nutrient_type == 'nitrogen':
        if value < 50:
            return "Low"
        elif 50 <= value < 100:
            return "Medium"
        elif 100 <= value <= 200:
            return "Optimal"
        else:
            return "High"
    elif nutrient_type == 'phosphorus':
        if value < 10:
            return "Low"
        elif 10 <= value < 20:
            return "Medium"
        elif 20 <= value <= 40:
            return "Optimal"
        else:
            return "High"
    elif nutrient_type == 'potassium':
        if value < 50:
            return "Low"
        elif 50 <= value < 100:
            return "Medium"
        elif 100 <= value <= 300:
            return "Optimal"
        else:
            return "High"
    return "Unknown"

def get_organic_matter_status(value):
    """Determine organic matter status based on value."""
    if value < 1.5:
        return "Very Low"
    elif 1.5 <= value < 3:
        return "Low"
    elif 3 <= value <= 6:
        return "Optimal"
    elif 6 < value <= 8:
        return "High"
    else:
        return "Very High"

def get_moisture_status(value):
    """Determine moisture status based on value."""
    if value < 10:
        return "Very Dry"
    elif 10 <= value < 20:
        return "Dry"
    elif 20 <= value <= 60:
        return "Optimal"
    elif 60 < value <= 70:
        return "Moist"
    else:
        return "Waterlogged"

def get_crop_recommendations_for_soil(ph, nitrogen, phosphorus, potassium, soil_type):
    """Get crop recommendations based on soil parameters."""
    recommendations = []
    
    # Check for acidic soil conditions
    if ph < 6.0:
        recommendations.extend([
            {"crop": "Blueberries", "suitability": "Excellent", "notes": "Thrive in acidic soil with pH 4.5-5.5"},
            {"crop": "Potatoes", "suitability": "Good", "notes": "Prefer slightly acidic soil with pH 5.0-6.0"},
            {"crop": "Sweet Potatoes", "suitability": "Good", "notes": "Grow well in sandy acidic soils"},
            {"crop": "Azaleas", "suitability": "Excellent", "notes": "Flourish in acidic soils"}
        ])
    
    # Check for alkaline soil conditions
    elif ph > 7.5:
        recommendations.extend([
            {"crop": "Lettuce", "suitability": "Excellent", "notes": "Tolerates alkaline soils well"},
            {"crop": "Spinach", "suitability": "Good", "notes": "Prefers slightly alkaline soil conditions"},
            {"crop": "Cabbage", "suitability": "Good", "notes": "Grows well in alkaline soil with good organic matter"},
            {"crop": "Cauliflower", "suitability": "Good", "notes": "Suitable for alkaline soils"}
        ])
    
    # Neutral pH
    else:
        recommendations.extend([
            {"crop": "Tomatoes", "suitability": "Excellent", "notes": "Thrive in neutral soil with adequate nitrogen"},
            {"crop": "Peppers", "suitability": "Good", "notes": "Prefer well-drained soil with neutral pH"},
            {"crop": "Corn", "suitability": "Good", "notes": "Requires nitrogen-rich soil"},
            {"crop": "Beans", "suitability": "Excellent", "notes": "Fix nitrogen in soil and grow well in neutral pH"}
        ])
    
    # Additional recommendations based on nutrients
    if nitrogen > 150 and phosphorus > 30 and potassium > 150:
        recommendations.extend([
            {"crop": "Leafy Greens", "suitability": "Excellent", "notes": "Thrive in nutrient-rich soil"},
            {"crop": "Broccoli", "suitability": "Excellent", "notes": "Requires nutrient-rich soil"}
        ])
    
    # Recommendations based on soil type
    if "Sandy" in soil_type:
        recommendations.extend([
            {"crop": "Carrots", "suitability": "Excellent", "notes": "Grow well in sandy soil"},
            {"crop": "Radishes", "suitability": "Excellent", "notes": "Prefer sandy, well-drained soil"}
        ])
    elif "Clay" in soil_type:
        recommendations.extend([
            {"crop": "Wheat", "suitability": "Good", "notes": "Tolerates clay soil well"},
            {"crop": "Rice", "suitability": "Excellent", "notes": "Ideal for water-retaining clay soils"}
        ])
    elif "Loam" in soil_type:
        recommendations.extend([
            {"crop": "Cucumbers", "suitability": "Excellent", "notes": "Thrive in loamy soil"},
            {"crop": "Squash", "suitability": "Excellent", "notes": "Grows best in fertile loam"}
        ])
    
    # Return top 5 recommendations
    return recommendations[:5]

def generate_soil_improvement_tips(ph, nitrogen, phosphorus, potassium, organic_matter, moisture):
    """Generate tips for improving soil health based on analysis."""
    tips = []
    
    # pH improvement tips
    if ph < 5.5:
        tips.append({
            "category": "pH Improvement",
            "title": "Raise Soil pH",
            "description": "Add agricultural lime to increase pH. Apply according to soil test recommendations.",
            "priority": "High"
        })
    elif ph > 7.5:
        tips.append({
            "category": "pH Improvement",
            "title": "Lower Soil pH",
            "description": "Add elemental sulfur, aluminum sulfate, or organic matter like pine needles to gradually lower pH.",
            "priority": "High"
        })
    
    # Nitrogen improvement
    if nitrogen < 50:
        tips.append({
            "category": "Nitrogen Management",
            "title": "Increase Nitrogen",
            "description": "Add nitrogen-rich fertilizers, incorporate legume cover crops, or add composted manure.",
            "priority": "High" if nitrogen < 30 else "Medium"
        })
    elif nitrogen > 200:
        tips.append({
            "category": "Nitrogen Management",
            "title": "Reduce Nitrogen",
            "description": "Plant nitrogen-consuming crops and avoid adding nitrogen fertilizers temporarily.",
            "priority": "Medium"
        })
    
    # Phosphorus improvement
    if phosphorus < 10:
        tips.append({
            "category": "Phosphorus Management",
            "title": "Increase Phosphorus",
            "description": "Add rock phosphate, bone meal, or phosphorus-containing fertilizers.",
            "priority": "High" if phosphorus < 5 else "Medium"
        })
    
    # Potassium improvement
    if potassium < 50:
        tips.append({
            "category": "Potassium Management",
            "title": "Increase Potassium",
            "description": "Add wood ash, kelp meal, or potassium-containing fertilizers like potassium sulfate.",
            "priority": "High" if potassium < 30 else "Medium"
        })
    
    # Organic matter improvement
    if organic_matter < 3:
        tips.append({
            "category": "Organic Matter",
            "title": "Increase Organic Content",
            "description": "Add compost, plant cover crops, use mulch, and incorporate crop residues into soil.",
            "priority": "High" if organic_matter < 1.5 else "Medium"
        })
    
    # Moisture management
    if moisture < 20:
        tips.append({
            "category": "Moisture Management",
            "title": "Improve Water Retention",
            "description": "Add organic matter, use mulch, consider drip irrigation, and install water-conserving practices.",
            "priority": "High" if moisture < 10 else "Medium"
        })
    elif moisture > 60:
        tips.append({
            "category": "Moisture Management",
            "title": "Improve Drainage",
            "description": "Install drainage systems, create raised beds, and add organic matter to improve soil structure.",
            "priority": "High" if moisture > 70 else "Medium"
        })
    
    # General soil health tip
    tips.append({
        "category": "General Soil Health",
        "title": "Regular Soil Testing",
        "description": "Test your soil annually to monitor changes and adjust management practices accordingly.",
        "priority": "Medium"
    })
    
    return tips

@app.route('/api/set_language', methods=['POST'])
def set_language():
    """Set the application language for UI and responses."""
    try:
        # First get JSON data with error handling
        try:
            data = request.get_json(silent=True)
            if not data:
                print("Invalid or missing JSON data in set_language request")
                return jsonify({'status': 'error', 'message': 'Invalid or missing request data'}), 400
        except Exception as json_err:
            print(f"JSON parsing error: {str(json_err)}")
            return jsonify({'status': 'error', 'message': 'Invalid JSON format'}), 400
            
        # Check for language parameter
        if 'language' not in data:
            print("No language specified in request")
            return jsonify({'status': 'error', 'message': 'No language specified'}), 400
        
        # Get language code and normalize it
        language = str(data.get('language', '')).lower().strip()
        if not language:
            print("Empty language code provided")
            return jsonify({'status': 'error', 'message': 'Empty language code'}), 400
            
        print(f"Setting application language to: {language}")
        
        # Validate language code
        valid_languages = ['en', 'hi', 'bn', 'ta', 'te', 'mr']
        if language not in valid_languages:
            print(f"Invalid language code: {language}")
            return jsonify({
                'status': 'error', 
                'message': f'Invalid language code. Supported languages are: {", ".join(valid_languages)}'
            }), 400
        
        # Store language in session
        old_language = session.get('app_language', 'en')
        session['app_language'] = language
        print(f"Updated session language from {old_language} to {language}")
        
        # Setup voice settings
        voice_language = None
        try:
            # Initialize voice settings structure if needed
            if 'user_voice_settings' not in session:
                session['user_voice_settings'] = {}
            
            # Map language codes to voice language codes
            voice_lang_map = {
                'en': 'en-US',
                'hi': 'hi-IN',
                'bn': 'bn-IN', 
                'ta': 'ta-IN',
                'te': 'te-IN',
                'mr': 'mr-IN'
            }
            
            voice_language = voice_lang_map.get(language, 'en-US')
            session['user_voice_settings']['language'] = voice_language
            print(f"Set voice language to: {voice_language}")
        except Exception as voice_settings_err:
            print(f"Error setting voice settings: {str(voice_settings_err)}")
            # Not critical, continue
        
        # Update external components (non-critical operations)
        components_updated = []
        
        # Try to update settings in voice manager if available
        try:
            if 'voice_mgr' in globals() and voice_mgr is not None:
                if hasattr(voice_mgr, 'set_language'):
                    print(f"Setting voice manager language to: {language}")
                    voice_mgr.set_language(language)
                    components_updated.append('voice_manager')
                elif hasattr(voice_mgr, 'update_settings'):
                    print(f"Updating voice manager settings for language: {language}")
                    voice_mgr.update_settings(session.get('user_id', 'default_user'), {'language': language})
                    components_updated.append('voice_manager')
        except Exception as voice_err:
            print(f"Voice manager language update error (non-critical): {str(voice_err)}")
        
        # Update chatbot language if global instance is available
        try:
            if 'chatbot' in globals() and chatbot is not None and hasattr(chatbot, 'set_language_preference'):
                print(f"Setting chatbot language preference to: {language}")
                chatbot.set_language_preference(language)
                components_updated.append('chatbot')
            
            # Also update the ChatbotManager fallback
            if 'chatbot_manager' in globals() and chatbot_manager is not None:
                print(f"Setting ChatbotManager language preference to: {language}")
                # The ChatbotManager doesn't store a language preference itself,
                # but we'll remember for next time
                components_updated.append('chatbot_manager')
        except Exception as chatbot_err:
            print(f"Chatbot language update error (non-critical): {str(chatbot_err)}")
        
        # Return success response with details
        response = {
            'status': 'success',
            'language': language,
            'previous_language': old_language,
            'components_updated': components_updated
        }
        
        if voice_language:
            response['voice_language'] = voice_language
            
        return jsonify(response)
        
    except Exception as e:
        print(f"Unhandled error in set_language route: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Could not set language: {str(e)}'
        }), 500

# Look for PyTorch model in models directory
try:
    pytorch_model_path = os.path.join(os.path.dirname(__file__), "models", "trained_plant_disease_model.pth")
    if os.path.exists(pytorch_model_path):
        print(f"Found PyTorch model at {pytorch_model_path}")
        model_type = 'pytorch'
        print(f"Successfully found PyTorch plant disease model")
    else:
        print(f"Warning: PyTorch model not found at {pytorch_model_path}.", file=sys.stderr)
except Exception as e:
    print(f"Warning: Failed to check for PyTorch model: {str(e)}", file=sys.stderr)

@app.route('/api/tts-test', methods=['GET'])
def test_tts():
    print("TTS Test endpoint called")
    
    try:
        # Test text in English
        test_text = "This is a test of the text to speech system"
        
        # Create a temporary file
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        
        # Generate speech
        from gtts import gTTS
        tts = gTTS(text=test_text, lang='en', slow=False)
        tts.save(temp_filename)
        
        # Return the file
        response = send_file(
            temp_filename,
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name='tts_test.mp3'
        )
        
        # Delete file after sending
        @response.call_on_close
        def cleanup():
            import os
            try:
                os.unlink(temp_filename)
            except:
                pass
                
        return response
    except Exception as e:
        print(f"Error in TTS test: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/test')
def test_index():
    return render_template('test_index.html')

@app.route('/tts-test')
def tts_test_page():
    return render_template('tts_test.html')

@app.route('/tts-debug')
def tts_debug_page():
    """Render the TTS debugging page."""
    return render_template('tts_debug.html')

@app.route('/api/check-filesystem', methods=['GET'])
def check_filesystem():
    """Check if the application has proper permissions to create and delete temporary files."""
    try:
        import os
        import tempfile
        
        # Check temp directory
        temp_dir = os.path.join(os.getcwd(), 'temp_audio')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Test file creation and deletion
        try:
            # Create a test file
            test_file = tempfile.NamedTemporaryFile(dir=temp_dir, suffix='.test', delete=False)
            test_filename = test_file.name
            test_file.write(b'test')
            test_file.close()
            
            # Check if file exists
            file_exists = os.path.exists(test_filename)
            
            # Try to delete the file
            os.remove(test_filename)
            delete_success = not os.path.exists(test_filename)
            
            return jsonify({
                'success': True,
                'temp_dir': temp_dir,
                'write_access': file_exists,
                'delete_access': delete_success
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'temp_dir': temp_dir,
                'error': f'File operation error: {str(e)}'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Filesystem check failed: {str(e)}'
        }), 500

@app.route('/api/weather_forecast', methods=['POST'])
def get_weather_forecast():
    """Get weather forecast data for a location."""
    try:
        print("Weather forecast API called")
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        print(f"Weather request for coordinates: {latitude}, {longitude}")
        
        if not latitude or not longitude:
            print("Missing latitude or longitude")
            return jsonify({
                'status': 'error',
                'message': 'Latitude and longitude are required'
            }), 400
        
        # Generate mock data for reliable functioning
        mock_data = generate_mock_weather_data(latitude, longitude)
            
        # Try WeatherAPI.com if possible, but use mock data as fallback
        try:
            # WeatherAPI.com - using their forecast API
            api_key = "eeadf20837fe4071a9f03723253103"  # Updated API key
            url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={latitude},{longitude}&days=7&aqi=no&alerts=no"
            
            print(f"Making weather API request to: {url}")
            
            # Make the actual API call
            import requests
            response = requests.get(url, timeout=5)  # Short timeout to prevent delays
            print(f"API response status: {response.status_code}")
            
            if response.status_code == 200:
                weather_data = response.json()
                print(f"Weather API success: {weather_data}")
                
                # Format data to match our expected structure
                formatted_data = {
                    "current": {
                        "dt": weather_data.get("current", {}).get("last_updated_epoch", int(time.time())),
                        "temp": weather_data.get("current", {}).get("temp_c", 25),
                        "humidity": weather_data.get("current", {}).get("humidity", 65),
                        "wind_speed": weather_data.get("current", {}).get("wind_kph", 5.2),
                        "weather": [{
                            "main": weather_data.get("current", {}).get("condition", {}).get("text", "Clear"),
                            "description": weather_data.get("current", {}).get("condition", {}).get("text", "clear sky"),
                            "icon": weather_data.get("current", {}).get("condition", {}).get("icon", "//cdn.weatherapi.com/weather/64x64/day/113.png").replace("//cdn.weatherapi.com/weather/64x64/", "")
                        }]
                    },
                    "daily": [],
                    "location": {
                        "name": weather_data.get("location", {}).get("name", "Unknown Location"),
                        "country": weather_data.get("location", {}).get("country", "")
                    }
                }
                
                # Process forecast data
                forecast_days = weather_data.get("forecast", {}).get("forecastday", [])
                for day in forecast_days:
                    formatted_day = {
                        "dt": day.get("date_epoch", int(time.time())),
                        "temp": {
                            "day": day.get("day", {}).get("avgtemp_c", 25),
                            "min": day.get("day", {}).get("mintemp_c", 20),
                            "max": day.get("day", {}).get("maxtemp_c", 30)
                        },
                        "humidity": day.get("day", {}).get("avghumidity", 65),
                        "wind_speed": day.get("day", {}).get("maxwind_kph", 5.2),
                        "weather": [{
                            "main": day.get("day", {}).get("condition", {}).get("text", "Clear"),
                            "description": day.get("day", {}).get("condition", {}).get("text", "clear sky"),
                            "icon": day.get("day", {}).get("condition", {}).get("icon", "//cdn.weatherapi.com/weather/64x64/day/113.png").replace("//cdn.weatherapi.com/weather/64x64/", "")
                        }]
                    }
                    formatted_data["daily"].append(formatted_day)
                
                # Fill remaining days if needed
                while len(formatted_data["daily"]) < 7:
                    # Clone the last day with slight variations if we have at least one day
                    if formatted_data["daily"]:
                        last_day = formatted_data["daily"][-1]
                        new_day = {
                            "dt": last_day["dt"] + 86400,
                            "temp": {
                                "day": last_day["temp"]["day"] + random.uniform(-2, 2),
                                "min": last_day["temp"]["min"] + random.uniform(-1, 1),
                                "max": last_day["temp"]["max"] + random.uniform(-1, 1)
                            },
                            "humidity": last_day["humidity"] + random.randint(-5, 5),
                            "wind_speed": last_day["wind_speed"] + random.uniform(-1, 1),
                            "weather": last_day["weather"]
                        }
                        formatted_data["daily"].append(new_day)
                    else:
                        # Create a default day if no forecast data at all
                        mock_day = mock_data["daily"][0]
                        formatted_data["daily"].append(mock_day)
                
                # Add farming recommendations based on weather conditions
                formatted_data["farming_recommendations"] = generate_farming_recommendations(
                    formatted_data["current"]["temp"],
                    formatted_data["current"]["weather"][0]["main"],
                    formatted_data["current"]["wind_speed"]
                )
                
                return jsonify({
                    'status': 'success',
                    'data': formatted_data
                })
        except Exception as api_error:
            # If any API error occurs, use mock data
            print(f"API exception, using mock data instead: {str(api_error)}")
            return jsonify({
                'status': 'success',
                'data': mock_data
            })
            
    except Exception as e:
        print(f"Unexpected error in weather_forecast: {str(e)}")
        # Generate emergency mock data even if the main function fails
        emergency_mock = generate_mock_weather_data(latitude or 20.5937, longitude or 78.9629)
        return jsonify({
            'status': 'success',
            'data': emergency_mock
        })

def generate_mock_weather_data(latitude, longitude):
    """Generate reliable mock weather data for a given location."""
    try:
        # Create a geocoder to get location name if possible
        location_name = "Unknown Location"
        country_code = "IN"
        
        try:
            import requests
            # Try to get location name from coordinates using a free API
            nominatim_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}"
            location_response = requests.get(nominatim_url, headers={'User-Agent': 'EDU SPARK Weather App'}, timeout=3)
            if location_response.status_code == 200:
                location_data = location_response.json()
                if 'display_name' in location_data:
                    location_name = location_data.get('display_name', '').split(',')[0]
                    country_code = location_data.get('address', {}).get('country_code', 'IN').upper()
        except Exception as loc_error:
            print(f"Error getting location name: {str(loc_error)}")
        
        # Current date/time
        current_time = int(time.time())
        
        # Generate varying weather based on latitude (more rain near equator, colder near poles)
        temp_baseline = max(5, min(35, 30 - abs(latitude) / 2))  # Hotter near equator
        humidity_baseline = max(40, min(90, 60 + abs(latitude)))
        
        # Some simple seasonal variation - warmer in local summer
        # Northern hemisphere summer: June-August, Southern hemisphere summer: December-February
        current_month = datetime.now().month
        is_summer_north = 5 <= current_month <= 8
        is_summer_south = current_month <= 2 or current_month >= 11
        
        if (latitude > 0 and is_summer_north) or (latitude < 0 and is_summer_south):
            temp_modifier = 5  # Warmer in summer
        else:
            temp_modifier = -5  # Cooler in winter
            
        # Generate the mock data structure
        daily_forecast = []
        
        for i in range(7):
            # Add some random variation for each day, more variation further in future
            day_variation = (i + 1) * 0.5
            daily_temp = temp_baseline + temp_modifier + random.uniform(-day_variation, day_variation)
            
            # Weather types with probabilities
            weather_types = [
                {"main": "Clear", "description": "clear sky", "icon": "01d", "prob": 0.4},
                {"main": "Clouds", "description": "scattered clouds", "icon": "03d", "prob": 0.3},
                {"main": "Rain", "description": "light rain", "icon": "10d", "prob": 0.2},
                {"main": "Thunderstorm", "description": "thunderstorm", "icon": "11d", "prob": 0.1}
            ]
            
            # Choose weather type based on probabilities
            rand = random.random()
            cumulative_prob = 0
            weather_type = weather_types[-1]
            for wt in weather_types:
                cumulative_prob += wt["prob"]
                if rand <= cumulative_prob:
                    weather_type = wt
                    break
                    
            daily_forecast.append({
                "dt": current_time + (i * 86400),
                "temp": {
                    "day": daily_temp,
                    "min": daily_temp - random.uniform(3, 8),
                    "max": daily_temp + random.uniform(2, 6)
                },
                "humidity": humidity_baseline + random.randint(-10, 10),
                "wind_speed": 5 + random.uniform(-2, 4),
                "weather": [
                    {
                        "main": weather_type["main"],
                        "description": weather_type["description"],
                        "icon": weather_type["icon"]
                    }
                ]
            })
            
        # Generate realistic farming recommendations
        mock_data = {
            "current": {
                "dt": current_time,
                "temp": daily_forecast[0]["temp"]["day"],
                "humidity": daily_forecast[0]["humidity"],
                "wind_speed": daily_forecast[0]["wind_speed"],
                "weather": daily_forecast[0]["weather"]
            },
            "daily": daily_forecast,
            "location": {
                "name": location_name,
                "country": country_code
            },
            "farming_recommendations": generate_farming_recommendations(
                daily_forecast[0]["temp"]["day"],
                daily_forecast[0]["weather"][0]["main"],
                daily_forecast[0]["wind_speed"]
            )
        }
        
        return mock_data
    except Exception as e:
        print(f"Error in generate_mock_weather_data: {str(e)}")
        # Absolute emergency fallback data
        return {
            "current": {"dt": int(time.time()), "temp": 25, "humidity": 65, "wind_speed": 5, 
                       "weather": [{"main": "Clear", "description": "clear sky", "icon": "01d"}]},
            "daily": [{"dt": int(time.time()) + (i * 86400), 
                      "temp": {"day": 25, "min": 20, "max": 30}, 
                      "humidity": 65, "wind_speed": 5,
                      "weather": [{"main": "Clear", "description": "clear sky", "icon": "01d"}]
                      } for i in range(7)],
            "location": {"name": "Emergency Backup Mode", "country": ""},
            "farming_recommendations": [
                "Weather data unavailable. Consider checking other sources for current conditions.",
                "Maintain regular monitoring of your crops and soil moisture.",
                "Follow standard seasonal practices for your region."
            ]
        }

def generate_farming_recommendations(temp, weather_condition, wind_speed):
    """Generate farming recommendations based on weather conditions."""
    recommendations = []
    
    # Temperature-based recommendations
    if temp > 30:
        recommendations.append("High temperatures detected. Consider additional irrigation and providing shade for sensitive crops.")
    elif temp < 15:
        recommendations.append("Cool temperatures detected. Protect temperature-sensitive crops from potential frost.")
    else:
        recommendations.append("Moderate temperatures are optimal for most crops. Maintain regular irrigation schedule.")
    
    # Weather condition recommendations
    if weather_condition == "Rain":
        recommendations.append("Rainy conditions detected. Consider delaying fertilizer application and monitoring for potential disease pressure.")
    elif weather_condition == "Clear":
        recommendations.append("Clear conditions detected. Ideal for field operations and harvesting if crops are ready.")
    elif weather_condition == "Clouds":
        recommendations.append("Cloudy conditions detected. Good time for transplanting seedlings to minimize transplant shock.")
    elif weather_condition == "Thunderstorm":
        recommendations.append("Thunderstorms detected. Secure farm structures and equipment. Check fields for standing water after storms.")
    
    # Wind recommendations
    if wind_speed > 8:
        recommendations.append("Strong winds detected. Consider delaying pesticide application and protecting young plants.")
    
    # Add general seasonal recommendation based on current month
    month = datetime.now().month
    if 3 <= month <= 5:  # Spring
        recommendations.append("Spring season is ideal for preparing beds, sowing seeds, and transplanting seedlings.")
    elif 6 <= month <= 8:  # Summer
        recommendations.append("Summer heat requires vigilant irrigation management. Consider mulching to conserve soil moisture.")
    elif 9 <= month <= 11:  # Fall
        recommendations.append("Fall is harvest season for many crops. Consider soil testing and amendments post-harvest.")
    else:  # Winter
        recommendations.append("Winter is planning season. Maintain equipment and plan crop rotations for the coming season.")
    
    return recommendations

@app.route('/api/nearby_selling_points', methods=['POST'])
def get_nearby_selling_points():
    """Get nearby agricultural selling points with price information."""
    try:
        print("Nearby selling points API called")
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        radius = data.get('radius', 10000)  # Default radius 10km
        
        print(f"Market request for coordinates: {latitude}, {longitude}, radius: {radius}")
        
        if not latitude or not longitude:
            print("Missing latitude or longitude")
            return jsonify({
                'status': 'error',
                'message': 'Latitude and longitude are required'
            }), 400
        
        # Convert to float to prevent NaN issues
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (ValueError, TypeError):
            print(f"Invalid coordinates: {latitude}, {longitude}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid coordinates format'
            }), 400
            
        # In a real implementation, you would query a database or external API
        # For demo purposes, we'll return mock data with market prediction information
        mock_markets = [
            {
                "id": 1,
                "name": "Farmers Market Central",
                "location": {
                    "lat": latitude + (random.uniform(-0.02, 0.02)),
                    "lng": longitude + (random.uniform(-0.02, 0.02))
                },
                "distance": round(random.uniform(0.5, 4.5), 1),
                "rating": round(random.uniform(3.5, 5.0), 1),
                "market_status": {
                    "current_price": round(random.uniform(30, 45), 2),
                    "unit": "kg",
                    "change_percent": round(random.uniform(-5, 8), 1),
                    "trend": random.choice(["up", "down", "stable"])
                },
                "market_trends": {
                    "current_price": round(random.uniform(30, 45), 2),
                    "unit": "kg",
                    "change_percent": round(random.uniform(-5, 8), 1),
                    "forecast": "rising"
                },
                "produce": [
                    {"name": "Rice", "price": round(random.uniform(30, 45), 2), "unit": "kg", "trend": random.choice(["up", "down", "stable"])},
                    {"name": "Wheat", "price": round(random.uniform(25, 35), 2), "unit": "kg", "trend": random.choice(["up", "down", "stable"])},
                    {"name": "Potatoes", "price": round(random.uniform(15, 25), 2), "unit": "kg", "trend": random.choice(["up", "down", "stable"])}
                ]
            },
            {
                "id": 2,
                "name": "Village Agricultural Hub",
                "location": {
                    "lat": latitude + (random.uniform(-0.03, 0.03)),
                    "lng": longitude + (random.uniform(-0.03, 0.03))
                },
                "distance": round(random.uniform(1.0, 6.0), 1),
                "rating": round(random.uniform(3.5, 5.0), 1),
                "market_status": {
                    "current_price": round(random.uniform(28, 42), 2),
                    "unit": "kg",
                    "change_percent": round(random.uniform(-5, 8), 1),
                    "trend": random.choice(["up", "down", "stable"])
                },
                "market_trends": {
                    "current_price": round(random.uniform(28, 42), 2),
                    "unit": "kg",
                    "change_percent": round(random.uniform(-5, 8), 1),
                    "forecast": "stable"
                },
                "produce": [
                    {"name": "Rice", "price": round(random.uniform(28, 42), 2), "unit": "kg", "trend": random.choice(["up", "down", "stable"])},
                    {"name": "Wheat", "price": round(random.uniform(24, 33), 2), "unit": "kg", "trend": random.choice(["up", "down", "stable"])},
                    {"name": "Tomatoes", "price": round(random.uniform(18, 30), 2), "unit": "kg", "trend": random.choice(["up", "down", "stable"])}
                ]
            },
            {
                "id": 3,
                "name": "Regional Crop Exchange",
                "location": {
                    "lat": latitude + (random.uniform(-0.04, 0.04)),
                    "lng": longitude + (random.uniform(-0.04, 0.04))
                },
                "distance": round(random.uniform(2.0, 8.0), 1),
                "rating": round(random.uniform(3.5, 5.0), 1),
                "market_status": {
                    "current_price": round(random.uniform(32, 40), 2),
                    "unit": "kg",
                    "change_percent": round(random.uniform(-5, 8), 1),
                    "trend": random.choice(["up", "down", "stable"])
                },
                "market_trends": {
                    "current_price": round(random.uniform(32, 40), 2),
                    "unit": "kg",
                    "change_percent": round(random.uniform(-5, 8), 1),
                    "forecast": "falling"
                },
                "produce": [
                    {"name": "Rice", "price": round(random.uniform(32, 40), 2), "unit": "kg", "trend": random.choice(["up", "down", "stable"])},
                    {"name": "Corn", "price": round(random.uniform(15, 28), 2), "unit": "kg", "trend": random.choice(["up", "down", "stable"])},
                    {"name": "Onions", "price": round(random.uniform(12, 22), 2), "unit": "kg", "trend": random.choice(["up", "down", "stable"])}
                ]
            }
        ]
        
        # Add best price indicator
        crops = {"Rice": [], "Wheat": [], "Tomatoes": [], "Potatoes": [], "Corn": [], "Onions": []}
        
        # Collect all prices
        for market in mock_markets:
            for produce in market["produce"]:
                if produce["name"] in crops:
                    crops[produce["name"]].append({"price": produce["price"], "market_id": market["id"]})
        
        # Find lowest price for each crop
        best_prices = {}
        for crop, prices in crops.items():
            if prices:
                best_price = min(prices, key=lambda x: x["price"])
                best_prices[crop] = best_price["market_id"]
        
        # Mark best prices
        for market in mock_markets:
            for produce in market["produce"]:
                if produce["name"] in best_prices and best_prices[produce["name"]] == market["id"]:
                    produce["best_price"] = True
                else:
                    produce["best_price"] = False
        
        # Add overall market summary for the UI
        market_summary = {
            "overall_trend": random.choice(["rising", "falling", "stable"]),
            "price_range": {
                "min": 15.0,
                "max": 45.0,
                "avg": 28.5,
                "unit": "₹/kg"
            },
            "recommendation": random.choice([
                "Prices are trending upward. Consider holding your crop for a few more days for better returns.",
                "Prices are relatively stable. Good time to sell if you need immediate funds.",
                "Prices might decrease soon. Consider selling now to avoid potential losses."
            ])
        }
        
        return jsonify({
            'status': 'success',
            'data': mock_markets,
            'summary': market_summary
        })
        
    except Exception as e:
        print(f"Error fetching selling points: {str(e)}")
        # Return fallback data even on error
        fallback_market = {
            "id": 1,
            "name": "Fallback Market",
            "location": {
                "lat": latitude if latitude else 20.5937,
                "lng": longitude if longitude else 78.9629
            },
            "market_status": {
                "current_price": 35.5,
                "unit": "kg",
                "change_percent": 2.5,
                "trend": "stable"
            },
            "market_trends": {
                "current_price": 35.5,
                "unit": "kg",
                "change_percent": 2.5,
                "forecast": "stable"
            },
            "produce": [
                {"name": "Rice", "price": 35.50, "unit": "kg", "trend": "stable", "best_price": True}
            ]
        }
        
        return jsonify({
            'status': 'success',
            'data': [fallback_market],
            'summary': {
                "overall_trend": "stable",
                "price_range": {"min": 30.0, "max": 40.0, "avg": 35.0, "unit": "₹/kg"},
                "recommendation": "Market data is limited. Consider checking multiple sources before making decisions."
            }
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000) 