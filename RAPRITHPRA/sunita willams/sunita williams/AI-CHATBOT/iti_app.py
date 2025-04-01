#!/usr/bin/env python3
"""
AI Chatbot Application

This file contains an advanced implementation of an AI chatbot using Google's Gemini AI
with continuous learning capabilities for context.
"""

import os
import sys
import argparse
import json
import random
from typing import List, Dict, Optional, Tuple, Any
import time
from datetime import datetime
import textwrap
import colorama
from colorama import Fore, Style
from dotenv import load_dotenv
import numpy as np

# New imports for enhanced features
import requests
import gtts
from gtts import gTTS
import speech_recognition as sr
import pygame
from googletrans import Translator
from youtube_transcript_api import YouTubeTranscriptApi
import googleapiclient.discovery
from googleapiclient.errors import HttpError

# Add core directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

# Import core modules
from core.context_manager import ContextManager
from core.voice_manager import VoiceInteractionManager  # Import from core/voice_manager
from core.plant_disease_detector import PlantDiseaseDetector  # Import the plant disease detector
from core.market_predictor import MarketPredictor  # Import market predictor

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Load environment variables from .env file
load_dotenv()

# Check for API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print(f"{Fore.RED}Error: GEMINI_API_KEY not found in environment variables.{Style.RESET_ALL}")
    print("Please add your Gemini API key to the .env file.")
    sys.exit(1)

# Check for YouTube API key
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    print(f"{Fore.YELLOW}Warning: YOUTUBE_API_KEY not found. YouTube features will be limited.{Style.RESET_ALL}")

# Try to import Google Generative AI
try:
    import google.generativeai as genai
except ImportError:
    print(f"{Fore.RED}Error: google.generativeai package not found.{Style.RESET_ALL}")
    print("Please run: pip install google-generativeai==0.3.1")
    sys.exit(1)

# Constants
CONVERSATION_HISTORY_PATH = "conversation_history.json"
QA_DATASET_PATH = "qa_dataset.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CONTEXT_LENGTH = 2096  # Maximum token context length
SYSTEM_PROMPT = """You are an advanced AI assistant built to be helpful, harmless, and honest.
You excel at providing accurate information and solving problems across a wide range of topics.
When you don't know something, you acknowledge your limitations rather than making up answers.
You communicate in a friendly, concise, and professional manner."""

# Define ITI topics and subtopics for structured learning
ITI_TOPICS = {
    "technical_subjects": [
        "Mechanical Engineering", "Electrical Engineering", "Electronics", "Computer Science",
        "Civil Engineering", "Automobile Engineering", "Refrigeration & Air Conditioning"
    ],
    "agricultural_topics": [
        "Modern Farming Techniques", "Crop Management", "Irrigation Systems", 
        "Sustainable Agriculture", "Organic Farming", "Pest Management", "Plant Disease Detection"
    ],
    "vocational_skills": [
        "Welding", "Carpentry", "Plumbing", "Tailoring", "Electrician Skills",
        "Computer Hardware", "Food Processing", "Textile Technology"
    ]
}

# User progress tracking database path
USER_PROGRESS_PATH = "user_progress.json"
STUDY_MATERIALS_PATH = "study_materials.json"
QUIZ_DATABASE_PATH = "quiz_database.json"


class ContextualModel:
    """Model for understanding and retrieving contextual information."""

    def __init__(self, model_path=None):
        """Initialize the contextual model.

        Args:
            model_path: Path to saved model or model identifier
        """
        self.model = None
        self.context_store = []
        
        try:
            # Try to import required packages
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            from sentence_transformers import SentenceTransformer
            
            print(f"{Fore.YELLOW}Loading contextual understanding model...{Style.RESET_ALL}")
            
            # Use sentence transformers for creating embeddings
            if not model_path or not os.path.exists(model_path):
                model_path = EMBEDDING_MODEL

            self.model = SentenceTransformer(model_path)
            self._load_context_store()
            print(f"{Fore.GREEN}✓ Successfully loaded contextual model{Style.RESET_ALL}")

        except ImportError as e:
            print(f"{Fore.RED}Error setting up contextual model: {e}{Style.RESET_ALL}")
            print("Please run: pip install sentence-transformers==2.3.0 scikit-learn==1.2.2")
            print("Continuing without contextual understanding capabilities...")
        except Exception as e:
            print(f"{Fore.RED}Error setting up contextual model: {e}{Style.RESET_ALL}")
            print("Continuing without contextual understanding capabilities...")

    def _load_context_store(self):
        """Load saved context information."""
        try:
            if os.path.exists("context_store.json"):
                with open("context_store.json", "r", encoding="utf-8") as f:
                    self.context_store = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded {len(self.context_store)} contextual entries{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load context store: {e}{Style.RESET_ALL}")
            self.context_store = []

    def _save_context_store(self):
        """Save context information."""
        try:
            with open("context_store.json", "w", encoding="utf-8") as f:
                json.dump(self.context_store, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save context store: {e}{Style.RESET_ALL}")

    def add_context(self, text, metadata=None):
        """Add new contextual information.

        Args:
            text: Text to add to context store
            metadata: Additional information about this context
        """
        if not self.model:
            return

        try:
            # Try to import required package
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create embedding for the text
            embedding = self.model.encode(text)
            embedding_list = embedding.tolist()

            # Add to context store
            context_entry = {
                "text": text,
                "embedding": embedding_list,
                "metadata": metadata or {},
                "added_at": datetime.now().isoformat()
            }

            self.context_store.append(context_entry)
            self._save_context_store()

        except Exception as e:
            print(f"{Fore.YELLOW}Could not add context: {e}{Style.RESET_ALL}")

    def retrieve_relevant_context(self, query, top_k=3):
        """Retrieve contextual information relevant to a query.

        Args:
            query: The query to match against context
            top_k: Number of results to return

        Returns:
            List of relevant context entries
        """
        if not self.model or not self.context_store:
            return []

        try:
            # Try to import required package
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create embedding for query
            query_embedding = self.model.encode(query)

            # Calculate similarity with all entries
            similarities = []
            for entry in self.context_store:
                entry_embedding = entry["embedding"]
                similarity = cosine_similarity([query_embedding], [entry_embedding])[0][0]
                similarities.append((similarity, entry))

            # Sort by similarity and return top_k
            similarities.sort(reverse=True, key=lambda x: x[0])

            # Return relevant contexts
            return [entry["text"] for _, entry in similarities[:top_k] if _ > 0.6]

        except Exception as e:
            print(f"{Fore.YELLOW}Could not retrieve context: {e}{Style.RESET_ALL}")
            return []

    def extract_entities(self, text):
        """Extract key entities from text for contextual understanding.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of extracted entities
        """
        # A simple implementation - in production you would use a proper NER model
        entities = {}

        # Look for temporal expressions
        if "yesterday" in text.lower():
            entities["time"] = "yesterday"
        elif "today" in text.lower():
            entities["time"] = "today"
        elif "tomorrow" in text.lower():
            entities["time"] = "tomorrow"

        # Look for user preferences that might have been expressed
        if "i like" in text.lower() or "i prefer" in text.lower():
            lower_text = text.lower()
            start_idx = max(lower_text.find("i like"), lower_text.find("i prefer"))
            if start_idx != -1:
                end_idx = lower_text.find(".", start_idx)
                if end_idx == -1:
                    end_idx = len(text)
                preference = text[start_idx:end_idx].strip()
                entities["preference"] = preference

        return entities


class YouTubeResourceManager:
    """Manages YouTube video resources for educational topics."""
    
    def __init__(self, api_key=None):
        """Initialize the YouTube resource manager.
        
        Args:
            api_key: YouTube Data API key
        """
        self.api_key = api_key
        self.youtube = None
        self.video_cache = {}
        self.snippet_cache = {}
        self.cache_duration = 3600  # Cache duration in seconds (1 hour)
        
        if self.api_key:
            self._setup_youtube_api()
    
    def _setup_youtube_api(self):
        """Setup the YouTube API client."""
        try:
            self.youtube = googleapiclient.discovery.build(
                "youtube", "v3", developerKey=self.api_key,
                cache_discovery=False  # Disable cache warning
            )
            print(f"{Fore.GREEN}✓ Successfully initialized YouTube API{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}YouTube API setup failed: {e}. Some features may be limited.{Style.RESET_ALL}")
            self.youtube = None

    def _clear_expired_cache(self):
        """Clear expired items from cache."""
        current_time = time.time()
        # Clear video cache
        self.video_cache = {
            k: v for k, v in self.video_cache.items()
            if v.get('timestamp', 0) + self.cache_duration > current_time
        }
        # Clear snippet cache
        self.snippet_cache = {
            k: v for k, v in self.snippet_cache.items()
            if v.get('timestamp', 0) + self.cache_duration > current_time
        }

    def search_videos(self, query, max_results=5, language="en"):
        """Search for educational videos related to a query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            language: Language preference for results
            
        Returns:
            List of video information dictionaries
        """
        if not self.youtube:
            print(f"{Fore.YELLOW}YouTube API not initialized. Using fallback search.{Style.RESET_ALL}")
            return self._fallback_search(query, max_results)
            
        try:
            # Clear expired cache entries
            self._clear_expired_cache()
            
            # Cache check
            cache_key = f"{query}_{max_results}_{language}"
            if cache_key in self.video_cache:
                cached_data = self.video_cache[cache_key]
                if cached_data.get('timestamp', 0) + self.cache_duration > time.time():
                    print(f"{Fore.GREEN}Using cached results for: {query}{Style.RESET_ALL}")
                    return cached_data['videos']
                
            print(f"{Fore.CYAN}Searching YouTube for: {query}{Style.RESET_ALL}")
            
            # Perform YouTube search
            try:
                search_response = self.youtube.search().list(
                    q=query,
                    part="id,snippet",
                    maxResults=max_results,
                    type="video",
                    relevanceLanguage=language,
                    videoEmbeddable="true",
                    videoCategoryId="27",  # Education category
                    fields="items(id(videoId),snippet(title,description,thumbnails/default/url,channelTitle))"
                ).execute()
            except Exception as search_error:
                print(f"{Fore.RED}YouTube search error: {search_error}{Style.RESET_ALL}")
                return self._fallback_search(query, max_results)
            
            # Get video IDs for additional details
            video_ids = [
                item["id"]["videoId"] 
                for item in search_response.get("items", []) 
                if "videoId" in item.get("id", {})
            ]
            
            details_map = {}
            if video_ids:
                try:
                    # Get additional video details
                    video_details = self.youtube.videos().list(
                        part="contentDetails,statistics",
                        id=",".join(video_ids),
                        fields="items(id,contentDetails(duration),statistics(viewCount,likeCount))"
                    ).execute()
                    
                    # Create a mapping of video details
                    details_map = {
                        item["id"]: item 
                        for item in video_details.get("items", [])
                    }
                except Exception as details_error:
                    print(f"{Fore.YELLOW}Error fetching video details: {details_error}{Style.RESET_ALL}")
            
            # Process results
            videos = []
            for item in search_response.get("items", []):
                try:
                    if "videoId" in item.get("id", {}):
                        video_id = item["id"]["videoId"]
                        details = details_map.get(video_id, {})
                        
                        # Extract duration in a readable format
                        duration = details.get("contentDetails", {}).get("duration", "")
                        try:
                            # Handle duration format more robustly
                            duration = duration.replace("PT", "")
                            hours = "00"
                            minutes = "00"
                            seconds = "00"
                            
                            if "H" in duration:
                                hours, duration = duration.split("H")
                                hours = hours.zfill(2)
                            if "M" in duration:
                                minutes, duration = duration.split("M")
                                minutes = minutes.zfill(2)
                            if "S" in duration:
                                seconds = duration.replace("S", "").zfill(2)
                            
                            formatted_duration = f"{hours}:{minutes}:{seconds}" if hours != "00" else f"{minutes}:{seconds}"
                        except Exception:
                            formatted_duration = "00:00"
                        
                        # Get view count and other statistics safely
                        statistics = details.get("statistics", {})
                        try:
                            view_count = int(statistics.get("viewCount", 0))
                            like_count = int(statistics.get("likeCount", 0))
                        except (ValueError, TypeError):
                            view_count = 0
                            like_count = 0
                        
                        # Create video info dictionary
                        video_info = {
                            "id": video_id,
                            "title": item["snippet"]["title"],
                            "description": item["snippet"]["description"],
                            "thumbnail": item["snippet"]["thumbnails"]["default"]["url"],
                            "channel": item["snippet"]["channelTitle"],
                            "url": f"https://www.youtube.com/watch?v={video_id}",
                            "duration": formatted_duration,
                            "views": view_count,
                            "likes": like_count
                        }
                        
                        videos.append(video_info)
                except Exception as video_error:
                    print(f"{Fore.YELLOW}Error processing video: {video_error}{Style.RESET_ALL}")
                    continue
            
            # Cache results with timestamp
            self.video_cache[cache_key] = {
                'videos': videos,
                'timestamp': time.time()
            }
            
            return videos
            
        except Exception as e:
            print(f"{Fore.RED}Error in search_videos: {e}{Style.RESET_ALL}")
            return self._fallback_search(query, max_results)
    
    def _calculate_educational_score(self, title, description, tags, view_count, like_count):
        """Calculate an educational relevance score for a video.
        
        Args:
            title: Video title
            description: Video description
            tags: Video tags
            view_count: Number of views
            like_count: Number of likes
            
        Returns:
            Float score between 0 and 1
        """
        try:
            score = 0.0
            
            # Educational keywords with weights
            educational_keywords = {
                'high': [
                    "tutorial", "course", "lesson", "education", "training",
                    "learn", "guide", "howto", "explanation", "lecture"
                ],
                'medium': [
                    "workshop", "classroom", "teaching", "instruction", "study",
                    "demo", "demonstrate", "example", "practice", "exercise"
                ],
                'low': [
                    "tips", "tricks", "basics", "introduction", "beginner",
                    "advanced", "master", "skills", "technique", "method"
                ]
            }
            
            # Check title (higher weight)
            title_lower = title.lower()
            for keyword in educational_keywords['high']:
                if keyword in title_lower:
                    score += 0.3
                    break
            for keyword in educational_keywords['medium']:
                if keyword in title_lower:
                    score += 0.2
                    break
            for keyword in educational_keywords['low']:
                if keyword in title_lower:
                    score += 0.1
                    break
            
            # Check description
            desc_lower = description.lower()
            for keyword in educational_keywords['high']:
                if keyword in desc_lower:
                    score += 0.15
                    break
            for keyword in educational_keywords['medium']:
                if keyword in desc_lower:
                    score += 0.1
                    break
            
            # Check tags
            for tag in tags:
                tag_lower = tag.lower()
                if any(keyword in tag_lower for keyword in educational_keywords['high']):
                    score += 0.1
                    break
                elif any(keyword in tag_lower for keyword in educational_keywords['medium']):
                    score += 0.05
                    break
            
            # Factor in engagement metrics
            if view_count > 0 and like_count > 0:
                like_ratio = like_count / view_count
                engagement_score = min(0.2, like_ratio * 1000)
                score += engagement_score
            
            return min(1.0, score)  # Normalize to 0-1
            
        except Exception as e:
            print(f"{Fore.YELLOW}Error calculating educational score: {e}{Style.RESET_ALL}")
            return 0.0
    
    def _get_relevant_snippets(self, video_id, query, max_snippets=3):
        """Get relevant transcript snippets based on the query.
        
        Args:
            video_id: YouTube video ID
            query: Search query
            max_snippets: Maximum number of snippets to return
            
        Returns:
            List of relevant transcript snippets with timestamps
        """
        try:
            # Check snippet cache
            cache_key = f"{video_id}_{query}"
            if cache_key in self.snippet_cache:
                return self.snippet_cache[cache_key]
            
            transcript_text = self.get_video_transcript(video_id)
            if not transcript_text:
                return None
            
            # Split transcript into chunks
            chunks = []
            current_chunk = []
            current_time = 0
            
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            for entry in transcript:
                current_chunk.append(entry["text"])
                if len(" ".join(current_chunk)) >= 200:  # Chunk size ~200 chars
                    chunks.append({
                        "text": " ".join(current_chunk),
                        "start": current_time,
                        "duration": entry["start"] - current_time
                    })
                    current_chunk = []
                    current_time = entry["start"]
            
            # Add remaining chunk
            if current_chunk:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "start": current_time,
                    "duration": transcript[-1]["start"] + transcript[-1]["duration"] - current_time
                })
            
            # Score chunks based on query relevance
            scored_chunks = []
            query_terms = query.lower().split()
            
            for chunk in chunks:
                score = 0
                chunk_text = chunk["text"].lower()
                
                # Score based on term frequency
                for term in query_terms:
                    score += chunk_text.count(term)
                
                # Bonus for exact phrase match
                if query.lower() in chunk_text:
                    score += 5
                
                if score > 0:
                    scored_chunks.append((score, chunk))
            
            # Sort and get top snippets
            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            snippets = []
            
            for _, chunk in scored_chunks[:max_snippets]:
                timestamp = self._format_timestamp(chunk["start"])
                snippet = {
                    "text": chunk["text"],
                    "timestamp": timestamp,
                    "url": f"https://www.youtube.com/watch?v={video_id}&t={int(chunk['start'])}"
                }
                snippets.append(snippet)
            
            # Cache snippets
            self.snippet_cache[cache_key] = snippets
            return snippets
            
        except Exception as e:
            print(f"{Fore.YELLOW}Error getting snippets: {e}{Style.RESET_ALL}")
            return None
    
    def _format_timestamp(self, seconds):
        """Format seconds into HH:MM:SS timestamp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def _fallback_search(self, query, max_results=5):
        """Fallback method when YouTube API is unavailable.
        
        Uses a basic web search to find YouTube videos.
        """
        try:
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": GEMINI_API_KEY,
                "cx": "015786823554162166635:wnpqmws6nma",  # Custom search engine ID for YouTube
                "q": f"{query} site:youtube.com",
                "num": max_results
            }
            
            response = requests.get(search_url, params=params)
            data = response.json()
            
            videos = []
            if "items" in data:
                for item in data["items"]:
                    if "youtube.com/watch" in item.get("link", ""):
                        video_id = item["link"].split("v=")[1].split("&")[0]
                        video_info = {
                            "id": video_id,
                            "title": item.get("title", ""),
                            "description": item.get("snippet", ""),
                            "url": item.get("link", "")
                        }
                        videos.append(video_info)
            
            return videos
            
        except Exception as e:
            print(f"{Fore.YELLOW}Fallback search error: {e}{Style.RESET_ALL}")
            return []
    
    def get_video_transcript(self, video_id, language="en"):
        """Get the transcript of a YouTube video if available.
        
        Args:
            video_id: YouTube video ID
            language: Preferred language for transcript
            
        Returns:
            Transcript text or None if unavailable
        """
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get transcript in preferred language
            try:
                transcript = transcript_list.find_transcript([language])
            except:
                # Fall back to any available transcript
                transcript = transcript_list.find_transcript(["en"])
                
            transcript_data = transcript.fetch()
            transcript_text = " ".join([item["text"] for item in transcript_data])
            
            return transcript_text
            
        except Exception as e:
            print(f"{Fore.YELLOW}Could not retrieve transcript: {e}{Style.RESET_ALL}")
            return None


class MultilingualSupport:
    """Provides multilingual capabilities to the chatbot."""
    
    def __init__(self):
        """Initialize the multilingual support system."""
        self.translator = Translator()
        self.supported_languages = {
            "en": "English",
            "hi": "Hindi",
            "bn": "Bengali",
            "te": "Telugu", 
            "mr": "Marathi",
            "ta": "Tamil",
            "ur": "Urdu",
            "gu": "Gujarati",
            "kn": "Kannada",
            "ml": "Malayalam",
            "pa": "Punjabi",
            "or": "Odia"
        }
        self.is_available = self._check_availability()
    
    def _check_availability(self):
        """Check if translation service is available."""
        try:
            test_result = self.translator.translate("test", src="en", dest="hi")
            return True
        except Exception as e:
            print(f"{Fore.YELLOW}Translation service unavailable: {e}{Style.RESET_ALL}")
            return False
    
    def translate(self, text, target_language="en", source_language=None):
        """Translate text to the target language.
        
        Args:
            text: Text to translate
            target_language: Language code to translate to
            source_language: Language code to translate from (auto-detect if None)
            
        Returns:
            Translated text
        """
        if not self.is_available:
            return text
            
        try:
            result = self.translator.translate(text, src=source_language, dest=target_language)
            return result.text
        except Exception as e:
            print(f"{Fore.YELLOW}Translation error: {e}{Style.RESET_ALL}")
            return text
    
    def detect_language(self, text):
        """Detect the language of input text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code or "en" if detection fails
        """
        if not self.is_available or not text:
            return "en"
            
        try:
            result = self.translator.detect(text)
            return result.lang
        except Exception as e:
            print(f"{Fore.YELLOW}Language detection error: {e}{Style.RESET_ALL}")
            return "en"
    
    def get_supported_languages(self):
        """Get a list of supported languages.
        
        Returns:
            Dictionary of language codes and names
        """
        return self.supported_languages


class ProgressTracker:
    """Tracks user learning progress across topics."""
    
    def __init__(self, user_id="default_user"):
        """Initialize the progress tracker.
        
        Args:
            user_id: Identifier for the user
        """
        self.user_id = user_id
        self.progress_data = {}
        self.quiz_results = {}
        self._load_progress_data()
    
    def _load_progress_data(self):
        """Load progress data from file if available."""
        try:
            if os.path.exists(USER_PROGRESS_PATH):
                with open(USER_PROGRESS_PATH, 'r', encoding='utf-8') as f:
                    all_user_data = json.load(f)
                    self.progress_data = all_user_data.get(self.user_id, {})
                    print(f"{Fore.GREEN}✓ Loaded progress data for user {self.user_id}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load progress data: {e}{Style.RESET_ALL}")
            self.progress_data = {}
    
    def _save_progress_data(self):
        """Save progress data to file."""
        try:
            # Load all user data first
            all_user_data = {}
            if os.path.exists(USER_PROGRESS_PATH):
                with open(USER_PROGRESS_PATH, 'r', encoding='utf-8') as f:
                    all_user_data = json.load(f)
            
            # Update with current user's data
            all_user_data[self.user_id] = self.progress_data
            
            # Save back to file
            with open(USER_PROGRESS_PATH, 'w', encoding='utf-8') as f:
                json.dump(all_user_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save progress data: {e}{Style.RESET_ALL}")
    
    def update_topic_progress(self, topic, subtopic, progress_value):
        """Update progress for a specific topic.
        
        Args:
            topic: Main topic category
            subtopic: Specific subtopic
            progress_value: Value between 0-100 indicating completion percentage
        """
        if topic not in self.progress_data:
            self.progress_data[topic] = {}
            
        if subtopic not in self.progress_data[topic]:
            self.progress_data[topic][subtopic] = {
                "progress": 0,
                "last_accessed": None,
                "quiz_scores": [],
                "completion_time": 0
            }
        
        # Update progress
        self.progress_data[topic][subtopic]["progress"] = progress_value
        self.progress_data[topic][subtopic]["last_accessed"] = datetime.now().isoformat()
        
        # Save updated data
        self._save_progress_data()
    
    def record_quiz_result(self, topic, subtopic, score, total_questions):
        """Record quiz results for a topic.
        
        Args:
            topic: Main topic category
            subtopic: Specific subtopic
            score: Number of correct answers
            total_questions: Total number of questions
        """
        if topic not in self.progress_data:
            self.progress_data[topic] = {}
            
        if subtopic not in self.progress_data[topic]:
            self.progress_data[topic][subtopic] = {
                "progress": 0,
                "last_accessed": None,
                "quiz_scores": [],
                "completion_time": 0
            }
        
        # Add quiz result
        percentage = (score / total_questions) * 100 if total_questions > 0 else 0
        quiz_entry = {
            "date": datetime.now().isoformat(),
            "score": score,
            "total": total_questions,
            "percentage": percentage
        }
        
        self.progress_data[topic][subtopic]["quiz_scores"].append(quiz_entry)
        
        # Update progress based on quiz performance
        current_progress = self.progress_data[topic][subtopic]["progress"]
        new_progress = max(current_progress, min(percentage, 100))
        self.progress_data[topic][subtopic]["progress"] = new_progress
        
        # Save updated data
        self._save_progress_data()
    
    def get_progress_summary(self):
        """Get a summary of user progress across all topics.
        
        Returns:
            Dictionary containing progress summary
        """
        summary = {
            "user_id": self.user_id,
            "topics": {},
            "overall_progress": 0,
            "strengths": [],
            "areas_for_improvement": []
        }
        
        total_progress = 0
        topic_count = 0
        
        # Calculate progress for each topic
        for topic, subtopics in self.progress_data.items():
            topic_progress = 0
            subtopic_count = 0
            
            summary["topics"][topic] = {
                "progress": 0,
                "subtopics": {}
            }
            
            for subtopic, data in subtopics.items():
                subtopic_progress = data["progress"]
                summary["topics"][topic]["subtopics"][subtopic] = subtopic_progress
                
                topic_progress += subtopic_progress
                subtopic_count += 1
            
            # Calculate average progress for the topic
            avg_topic_progress = topic_progress / subtopic_count if subtopic_count > 0 else 0
            summary["topics"][topic]["progress"] = avg_topic_progress
            
            total_progress += avg_topic_progress
            topic_count += 1
        
        # Calculate overall progress
        summary["overall_progress"] = total_progress / topic_count if topic_count > 0 else 0
        
        # Identify strengths and areas for improvement
        all_subtopics = []
        for topic, subtopics in self.progress_data.items():
            for subtopic, data in subtopics.items():
                all_subtopics.append((topic, subtopic, data["progress"]))
        
        # Sort by progress
        all_subtopics.sort(key=lambda x: x[2], reverse=True)
        
        # Get top 3 strengths
        for topic, subtopic, progress in all_subtopics[:3]:
            if progress > 70:  # Only consider as strength if progress is good
                summary["strengths"].append({
                    "topic": topic,
                    "subtopic": subtopic, 
                    "progress": progress
                })
        
        # Get bottom 3 areas for improvement
        for topic, subtopic, progress in all_subtopics[-3:]:
            if progress < 80:  # Only consider as improvement area if not mastered
                summary["areas_for_improvement"].append({
                    "topic": topic,
                    "subtopic": subtopic,
                    "progress": progress
                })
        
        return summary


class StudyMaterialGenerator:
    """Generates and manages study materials for educational topics."""
    
    def __init__(self):
        """Initialize the study material generator."""
        self.materials = {}
        self._load_materials()
    
    def _load_materials(self):
        """Load study materials from file if available."""
        try:
            if os.path.exists(STUDY_MATERIALS_PATH):
                with open(STUDY_MATERIALS_PATH, 'r', encoding='utf-8') as f:
                    self.materials = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded study materials database{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load study materials: {e}{Style.RESET_ALL}")
            self.materials = {}
    
    def _save_materials(self):
        """Save study materials to file."""
        try:
            with open(STUDY_MATERIALS_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.materials, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save study materials: {e}{Style.RESET_ALL}")
    
    def generate_summary(self, topic, subtopic, content, model):
        """Generate a summary for study material.
        
        Args:
            topic: Main topic category
            subtopic: Specific subtopic
            content: Content to summarize
            model: GenAI model to use for summarization
            
        Returns:
            Generated summary text
        """
        try:
            prompt = f"""
            Please create a concise summary of the following content about {subtopic} 
            (which is part of {topic}). Focus on key points and concepts:
            
            {content}
            
            Organize the summary with bullet points for main concepts and include 
            2-3 practice questions at the end.
            """
            
            response = model.generate_content(prompt)
            summary = response.text
            
            # Store the generated summary
            if topic not in self.materials:
                self.materials[topic] = {}
                
            if subtopic not in self.materials[topic]:
                self.materials[topic][subtopic] = []
                
            material_entry = {
                "type": "summary",
                "content": summary,
                "original": content[:500] + "..." if len(content) > 500 else content,
                "date_created": datetime.now().isoformat()
            }
            
            self.materials[topic][subtopic].append(material_entry)
            self._save_materials()
            
            return summary
            
        except Exception as e:
            print(f"{Fore.YELLOW}Error generating summary: {e}{Style.RESET_ALL}")
            return f"Sorry, I couldn't generate a summary due to an error: {str(e)}"
    
    def create_revision_notes(self, topic, subtopic, model):
        """Create revision notes for a specific topic.
        
        Args:
            topic: Main topic category
            subtopic: Specific subtopic
            model: GenAI model to use for note generation
            
        Returns:
            Generated revision notes
        """
        try:
            prompt = f"""
            Create comprehensive revision notes for {subtopic} (part of {topic}).
            Include:
            
            1. Key definitions and concepts
            2. Important formulas or principles (if applicable)
            3. Step-by-step procedures (if applicable)
            4. Common mistakes to avoid
            5. Practical applications
            6. Memory aids or mnemonics
            
            Format the notes in a clear, structured way that's easy to study from.
            """
            
            response = model.generate_content(prompt)
            notes = response.text
            
            # Store the generated notes
            if topic not in self.materials:
                self.materials[topic] = {}
                
            if subtopic not in self.materials[topic]:
                self.materials[topic][subtopic] = []
                
            material_entry = {
                "type": "revision_notes",
                "content": notes,
                "date_created": datetime.now().isoformat()
            }
            
            self.materials[topic][subtopic].append(material_entry)
            self._save_materials()
            
            return notes
            
        except Exception as e:
            print(f"{Fore.YELLOW}Error creating revision notes: {e}{Style.RESET_ALL}")
            return f"Sorry, I couldn't create revision notes due to an error: {str(e)}"
    
    def get_study_materials(self, topic=None, subtopic=None):
        """Get study materials for a topic.
        
        Args:
            topic: Main topic category (optional)
            subtopic: Specific subtopic (optional)
            
        Returns:
            Dictionary of study materials
        """
        if topic and subtopic:
            # Return materials for specific topic and subtopic
            return self.materials.get(topic, {}).get(subtopic, [])
            
        elif topic:
            # Return all materials for a topic
            return self.materials.get(topic, {})
            
        else:
            # Return all materials
            return self.materials


class QuizManager:
    """Manages quizzes for educational assessment."""
    
    def __init__(self):
        """Initialize the quiz manager."""
        self.quizzes = {}
        self._load_quizzes()
    
    def _load_quizzes(self):
        """Load quiz database from file if available."""
        try:
            if os.path.exists(QUIZ_DATABASE_PATH):
                with open(QUIZ_DATABASE_PATH, 'r', encoding='utf-8') as f:
                    self.quizzes = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded quiz database{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load quiz database: {e}{Style.RESET_ALL}")
            self.quizzes = {}
    
    def _save_quizzes(self):
        """Save quiz database to file."""
        try:
            with open(QUIZ_DATABASE_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.quizzes, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save quiz database: {e}{Style.RESET_ALL}")
    
    def generate_quiz(self, topic, subtopic, difficulty, num_questions, model):
        """Generate a new quiz for a topic.
        
        Args:
            topic: Main topic category
            subtopic: Specific subtopic
            difficulty: Quiz difficulty level
            num_questions: Number of questions to generate
            model: GenAI model to use for quiz generation
            
        Returns:
            Generated quiz
        """
        try:
            prompt = f"""
            Create a quiz for {subtopic} (part of {topic}) with {num_questions} multiple-choice questions.
            Difficulty level: {difficulty}.
            
            For each question:
            1. Provide a clear question statement
            2. Give 4 answer options (A, B, C, D)
            3. Indicate the correct answer
            4. Include a brief explanation for why the answer is correct
            
            Format each question as JSON with the following structure:
            {{
                "question": "Question text",
                "options": {{
                    "A": "First option",
                    "B": "Second option",
                    "C": "Third option",
                    "D": "Fourth option"
                }},
                "correct_answer": "The correct letter (A, B, C, or D)",
                "explanation": "Explanation of why this answer is correct"
            }}
            
            Return a JSON array of {num_questions} questions.
            """
            
            response = model.generate_content(prompt)
            try:
                # Try to extract JSON from the response
                response_text = response.text
                # Find JSON content (between [ and ])
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_content = response_text[start_idx:end_idx]
                    quiz_questions = json.loads(json_content)
                else:
                    raise ValueError("Couldn't find valid JSON in the response")
                    
            except Exception as json_error:
                print(f"{Fore.YELLOW}Error parsing quiz JSON: {json_error}. Regenerating...{Style.RESET_ALL}")
                # Try more explicit formatting prompt as a fallback
                fallback_prompt = f"""
                Generate a quiz about {subtopic} (topic: {topic}) with exactly {num_questions} multiple-choice questions.
                Difficulty: {difficulty}
                
                Format your response as VALID JSON ONLY, with no other text before or after.
                The JSON should be an array of question objects with this exact structure:
                [
                  {{
                    "question": "Question text here",
                    "options": {{
                      "A": "Option A text",
                      "B": "Option B text",
                      "C": "Option C text",
                      "D": "Option D text"
                    }},
                    "correct_answer": "A",
                    "explanation": "Explanation text here"
                  }},
                  ...more questions...
                ]
                
                ONLY return the JSON array, nothing else.
                """
                response = model.generate_content(fallback_prompt)
                response_text = response.text.strip()
                
                # Handle case where the model might add backticks
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                    
                quiz_questions = json.loads(response_text.strip())
            
            # Store the generated quiz
            quiz_id = f"{topic}_{subtopic}_{difficulty}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            quiz_entry = {
                "id": quiz_id,
                "topic": topic,
                "subtopic": subtopic,
                "difficulty": difficulty,
                "questions": quiz_questions,
                "date_created": datetime.now().isoformat()
            }
            
            if topic not in self.quizzes:
                self.quizzes[topic] = {}
                
            if subtopic not in self.quizzes[topic]:
                self.quizzes[topic][subtopic] = []
                
            self.quizzes[topic][subtopic].append(quiz_entry)
            self._save_quizzes()
            
            return quiz_entry
            
        except Exception as e:
            print(f"{Fore.YELLOW}Error generating quiz: {e}{Style.RESET_ALL}")
            return {"error": f"Could not generate quiz: {str(e)}"}
    
    def get_available_quizzes(self, topic=None, subtopic=None):
        """Get available quizzes for a topic.
        
        Args:
            topic: Main topic category (optional)
            subtopic: Specific subtopic (optional)
            
        Returns:
            List of available quiz metadata
        """
        if topic and subtopic:
            # Return quizzes for specific topic and subtopic
            quizzes = self.quizzes.get(topic, {}).get(subtopic, [])
            return [{"id": q["id"], "topic": topic, "subtopic": subtopic, 
                     "difficulty": q["difficulty"], "question_count": len(q["questions"]),
                     "date_created": q["date_created"]} for q in quizzes]
            
        elif topic:
            # Return all quizzes for a topic
            result = []
            for subtopic, quizzes in self.quizzes.get(topic, {}).items():
                for q in quizzes:
                    result.append({"id": q["id"], "topic": topic, "subtopic": subtopic, 
                                  "difficulty": q["difficulty"], "question_count": len(q["questions"]),
                                  "date_created": q["date_created"]})
            return result
            
        else:
            # Return all quizzes
            result = []
            for topic, subtopics in self.quizzes.items():
                for subtopic, quizzes in subtopics.items():
                    for q in quizzes:
                        result.append({"id": q["id"], "topic": topic, "subtopic": subtopic, 
                                      "difficulty": q["difficulty"], "question_count": len(q["questions"]),
                                      "date_created": q["date_created"]})
            return result
    
    def get_quiz_by_id(self, quiz_id):
        """Get a specific quiz by ID.
        
        Args:
            quiz_id: The ID of the quiz to retrieve
            
        Returns:
            Quiz data or None if not found
        """
        for topic, subtopics in self.quizzes.items():
            for subtopic, quizzes in subtopics.items():
                for quiz in quizzes:
                    if quiz["id"] == quiz_id:
                        return quiz
        
        return None


class ChatBot:
    """AI Chatbot with continuous learning capabilities."""

    def __init__(self, continuous_learning=False, user_id="default_user"):
        """Initialize chatbot with optional continuous learning.

        Args:
            continuous_learning: Whether to enable continuous learning mode
            user_id: Unique identifier for the user
        """
        self.user_id = user_id
        self.continuous_learning = continuous_learning
        self.conversation_history = []
        self.qa_dataset = []
        self.model = None
        self.context_model = ContextualModel()
        self.youtube_manager = YouTubeResourceManager(api_key=YOUTUBE_API_KEY)
        self.multilingual = MultilingualSupport()
        self.progress_tracker = ProgressTracker(user_id=user_id)
        self.study_material_generator = StudyMaterialGenerator()
        self.quiz_manager = QuizManager()
        self.voice_manager = VoiceInteractionManager()
        self.plant_disease_detector = PlantDiseaseDetector()  # Initialize plant disease detector
        self.market_predictor = MarketPredictor()  # Initialize market predictor
        self.ITI_TOPICS= ITI_TOPICS
        # Load user preferences
        self.user_preferences = {
            "language": "en",
            "response_length": "medium",  # concise, medium, detailed
            "difficulty_level": "medium",  # easy, medium, hard
            "voice_enabled": False,
            "voice_wake_word": "assistant",
            "theme": "default"
        }
        
        # Customize file paths based on user_id
        self.conversation_history_path = f"conversation_history_{user_id}.json"
        self.user_preferences_path = f"user_preferences_{user_id}.json"
        
        self._load_user_preferences()
        self._setup_gemini_model()
        self._load_conversation_history()
        
        if continuous_learning:
            self._load_qa_dataset()
            print(f"{Fore.GREEN}Continuous learning mode enabled. Learning from context and history.{Style.RESET_ALL}")

    def _load_user_preferences(self):
        """Load user preferences from file if available."""
        try:
            preferences_path = f"user_preferences_{self.user_id}.json"
            if os.path.exists(preferences_path):
                with open(preferences_path, 'r', encoding='utf-8') as f:
                    self.user_preferences = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded preferences for user {self.user_id}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load user preferences: {e}{Style.RESET_ALL}")
            # Keep default preferences

    def _save_user_preferences(self):
        """Save user preferences to file."""
        try:
            preferences_path = f"user_preferences_{self.user_id}.json"
            with open(preferences_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_preferences, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save user preferences: {e}{Style.RESET_ALL}")

    def _load_qa_dataset(self):
        """Load the QA dataset from file if available."""
        try:
            if os.path.exists(QA_DATASET_PATH):
                with open(QA_DATASET_PATH, 'r', encoding='utf-8') as f:
                    self.qa_dataset = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded QA dataset with {len(self.qa_dataset)} examples{Style.RESET_ALL}")
            else:
                self.qa_dataset = []
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load QA dataset: {e}{Style.RESET_ALL}")
            self.qa_dataset = []

    def _save_qa_dataset(self):
        """Save the QA dataset to file."""
        try:
            with open(QA_DATASET_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.qa_dataset, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save QA dataset: {e}{Style.RESET_ALL}")

    def _setup_gemini_model(self) -> None:
        """Configure the Gemini AI model."""
        try:
            # Check if the google-generativeai package is installed
            try:
                import google.generativeai as genai
            except ImportError:
                print(f"{Fore.RED}The 'google-generativeai' package is not installed. Please install it with 'pip install google-generativeai'.{Style.RESET_ALL}")
                self.gemini_model = False
                return
                
            # Check if API key is available
            if not GEMINI_API_KEY:
                print(f"{Fore.RED}Gemini API key is not set. Please set the GEMINI_API_KEY environment variable.{Style.RESET_ALL}")
                self.gemini_model = False
                return
                
            genai.configure(api_key=GEMINI_API_KEY)
            generation_config = {
                "temperature": 0.8,  # Balanced between creativity and consistency
                "top_p": 0.95,      # Slightly increased for more diverse responses
                "top_k": 40,        # Keep top 40 tokens for sampling
                "max_output_tokens": 2096,  # Increased for longer responses
                "candidate_count": 1,  # Number of response candidates to generate
            }
            
            # Use gemini-1.5-pro which has better multimodal support
            # The flash-lite model doesn't fully support image inputs correctly
            try:
                # First try to use gemini-1.5-pro which has excellent multimodal support
                model_name = "gemini-1.5-pro"
                self.model = genai.GenerativeModel(model_name=model_name, 
                                                generation_config=generation_config)
                print(f"{Fore.GREEN}✓ Successfully configured Gemini model: {model_name}{Style.RESET_ALL}")
            except Exception as model_err:
                print(f"{Fore.YELLOW}Could not initialize primary model, trying fallback: {str(model_err)}{Style.RESET_ALL}")
                # Fall back to gemini-pro if 1.5 isn't available
                try:
                    fallback_model = "gemini-pro"
                    self.model = genai.GenerativeModel(model_name=fallback_model, 
                                                    generation_config=generation_config)
                    print(f"{Fore.GREEN}✓ Successfully configured fallback Gemini model: {fallback_model}{Style.RESET_ALL}")
                except Exception as fallback_err:
                    print(f"{Fore.RED}Failed to initialize Gemini models: {str(fallback_err)}{Style.RESET_ALL}")
                    self.gemini_model = False
                    return
            
            self.gemini_model = True
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]
            
            print(f"{Fore.GREEN}✓ Successfully configured Gemini API{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error configuring Gemini API: {str(e)}{Style.RESET_ALL}")
            self.gemini_model = False

    def _load_conversation_history(self) -> None:
        """Load conversation history from file if available."""
        try:
            history_path = f"conversation_history_{self.user_id}.json"
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                print(
                    f"{Fore.GREEN}✓ Loaded conversation history with {len(self.conversation_history)} exchanges{Style.RESET_ALL}")
            elif os.path.exists(CONVERSATION_HISTORY_PATH):
                # Fall back to default history file
                with open(CONVERSATION_HISTORY_PATH, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                print(
                    f"{Fore.GREEN}✓ Loaded default conversation history with {len(self.conversation_history)} exchanges{Style.RESET_ALL}")
            else:
                self.conversation_history = []
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load conversation history: {e}{Style.RESET_ALL}")
            self.conversation_history = []

    def _save_conversation_history(self) -> None:
        """Save conversation history to file."""
        try:
            history_path = f"conversation_history_{self.user_id}.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save conversation history: {e}{Style.RESET_ALL}")

    def add_to_dataset(self, question: str, answer: str, context: str = "") -> None:
        """Add a new question-answer pair to the dataset for contextual learning.

        Args:
            question: User's question
            answer: AI's answer
            context: Relevant context for this QA pair
        """
        # Add to QA dataset
        qa_pair = {
            "question": question,
            "answer": answer,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }

        self.qa_dataset.append(qa_pair)
        self._save_qa_dataset()
        
        # Update learning statistics
        self.learning_stats["dataset_growth"].append({
            "timestamp": datetime.now().isoformat(),
            "dataset_size": len(self.qa_dataset)
        })

        # Add to context model for future retrieval
        self.context_model.add_context(
            f"Q: {question}\nA: {answer}",
            metadata={"type": "qa_pair", "timestamp": datetime.now().isoformat()}
        )
        
        # Update learning statistics
        self.learning_stats["context_items_added"] += 1
        self.learning_stats["last_context_update"] = datetime.now().isoformat()

    def _process_query(self, query: str) -> str:
        """Process a user query with enhanced context understanding.
        
        Args:
            query: The user's input query
            
        Returns:
            The chatbot's response
        """
        # Add query to context
        self.context_model.add_context(query, {"role": "user"})
        
        # Detect intent
        intent = self.context_model.detect_intent(query)
        
        # Extract entities
        entities = self.context_model.extract_entities(query)
        
        # Get relevant context
        relevant_context = self.context_model.retrieve_relevant_context(query)
        
        # Use the Gemini model for response generation
        return self.generate_response(query, voice_input=False, 
                                      additional_context=relevant_context,
                                      detected_intent=intent,
                                      extracted_entities=entities)
                                      
    def generate_response(self, query: str, voice_input: bool = False, 
                          additional_context: List[str] = None,
                          detected_intent: str = None, 
                          extracted_entities: Dict[str, List[str]] = None,
                          image_path: str = None) -> str:
        """Generate a response to a user query.
        
        Args:
            query: The user's query
            voice_input: Whether the input is from voice (affects response style)
            additional_context: Optional list of contextual information to include
            detected_intent: Optional detected intent from NLU
            extracted_entities: Optional dictionary of extracted entities
            image_path: Optional path to an image for multimodal queries
            
        Returns:
            The generated response
        """
        try:
            # Special case for image analysis - use direct multimodal handling
            if image_path and os.path.exists(image_path):
                print(f"Image detected, using direct multimodal generation: {image_path}")
                try:
                    if not query or query.strip() == "":
                        query = "What can you tell me about this image?"
                    
                    return self.generate_multimodal_response(query, image_path)
                except Exception as img_err:
                    print(f"Error in multimodal generation: {str(img_err)}")
                    # Fall back to regular text response
            
            # Record time for performance tracking
            start_time = time.time()
            
            # Format query for readability in conversation history
            formatted_query = query.strip()
            
            # Start building the instruction for the model
            instruction = f"Query: {formatted_query}\n\n"
            
            # Add any additional context
            if additional_context and any(additional_context):
                valid_contexts = [ctx for ctx in additional_context if ctx]
                if valid_contexts:
                    instruction += "Context:\n" + "\n".join(valid_contexts) + "\n\n"
                    
            # Add intent if available
            if detected_intent:
                instruction += f"Detected intent: {detected_intent}\n\n"
                
            # Add entities if available
            if extracted_entities and any(extracted_entities.values()):
                entity_str = ""
                for entity_type, entities in extracted_entities.items():
                    if entities:
                        entity_str += f"- {entity_type}: {', '.join(entities)}\n"
                if entity_str:
                    instruction += f"Extracted entities:\n{entity_str}\n"
            
            # Add the instruction for educational context
            educational_context = self._get_educational_context(query)
            if educational_context:
                instruction += f"Educational context:\n{educational_context}\n\n"
                
            # Add instructions for response formatting based on mode
            instruction += "Instructions:\n"
            
            # Add voice-specific instructions if from voice input
            if voice_input:
                instruction += "- This query came from voice input. Keep your response concise and easy to listen to.\n"
                instruction += "- Use shorter sentences and simpler language.\n"
                instruction += "- Avoid listing too many items.\n"
            else:
                instruction += "- Provide a helpful, educational response.\n"
                instruction += "- Use markdown formatting when appropriate for clarity.\n"
                
            # General instruction continuation
            instruction += "- Focus on agricultural, educational, or vocational topics.\n"
            instruction += "- If the query is unclear, ask for clarification.\n"
            instruction += "- If the query is not related to agriculture, education, or vocational training, " \
                          "politely redirect to these topics.\n"
            
            # Generate response using Gemini
            if self.gemini_model:
                try:
                    # Use the text-only generation method 
                    response = self._generate_with_gemini(instruction)
                except Exception as e:
                    print(f"Error generating with Gemini: {e}")
                    # Fall back to basic keyword matching
                    response = "I'm sorry, I'm having trouble generating a response right now. " \
                               "Please try again with a different question."
            else:
                # Fallback to a basic response if model is not available
                response = "I'm sorry, the AI model is not available right now. " \
                           "Please try again later or contact support if this issue persists."
            
            # Record request in conversation history
            self.conversation_history.append({
                "query": formatted_query,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save conversation periodically 
            if len(self.conversation_history) % 10 == 0:
                self._save_conversation_history()

            # If continuous learning is enabled, update progress data
            if self.continuous_learning:
                self._update_progress_from_query(query, response)
                
            # Track performance
            duration = time.time() - start_time
            print(f"Response generated in {duration:.2f} seconds")
                
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try asking in a different way."
    
    def generate_multimodal_response(self, query: str, image_path: str) -> str:
        """Generate a response that includes image analysis.
        
        This is a completely separate method to avoid any recursion issues.
        
        Args:
            query: The text query accompanying the image
            image_path: Path to the image file
            
        Returns:
            A text response analyzing the image
        """
        try:
            # Basic validation
            if not self.gemini_model or not self.model:
                return "Image analysis is not available because the AI model is not configured correctly."
                
            if not os.path.exists(image_path):
                return "I couldn't find the image file. Please try uploading it again."
            
            # Import PIL for image handling
            try:
                from PIL import Image
            except ImportError:
                return "I need the Python Imaging Library (PIL) to analyze images. Please install it with 'pip install pillow'."
            
            # Prepare the image and prompt
            try:
                # Load the image using PIL
                image = Image.open(image_path)
                print(f"Loaded image with size {image.size}")
                
                # Prepare the multimodal prompt
                prompt = [
                    f"Please analyze this image. {query}",  # Text instruction
                    image                                   # The image data
                ]
                
                # Generate the response
                print("Sending multimodal prompt to Gemini")
                generation_result = self.model.generate_content(prompt)
                
                # Process the result
                if generation_result and hasattr(generation_result, 'text'):
                    print("Successfully received multimodal response")
                    return generation_result.text
                else:
                    print("Received empty response from Gemini")
                    return "I couldn't analyze this image properly. The model returned an empty response."
                    
            except Exception as e:
                print(f"Error during multimodal generation: {str(e)}")
                return f"I had trouble analyzing this image: {str(e)}"
        
        except Exception as e:
            print(f"Top-level error in multimodal generation: {str(e)}")
            return "I encountered an error while trying to analyze this image. Please try again with a different image or question."

    def _generate_with_gemini(self, model_input: str) -> str:
        """Generate a response using the Gemini API with text input only.
        
        Args:
            model_input: The text input to the model
            
        Returns:
            The generated response text
        """
        try:
            if not self.gemini_model or not self.model:
                return "Gemini model is not available. Please check your API key and configuration."
                
            # Text-only generation
            print("Generating with Gemini (text-only)")
            response = self.model.generate_content(model_input)
            return response.text
        except Exception as e:
            print(f"Error generating with Gemini: {str(e)}")
            # Return a fallback response
            return f"I'm having trouble generating a response at the moment. Error: {str(e)}"
    
    def _get_educational_context(self, query):
        """Get educational context relevant to the query.
        
        Args:
            query: The user's query
            
        Returns:
            List of relevant educational context strings
        """
        context = []
        
        # Check if query is related to any ITI topics
        for category, topics in ITI_TOPICS.items():
            for topic in topics:
                if topic.lower() in query.lower():
                    # Get study materials for this topic
                    materials = self.study_material_generator.get_study_materials(category, topic)
                    if materials:
                        for material in materials[:2]:  # Limit to 2 materials
                            context.append(f"Study Material ({material['type']}): {material['content'][:500]}...")
                    
                    # Get progress information
                    progress_data = self.progress_tracker.progress_data.get(category, {}).get(topic, {})
                    if progress_data:
                        progress_info = f"Progress in {topic}: {progress_data.get('progress', 0)}% complete"
                        context.append(progress_info)
                    
                    break
        
        return context
    
    def _get_youtube_recommendations(self, query):
        """Get YouTube video recommendations for the query.
        
        Args:
            query: The user's query
            
        Returns:
            String with YouTube recommendations or None
        """
        # Expanded list of educational terms
        educational_terms = [
            "how to", "explain", "tutorial", "learn", "course", 
            "training", "guide", "lesson", "example", "demonstration",
            "lecture", "workshop", "classroom", "teaching", "instruction",
            "video", "watch", "show", "demonstrate", "teach", "help",
            "understand", "explanation", "step by step", "process",
            "procedure", "method", "technique", "practice", "exercise"
        ]
        
        # More lenient check for educational queries
        query_lower = query.lower()
        is_educational = (
            any(term in query_lower for term in educational_terms) or
            any(topic.lower() in query_lower for topic in [topic for topics in ITI_TOPICS.values() for topic in topics])
        )
        
        if not is_educational:
            return None
            
        videos = self.youtube_manager.search_videos(query, max_results=3)
        
        if not videos:
            return None
            
        recommendations = f"\n{Fore.CYAN}📚 Recommended Educational Videos:{Style.RESET_ALL}\n"
        
        for i, video in enumerate(videos, 1):
            # Format video duration and views
            views = f"{video['views']:,}" if 'views' in video else "N/A"
            duration = video['duration'] if 'duration' in video else "N/A"
            
            recommendations += f"\n{i}. {Fore.YELLOW}{video['title']}{Style.RESET_ALL}"
            recommendations += f"\n   👤 {video['channel']} | ⏱️ {duration} | 👁️ {views} views"
            recommendations += f"\n   🔗 Watch: {video['url']}\n"
        
        return recommendations
    
    def _update_progress_from_query(self, query: str, response: str) -> None:
        """Update progress tracking with enhanced entity awareness.
        
        Args:
            query: The user's query
            response: The assistant's response
        """
        # Extract entities from query
        entities = self.context_model.extract_entities(query)
        
        # Check for ITI topics in entities and query
        for category, topics in ITI_TOPICS.items():
            for topic in topics:
                topic_lower = topic.lower()
                if (topic_lower in query.lower() or
                    any(topic_lower in entity.lower() for entity_list in entities.values() for entity in entity_list)):
                    # Calculate response quality based on length and entity relevance
                    response_quality = min(len(response) / 1000, 1.0)  # Normalize to 0-1
                    
                    # Get current progress
                    current_progress = self.progress_tracker.progress_data.get(category, {}).get(topic, {}).get("progress", 0)
                    
                    # Increment progress based on response quality (max 5% per interaction)
                    progress_increment = min(response_quality * 5, 5)
                    new_progress = min(current_progress + progress_increment, 100)
                    
                    self.progress_tracker.update_topic_progress(category, topic, new_progress)
                    break
    
    def listen_for_voice_query(self):
        """Listen for a voice query using the enhanced voice manager.
        
        Returns:
            Recognized text or None if recognition failed
        """
        if not self.user_preferences["voice_enabled"] or not self.voice_manager.is_available:
            print(f"{Fore.YELLOW}[VOICE] Voice interaction is disabled or unavailable{Style.RESET_ALL}")
            return None
            
        # Use the enhanced voice manager with proper language mapping
        lang_code = self.user_preferences["language"]
        # Map language code to speech recognition format if needed (e.g., 'en' to 'en-US')
        speech_lang = lang_code
        if len(lang_code) == 2:
            speech_lang = f"{lang_code}-{lang_code.upper()}"
            
        # Show clear debug messages about voice input process
        print(f"{Fore.GREEN}[VOICE] 🎤 Listening for voice input in {speech_lang}...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[VOICE] Please speak clearly into your microphone...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[VOICE] Adjusting for ambient noise...{Style.RESET_ALL}")
        
        # Use a short timeout (5 seconds) to quickly fall back to typing if no speech is detected
        # This helps prevent long waits when the user intends to type
        try:
            print(f"{Fore.CYAN}[VOICE] Ready to capture speech (timeout: 5s){Style.RESET_ALL}")
            result = self.voice_manager.listen(language=speech_lang, timeout=5)
            
            if result:
                print(f"{Fore.GREEN}[VOICE] ✓ Successfully captured: \"{result}\"{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}[VOICE] ⚠ No speech detected or recognition failed{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}[VOICE] Please check your microphone and try again{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}[VOICE] ❌ Error during voice recognition: {str(e)}{Style.RESET_ALL}")
            result = None
            
        return result
    
    def start_continuous_voice_interaction(self):
        """Start continuous voice interaction mode using event-based architecture.
        
        This will continuously listen for voice input and process it.
        """
        if not self.user_preferences["voice_enabled"] or not self.voice_manager.is_available:
            return False
            
        # Map language code to speech recognition format
        lang_map = {
            'en': 'en-US',
            'es': 'es-ES',
            'fr': 'fr-FR',
            'de': 'de-DE',
            'it': 'it-IT',
            'pt': 'pt-PT',
            'zh': 'zh-CN',
            'ja': 'ja-JP',
            'ko': 'ko-KR',
            'ru': 'ru-RU'
        }
        
        lang_code = self.user_preferences["language"]
        recognition_lang = lang_map.get(lang_code, 'en-US')
        
        # Display wake word information if set
        wake_word = self.voice_manager.settings.get("wake_word")
        if wake_word:
            print(f"{Fore.CYAN}Wake word is set to: '{wake_word}' - Say this to activate voice recognition{Style.RESET_ALL}")
        
        # Remove any existing event handlers to prevent duplicates
        self.voice_manager.off("recognition_completed")
        self.voice_manager.off("error_occurred")
        
        # Set up event handlers
        self.voice_manager.on("recognition_completed", self._handle_voice_recognition)
        self.voice_manager.on("error_occurred", self._handle_voice_error)
        
        # Start continuous listening
        session = self.voice_manager.start_continuous_listening(language=recognition_lang)
        
        if session:
            print(f"{Fore.CYAN}Voice session started: {session.id}{Style.RESET_ALL}")
            return True
        return False
    
    def _handle_voice_recognition(self, data):
        """Handle voice recognition event.
        
        Args:
            data: Recognition event data containing session_id and text
        """
        if not data or "text" not in data or not data["text"]:
            return
            
        # Process the recognized text
        text = data["text"]
        print(f"\n{Fore.GREEN}Recognized: \"{text}\"{Style.RESET_ALL}")
        
        # Check for wake word if set
        wake_word = self.voice_manager.settings.get("wake_word")
        if wake_word and wake_word.lower() not in text.lower():
            print(f"{Fore.YELLOW}Wake word not detected. Say '{wake_word}' to activate.{Style.RESET_ALL}")
            return
            
        # If wake word is present, process the command (optionally remove the wake word from the command)
        if wake_word and wake_word.lower() in text.lower():
            # Remove the wake word from the text
            command = text.lower().replace(wake_word.lower(), "").strip()
            if command:
                print(f"{Fore.CYAN}Processing command: \"{command}\"{Style.RESET_ALL}")
                text = command
            else:
                # If only wake word was spoken, prompt for a command
                self.voice_manager.speak("Yes, how can I help you?", language=self.user_preferences["language"])
                return
        
        # Process the query using ChatBot's processing logic
        response = self._process_query(text)
        
        # Format and display the response
        if response:
            formatted_response = format_response(response)
            print(f"\n{Fore.BLUE}AI Assistant:{Style.RESET_ALL}")
            print(formatted_response)
            
            # Speak the response
            lang_code = self.user_preferences["language"]
            self.voice_manager.speak(response, language=lang_code)
    
    def _process_query(self, text):
        """Process a query and generate a response.
        
        Args:
            text: The query text to process
            
        Returns:
            Generated response text
        """
        # Process the query and get response
        response = self.generate_response(text)
        
        # Save to conversation history
        self.conversation_history.append({
            "user": text,
            "assistant": response,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        self._save_conversation_history()
        
        return response
    
    def _handle_voice_error(self, data):
        """Handle voice recognition error event.
        
        Args:
            data: Error event data
        """
        if not data or "error" not in data:
            return
            
        error = data["error"]
        source = data.get("source", "unknown")
        
        if source == "listen" and "Could not understand audio" in error:
            # Don't show this common error to avoid clutter
            return
            
        print(f"{Fore.YELLOW}Voice error in {source}: {error}{Style.RESET_ALL}")
    
    def stop_continuous_voice_interaction(self):
        """Stop continuous voice interaction mode."""
        if self.voice_manager:
            # Remove event handlers
            self.voice_manager.off("recognition_completed")
            self.voice_manager.off("error_occurred")
            
            # Stop listening
            self.voice_manager.stop_continuous_listening()
            print(f"{Fore.YELLOW}Voice interaction stopped{Style.RESET_ALL}")
            return True
        return False
    
    def set_language_preference(self, language_code):
        """Set the user's preferred language.
        
        Args:
            language_code: ISO language code
            
        Returns:
            Success message
        """
        if language_code in self.multilingual.supported_languages:
            self.user_preferences["language"] = language_code
            self._save_user_preferences()
            language_name = self.multilingual.supported_languages[language_code]
            return f"Language preference set to {language_name}"
        else:
            supported = ", ".join(f"{code} ({name})" for code, name in self.multilingual.supported_languages.items())
            return f"Unsupported language code. Supported languages are: {supported}"
    
    def toggle_voice_interaction(self):
        """Toggle voice interaction on/off.
        
        Returns:
            Status message
        """
        if not self.voice_manager.is_available:
            return "Voice interaction is not available on this system"
            
        self.user_preferences["voice_enabled"] = not self.user_preferences["voice_enabled"]
        self._save_user_preferences()
        
        status = "enabled" if self.user_preferences["voice_enabled"] else "disabled"
        return f"Voice interaction {status}"
    
    def set_difficulty_level(self, level):
        """Set the difficulty level for educational content.
        
        Args:
            level: Difficulty level (easy, medium, hard)
            
        Returns:
            Success message
        """
        valid_levels = ["easy", "medium", "hard"]
        if level.lower() in valid_levels:
            self.user_preferences["difficulty_level"] = level.lower()
            self._save_user_preferences()
            return f"Difficulty level set to {level}"
        else:
            return f"Invalid difficulty level. Choose from: {', '.join(valid_levels)}"
    
    def generate_quiz(self, topic, subtopic, difficulty="medium", num_questions=5):
        """Generate a quiz for a specific topic.
        
        Args:
            topic: Main topic category
            subtopic: Specific subtopic
            difficulty: Quiz difficulty level
            num_questions: Number of questions
            
        Returns:
            Generated quiz
        """
        return self.quiz_manager.generate_quiz(
            topic, subtopic, difficulty, num_questions, self.model
        )
    
    def take_quiz(self, quiz_id):
        """Prepare a quiz for the user to take.
        
        Args:
            quiz_id: ID of the quiz to take
            
        Returns:
            Quiz data formatted for display
        """
        quiz = self.quiz_manager.get_quiz_by_id(quiz_id)
        if not quiz:
            return {"error": "Quiz not found"}
            
        # Format for display (without showing correct answers)
        display_quiz = {
            "id": quiz["id"],
            "topic": quiz["topic"],
            "subtopic": quiz["subtopic"],
            "difficulty": quiz["difficulty"],
            "questions": []
        }
        
        for q in quiz["questions"]:
            display_quiz["questions"].append({
                "question": q["question"],
                "options": q["options"]
            })
            
        return display_quiz
    
    def check_quiz_answers(self, quiz_id, user_answers):
        """Check user's answers for a quiz.
        
        Args:
            quiz_id: ID of the quiz
            user_answers: Dictionary mapping question indices to user's answers
            
        Returns:
            Quiz results
        """
        quiz = self.quiz_manager.get_quiz_by_id(quiz_id)
        if not quiz:
            return {"error": "Quiz not found"}
            
        correct_count = 0
        results = []
        
        for i, (q_idx, user_answer) in enumerate(user_answers.items()):
            q_idx = int(q_idx)
            if q_idx < len(quiz["questions"]):
                question = quiz["questions"][q_idx]
                is_correct = user_answer == question["correct_answer"]
                
                if is_correct:
                    correct_count += 1
                    
                results.append({
                    "question": question["question"],
                    "user_answer": user_answer,
                    "correct_answer": question["correct_answer"],
                    "is_correct": is_correct,
                    "explanation": question["explanation"]
                })
        
        # Record quiz result in progress tracker
        self.progress_tracker.record_quiz_result(
            quiz["topic"], 
            quiz["subtopic"], 
            correct_count, 
            len(results)
        )
        
        return {
            "quiz_id": quiz_id,
            "score": correct_count,
            "total": len(results),
            "percentage": (correct_count / len(results)) * 100 if results else 0,
            "results": results
        }
    
    def generate_study_materials(self, topic, subtopic, content=None):
        """Generate study materials for a topic.
        
        Args:
            topic: Main topic category
            subtopic: Specific subtopic
            content: Optional content to summarize
            
        Returns:
            Generated study materials
        """
        if content:
            # Generate summary of provided content
            return self.study_material_generator.generate_summary(
                topic, subtopic, content, self.model
            )
        else:
            # Generate revision notes for the topic
            return self.study_material_generator.create_revision_notes(
                topic, subtopic, self.model
            )
    
    def get_progress_report(self):
        """Get a progress report for the user.
        
        Returns:
            Progress summary
        """
        return self.progress_tracker.get_progress_summary()
    
    def recommend_next_topics(self):
        """Recommend next topics to study based on progress.
        
        Returns:
            List of recommended topics
        """
        progress_summary = self.progress_tracker.get_progress_summary()
        recommendations = []
        
        # Recommend topics from areas for improvement
        for area in progress_summary.get("areas_for_improvement", []):
            recommendations.append({
                "topic": area["topic"],
                "subtopic": area["subtopic"],
                "reason": "This is an area where you could improve",
                "current_progress": f"{area['progress']:.1f}%"
            })
        
        # If not enough recommendations, add some from unexplored topics
        if len(recommendations) < 3:
            for category, topics in ITI_TOPICS.items():
                for topic in topics:
                    # Check if this topic is already in progress
                    if category in progress_summary.get("topics", {}) and topic in progress_summary["topics"][category].get("subtopics", {}):
                        continue
                        
                    # Add as recommendation
                    recommendations.append({
                        "topic": category,
                        "subtopic": topic,
                        "reason": "New topic to explore",
                        "current_progress": "0%"
                    })
                    
                    if len(recommendations) >= 3:
                        break
                        
                if len(recommendations) >= 3:
                    break
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        self._save_conversation_history()
        print(f"{Fore.GREEN}Conversation history cleared.{Style.RESET_ALL}")

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the user's learning progress."""
        # Get progress data from the progress tracker
        progress_data = self.progress_tracker.get_progress_summary()
        
        # Calculate overall stats
        completed_topics = sum(1 for topic in progress_data.get("topics", {}).values() 
                              if topic.get("completion", 0) == 100)
        total_topics = len(progress_data.get("topics", {}))
        
        # Calculate quiz performance
        quiz_results = progress_data.get("quiz_results", [])
        avg_score = sum(result.get("score", 0) for result in quiz_results) / len(quiz_results) if quiz_results else 0
        
        stats = {
            "completed_topics": completed_topics,
            "total_topics": total_topics,
            "completion_percentage": (completed_topics / total_topics * 100) if total_topics > 0 else 0,
            "quiz_count": len(quiz_results),
            "average_quiz_score": avg_score,
            "strong_topics": progress_data.get("strong_topics", []),
            "weak_topics": progress_data.get("weak_topics", []),
        }
        
        return stats
    
    def predict_crop_prices(self, crop: str, days_ahead: int = 7) -> Dict:
        """Predict future prices for a specific crop.
        
        Args:
            crop: The crop to predict prices for
            days_ahead: Number of days to predict ahead
            
        Returns:
            Dictionary with prediction results
        """
        return self.market_predictor.predict_price(crop, days_ahead)
    
    def get_market_data(self, crop: str, location: str = None) -> Dict:
        """Get current market data for a crop.
        
        Args:
            crop: The crop to get data for
            location: Optional location filter
            
        Returns:
            Dictionary with market data
        """
        return self.market_predictor.fetch_market_data(crop, location)
    
    def setup_price_alerts(self, crop: str, farmer_id: str, contact_info: Dict) -> Dict:
        """Set up price alerts for a farmer.
        
        Args:
            crop: The crop to monitor
            farmer_id: Farmer's identifier
            contact_info: Contact information for alerts
            
        Returns:
            Alert setup status
        """
        return self.market_predictor.generate_farmer_alert(crop, farmer_id, contact_info)
    
    def get_n8n_workflow(self) -> Dict:
        """Get n8n workflow configuration for price predictions and alerts.
        
        Returns:
            n8n workflow configuration
        """
        return self.market_predictor.get_n8n_workflow_config()

    def detect_plant_disease(self, image_path):
        """Detect plant disease from an image.
        
        Args:
            image_path (str): Path to the plant image file
            
        Returns:
            str: Formatted response with disease detection results
        """
        if not os.path.exists(image_path):
            return f"{Fore.RED}Error: Image file not found at {image_path}{Style.RESET_ALL}"
        
        # Analyze the image
        result = self.plant_disease_detector.predict_disease(image_path)
        
        if result["status"] == "error":
            return f"{Fore.RED}Error analyzing image: {result['error']}{Style.RESET_ALL}"
        
        # Format the response
        response = []
        response.append(f"\n{Fore.CYAN}🔬 Plant Disease Analysis Results:{Style.RESET_ALL}\n")
        response.append(f"{Fore.GREEN}📊 Prediction:{Style.RESET_ALL} {result['formatted_class']}")
        response.append(f"{Fore.GREEN}🎯 Confidence:{Style.RESET_ALL} {result['confidence']:.2f}%")
        
        # Check if plant is healthy or diseased
        if result["is_healthy"]:
            response.append(f"\n{Fore.GREEN}✅ Good news! Your plant appears to be healthy.{Style.RESET_ALL}")
        else:
            # Get treatment recommendations
            treatment = self.plant_disease_detector.get_treatment_recommendation(result["predicted_class"])
            
            response.append(f"\n{Fore.YELLOW}⚠️ Disease detected: {treatment['disease']}{Style.RESET_ALL}")
            response.append(f"\n{Fore.BLUE}📋 Description:{Style.RESET_ALL}")
            response.append(f"{treatment['description']}")
            
            response.append(f"\n{Fore.BLUE}💊 Treatment:{Style.RESET_ALL}")
            response.append(f"{treatment['treatment']}")
            
            response.append(f"\n{Fore.BLUE}🛡️ Prevention:{Style.RESET_ALL}")
            response.append(f"{treatment['prevention']}")
        
        # Add top alternative predictions if confidence is not very high
        if result["confidence"] < 90 and len(result["all_predictions"]) > 1:
            response.append(f"\n{Fore.CYAN}📊 Alternative possibilities:{Style.RESET_ALL}")
            alt_predictions = sorted(result["all_predictions"], key=lambda x: x["confidence"], reverse=True)[1:4]
            for i, pred in enumerate(alt_predictions):
                if pred["confidence"] > 5:  # Only show alternatives with >5% confidence
                    response.append(f"  {i+1}. {pred['class']} ({pred['confidence']:.2f}%)")
        
        # Add to history and update progress tracker
        self._update_progress_from_query(
            f"Analyzed plant image for diseases: {result['formatted_class']}", 
            "\n".join(response)
        )
        
        return "\n".join(response)


def display_welcome_message() -> None:
    """Display a welcome message with ASCII art."""
    welcome_text = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║     🧠 ITI & AGRICULTURAL EDUCATION AI ASSISTANT 🧠                 ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Type your messages to chat with the AI assistant.
    
    Special commands:
     - !help     : Show this help message
     - !clear    : Clear conversation history
     - !stats    : Show learning statistics
     - !exit     : Exit the application
     
    Educational features:
     - !voice    : Toggle voice interaction
     - !wake <phrase> : Set wake word for voice activation (e.g., !wake utho iti)
     - !language <code> : Set language (e.g., !language hi for Hindi)
     - !length <option> : Set response length (concise, medium, detailed)
     - !difficulty <level> : Set difficulty (easy, medium, hard)
     - !quiz <topic> <subtopic> : Generate a quiz
     - !study <topic> <subtopic> : Generate study materials
     - !progress : Show your learning progress
     - !recommend : Get topic recommendations
     - !topics   : List available topics
     - !plantdisease <image_path> : Detect plant diseases from an image
     
    Start with:
     - python iti_app.py --continuous-learning  : Enable contextual learning
     - python iti_app.py --user-id <n>       : Use a specific user profile
    """
    print(f"{Fore.CYAN}{welcome_text}{Style.RESET_ALL}")


def format_response(text: str) -> str:
    """Format the response for better readability."""
    paragraphs = text.split('\n\n')
    formatted_paragraphs = []

    for paragraph in paragraphs:
        if paragraph.strip():
            wrapped = textwrap.fill(paragraph, width=80)
            formatted_paragraphs.append(wrapped)

    return '\n\n'.join(formatted_paragraphs)


def display_topics():
    """Display available ITI topics."""
    print(f"\n{Fore.CYAN}Available Topics:{Style.RESET_ALL}")
    
    for category, topics in ITI_TOPICS.items():
        print(f"\n{Fore.YELLOW}{category.replace('_', ' ').title()}:{Style.RESET_ALL}")
        for i, topic in enumerate(topics):
            print(f"  {i+1}. {topic}")


def take_quiz_interactive(chatbot, quiz):
    """Interactive quiz-taking function.
    
    Args:
        chatbot: ChatBot instance
        quiz: Quiz data
    """
    if "error" in quiz:
        print(f"{Fore.RED}{quiz['error']}{Style.RESET_ALL}")
        return
        
    print(f"\n{Fore.CYAN}Quiz: {quiz['topic']} - {quiz['subtopic']} (Difficulty: {quiz['difficulty']}){Style.RESET_ALL}")
    print(f"Answer each question by typing the letter of your choice (A, B, C, or D).\n")
    
    user_answers = {}
    
    for i, question in enumerate(quiz["questions"]):
        print(f"\n{Fore.YELLOW}Question {i+1}:{Style.RESET_ALL} {question['question']}")
        
        for option, text in question["options"].items():
            print(f"  {option}: {text}")
            
        while True:
            answer = input(f"\n{Fore.GREEN}Your answer:{Style.RESET_ALL} ").strip().upper()
            if answer in ["A", "B", "C", "D"]:
                user_answers[str(i)] = answer
                break
            else:
                print(f"{Fore.RED}Please enter A, B, C, or D.{Style.RESET_ALL}")
    
    # Check answers
    results = chatbot.check_quiz_answers(quiz["id"], user_answers)
    
    # Display results
    print(f"\n{Fore.CYAN}Quiz Results:{Style.RESET_ALL}")
    print(f"Score: {results['score']}/{results['total']} ({results['percentage']:.1f}%)")
    
    print(f"\n{Fore.CYAN}Detailed Results:{Style.RESET_ALL}")
    for i, result in enumerate(results["results"]):
        if result["is_correct"]:
            print(f"\n{Fore.GREEN}✓ Question {i+1}: Correct!{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}✗ Question {i+1}: Incorrect{Style.RESET_ALL}")
            print(f"  Your answer: {result['user_answer']}")
            print(f"  Correct answer: {result['correct_answer']}")
        
        print(f"  {Fore.YELLOW}Explanation:{Style.RESET_ALL} {result['explanation']}")


def run_cli_loop(chatbot, output_format="text"):
    """Main command-line interface loop."""
    
    def display_voice_status(status):
        if status:
            return f"{Fore.GREEN}✓ Voice input enabled{Style.RESET_ALL}"
        else:
            return f"{Fore.YELLOW}✗ Voice input disabled{Style.RESET_ALL}"
    
    # Display welcome message
    display_welcome_message()
    
    # Display voice status
    print(f"Voice interaction: {display_voice_status(chatbot.user_preferences['voice_enabled'])}")
    
    continuous_voice = False
    should_exit = False
    
    while not should_exit:
        query = input(f"\n{Fore.GREEN}You: {Style.RESET_ALL}")
        
        if query.lower() in ["exit", "quit", "bye"]:
            should_exit = True
            print(f"\n{Fore.GREEN}Thank you for using the AI Chatbot. Goodbye!{Style.RESET_ALL}")
            continue
        
        if query.lower() == "voice":
            print(f"{Fore.CYAN}[VOICE] Activating voice input mode...{Style.RESET_ALL}")
            print(f"{Fore.CYAN}[VOICE] Initializing microphone and speech recognition...{Style.RESET_ALL}")
            
            voice_query = chatbot.listen_for_voice_query()
            
            if voice_query:
                query = voice_query
                print(f"\n{Fore.GREEN}You (voice): {voice_query}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}[VOICE] Processing voice query...{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.RED}[VOICE] No voice input detected or recognition failed.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}[VOICE] Please try again or type your query.{Style.RESET_ALL}")
                continue
        
        # Process special commands
        if query.startswith("!"):
            command = query[1:].lower()
            
            if command == "help":
                display_welcome_message()
                continue
            elif command == "clear":
                chatbot.clear_history()
                continue
            elif command == "stats":
                stats = chatbot.get_learning_stats()
                print(f"\n{Fore.GREEN}Learning Statistics:{Style.RESET_ALL}")
                print(f"Completed Topics: {stats['completed_topics']}/{stats['total_topics']}")
                print(f"Completion: {stats['completion_percentage']:.1f}%")
                print(f"Quizzes Taken: {stats['quiz_count']}")
                print(f"Average Quiz Score: {stats['average_quiz_score']:.1f}%")
                continue
            elif command == "topics":
                display_topics()
                continue
            elif command.startswith("quiz"):
                parts = command.split()
                if len(parts) >= 2:
                    topic = parts[1]
                    subtopic = parts[2] if len(parts) > 2 else None
                    quiz = chatbot.generate_quiz(topic, subtopic)
                    if quiz:
                        print(f"\n{Fore.GREEN}Quiz generated successfully!{Style.RESET_ALL}")
                        take_quiz = input(f"{Fore.CYAN}Do you want to take the quiz now? (y/n): {Style.RESET_ALL}")
                        if take_quiz.lower() == 'y':
                            take_quiz_interactive(chatbot, quiz)
                continue
            elif command.startswith("study"):
                parts = command.split()
                if len(parts) >= 2:
                    topic = parts[1]
                    subtopic = parts[2] if len(parts) > 2 else None
                    materials = chatbot.generate_study_materials(topic, subtopic)
                    if materials:
                        print(f"\n{Fore.GREEN}Study materials generated:{Style.RESET_ALL}")
                        for mat in materials:
                            print(f"\n{Fore.YELLOW}{mat['title']}{Style.RESET_ALL}")
                            print(f"{mat['content']}")
                continue
            elif command == "progress":
                report = chatbot.get_progress_report()
                print(f"\n{Fore.GREEN}Progress Report:{Style.RESET_ALL}")
                for topic, data in report.get("topics", {}).items():
                    print(f"\n{Fore.YELLOW}{topic}: {data['completion']}% complete{Style.RESET_ALL}")
                    for subtopic, completion in data.get("subtopics", {}).items():
                        if subtopic != "completion":
                            print(f"  - {subtopic}: {completion}% complete")
                print(f"\n{Fore.GREEN}Overall Progress: {report.get('overall_progress', 0)}%{Style.RESET_ALL}")
                continue
            elif command.startswith("plantdisease"):
                parts = command.split()
                if len(parts) >= 2:
                    image_path = parts[1]
                    if not os.path.exists(image_path):
                        print(f"{Fore.RED}Image file not found. Please check the path and try again.{Style.RESET_ALL}")
                        continue
                    print(f"{Fore.YELLOW}Analyzing image...{Style.RESET_ALL}")
                    result = chatbot.detect_plant_disease(image_path)
                    if result:
                        print(f"\n{Fore.GREEN}Disease Detection Results:{Style.RESET_ALL}")
                        # Check if result is a string or dictionary
                        if isinstance(result, dict):
                            print(f"Disease: {result.get('disease', 'Unknown')}")
                            print(f"Confidence: {result.get('confidence', 0) * 100:.1f}%")
                            if result.get('recommendations'):
                                print(f"\n{Fore.YELLOW}Recommendations:{Style.RESET_ALL}")
                                for rec in result.get('recommendations'):
                                    print(f"- {rec}")
                        else:
                            # Handle result as a string
                            print(result)
                    else:
                        print(f"{Fore.RED}Failed to analyze image. Please try again with a clearer image.{Style.RESET_ALL}")
                continue
            elif command.startswith("market"):
                parts = command.split()
                if len(parts) >= 2:
                    subcommand = parts[1]
                    if subcommand == "data":
                        crop = input(f"\n{Fore.CYAN}Enter crop name: {Style.RESET_ALL}")
                        location = input(f"{Fore.CYAN}Enter location (optional, press Enter to skip): {Style.RESET_ALL}")
                        market_data = chatbot.get_market_data(crop, location if location else None)
                        if "error" not in market_data:
                            print(f"\n{Fore.GREEN}Market Data for {crop}:{Style.RESET_ALL}")
                            print(f"Price: ₹{market_data['price']:.2f}")
                            print(f"Demand: {market_data['demand']}")
                            print(f"Location: {market_data['location'] if market_data['location'] else 'All Regions'}")
                            print(f"Timestamp: {market_data['timestamp']}")
                            print(f"Source: {market_data['source']}")
                        else:
                            print(f"{Fore.RED}Error fetching market data: {market_data['error']}{Style.RESET_ALL}")
                    elif subcommand == "predict":
                        crop = input(f"\n{Fore.CYAN}Enter crop name: {Style.RESET_ALL}")
                        days = input(f"{Fore.CYAN}Enter days to predict ahead (default: 7): {Style.RESET_ALL}")
                        if not days.isdigit():
                            days = 7
                        else:
                            days = int(days)
                        prediction = chatbot.predict_crop_prices(crop, days)
                        if prediction.get("status") == "success":
                            print(f"\n{Fore.GREEN}Price Prediction for {crop}:{Style.RESET_ALL}")
                            print(f"Current Price: ₹{prediction['current_price']:.2f}")
                            print(f"Price Trend: {prediction['price_trend'].title()}")
                            print(f"Best Selling Date: {prediction['best_selling_date']}")
                            print(f"Best Selling Price: ₹{prediction['best_selling_price']:.2f}")
                            print(f"\n{Fore.YELLOW}Day-by-Day Predictions:{Style.RESET_ALL}")
                            for day in prediction['predictions']:
                                print(f"{day['date']}: ₹{day['predicted_price']:.2f} (Confidence: {day['confidence'] * 100:.1f}%)")
                        else:
                            print(f"{Fore.RED}Error generating prediction: {prediction.get('error', 'Unknown error')}{Style.RESET_ALL}")
                    elif subcommand == "alert":
                        crop = input(f"\n{Fore.CYAN}Enter crop name: {Style.RESET_ALL}")
                        farmer_id = input(f"{Fore.CYAN}Enter farmer ID: {Style.RESET_ALL}")
                        phone = input(f"{Fore.CYAN}Enter phone number: {Style.RESET_ALL}")
                        contact_info = {
                            "phone": phone,
                            "language": chatbot.user_preferences.get("language", "en")
                        }
                        alert_result = chatbot.setup_price_alerts(crop, farmer_id, contact_info)
                        if alert_result.get("status") == "ready_to_send":
                            print(f"\n{Fore.GREEN}Price alert set up successfully!{Style.RESET_ALL}")
                            print(f"Alert Message: {alert_result['message']}")
                        elif alert_result.get("status") == "no_alert_needed":
                            print(f"\n{Fore.YELLOW}No alert needed at this time: {alert_result['reason']}{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.RED}Error setting up alert: {alert_result.get('error', 'Unknown error')}{Style.RESET_ALL}")
                    elif subcommand == "trends":
                        # Get market trends for top crops
                        print(f"{Fore.YELLOW}Analyzing market trends...{Style.RESET_ALL}")
                        try:
                            trends = chatbot.market_predictor.get_market_trends()
                            print(f"\n{Fore.GREEN}Current Market Trends:{Style.RESET_ALL}")
                            for crop, trend in trends.items():
                                direction = "↑" if trend['direction'] == 'up' else "↓" if trend['direction'] == 'down' else "→"
                                print(f"{crop}: {direction} {trend['percentage']}% ({trend['description']})")
                        except Exception as e:
                            print(f"{Fore.RED}Error fetching market trends: {str(e)}{Style.RESET_ALL}")
                    elif subcommand == "compare":
                        # Compare prices between different crops
                        crops = input(f"\n{Fore.CYAN}Enter comma-separated crop names to compare: {Style.RESET_ALL}")
                        crop_list = [c.strip() for c in crops.split(",")]
                        if len(crop_list) < 2:
                            print(f"{Fore.RED}Please provide at least two crops to compare.{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.YELLOW}Comparing crop prices...{Style.RESET_ALL}")
                            try:
                                comparison = chatbot.market_predictor.compare_crops(crop_list)
                                print(f"\n{Fore.GREEN}Crop Price Comparison:{Style.RESET_ALL}")
                                print(f"{'Crop':<15} {'Current Price':<15} {'7-Day Trend':<15} {'Profit Margin':<15}")
                                print(f"{'-'*60}")
                                for crop, data in comparison.items():
                                    trend = "↑" if data['trend'] > 0 else "↓" if data['trend'] < 0 else "→"
                                    print(f"{crop:<15} ₹{data['price']:<14.2f} {trend} {abs(data['trend']):<10.1f}% {data['profit_margin']:<15.1f}%")
                                
                                # Show recommendation
                                if 'recommendation' in comparison:
                                    print(f"\n{Fore.YELLOW}Recommendation:{Style.RESET_ALL} {comparison['recommendation']}")
                            except Exception as e:
                                print(f"{Fore.RED}Error comparing crops: {str(e)}{Style.RESET_ALL}")
                    elif subcommand == "history":
                        # Get historical price data for a crop
                        crop = input(f"\n{Fore.CYAN}Enter crop name: {Style.RESET_ALL}")
                        months = input(f"{Fore.CYAN}Enter number of months of history (default: 6): {Style.RESET_ALL}")
                        
                        if not months.isdigit():
                            months = 6
                        else:
                            months = int(months)
                        
                        print(f"{Fore.YELLOW}Fetching historical data for {crop}...{Style.RESET_ALL}")
                        try:
                            history = chatbot.market_predictor.get_price_history(crop, months)
                            print(f"\n{Fore.GREEN}Historical Price Data for {crop} (Last {months} months):{Style.RESET_ALL}")
                            print(f"{'Month':<10} {'Avg. Price':<15} {'Min Price':<15} {'Max Price':<15} {'% Change':<15}")
                            print(f"{'-'*70}")
                            
                            prev_price = None
                            for month_data in history:
                                month = month_data['month']
                                avg_price = month_data['avg_price']
                                
                                change = ""
                                if prev_price is not None:
                                    pct_change = ((avg_price - prev_price) / prev_price) * 100
                                    change = f"{pct_change:+.1f}%"
                                
                                prev_price = avg_price
                                print(f"{month:<10} ₹{avg_price:<14.2f} ₹{month_data['min_price']:<14.2f} ₹{month_data['max_price']:<14.2f} {change:<15}")
                            
                            # Show seasonality insights
                            if 'seasonality' in history[0]:
                                print(f"\n{Fore.YELLOW}Seasonality Insights:{Style.RESET_ALL} {history[0]['seasonality']}")
                        except Exception as e:
                            print(f"{Fore.RED}Error fetching price history: {str(e)}{Style.RESET_ALL}")
                    elif subcommand == "recommendations":
                        # Get personalized crop recommendations
                        location = input(f"\n{Fore.CYAN}Enter your location: {Style.RESET_ALL}")
                        season = input(f"{Fore.CYAN}Enter current season (summer/winter/rainy): {Style.RESET_ALL}")
                        
                        print(f"{Fore.YELLOW}Generating crop recommendations...{Style.RESET_ALL}")
                        try:
                            recommendations = chatbot.market_predictor.get_crop_recommendations(location, season)
                            print(f"\n{Fore.GREEN}Recommended Crops for {location} ({season.title()} season):{Style.RESET_ALL}")
                            print(f"\n{Fore.YELLOW}Top Recommendations:{Style.RESET_ALL}")
                            
                            for i, crop in enumerate(recommendations['top_crops'], 1):
                                print(f"{i}. {crop['name']}")
                                print(f"   Expected Price: ₹{crop['expected_price']:.2f} per {crop['unit']}")
                                print(f"   Growing Period: {crop['growing_period']} days")
                                print(f"   Profit Potential: {crop['profit_potential']}%")
                                print(f"   Reason: {crop['reason']}")
                                print()
                            
                            if 'market_insights' in recommendations:
                                print(f"{Fore.YELLOW}Market Insights:{Style.RESET_ALL} {recommendations['market_insights']}")
                        except Exception as e:
                            print(f"{Fore.RED}Error generating recommendations: {str(e)}{Style.RESET_ALL}")
                    elif subcommand == "help" or subcommand == "?":
                        # Show market command help
                        print(f"\n{Fore.CYAN}Available Market Predictor Commands:{Style.RESET_ALL}")
                        print(f"  !market data - Get current market data for a crop")
                        print(f"  !market predict - Predict future prices for a crop")
                        print(f"  !market alert - Set up price alerts for a farmer")
                        print(f"  !market trends - Show trending crops and market trends")
                        print(f"  !market compare - Compare prices of different crops")
                        print(f"  !market history - Show historical price data for a crop")
                        print(f"  !market recommendations - Get personalized crop recommendations")
                        print(f"  !market help - Show this help message")
                    else:
                        print(f"{Fore.RED}Unknown market subcommand. Type !market help for available commands.{Style.RESET_ALL}")
                else:
                    # Display market command help if no subcommand is specified
                    print(f"\n{Fore.CYAN}Available Market Predictor Commands:{Style.RESET_ALL}")
                    print(f"  !market data - Get current market data for a crop")
                    print(f"  !market predict - Predict future prices for a crop")
                    print(f"  !market alert - Set up price alerts for a farmer")
                    print(f"  !market trends - Show trending crops and market trends")
                    print(f"  !market compare - Compare prices of different crops")
                    print(f"  !market history - Show historical price data for a crop")
                    print(f"  !market recommendations - Get personalized crop recommendations")
                    print(f"  !market help - Show this help message")
                continue
            elif command.startswith("language"):
                parts = command.split()
                if len(parts) >= 2:
                    lang_code = parts[1]
                    chatbot.set_language_preference(lang_code)
                    print(f"{Fore.GREEN}Language set to: {lang_code}{Style.RESET_ALL}")
                continue
            elif command.startswith("difficulty"):
                parts = command.split()
                if len(parts) >= 2:
                    difficulty = parts[1]
                    chatbot.set_difficulty_level(difficulty)
                    print(f"{Fore.GREEN}Difficulty level set to: {difficulty}{Style.RESET_ALL}")
                continue
            elif command == "voice":
                # Toggle voice interaction
                status = chatbot.toggle_voice_interaction()
                is_enabled = chatbot.user_preferences["voice_enabled"]
                if is_enabled:
                    print(f"{Fore.GREEN}[VOICE] ✓ Voice input enabled{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}[VOICE] You can now use voice commands by typing 'voice'{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}[VOICE] Wake word is set to: '{chatbot.voice_manager.settings.get('wake_word', 'None')}'{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}[VOICE] ✗ Voice input disabled{Style.RESET_ALL}")
                continue
        
        # Process regular query
        response = chatbot.generate_response(query)
        formatted_response = format_response(response)
        print(f"\n{Fore.BLUE}AI: {formatted_response}{Style.RESET_ALL}")


def main():
    """Main entry point for the chatbot application"""
    parser = argparse.ArgumentParser(description="ITI Chatbot Application")
    parser.add_argument('--no-voice', action='store_true', help='Disable voice interaction')
    parser.add_argument('--output', type=str, choices=['text', 'json'], default='text', help='Output format')
    parser.add_argument('--language', type=str, default=None, help='Language code (e.g., en, fr, es)')
    parser.add_argument('--continuous-learning', action='store_true', help='Enable continuous learning')
    parser.add_argument('--custom-dataset', type=str, help='Path to custom dataset file')
    parser.add_argument('--user-id', type=str, default="default_user", help='User identifier for personalized experience')
    parser.add_argument('--voice-enabled', action='store_true', default=True, help='Enable voice interaction by default')
    parser.add_argument('--continuous-voice', action='store_true', help='Enable continuous voice interaction mode')
    parser.add_argument('--model', type=str, default=None, help='AI model to use (if available)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create and configure chatbot
    chatbot = ChatBot(
        continuous_learning=args.continuous_learning,
        user_id=args.user_id
    )
    
    # Configure voice settings
    if args.voice_enabled and not chatbot.user_preferences["voice_enabled"]:
        chatbot.user_preferences["voice_enabled"] = True
        chatbot._save_user_preferences()
    
    # Set language if provided
    if args.language:
        chatbot.set_language_preference(args.language)
    
    # Load custom dataset if provided
    if args.custom_dataset and os.path.exists(args.custom_dataset):
        try:
            print(f"{Fore.YELLOW}Loading custom dataset from {args.custom_dataset}...{Style.RESET_ALL}")
            with open(args.custom_dataset, 'r', encoding='utf-8') as f:
                custom_data = json.load(f)
                if isinstance(custom_data, list):
                    # Add each item to the QA dataset
                    for item in custom_data:
                        if isinstance(item, dict) and 'question' in item and 'answer' in item:
                            chatbot.qa_dataset.append(item)
                    print(f"{Fore.GREEN}✓ Loaded {len(custom_data)} examples from custom dataset{Style.RESET_ALL}")
                    # Save the expanded dataset
                    chatbot._save_qa_dataset()
                else:
                    print(f"{Fore.RED}Custom dataset must be a list of QA pairs{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading custom dataset: {e}{Style.RESET_ALL}")
    
    # Display welcome message
    display_welcome_message()
    print(f"User ID: {Fore.CYAN}{args.user_id}{Style.RESET_ALL}")
    print(f"Continuous learning: {Fore.CYAN}{'Enabled' if args.continuous_learning else 'Disabled'}{Style.RESET_ALL}")
    print(f"Voice interaction: {Fore.CYAN}{'Enabled' if chatbot.user_preferences['voice_enabled'] else 'Disabled'}{Style.RESET_ALL}")
    print(f"Enhanced voice: {Fore.CYAN}Enabled{Style.RESET_ALL}")
    
    # Set response length to concise by default
    chatbot.user_preferences["response_length"] = "concise"
    chatbot._save_user_preferences()
    print(f"Response length: {Fore.CYAN}{chatbot.user_preferences['response_length']}{Style.RESET_ALL}")
    
    if args.continuous_voice and chatbot.user_preferences['voice_enabled']:
        print(f"Continuous voice mode: {Fore.CYAN}Enabled{Style.RESET_ALL}")
    
    print(f"Language: {Fore.CYAN}{chatbot.multilingual.supported_languages.get(chatbot.user_preferences['language'], 'English')}{Style.RESET_ALL}")
    
    # Start continuous voice mode if requested
    if args.continuous_voice and chatbot.user_preferences['voice_enabled']:
        chatbot.start_continuous_voice_interaction()
        print(f"{Fore.CYAN}Continuous voice interaction mode enabled. Speak anytime to interact.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Press Ctrl+C to exit continuous voice mode.{Style.RESET_ALL}")
        
        # Keep the application running until Ctrl+C with status updates
        try:
            status_messages = [
                "🎤 Listening for voice input...",
                "Say something to interact...",
                "Voice recognition active...",
                "Enhanced voice mode active...",
                "Ready for your commands..."
            ]
            i = 0
            while True:
                # Show a rotating status message
                print(f"\r{Fore.GREEN}{status_messages[i % len(status_messages)]}{Style.RESET_ALL}", end="", flush=True)
                time.sleep(1.5)
                i += 1
        except KeyboardInterrupt:
            chatbot.stop_continuous_voice_interaction()
            print(f"\n{Fore.YELLOW}Continuous voice mode disabled. Reverting to normal interaction.{Style.RESET_ALL}")
    
    # Run CLI loop if not in continuous voice mode
    if not args.continuous_voice or not chatbot.user_preferences['voice_enabled']:
        run_cli_loop(chatbot, args.output)


if __name__ == "__main__":
    sys.exit(main())