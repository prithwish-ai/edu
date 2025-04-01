"""
Context manager for the AI Chatbot.

This module provides context management functionality, enabling the chatbot
to maintain context over conversations and perform tasks like entity extraction,
intent detection, and relevant context retrieval.
"""

import os
import json
import time
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from colorama import Fore, Style

class ContextManager:
    """Manages contextual information for enhanced conversation capabilities."""
    
    def __init__(self, max_context_entries=1000, save_interval=10):
        """Initialize the context manager.
        
        Args:
            max_context_entries: Maximum number of context entries to store
            save_interval: Number of updates before auto-saving
        """
        self.max_entries = max_context_entries
        self.save_interval = save_interval
        self.update_counter = 0
        self.last_save_time = time.time()
        
        # Context storage
        self.context_store = []
        
        # Entity storage
        self.entity_store = {}
        
        # NLP components (lazy loaded)
        self.nlp = None
        self.vectorizer = None
        self.embedding_model = None
        
        # Load existing data
        self._load_context()
        self._load_entity_store()
        
        print(f"{Fore.GREEN}✓ Context manager initialized{Style.RESET_ALL}")
    
    def _load_spacy(self):
        """Load spaCy NLP model on first use."""
        if self.nlp is None:
            try:
                import spacy
                print(f"{Fore.YELLOW}Loading NLP model...{Style.RESET_ALL}")
                try:
                    # Try to load medium model first
                    self.nlp = spacy.load("en_core_web_md")
                except OSError:
                    # Fall back to small model
                    self.nlp = spacy.load("en_core_web_sm")
                
                print(f"{Fore.GREEN}✓ NLP model loaded{Style.RESET_ALL}")
            except ImportError:
                print(f"{Fore.YELLOW}spaCy not available, using simplified entity extraction{Style.RESET_ALL}")
                print("Install with: pip install spacy")
                print("Then: python -m spacy download en_core_web_md (or en_core_web_sm)")
                
    def _load_vectorizer(self):
        """Load or create text vectorizer."""
        if self.vectorizer is None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                self.vectorizer = TfidfVectorizer(
                    lowercase=True, 
                    strip_accents='unicode',
                    ngram_range=(1, 2),
                    max_features=10000
                )
                
                print(f"{Fore.GREEN}✓ Text vectorizer initialized{Style.RESET_ALL}")
            except ImportError:
                print(f"{Fore.YELLOW}scikit-learn not available, vectorization disabled{Style.RESET_ALL}")
                print("Install with: pip install scikit-learn")
                
    def _load_embedding_model(self):
        """Load sentence embedding model."""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                print(f"{Fore.GREEN}✓ Embedding model loaded{Style.RESET_ALL}")
            except ImportError:
                print(f"{Fore.YELLOW}SentenceTransformer not available, semantic search disabled{Style.RESET_ALL}")
                print("Install with: pip install sentence-transformers")
    
    def _load_context(self):
        """Load context data from file if it exists."""
        try:
            if os.path.exists("context_store.json"):
                with open("context_store.json", "r", encoding="utf-8") as f:
                    self.context_store = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded context store with {len(self.context_store)} entries{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load context store: {e}{Style.RESET_ALL}")
            self.context_store = []
    
    def _save_context(self, force=False):
        """Save context data to file.
        
        Args:
            force: Whether to force saving regardless of interval
        """
        current_time = time.time()
        self.update_counter += 1
        
        # Save if forced, counter interval reached, or time interval reached
        if (force or 
            self.update_counter >= self.save_interval or
            current_time - self.last_save_time > 300):  # 5 minutes
            try:
                # Limit to max entries before saving
                if len(self.context_store) > self.max_entries:
                    self.context_store = self.context_store[-self.max_entries:]
                
                with open("context_store.json", "w", encoding="utf-8") as f:
                    json.dump(self.context_store, f, ensure_ascii=False, indent=2)
                    
                self.update_counter = 0
                self.last_save_time = current_time
                return True
            except Exception as e:
                print(f"{Fore.YELLOW}Could not save context store: {e}{Style.RESET_ALL}")
                return False
        
        return False
    
    def _load_entity_store(self):
        """Load entity data from file if it exists."""
        try:
            if os.path.exists("entity_store.json"):
                with open("entity_store.json", "r", encoding="utf-8") as f:
                    self.entity_store = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded entity store with {len(self.entity_store)} entity types{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load entity store: {e}{Style.RESET_ALL}")
            self.entity_store = {}
    
    def _save_entity_store(self):
        """Save entity data to file."""
        try:
            with open("entity_store.json", "w", encoding="utf-8") as f:
                json.dump(self.entity_store, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save entity store: {e}{Style.RESET_ALL}")
    
    def add_context(self, text, metadata=None):
        """Add text to context store.
        
        Args:
            text: Text to add
            metadata: Optional metadata about the text
        """
        if not text or not text.strip():
            return
            
        # Add timestamp if not provided in metadata
        if not metadata:
            metadata = {}
        
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
            
        # Extract entities from text
        entities = self.extract_entities(text)
        
        # Add to entity store
        for entity_type, entity_values in entities.items():
            if entity_type not in self.entity_store:
                self.entity_store[entity_type] = {}
                
            for entity_value in entity_values:
                if entity_value not in self.entity_store[entity_type]:
                    self.entity_store[entity_type][entity_value] = []
                    
                # Add occurrence with snippet
                self.entity_store[entity_type][entity_value].append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "timestamp": metadata.get("timestamp")
                })
                
        # Create the context entry
        context_entry = {
            "text": text,
            "metadata": metadata,
            "entities": entities
        }
        
        # Try to create embedding if the model is available
        try:
            # Lazy load embedding model if needed
            if self.embedding_model is None:
                self._load_embedding_model()
                
            if self.embedding_model:
                embedding = self.embedding_model.encode(text)
                context_entry["embedding"] = embedding.tolist()
        except Exception as e:
            print(f"{Fore.YELLOW}Could not create embedding: {e}{Style.RESET_ALL}")
        
        # Add to context store
        self.context_store.append(context_entry)
        
        # Save periodically
        self._save_context()
        if len(entities) > 0:
            self._save_entity_store()
    
    def get_relevant_context(self, query, max_items=5, threshold=0.2):
        """Get context entries relevant to a query.
        
        Args:
            query: The query to match against
            max_items: Maximum number of context items to return
            threshold: Minimum similarity score to include
            
        Returns:
            List of relevant context entries
        """
        if not query or not self.context_store:
            return []
            
        # Try embedding-based retrieval first
        embedding_results = []
        
        try:
            if self.embedding_model is None:
                self._load_embedding_model()
                
            if self.embedding_model:
                # Create query embedding
                query_embedding = self.embedding_model.encode(query)
                
                # Find entries with embeddings
                entries_with_embeddings = [
                    (i, entry) for i, entry in enumerate(self.context_store)
                    if "embedding" in entry
                ]
                
                if entries_with_embeddings:
                    # Calculate cosine similarities
                    from sklearn.metrics.pairwise import cosine_similarity
                    
                    similarities = []
                    for i, entry in entries_with_embeddings:
                        entry_embedding = np.array(entry["embedding"])
                        similarity = cosine_similarity([query_embedding], [entry_embedding])[0][0]
                        similarities.append((i, similarity))
                    
                    # Sort by similarity
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    # Get top results above threshold
                    embedding_results = [
                        (i, score) for i, score in similarities[:max_items]
                        if score >= threshold
                    ]
        except Exception as e:
            print(f"{Fore.YELLOW}Embedding search failed: {e}{Style.RESET_ALL}")
            
        # Fallback to vectorizer if no results
        if not embedding_results:
            try:
                if self.vectorizer is None:
                    self._load_vectorizer()
                    
                if self.vectorizer:
                    # Create document matrix
                    texts = [entry["text"] for entry in self.context_store]
                    
                    # Fit vectorizer and transform
                    X = self.vectorizer.fit_transform(texts)
                    query_vec = self.vectorizer.transform([query])
                    
                    # Calculate cosine similarities
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarities = cosine_similarity(query_vec, X)[0]
                    
                    # Get indices of top results
                    top_indices = np.argsort(similarities)[::-1][:max_items]
                    
                    # Filter by threshold
                    embedding_results = [
                        (i, similarities[i]) for i in top_indices
                        if similarities[i] >= threshold
                    ]
            except Exception as e:
                print(f"{Fore.YELLOW}Vectorizer search failed: {e}{Style.RESET_ALL}")
                
        # Return relevant context entries
        result = []
        for i, _ in embedding_results:
            result.append(self.context_store[i]["text"])
            
        return result
    
    def extract_entities(self, text):
        """Extract entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entities by type
        """
        # Try spaCy extraction first
        try:
            if self.nlp is None:
                self._load_spacy()
                
            if self.nlp:
                # For longer text, split and process separately to avoid memory issues
                if len(text) > 1000:
                    # Process first and last portions where entities are often found
                    first_chunk = text[:1000]
                    last_chunk = text[-1000:]
                    
                    doc_first = self.nlp(first_chunk)
                    doc_last = self.nlp(last_chunk)
                    
                    entities = {}
                    for ent in list(doc_first.ents) + list(doc_last.ents):
                        if ent.label_ not in entities:
                            entities[ent.label_] = []
                        if ent.text not in entities[ent.label_]:
                            entities[ent.label_].append(ent.text)
                else:
                    # Process the whole text for shorter content
                    doc = self.nlp(text)
                    entities = {}
                    for ent in doc.ents:
                        if ent.label_ not in entities:
                            entities[ent.label_] = []
                        if ent.text not in entities[ent.label_]:
                            entities[ent.label_].append(ent.text)
                            
                return entities
        except Exception as e:
            print(f"{Fore.YELLOW}Entity extraction with spaCy failed: {e}{Style.RESET_ALL}")
            
        # Fall back to simple extraction
        return self._extract_entities_simple(text)
        
    def _extract_entities_simple(self, text):
        """Simple entity extraction for when spaCy is not available.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entities by type
        """
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],  # Geo-political entities (countries, cities)
            "DATE": [],
            "PRODUCT": []
        }
        
        # Simple check for dates (MM/DD/YYYY or similar)
        import re
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
            r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                entities["DATE"].append(match.group())
                
        # Look for capitalized phrases as potential named entities
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
            entity = match.group()
            
            # Skip single words less than 4 chars as they're often not entities
            if ' ' not in entity and len(entity) < 4:
                continue
                
            # Guess entity type based on context
            if any(term in text.lower()[:match.start()] for term in ["company", "organization", "corporation", "inc", "ltd"]):
                entities["ORG"].append(entity)
            elif any(term in text.lower()[:match.start()] for term in ["city", "country", "town", "nation", "state"]):
                entities["GPE"].append(entity)
            elif any(term in text.lower()[:match.start()] for term in ["mr", "mrs", "ms", "dr", "prof", "miss"]):
                entities["PERSON"].append(entity)
            else:
                # Look for product indicators
                if any(term in text.lower()[:match.start()] for term in ["product", "model", "brand", "device"]):
                    entities["PRODUCT"].append(entity)
                elif ' ' in entity:  # Multi-word capitalized phrases are likely entities
                    entities["ORG"].append(entity)
                    
        # Remove duplicates
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
            
        return entities
    
    def get_entity_info(self, entity_type=None, entity_name=None):
        """Get information about entities.
        
        Args:
            entity_type: Type of entity to query (optional)
            entity_name: Specific entity name to query (optional)
            
        Returns:
            Entity information based on query
        """
        if entity_type and entity_name:
            # Get specific entity
            return self.entity_store.get(entity_type, {}).get(entity_name, [])
        elif entity_type:
            # Get all entities of a type
            return self.entity_store.get(entity_type, {})
        else:
            # Get summary of entity types
            return {
                entity_type: len(entities)
                for entity_type, entities in self.entity_store.items()
            }
    
    def get_related_entities(self, entity_name, threshold=0.7):
        """Find entities related to the given entity.
        
        Args:
            entity_name: Name of the entity to find relations for
            threshold: Minimum similarity score to consider related
            
        Returns:
            Dictionary of related entities by type
        """
        related = {}
        
        # Look for direct mentions in the same contexts
        entity_occurrences = []
        
        for entity_type, entities in self.entity_store.items():
            if entity_name in entities:
                # Found the entity, remember its occurrences
                entity_occurrences = entities[entity_name]
                break
                
        if not entity_occurrences:
            return related
            
        # Get text snippets containing this entity
        entity_contexts = [occurrence["text"] for occurrence in entity_occurrences]
        
        # Look for other entities in the same contexts
        for entity_type, entities in self.entity_store.items():
            related[entity_type] = []
            
            for other_entity, occurrences in entities.items():
                # Skip the entity itself
                if other_entity == entity_name:
                    continue
                    
                # Get text snippets containing the other entity
                other_contexts = [occurrence["text"] for occurrence in occurrences]
                
                # Calculate how many times they co-occur
                common_contexts = set(entity_contexts).intersection(set(other_contexts))
                
                if common_contexts:
                    # Calculate co-occurrence score
                    score = len(common_contexts) / min(len(entity_contexts), len(other_contexts))
                    
                    if score >= threshold:
                        related[entity_type].append({
                            "entity": other_entity,
                            "score": score,
                            "co_occurrences": len(common_contexts)
                        })
            
            # Sort by score
            related[entity_type].sort(key=lambda x: x["score"], reverse=True)
            
            # Remove empty categories
            if not related[entity_type]:
                del related[entity_type]
                
        return related
    
    def detect_intent(self, text):
        """Detect the likely intent of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with intent information
        """
        text_lower = text.lower()
        
        # Define intent keywords
        intent_patterns = {
            "greeting": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"],
            "farewell": ["bye", "goodbye", "see you", "talk to you later", "until next time"],
            "question": ["what", "why", "how", "when", "where", "who", "is it", "are there", "can you", "could you"],
            "command": ["show", "find", "list", "search", "get", "tell me", "give me", "display"],
            "comparison": ["versus", "vs", "compare", "difference", "similarities", "better", "worse", "best"],
            "definition": ["define", "explain", "what is", "what are", "meaning of", "definition"],
            "opinion": ["think", "believe", "opinion", "feel about", "thoughts on", "stance on"],
            "clarification": ["mean", "understand", "clarify", "explain", "confused"],
            "agreement": ["agree", "yes", "correct", "right", "exactly", "sure"],
            "disagreement": ["disagree", "no", "incorrect", "wrong", "not right", "not exactly"],
            "gratitude": ["thanks", "thank you", "appreciate", "grateful"],
            "request": ["please", "would you", "can you", "could you", "would like"]
        }
        
        # Analyze the text for each intent
        intent_scores = {}
        
        for intent, keywords in intent_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # More specific phrases get higher scores
                    score += len(keyword.split())
                    
            if score > 0:
                intent_scores[intent] = score
                
        # Get the top intents
        result = {
            "top_intents": [],
            "has_question": "?" in text,
            "sentiment": "neutral"  # Basic, would need sentiment analysis for better results
        }
        
        if intent_scores:
            # Sort by score
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            result["top_intents"] = [intent for intent, _ in sorted_intents[:3]]
            
            # Determine primary intent
            result["primary_intent"] = sorted_intents[0][0]
        else:
            result["primary_intent"] = "unknown"
            
        # Basic sentiment analysis
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "happy", "like", "love", "best"]
        negative_words = ["bad", "terrible", "awful", "horrible", "sad", "hate", "dislike", "worst"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            result["sentiment"] = "positive"
        elif negative_count > positive_count:
            result["sentiment"] = "negative"
            
        return result
    
    def summarize_context(self, max_items=10):
        """Provide a summary of the current context.
        
        Args:
            max_items: Maximum number of recent context entries to summarize
            
        Returns:
            Context summary dictionary
        """
        if not self.context_store:
            return {"entries": 0}
            
        # Basic summary stats
        summary = {
            "entries": len(self.context_store),
            "earliest": None,
            "latest": None,
            "entity_types": len(self.entity_store),
            "top_entities": {},
            "recent_entries": []
        }
        
        # Get timestamps
        timestamps = []
        for entry in self.context_store:
            if "metadata" in entry and "timestamp" in entry["metadata"]:
                timestamps.append(entry["metadata"]["timestamp"])
                
        if timestamps:
            summary["earliest"] = min(timestamps)
            summary["latest"] = max(timestamps)
            
        # Get top entity counts
        for entity_type, entities in self.entity_store.items():
            # Sort entities by occurrence count
            sorted_entities = sorted(
                [(entity, len(occurrences)) for entity, occurrences in entities.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            if sorted_entities:
                summary["top_entities"][entity_type] = sorted_entities[:5]
                
        # Get recent entries
        recent = self.context_store[-max_items:]
        summary["recent_entries"] = [
            {
                "text": entry["text"][:100] + "..." if len(entry["text"]) > 100 else entry["text"],
                "timestamp": entry.get("metadata", {}).get("timestamp", "unknown")
            }
            for entry in recent
        ]
        
        return summary
    
    def cleanup(self):
        """Clean up old or redundant context entries."""
        if not self.context_store:
            return
            
        # Remove very old entries beyond the max_entries limit
        if len(self.context_store) > self.max_entries:
            # Sort by timestamp if available
            sorted_entries = []
            for i, entry in enumerate(self.context_store):
                timestamp = entry.get("metadata", {}).get("timestamp", "")
                sorted_entries.append((i, timestamp))
                
            if sorted_entries and sorted_entries[0][1]:
                # Sort by timestamp
                sorted_entries.sort(key=lambda x: x[1])
                
                # Keep only the most recent max_entries
                keep_indices = {i for i, _ in sorted_entries[-self.max_entries:]}
                self.context_store = [entry for i, entry in enumerate(self.context_store) if i in keep_indices]
            else:
                # Simply keep the most recent entries
                self.context_store = self.context_store[-self.max_entries:]
                
        # Save the cleaned context
        self._save_context(force=True) 