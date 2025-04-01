"""
Course finder module for the ITI Assistant.

This module provides functionality to help users find suitable ITI courses
based on their interests, qualifications, and career goals.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from colorama import Fore, Style

class CourseFinderManager:
    """Manages course recommendation functionality."""
    
    def __init__(self, courses_data_path="data/iti_courses.json", locations_data_path="data/iti_locations.json"):
        """Initialize the course finder manager.
        
        Args:
            courses_data_path: Path to courses data file
            locations_data_path: Path to locations data file
        """
        self.courses_data_path = courses_data_path
        self.locations_data_path = locations_data_path
        self.courses = {}
        self.locations = {}
        self.user_preferences = {}
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(courses_data_path), exist_ok=True)
        
        # Load data
        self._load_courses()
        self._load_locations()
        
        print(f"{Fore.GREEN}✓ Course finder initialized{Style.RESET_ALL}")
    
    def _load_courses(self):
        """Load courses data from file."""
        try:
            if os.path.exists(self.courses_data_path):
                with open(self.courses_data_path, "r", encoding="utf-8") as f:
                    self.courses = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded {len(self.courses)} ITI courses{Style.RESET_ALL}")
            else:
                # Initialize with default courses if file doesn't exist
                self._initialize_default_courses()
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load courses data: {e}{Style.RESET_ALL}")
            self._initialize_default_courses()
    
    def _initialize_default_courses(self):
        """Initialize with default ITI courses data."""
        self.courses = {
            "engineering": {
                "Fitter": {
                    "duration": "1 year",
                    "eligibility": "10th pass",
                    "description": "Learn to fit and assemble structural frameworks and housings using hand tools and power tools.",
                    "career_prospects": ["Manufacturing", "Automobile", "Production", "Maintenance"],
                    "skills_gained": ["Blueprint reading", "Precision measurement", "Hand tools", "Assembly"],
                    "avg_salary": "₹15,000 - ₹25,000 per month",
                    "difficulty": "Medium"
                },
                "Turner": {
                    "duration": "1 year",
                    "eligibility": "10th pass",
                    "description": "Learn to operate lathe machines to create precision components by removing metal.",
                    "career_prospects": ["Manufacturing", "Tool rooms", "Production units"],
                    "skills_gained": ["Lathe operation", "Precision measurement", "Blueprint reading"],
                    "avg_salary": "₹15,000 - ₹25,000 per month",
                    "difficulty": "Medium"
                },
                "Electrician": {
                    "duration": "2 years",
                    "eligibility": "10th pass",
                    "description": "Learn to install, maintain, and repair electrical systems in residential, commercial, and industrial settings.",
                    "career_prospects": ["Power companies", "Industries", "Self-employment", "Maintenance"],
                    "skills_gained": ["Circuit design", "Wiring", "Troubleshooting", "Safety protocols"],
                    "avg_salary": "₹15,000 - ₹30,000 per month",
                    "difficulty": "Medium"
                },
                "Mechanic (Motor Vehicle)": {
                    "duration": "2 years",
                    "eligibility": "10th pass",
                    "description": "Learn to repair and maintain vehicles including engines, transmissions, and electrical systems.",
                    "career_prospects": ["Automobile industry", "Service stations", "Self-employment"],
                    "skills_gained": ["Engine repair", "Diagnostics", "Electrical systems", "Preventive maintenance"],
                    "avg_salary": "₹15,000 - ₹30,000 per month",
                    "difficulty": "Medium"
                },
                "COPA (Computer Operator & Programming Assistant)": {
                    "duration": "1 year",
                    "eligibility": "10th pass",
                    "description": "Learn computer applications, programming, and office automation.",
                    "career_prospects": ["IT companies", "Office administration", "Data entry", "Software support"],
                    "skills_gained": ["Office software", "Basic programming", "Data management", "Computer hardware"],
                    "avg_salary": "₹12,000 - ₹25,000 per month",
                    "difficulty": "Low"
                }
            },
            "non_engineering": {
                "Dress Making": {
                    "duration": "1 year",
                    "eligibility": "8th pass",
                    "description": "Learn to design and create various types of clothing and garments.",
                    "career_prospects": ["Garment industry", "Boutiques", "Self-employment"],
                    "skills_gained": ["Pattern making", "Cutting", "Sewing", "Design basics"],
                    "avg_salary": "₹10,000 - ₹20,000 per month",
                    "difficulty": "Low"
                },
                "Digital Photography": {
                    "duration": "1 year",
                    "eligibility": "10th pass",
                    "description": "Learn digital photography techniques, editing, and studio management.",
                    "career_prospects": ["Photography studios", "Media", "Self-employment", "E-commerce"],
                    "skills_gained": ["Camera operation", "Lighting", "Composition", "Photo editing"],
                    "avg_salary": "₹15,000 - ₹30,000 per month",
                    "difficulty": "Low"
                }
            },
            "advanced": {
                "CNC Programming": {
                    "duration": "1 year",
                    "eligibility": "ITI in Mechanical trade",
                    "description": "Learn to program and operate Computer Numerical Control machines.",
                    "career_prospects": ["Manufacturing", "Tool rooms", "Production units"],
                    "skills_gained": ["G-code programming", "CAD/CAM", "Precision machining"],
                    "avg_salary": "₹20,000 - ₹40,000 per month",
                    "difficulty": "High"
                },
                "Mechatronics": {
                    "duration": "1 year",
                    "eligibility": "ITI in Electrical/Electronics/Mechanical",
                    "description": "Learn integration of mechanical, electronic, and computer systems.",
                    "career_prospects": ["Automation industry", "Manufacturing", "Robotics"],
                    "skills_gained": ["PLC programming", "Hydraulics", "Pneumatics", "Robotics"],
                    "avg_salary": "₹25,000 - ₹45,000 per month",
                    "difficulty": "High"
                }
            }
        }
        
        # Save the default data
        self._save_courses()
        print(f"{Fore.GREEN}✓ Initialized default courses data{Style.RESET_ALL}")
    
    def _save_courses(self):
        """Save courses data to file."""
        try:
            os.makedirs(os.path.dirname(self.courses_data_path), exist_ok=True)
            with open(self.courses_data_path, "w", encoding="utf-8") as f:
                json.dump(self.courses, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save courses data: {e}{Style.RESET_ALL}")
    
    def _load_locations(self):
        """Load ITI locations data from file."""
        try:
            if os.path.exists(self.locations_data_path):
                with open(self.locations_data_path, "r", encoding="utf-8") as f:
                    self.locations = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded {len(self.locations)} ITI locations{Style.RESET_ALL}")
            else:
                # Initialize with default locations if file doesn't exist
                self._initialize_default_locations()
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load locations data: {e}{Style.RESET_ALL}")
            self._initialize_default_locations()
    
    def _initialize_default_locations(self):
        """Initialize with default ITI locations data."""
        self.locations = {
            "Maharashtra": [
                {
                    "name": "Government ITI, Mumbai",
                    "address": "Mumbai, Maharashtra",
                    "courses": ["Fitter", "Turner", "Electrician", "COPA"],
                    "contact": "022-12345678",
                    "website": "https://itimumbai.gov.in"
                },
                {
                    "name": "Government ITI, Pune",
                    "address": "Pune, Maharashtra",
                    "courses": ["Fitter", "Turner", "Electrician", "Mechanic (Motor Vehicle)"],
                    "contact": "020-12345678",
                    "website": "https://itipune.gov.in"
                }
            ],
            "Delhi": [
                {
                    "name": "Government ITI, Delhi",
                    "address": "Delhi",
                    "courses": ["Fitter", "Electrician", "COPA", "Digital Photography"],
                    "contact": "011-12345678",
                    "website": "https://itidelhi.gov.in"
                }
            ],
            "Tamil Nadu": [
                {
                    "name": "Government ITI, Chennai",
                    "address": "Chennai, Tamil Nadu",
                    "courses": ["Fitter", "Turner", "Electrician", "Dress Making"],
                    "contact": "044-12345678",
                    "website": "https://itichennai.gov.in"
                }
            ]
        }
        
        # Save the default data
        self._save_locations()
        print(f"{Fore.GREEN}✓ Initialized default locations data{Style.RESET_ALL}")
    
    def _save_locations(self):
        """Save locations data to file."""
        try:
            os.makedirs(os.path.dirname(self.locations_data_path), exist_ok=True)
            with open(self.locations_data_path, "w", encoding="utf-8") as f:
                json.dump(self.locations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save locations data: {e}{Style.RESET_ALL}")
    
    def get_all_courses(self):
        """Get all available courses.
        
        Returns:
            Dictionary of all courses
        """
        return self.courses
    
    def get_course_details(self, course_name):
        """Get details for a specific course.
        
        Args:
            course_name: Name of the course
            
        Returns:
            Course details or None if not found
        """
        for category, courses in self.courses.items():
            if course_name in courses:
                return {
                    "name": course_name,
                    "category": category,
                    **courses[course_name]
                }
        
        return None
    
    def find_courses_by_interest(self, interests):
        """Find courses based on user interests.
        
        Args:
            interests: List of interest keywords
            
        Returns:
            List of matching courses with scores
        """
        matches = []
        
        for category, courses in self.courses.items():
            for course_name, details in courses.items():
                score = 0
                
                # Check career prospects
                for interest in interests:
                    interest_lower = interest.lower()
                    
                    # Check in career prospects
                    for prospect in details.get("career_prospects", []):
                        if interest_lower in prospect.lower():
                            score += 2
                    
                    # Check in skills gained
                    for skill in details.get("skills_gained", []):
                        if interest_lower in skill.lower():
                            score += 2
                    
                    # Check in description
                    if interest_lower in details.get("description", "").lower():
                        score += 1
                
                if score > 0:
                    matches.append({
                        "name": course_name,
                        "category": category,
                        "score": score,
                        "details": details
                    })
        
        # Sort by score (highest first)
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        return matches
    
    def find_courses_by_eligibility(self, education_level):
        """Find courses based on user's education level.
        
        Args:
            education_level: User's education level (e.g., "10th pass")
            
        Returns:
            List of eligible courses
        """
        eligible_courses = []
        
        # Define education level hierarchy
        education_hierarchy = {
            "8th pass": 1,
            "10th pass": 2,
            "12th pass": 3,
            "ITI": 4,
            "Diploma": 5,
            "Graduate": 6
        }
        
        user_level = education_hierarchy.get(education_level, 0)
        
        for category, courses in self.courses.items():
            for course_name, details in courses.items():
                required_level = details.get("eligibility", "")
                
                # Check if user meets the eligibility
                for level, value in education_hierarchy.items():
                    if level in required_level and value <= user_level:
                        eligible_courses.append({
                            "name": course_name,
                            "category": category,
                            "details": details
                        })
                        break
        
        return eligible_courses
    
    def find_nearby_institutes(self, state, courses=None):
        """Find ITI institutes in a specific state offering specific courses.
        
        Args:
            state: State name
            courses: List of course names (optional)
            
        Returns:
            List of matching institutes
        """
        if state not in self.locations:
            return []
        
        institutes = self.locations[state]
        
        if not courses:
            return institutes
        
        # Filter by courses
        matching_institutes = []
        for institute in institutes:
            for course in courses:
                if course in institute.get("courses", []):
                    matching_institutes.append(institute)
                    break
        
        return matching_institutes
    
    def get_course_comparison(self, course_names):
        """Compare multiple courses side by side.
        
        Args:
            course_names: List of course names to compare
            
        Returns:
            Dictionary with comparison data
        """
        comparison = {
            "courses": [],
            "comparison_points": [
                "duration", "eligibility", "career_prospects", 
                "skills_gained", "avg_salary", "difficulty"
            ]
        }
        
        for course_name in course_names:
            course_details = self.get_course_details(course_name)
            if course_details:
                comparison["courses"].append(course_details)
        
        return comparison
    
    def save_user_preferences(self, user_id, preferences):
        """Save user course preferences.
        
        Args:
            user_id: User identifier
            preferences: Dictionary of user preferences
            
        Returns:
            True if successful, False otherwise
        """
        try:
            preferences_path = f"data/user_course_preferences_{user_id}.json"
            os.makedirs(os.path.dirname(preferences_path), exist_ok=True)
            
            with open(preferences_path, "w", encoding="utf-8") as f:
                json.dump(preferences, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save user preferences: {e}{Style.RESET_ALL}")
            return False
    
    def load_user_preferences(self, user_id):
        """Load user course preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of user preferences
        """
        try:
            preferences_path = f"data/user_course_preferences_{user_id}.json"
            
            if os.path.exists(preferences_path):
                with open(preferences_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            
            return {}
            
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load user preferences: {e}{Style.RESET_ALL}")
            return {} 