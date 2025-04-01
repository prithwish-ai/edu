"""
Exam Preparation Module for ITI Assistant.

This module provides features to help students prepare for ITI exams, including:
- Access to study materials
- Practice questions and quizzes
- Exam tips and strategies
- Mock tests and performance analysis
- Revision schedules
"""

import os
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from colorama import Fore, Style

class ExamPreparationManager:
    """Manages exam preparation features for ITI students."""
    
    def __init__(self, study_materials_path="data/study_materials.json", quiz_database_path="data/quiz_database.json", 
                 exam_schedule_path="data/exam_schedules.json", user_progress_path="data/user_progress.json"):
        """Initialize the exam preparation manager.
        
        Args:
            study_materials_path: Path to study materials JSON file
            quiz_database_path: Path to quiz questions JSON file
            exam_schedule_path: Path to exam schedules JSON file
            user_progress_path: Path to user progress tracking file
        """
        self.study_materials_path = study_materials_path
        self.quiz_database_path = quiz_database_path
        self.exam_schedule_path = exam_schedule_path
        self.user_progress_path = user_progress_path
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(study_materials_path), exist_ok=True)
        
        # Load data
        self.study_materials = self._load_study_materials()
        self.quiz_database = self._load_quiz_database()
        self.exam_schedules = self._load_exam_schedules()
        self.user_progress = self._load_user_progress()
        
        print(f"{Fore.GREEN}✓ Exam preparation manager initialized{Style.RESET_ALL}")
        
    def _load_study_materials(self) -> Dict:
        """Load study materials from file or initialize default data."""
        try:
            if os.path.exists(self.study_materials_path):
                with open(self.study_materials_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Initialize with default study materials
                default_materials = self._initialize_default_study_materials()
                self._save_study_materials(default_materials)
                return default_materials
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load study materials: {e}{Style.RESET_ALL}")
            default_materials = self._initialize_default_study_materials()
            return default_materials
            
    def _initialize_default_study_materials(self) -> Dict:
        """Initialize default study materials for common ITI trades."""
        return {
            "Electrician": {
                "modules": [
                    {
                        "title": "Basic Electricity",
                        "topics": [
                            "Electric Circuits", "Ohm's Law", "Kirchhoff's Laws",
                            "Resistors, Capacitors, and Inductors", "AC and DC Fundamentals"
                        ],
                        "resources": [
                            {"type": "text", "title": "Basic Electrical Theory", "content": "Comprehensive guide to electrical theory basics"},
                            {"type": "video", "title": "Understanding Ohm's Law", "url": "https://example.com/ohms-law-video"},
                            {"type": "practice", "title": "Circuit Calculations", "questions": 15}
                        ]
                    },
                    {
                        "title": "Electrical Measurements",
                        "topics": [
                            "Measuring Instruments", "Multimeters", "Oscilloscopes",
                            "Power Measurements", "Calibration Techniques"
                        ],
                        "resources": [
                            {"type": "text", "title": "Electrical Measurement Guide", "content": "Detailed guide to using electrical measuring instruments"},
                            {"type": "video", "title": "Multimeter Usage Tutorial", "url": "https://example.com/multimeter-tutorial"},
                            {"type": "practice", "title": "Measurement Practice", "questions": 10}
                        ]
                    }
                ],
                "important_topics": [
                    "Safety in Electrical Work", "AC Motor Control",
                    "Transformer Principles", "Power Distribution Systems"
                ],
                "exam_pattern": {
                    "theory_marks": 100,
                    "practical_marks": 100,
                    "passing_percentage": 40,
                    "duration_hours": 3,
                    "question_types": ["Multiple Choice", "Short Answer", "Long Answer", "Diagram-based"]
                }
            },
            "Fitter": {
                "modules": [
                    {
                        "title": "Fitting and Assembly",
                        "topics": [
                            "Hand Tools", "Marking Tools", "Measuring Instruments",
                            "Filing and Fitting", "Assembly Techniques"
                        ],
                        "resources": [
                            {"type": "text", "title": "Fitting Workshop Manual", "content": "Comprehensive guide to fitting techniques"},
                            {"type": "video", "title": "Precision Fitting Demonstration", "url": "https://example.com/fitting-demo"},
                            {"type": "practice", "title": "Measurement Exercises", "questions": 12}
                        ]
                    },
                    {
                        "title": "Machine Operations",
                        "topics": [
                            "Drilling", "Grinding", "Lathe Operations",
                            "Milling Basics", "CNC Introduction"
                        ],
                        "resources": [
                            {"type": "text", "title": "Machine Shop Operations", "content": "Guide to basic machine shop processes"},
                            {"type": "video", "title": "Lathe Operating Tutorial", "url": "https://example.com/lathe-tutorial"},
                            {"type": "practice", "title": "Machine Operation Quiz", "questions": 15}
                        ]
                    }
                ],
                "important_topics": [
                    "Workshop Safety", "Blueprint Reading",
                    "Precision Measurement", "Material Properties"
                ],
                "exam_pattern": {
                    "theory_marks": 100,
                    "practical_marks": 100,
                    "passing_percentage": 40,
                    "duration_hours": 3,
                    "question_types": ["Multiple Choice", "Short Answer", "Practical Demonstration", "Drawing"]
                }
            },
            "COPA": {
                "modules": [
                    {
                        "title": "Computer Fundamentals",
                        "topics": [
                            "Computer Architecture", "Operating Systems", "File Management",
                            "Software and Hardware", "Networking Basics"
                        ],
                        "resources": [
                            {"type": "text", "title": "Computer Fundamentals Guide", "content": "Comprehensive introduction to computers"},
                            {"type": "video", "title": "Computer Hardware Tour", "url": "https://example.com/hardware-tour"},
                            {"type": "practice", "title": "Computer Basics Quiz", "questions": 20}
                        ]
                    },
                    {
                        "title": "Office Applications",
                        "topics": [
                            "Word Processing", "Spreadsheets", "Presentations",
                            "Database Management", "Email and Communication"
                        ],
                        "resources": [
                            {"type": "text", "title": "Office Applications Manual", "content": "Guide to using office productivity software"},
                            {"type": "video", "title": "Advanced Excel Tutorial", "url": "https://example.com/excel-tutorial"},
                            {"type": "practice", "title": "Office Apps Exercises", "questions": 15}
                        ]
                    }
                ],
                "important_topics": [
                    "Programming Basics", "Internet Security",
                    "Web Development Fundamentals", "Troubleshooting"
                ],
                "exam_pattern": {
                    "theory_marks": 100,
                    "practical_marks": 100,
                    "passing_percentage": 40,
                    "duration_hours": 3,
                    "question_types": ["Multiple Choice", "Practical Lab Work", "Problem Solving", "Project"]
                }
            }
        }
    
    def _save_study_materials(self, materials: Dict) -> None:
        """Save study materials to file."""
        try:
            with open(self.study_materials_path, 'w', encoding='utf-8') as f:
                json.dump(materials, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save study materials: {e}{Style.RESET_ALL}")
    
    def _load_quiz_database(self) -> Dict:
        """Load quiz questions from file or initialize default data."""
        try:
            if os.path.exists(self.quiz_database_path):
                with open(self.quiz_database_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Initialize with default quiz questions
                default_quizzes = self._initialize_default_quizzes()
                self._save_quiz_database(default_quizzes)
                return default_quizzes
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load quiz database: {e}{Style.RESET_ALL}")
            default_quizzes = self._initialize_default_quizzes()
            return default_quizzes
    
    def _initialize_default_quizzes(self) -> Dict:
        """Initialize default quiz questions for common ITI trades."""
        return {
            "Electrician": [
                {
                    "topic": "Basic Electricity",
                    "difficulty": "easy",
                    "questions": [
                        {
                            "question": "What is Ohm's Law?",
                            "options": [
                                "V = IR",
                                "P = VI",
                                "E = mc²",
                                "F = ma"
                            ],
                            "correct_answer": 0,
                            "explanation": "Ohm's Law states that the current through a conductor is directly proportional to the voltage and inversely proportional to the resistance (V = IR)."
                        },
                        {
                            "question": "Which unit is used to measure electrical resistance?",
                            "options": [
                                "Volt",
                                "Ampere",
                                "Ohm",
                                "Watt"
                            ],
                            "correct_answer": 2,
                            "explanation": "Electrical resistance is measured in ohms, named after Georg Simon Ohm."
                        }
                    ]
                },
                {
                    "topic": "Safety",
                    "difficulty": "medium",
                    "questions": [
                        {
                            "question": "What should be your first action when someone experiences an electric shock?",
                            "options": [
                                "Pull them away with your bare hands",
                                "Throw water on them",
                                "Turn off the power source",
                                "Call for help first, then provide aid"
                            ],
                            "correct_answer": 2,
                            "explanation": "The first priority is to turn off the power source to prevent further injury to the victim and to make it safe for you to provide assistance."
                        }
                    ]
                }
            ],
            "Fitter": [
                {
                    "topic": "Measurement Techniques",
                    "difficulty": "easy",
                    "questions": [
                        {
                            "question": "Which tool is used for precise angular measurement?",
                            "options": [
                                "Vernier Caliper",
                                "Micrometer",
                                "Bevel Protractor",
                                "Try Square"
                            ],
                            "correct_answer": 2,
                            "explanation": "A bevel protractor is used for precise measurement of angles in fitting and machining work."
                        }
                    ]
                },
                {
                    "topic": "Hand Tools",
                    "difficulty": "medium",
                    "questions": [
                        {
                            "question": "Which file would be best for finishing work on soft metals?",
                            "options": [
                                "Bastard file",
                                "Second cut file",
                                "Smooth file",
                                "Rough file"
                            ],
                            "correct_answer": 2,
                            "explanation": "A smooth file has fine teeth and is used for finishing work, especially on softer metals where a fine finish is required."
                        }
                    ]
                }
            ],
            "COPA": [
                {
                    "topic": "Computer Basics",
                    "difficulty": "easy",
                    "questions": [
                        {
                            "question": "What does CPU stand for?",
                            "options": [
                                "Central Processing Unit",
                                "Computer Processing Unit",
                                "Central Program Utility",
                                "Central Processor Unit"
                            ],
                            "correct_answer": 0,
                            "explanation": "CPU stands for Central Processing Unit. It's the primary component of a computer that performs most of the processing."
                        }
                    ]
                },
                {
                    "topic": "Office Applications",
                    "difficulty": "medium",
                    "questions": [
                        {
                            "question": "Which Excel function would you use to add up a range of cells?",
                            "options": [
                                "AVERAGE",
                                "COUNT",
                                "SUM",
                                "TOTAL"
                            ],
                            "correct_answer": 2,
                            "explanation": "The SUM function in Excel is used to add all numbers in a range of cells."
                        }
                    ]
                }
            ]
        }
    
    def _save_quiz_database(self, quizzes: Dict) -> None:
        """Save quiz database to file."""
        try:
            with open(self.quiz_database_path, 'w', encoding='utf-8') as f:
                json.dump(quizzes, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save quiz database: {e}{Style.RESET_ALL}")
    
    def _load_exam_schedules(self) -> Dict:
        """Load exam schedules from file or initialize default data."""
        try:
            if os.path.exists(self.exam_schedule_path):
                with open(self.exam_schedule_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Initialize with default exam schedules
                default_schedules = self._initialize_default_exam_schedules()
                self._save_exam_schedules(default_schedules)
                return default_schedules
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load exam schedules: {e}{Style.RESET_ALL}")
            default_schedules = self._initialize_default_exam_schedules()
            return default_schedules
    
    def _initialize_default_exam_schedules(self) -> Dict:
        """Initialize default exam schedules for common ITI trades."""
        # Create a schedule for the next 6 months
        today = datetime.now()
        schedules = {
            "semester_exams": [
                {
                    "trade": "Electrician",
                    "semester": 1,
                    "date": (today + timedelta(days=45)).strftime("%Y-%m-%d"),
                    "subjects": [
                        {"name": "Trade Theory", "date": (today + timedelta(days=45)).strftime("%Y-%m-%d"), "time": "10:00 AM"},
                        {"name": "Workshop Calculation", "date": (today + timedelta(days=46)).strftime("%Y-%m-%d"), "time": "10:00 AM"},
                        {"name": "Engineering Drawing", "date": (today + timedelta(days=47)).strftime("%Y-%m-%d"), "time": "10:00 AM"}
                    ]
                },
                {
                    "trade": "Fitter",
                    "semester": 1,
                    "date": (today + timedelta(days=50)).strftime("%Y-%m-%d"),
                    "subjects": [
                        {"name": "Trade Theory", "date": (today + timedelta(days=50)).strftime("%Y-%m-%d"), "time": "10:00 AM"},
                        {"name": "Workshop Calculation", "date": (today + timedelta(days=51)).strftime("%Y-%m-%d"), "time": "10:00 AM"},
                        {"name": "Engineering Drawing", "date": (today + timedelta(days=52)).strftime("%Y-%m-%d"), "time": "10:00 AM"}
                    ]
                }
            ],
            "all_india_trade_tests": [
                {
                    "trade": "All Trades",
                    "year": today.year,
                    "season": "Summer",
                    "start_date": (today + timedelta(days=90)).strftime("%Y-%m-%d"),
                    "end_date": (today + timedelta(days=120)).strftime("%Y-%m-%d"),
                    "registration_deadline": (today + timedelta(days=30)).strftime("%Y-%m-%d")
                },
                {
                    "trade": "All Trades",
                    "year": today.year,
                    "season": "Winter",
                    "start_date": (today + timedelta(days=180)).strftime("%Y-%m-%d"),
                    "end_date": (today + timedelta(days=210)).strftime("%Y-%m-%d"),
                    "registration_deadline": (today + timedelta(days=120)).strftime("%Y-%m-%d")
                }
            ],
            "practical_exams": [
                {
                    "trade": "Electrician",
                    "semester": 1,
                    "date": (today + timedelta(days=60)).strftime("%Y-%m-%d"),
                    "time": "9:00 AM",
                    "venue": "Main Workshop",
                    "requirements": ["Safety equipment", "Basic tools", "ID card"]
                },
                {
                    "trade": "Fitter",
                    "semester": 1,
                    "date": (today + timedelta(days=65)).strftime("%Y-%m-%d"),
                    "time": "9:00 AM",
                    "venue": "Fitting Workshop",
                    "requirements": ["Safety equipment", "Basic tools", "ID card"]
                }
            ]
        }
        return schedules
    
    def _save_exam_schedules(self, schedules: Dict) -> None:
        """Save exam schedules to file."""
        try:
            with open(self.exam_schedule_path, 'w', encoding='utf-8') as f:
                json.dump(schedules, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save exam schedules: {e}{Style.RESET_ALL}")
    
    def _load_user_progress(self) -> Dict:
        """Load user progress data from file or initialize empty data."""
        try:
            if os.path.exists(self.user_progress_path):
                with open(self.user_progress_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Initialize with empty user progress
                empty_progress = {}
                self._save_user_progress(empty_progress)
                return empty_progress
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load user progress: {e}{Style.RESET_ALL}")
            return {}
    
    def _save_user_progress(self, progress: Dict) -> None:
        """Save user progress to file."""
        try:
            with open(self.user_progress_path, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save user progress: {e}{Style.RESET_ALL}")
    
    def get_study_materials(self, trade: str) -> Dict:
        """Get study materials for a specific trade.
        
        Args:
            trade: The ITI trade (e.g., "Electrician", "Fitter")
            
        Returns:
            Dictionary containing study materials for the trade
        """
        return self.study_materials.get(trade, {})
    
    def get_quiz_by_trade_and_topic(self, trade: str, topic: str = None, difficulty: str = None) -> List[Dict]:
        """Get quiz questions filtered by trade, topic, and difficulty.
        
        Args:
            trade: The ITI trade
            topic: Optional topic to filter by
            difficulty: Optional difficulty level (easy, medium, hard)
            
        Returns:
            List of quiz questions matching the criteria
        """
        trade_quizzes = self.quiz_database.get(trade, [])
        
        if not trade_quizzes:
            return []
            
        # Filter by topic if specified
        if topic:
            filtered_quizzes = [q for q in trade_quizzes if q["topic"].lower() == topic.lower()]
        else:
            filtered_quizzes = trade_quizzes.copy()
            
        # Filter by difficulty if specified
        if difficulty and filtered_quizzes:
            filtered_quizzes = [q for q in filtered_quizzes if q["difficulty"].lower() == difficulty.lower()]
            
        return filtered_quizzes
    
    def generate_mock_test(self, trade: str, num_questions: int = 10) -> List[Dict]:
        """Generate a mock test with random questions for a specific trade.
        
        Args:
            trade: The ITI trade
            num_questions: Number of questions to include
            
        Returns:
            List of question dictionaries for the mock test
        """
        trade_quizzes = self.quiz_database.get(trade, [])
        
        if not trade_quizzes:
            return []
            
        # Gather all questions from all topics
        all_questions = []
        for topic_quiz in trade_quizzes:
            for question in topic_quiz["questions"]:
                question_with_topic = question.copy()
                question_with_topic["topic"] = topic_quiz["topic"]
                question_with_topic["difficulty"] = topic_quiz["difficulty"]
                all_questions.append(question_with_topic)
                
        # Select random questions
        if len(all_questions) <= num_questions:
            return all_questions
            
        return random.sample(all_questions, num_questions)
    
    def save_quiz_results(self, user_id: str, trade: str, score: int, max_score: int, 
                          topic: str = None, difficulty: str = None) -> None:
        """Save quiz results to user progress data.
        
        Args:
            user_id: User identifier
            trade: The ITI trade
            score: Score achieved
            max_score: Maximum possible score
            topic: Optional topic of the quiz
            difficulty: Optional difficulty level
        """
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {"quizzes": [], "study_time": {}, "completion": {}}
            
        quiz_result = {
            "trade": trade,
            "score": score,
            "max_score": max_score,
            "percentage": round((score / max_score) * 100, 2),
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "difficulty": difficulty
        }
        
        self.user_progress[user_id]["quizzes"].append(quiz_result)
        self._save_user_progress(self.user_progress)
    
    def get_quiz_history(self, user_id: str, trade: str = None) -> List[Dict]:
        """Get quiz history for a user, optionally filtered by trade.
        
        Args:
            user_id: User identifier
            trade: Optional trade to filter by
            
        Returns:
            List of quiz result dictionaries
        """
        if user_id not in self.user_progress or "quizzes" not in self.user_progress[user_id]:
            return []
            
        quiz_history = self.user_progress[user_id]["quizzes"]
        
        if trade:
            return [q for q in quiz_history if q["trade"].lower() == trade.lower()]
        return quiz_history
    
    def log_study_time(self, user_id: str, trade: str, topic: str, minutes: int) -> None:
        """Log study time for a specific trade and topic.
        
        Args:
            user_id: User identifier
            trade: The ITI trade
            topic: The topic studied
            minutes: Time spent studying in minutes
        """
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {"quizzes": [], "study_time": {}, "completion": {}}
            
        if trade not in self.user_progress[user_id]["study_time"]:
            self.user_progress[user_id]["study_time"][trade] = {}
            
        if topic not in self.user_progress[user_id]["study_time"][trade]:
            self.user_progress[user_id]["study_time"][trade][topic] = 0
            
        self.user_progress[user_id]["study_time"][trade][topic] += minutes
        self._save_user_progress(self.user_progress)
    
    def get_study_time_summary(self, user_id: str, trade: str = None) -> Dict:
        """Get summary of study time for a user, optionally filtered by trade.
        
        Args:
            user_id: User identifier
            trade: Optional trade to filter by
            
        Returns:
            Dictionary with study time summary
        """
        if user_id not in self.user_progress or "study_time" not in self.user_progress[user_id]:
            return {"total_minutes": 0, "by_trade": {}, "by_topic": {}}
            
        study_time = self.user_progress[user_id]["study_time"]
        
        if trade:
            filtered_study_time = {trade: study_time.get(trade, {})}
        else:
            filtered_study_time = study_time
            
        # Calculate summary
        total_minutes = 0
        by_trade = {}
        by_topic = {}
        
        for t, topics in filtered_study_time.items():
            trade_total = sum(topics.values())
            total_minutes += trade_total
            by_trade[t] = trade_total
            
            for topic, minutes in topics.items():
                if topic not in by_topic:
                    by_topic[topic] = 0
                by_topic[topic] += minutes
                
        return {
            "total_minutes": total_minutes,
            "total_hours": round(total_minutes / 60, 1),
            "by_trade": by_trade,
            "by_topic": by_topic
        }
    
    def mark_topic_completed(self, user_id: str, trade: str, topic: str) -> None:
        """Mark a topic as completed for a user.
        
        Args:
            user_id: User identifier
            trade: The ITI trade
            topic: The completed topic
        """
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {"quizzes": [], "study_time": {}, "completion": {}}
            
        if trade not in self.user_progress[user_id]["completion"]:
            self.user_progress[user_id]["completion"][trade] = []
            
        if topic not in self.user_progress[user_id]["completion"][trade]:
            self.user_progress[user_id]["completion"][trade].append({
                "topic": topic,
                "completed_at": datetime.now().isoformat()
            })
            
        self._save_user_progress(self.user_progress)
    
    def get_completion_status(self, user_id: str, trade: str) -> Dict:
        """Get completion status for a user's trade.
        
        Args:
            user_id: User identifier
            trade: The ITI trade
            
        Returns:
            Dictionary with completion status information
        """
        if user_id not in self.user_progress or "completion" not in self.user_progress[user_id]:
            return {"completed_topics": [], "total_topics": 0, "completion_percentage": 0}
            
        trade_materials = self.get_study_materials(trade)
        if not trade_materials or "modules" not in trade_materials:
            return {"completed_topics": [], "total_topics": 0, "completion_percentage": 0}
            
        # Get all topics for the trade
        all_topics = []
        for module in trade_materials["modules"]:
            all_topics.extend(module["topics"])
            
        # Get completed topics
        completed_topics = []
        if trade in self.user_progress[user_id]["completion"]:
            completed_topics = [item["topic"] for item in self.user_progress[user_id]["completion"][trade]]
            
        # Calculate percentage
        total_topics = len(all_topics)
        completion_percentage = round((len(completed_topics) / total_topics) * 100, 1) if total_topics > 0 else 0
            
        return {
            "completed_topics": completed_topics,
            "total_topics": total_topics,
            "completion_percentage": completion_percentage
        }
    
    def get_upcoming_exams(self, trade: str = None) -> Dict:
        """Get upcoming exam schedules, optionally filtered by trade.
        
        Args:
            trade: Optional trade to filter by
            
        Returns:
            Dictionary with upcoming exam information
        """
        today = datetime.now()
        upcoming = {
            "semester_exams": [],
            "all_india_trade_tests": [],
            "practical_exams": []
        }
        
        # Filter semester exams
        for exam in self.exam_schedules.get("semester_exams", []):
            exam_date = datetime.strptime(exam["date"], "%Y-%m-%d")
            if exam_date > today and (not trade or exam["trade"] == trade):
                upcoming["semester_exams"].append(exam)
                
        # Filter trade tests
        for test in self.exam_schedules.get("all_india_trade_tests", []):
            test_date = datetime.strptime(test["start_date"], "%Y-%m-%d")
            if test_date > today and (not trade or test["trade"] == "All Trades" or test["trade"] == trade):
                upcoming["all_india_trade_tests"].append(test)
                
        # Filter practical exams
        for exam in self.exam_schedules.get("practical_exams", []):
            exam_date = datetime.strptime(exam["date"], "%Y-%m-%d")
            if exam_date > today and (not trade or exam["trade"] == trade):
                upcoming["practical_exams"].append(exam)
                
        return upcoming
    
    def generate_exam_tips(self, trade: str) -> List[str]:
        """Generate exam preparation tips for a specific trade.
        
        Args:
            trade: The ITI trade
            
        Returns:
            List of exam preparation tips
        """
        general_tips = [
            "Create a study schedule and stick to it",
            "Take regular breaks during study sessions to maintain focus",
            "Use flash cards for important formulas and definitions",
            "Join a study group to discuss difficult concepts",
            "Practice previous years' question papers",
            "Focus on understanding concepts rather than memorizing",
            "Get plenty of sleep before the exam",
            "Review your notes regularly, not just before exams",
            "Teach concepts to others to reinforce your understanding",
            "Use mind maps to connect related topics"
        ]
        
        # Trade-specific tips
        trade_tips = {
            "Electrician": [
                "Practice circuit diagrams regularly",
                "Memorize electrical codes and safety regulations",
                "Review Ohm's Law and Kirchhoff's Laws applications",
                "Practice troubleshooting scenarios for common electrical issues",
                "Study motor control circuits thoroughly"
            ],
            "Fitter": [
                "Practice reading engineering drawings",
                "Review tolerance and fit calculations",
                "Practice with measuring instruments for accuracy",
                "Study material properties and their applications",
                "Review assembly techniques and procedures"
            ],
            "COPA": [
                "Practice typing to improve speed and accuracy",
                "Review database management concepts",
                "Practice common office software operations",
                "Study computer troubleshooting procedures",
                "Review programming logic and basic algorithms"
            ]
        }
        
        # Combine general tips with trade-specific tips
        combined_tips = general_tips.copy()
        if trade in trade_tips:
            combined_tips.extend(trade_tips[trade])
            
        # Shuffle tips to provide variety
        random.shuffle(combined_tips)
        
        return combined_tips[:10]  # Return 10 tips
    
    def create_revision_schedule(self, user_id: str, trade: str, exam_date_str: str) -> Dict:
        """Create a revision schedule for a user leading up to an exam.
        
        Args:
            user_id: User identifier
            trade: The ITI trade
            exam_date_str: Exam date in YYYY-MM-DD format
            
        Returns:
            Dictionary with revision schedule information
        """
        try:
            exam_date = datetime.strptime(exam_date_str, "%Y-%m-%d")
        except ValueError:
            return {"error": "Invalid date format. Please use YYYY-MM-DD format."}
            
        today = datetime.now()
        if exam_date <= today:
            return {"error": "Exam date must be in the future."}
            
        days_until_exam = (exam_date - today).days
        if days_until_exam < 1:
            return {"error": "Exam is too soon to create a revision schedule."}
            
        # Get trade materials
        trade_materials = self.get_study_materials(trade)
        if not trade_materials or "modules" not in trade_materials:
            return {"error": f"No study materials found for {trade}."}
            
        # Get all topics and important topics
        all_topics = []
        for module in trade_materials["modules"]:
            for topic in module["topics"]:
                all_topics.append({"module": module["title"], "topic": topic})
                
        important_topics = trade_materials.get("important_topics", [])
        
        # Get completion status to prioritize incomplete topics
        completion_status = self.get_completion_status(user_id, trade)
        completed_topics = completion_status.get("completed_topics", [])
        
        # Prioritize topics: important uncompleted first, then other uncompleted, then completed
        priority_1 = []  # Important and not completed
        priority_2 = []  # Not important and not completed
        priority_3 = []  # Completed topics
        
        for topic_info in all_topics:
            topic = topic_info["topic"]
            if topic in important_topics and topic not in completed_topics:
                priority_1.append(topic_info)
            elif topic not in completed_topics:
                priority_2.append(topic_info)
            else:
                priority_3.append(topic_info)
                
        # Create schedule based on available days
        schedule = []
        
        # Distribute topics across days
        all_prioritized = priority_1 + priority_2 + priority_3
        topics_per_day = max(1, len(all_prioritized) // days_until_exam)
        
        for day in range(days_until_exam):
            date = today + timedelta(days=day+1)
            day_topics = []
            
            # Get topics for this day
            start_idx = day * topics_per_day
            end_idx = min(start_idx + topics_per_day, len(all_prioritized))
            
            for i in range(start_idx, end_idx):
                if i < len(all_prioritized):
                    day_topics.append(all_prioritized[i])
            
            # Add revision day
            if day_topics:
                schedule.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "day": day + 1,
                    "topics": day_topics,
                    "priority": "High" if any(t in priority_1 for t in day_topics) else "Medium"
                })
                
        # Add final revision of all important topics before exam
        schedule.append({
            "date": exam_date.strftime("%Y-%m-%d"),
            "day": days_until_exam,
            "topics": [{"module": "Final Revision", "topic": topic} for topic in important_topics],
            "priority": "Highest"
        })
                
        return {
            "trade": trade,
            "exam_date": exam_date_str,
            "days_until_exam": days_until_exam,
            "schedule": schedule
        } 