"""
Mentorship module for the ITI Assistant.

This module provides functionality to connect ITI students with industry mentors,
manage mentorship sessions, and track mentorship progress.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from colorama import Fore, Style

class MentorshipManager:
    """Manages mentorship functionality."""
    
    def __init__(self, mentors_data_path="data/mentors.json", 
                 mentorship_sessions_path="data/mentorship_sessions.json"):
        """Initialize the mentorship manager.
        
        Args:
            mentors_data_path: Path to mentors data file
            mentorship_sessions_path: Path to mentorship sessions file
        """
        self.mentors_data_path = mentors_data_path
        self.mentorship_sessions_path = mentorship_sessions_path
        self.mentors = {}
        self.mentorship_sessions = {}
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(mentors_data_path), exist_ok=True)
        
        # Load data
        self._load_mentors()
        self._load_mentorship_sessions()
        
        print(f"{Fore.GREEN}✓ Mentorship manager initialized{Style.RESET_ALL}")
    
    def _load_mentors(self):
        """Load mentors data from file."""
        try:
            if os.path.exists(self.mentors_data_path):
                with open(self.mentors_data_path, "r", encoding="utf-8") as f:
                    self.mentors = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded {len(self.mentors)} mentors{Style.RESET_ALL}")
            else:
                # Initialize with default mentors if file doesn't exist
                self._initialize_default_mentors()
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load mentors data: {e}{Style.RESET_ALL}")
            self._initialize_default_mentors()
    
    def _initialize_default_mentors(self):
        """Initialize with default mentors data."""
        self.mentors = {
            "MEN-001": {
                "id": "MEN-001",
                "name": "Rajiv Mehta",
                "profile_picture": "rajiv_mehta.jpg",
                "current_position": "Senior Production Engineer",
                "company": "Tata Motors",
                "years_of_experience": 15,
                "iti_background": True,
                "trade_expertise": ["Fitter", "Machinist"],
                "education": [
                    "ITI in Fitter from Government ITI, Mumbai (1998)",
                    "Diploma in Mechanical Engineering (2005)"
                ],
                "career_highlights": [
                    "Started as ITI apprentice at Tata Motors",
                    "Led major automation project that increased production efficiency by 25%",
                    "Mentored over 30 ITI graduates in the past 8 years"
                ],
                "mentorship_areas": [
                    "Career planning for ITI graduates",
                    "Advanced industrial fitting techniques",
                    "Transitioning from technician to engineering roles"
                ],
                "availability": "Weekends, 2 hours per week",
                "mentorship_style": "Practical guidance with focus on hands-on learning and industry-relevant skills",
                "languages": ["English", "Hindi", "Marathi"],
                "contact_info": {
                    "email": "rajiv.mehta@example.com",
                    "phone": "+91-9876543210"
                },
                "testimonials": [
                    {
                        "mentee_name": "Amit Kumar",
                        "testimonial": "Mr. Mehta's guidance helped me transition from being a shop floor technician to a supervisor role. His practical advice and industry insights were invaluable."
                    }
                ]
            },
            "MEN-002": {
                "id": "MEN-002",
                "name": "Priya Desai",
                "profile_picture": "priya_desai.jpg",
                "current_position": "Technical Trainer",
                "company": "Larsen & Toubro",
                "years_of_experience": 12,
                "iti_background": True,
                "trade_expertise": ["Electrician", "Electronics Mechanic"],
                "education": [
                    "ITI in Electrician from Government ITI, Pune (2002)",
                    "B.Tech in Electrical Engineering (2010)"
                ],
                "career_highlights": [
                    "First female electrician in her division at L&T",
                    "Developed training program for new ITI apprentices",
                    "Won 'Best Trainer' award in 2018"
                ],
                "mentorship_areas": [
                    "Women in technical trades",
                    "Advanced electrical troubleshooting",
                    "Continuing education opportunities for ITI graduates"
                ],
                "availability": "Weekday evenings, 3 hours per week",
                "mentorship_style": "Supportive and encouraging, with focus on building confidence and technical competence",
                "languages": ["English", "Hindi", "Gujarati"],
                "contact_info": {
                    "email": "priya.desai@example.com",
                    "phone": "+91-9876543211"
                },
                "testimonials": [
                    {
                        "mentee_name": "Neha Sharma",
                        "testimonial": "As a female student in a male-dominated trade, Ms. Desai's mentorship was exactly what I needed. She helped me navigate industry challenges and build my technical capabilities."
                    }
                ]
            },
            "MEN-003": {
                "id": "MEN-003",
                "name": "Mohammed Khan",
                "profile_picture": "mohammed_khan.jpg",
                "current_position": "Service Center Head",
                "company": "Maruti Suzuki",
                "years_of_experience": 18,
                "iti_background": True,
                "trade_expertise": ["Mechanic (Motor Vehicle)", "Diesel Mechanic"],
                "education": [
                    "ITI in Mechanic (Motor Vehicle) from Government ITI, Delhi (1996)",
                    "Advanced Certification in Automotive Technology (2005)"
                ],
                "career_highlights": [
                    "Started as apprentice mechanic and rose to head a major service center",
                    "Specialized in hybrid vehicle technology",
                    "Trains over 50 mechanics annually"
                ],
                "mentorship_areas": [
                    "Automotive industry career growth",
                    "Latest trends in vehicle technology",
                    "Customer service for technical professionals"
                ],
                "availability": "Flexible, 4 hours per week",
                "mentorship_style": "Direct and practical, with focus on real-world problem solving and customer satisfaction",
                "languages": ["English", "Hindi", "Urdu"],
                "contact_info": {
                    "email": "mohammed.khan@example.com",
                    "phone": "+91-9876543212"
                },
                "testimonials": [
                    {
                        "mentee_name": "Ravi Singh",
                        "testimonial": "Mr. Khan's mentorship transformed my approach to automotive repairs. His emphasis on both technical excellence and customer service helped me secure a position at a premium car dealership."
                    }
                ]
            },
            "MEN-004": {
                "id": "MEN-004",
                "name": "Sunita Verma",
                "profile_picture": "sunita_verma.jpg",
                "current_position": "IT Program Manager",
                "company": "Tech Mahindra",
                "years_of_experience": 10,
                "iti_background": True,
                "trade_expertise": ["COPA", "Information Technology"],
                "education": [
                    "ITI in COPA from Government ITI, Bengaluru (2005)",
                    "BCA (2009)",
                    "MBA in IT Management (2013)"
                ],
                "career_highlights": [
                    "Transitioned from ITI COPA graduate to IT management",
                    "Leads Tech Mahindra's ITI graduate recruitment program",
                    "Developed bridge course for ITI graduates entering IT sector"
                ],
                "mentorship_areas": [
                    "IT career pathways for ITI graduates",
                    "Skill enhancement for IT industry",
                    "Higher education opportunities while working"
                ],
                "availability": "Online mentoring, 5 hours per week",
                "mentorship_style": "Goal-oriented and structured, with focus on clear career planning and skill development",
                "languages": ["English", "Hindi", "Kannada"],
                "contact_info": {
                    "email": "sunita.verma@example.com",
                    "phone": "+91-9876543213"
                },
                "testimonials": [
                    {
                        "mentee_name": "Prakash Nair",
                        "testimonial": "Ms. Verma helped me create a clear roadmap from my ITI COPA qualification to a successful IT career. Her guidance on additional certifications and skills was spot-on."
                    }
                ]
            },
            "MEN-005": {
                "id": "MEN-005",
                "name": "Harpreet Singh",
                "profile_picture": "harpreet_singh.jpg",
                "current_position": "Lead Welder",
                "company": "BHEL",
                "years_of_experience": 14,
                "iti_background": True,
                "trade_expertise": ["Welder", "Sheet Metal Worker"],
                "education": [
                    "ITI in Welder from Government ITI, Amritsar (2000)",
                    "Advanced Welding Certification (2008)"
                ],
                "career_highlights": [
                    "Specialized in high-pressure welding for power plants",
                    "Recipient of BHEL's 'Craftsman of the Year' award twice",
                    "Represented India at International Skills Competition"
                ],
                "mentorship_areas": [
                    "Advanced welding techniques",
                    "Safety protocols in high-risk environments",
                    "Certification preparation for welders"
                ],
                "availability": "Weekends, 3 hours per week",
                "mentorship_style": "Hands-on demonstration and practice, with strict emphasis on quality and safety",
                "languages": ["English", "Hindi", "Punjabi"],
                "contact_info": {
                    "email": "harpreet.singh@example.com",
                    "phone": "+91-9876543214"
                },
                "testimonials": [
                    {
                        "mentee_name": "Manoj Kumar",
                        "testimonial": "Mr. Singh's attention to detail and emphasis on quality transformed my welding skills. His mentorship helped me qualify for specialized welding positions that I wouldn't have been eligible for otherwise."
                    }
                ]
            }
        }
        
        # Save the default data
        self._save_mentors()
        print(f"{Fore.GREEN}✓ Initialized default mentors data{Style.RESET_ALL}")
    
    def _save_mentors(self):
        """Save mentors data to file."""
        try:
            os.makedirs(os.path.dirname(self.mentors_data_path), exist_ok=True)
            with open(self.mentors_data_path, "w", encoding="utf-8") as f:
                json.dump(self.mentors, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save mentors data: {e}{Style.RESET_ALL}")
    
    def _load_mentorship_sessions(self):
        """Load mentorship sessions data from file."""
        try:
            if os.path.exists(self.mentorship_sessions_path):
                with open(self.mentorship_sessions_path, "r", encoding="utf-8") as f:
                    self.mentorship_sessions = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded mentorship sessions data{Style.RESET_ALL}")
            else:
                # Initialize with empty mentorship sessions if file doesn't exist
                self.mentorship_sessions = {}
                self._save_mentorship_sessions()
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load mentorship sessions data: {e}{Style.RESET_ALL}")
            self.mentorship_sessions = {}
            self._save_mentorship_sessions()
    
    def _save_mentorship_sessions(self):
        """Save mentorship sessions data to file."""
        try:
            os.makedirs(os.path.dirname(self.mentorship_sessions_path), exist_ok=True)
            with open(self.mentorship_sessions_path, "w", encoding="utf-8") as f:
                json.dump(self.mentorship_sessions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save mentorship sessions data: {e}{Style.RESET_ALL}")
    
    def get_mentors(self, trade=None, mentorship_area=None):
        """Get mentors information.
        
        Args:
            trade: Filter by trade expertise (optional)
            mentorship_area: Filter by mentorship area (optional)
            
        Returns:
            Dictionary of mentors
        """
        if not trade and not mentorship_area:
            return self.mentors
        
        filtered_mentors = {}
        for mentor_id, mentor_info in self.mentors.items():
            # Filter by trade if specified
            if trade and trade not in mentor_info.get("trade_expertise", []):
                continue
            
            # Filter by mentorship area if specified
            if mentorship_area:
                mentorship_areas = mentor_info.get("mentorship_areas", [])
                if not any(mentorship_area.lower() in area.lower() for area in mentorship_areas):
                    continue
            
            filtered_mentors[mentor_id] = mentor_info
        
        return filtered_mentors
  
        def get_mentor_details(self, mentor_id):
            """Get details for a specific mentor.
            
        Args:
            mentor_id: ID of the mentor
            
        Returns:
            Mentor details or None if not found
        """
        return self.mentors.get(mentor_id)
    
    def request_mentorship(self, user_id, mentor_id, request_data):
        """Request mentorship from a mentor.
        
        Args:
            user_id: User identifier
            mentor_id: Mentor ID
            request_data: Mentorship request details
            
        Returns:
            Session ID if successful, None otherwise
        """
        if mentor_id not in self.mentors:
            print(f"{Fore.YELLOW}Mentor ID {mentor_id} not found{Style.RESET_ALL}")
            return None
        
        try:
            # Generate session ID
            session_id = f"SESSION-{len(self.mentorship_sessions) + 1:03d}"
            
            # Create session
            session = {
                "session_id": session_id,
                "user_id": user_id,
                "mentor_id": mentor_id,
                "request_date": datetime.now().isoformat(),
                "status": "Requested",
                "request_data": request_data,
                "messages": [],
                "meetings": [],
                "goals": request_data.get("goals", []),
                "progress_notes": []
            }
            
            # Add to sessions
            self.mentorship_sessions[session_id] = session
            
            # Save to file
            self._save_mentorship_sessions()
            
            return session_id
            
        except Exception as e:
            print(f"{Fore.YELLOW}Could not create mentorship session: {e}{Style.RESET_ALL}")
            return None
    
    def update_session_status(self, session_id, new_status):
        """Update the status of a mentorship session.
        
        Args:
            session_id: Session ID
            new_status: New session status
            
        Returns:
            True if successful, False otherwise
        """
        if session_id not in self.mentorship_sessions:
            print(f"{Fore.YELLOW}Session ID {session_id} not found{Style.RESET_ALL}")
            return False
        
        try:
            # Update status
            self.mentorship_sessions[session_id]["status"] = new_status
            self.mentorship_sessions[session_id]["last_updated"] = datetime.now().isoformat()
            
            # Save to file
            self._save_mentorship_sessions()
            
            return True
            
        except Exception as e:
            print(f"{Fore.YELLOW}Could not update session status: {e}{Style.RESET_ALL}")
            return False
    
    def add_session_message(self, session_id, sender, message):
        """Add a message to a mentorship session.
        
        Args:
            session_id: Session ID
            sender: Message sender (user_id or mentor_id)
            message: Message content
            
        Returns:
            True if successful, False otherwise
        """
        if session_id not in self.mentorship_sessions:
            print(f"{Fore.YELLOW}Session ID {session_id} not found{Style.RESET_ALL}")
            return False
        
        try:
            # Create message
            message_entry = {
                "sender": sender,
                "timestamp": datetime.now().isoformat(),
                "content": message
            }
            
            # Add to session
            self.mentorship_sessions[session_id]["messages"].append(message_entry)
            self.mentorship_sessions[session_id]["last_updated"] = datetime.now().isoformat()
            
            # Save to file
            self._save_mentorship_sessions()
            
            return True
            
        except Exception as e:
            print(f"{Fore.YELLOW}Could not add session message: {e}{Style.RESET_ALL}")
            return False
    
    def schedule_meeting(self, session_id, meeting_data):
        """Schedule a meeting for a mentorship session.
        
        Args:
            session_id: Session ID
            meeting_data: Meeting details
            
        Returns:
            Meeting ID if successful, None otherwise
        """
        if session_id not in self.mentorship_sessions:
            print(f"{Fore.YELLOW}Session ID {session_id} not found{Style.RESET_ALL}")
            return None
        
        try:
            # Generate meeting ID
            meeting_id = f"MEET-{len(self.mentorship_sessions[session_id]['meetings']) + 1:03d}"
            
            # Create meeting
            meeting = {
                "meeting_id": meeting_id,
                "scheduled_date": meeting_data.get("scheduled_date"),
                "duration": meeting_data.get("duration", "1 hour"),
                "mode": meeting_data.get("mode", "Online"),
                "platform": meeting_data.get("platform", "Zoom"),
                "link": meeting_data.get("link", ""),
                "agenda": meeting_data.get("agenda", ""),
                "status": "Scheduled",
                "notes": "",
                "created_at": datetime.now().isoformat()
            }
            
            # Add to session
            self.mentorship_sessions[session_id]["meetings"].append(meeting)
            self.mentorship_sessions[session_id]["last_updated"] = datetime.now().isoformat()
            
            # Save to file
            self._save_mentorship_sessions()
            
            return meeting_id
            
        except Exception as e:
            print(f"{Fore.YELLOW}Could not schedule meeting: {e}{Style.RESET_ALL}")
            return None
    
    def update_meeting(self, session_id, meeting_id, update_data):
        """Update meeting details.
        
        Args:
            session_id: Session ID
            meeting_id: Meeting ID
            update_data: Updated meeting details
            
        Returns:
            True if successful, False otherwise
        """
        if session_id not in self.mentorship_sessions:
            print(f"{Fore.YELLOW}Session ID {session_id} not found{Style.RESET_ALL}")
            return False
        
        try:
            # Find meeting
            meeting_found = False
            for i, meeting in enumerate(self.mentorship_sessions[session_id]["meetings"]):
                if meeting["meeting_id"] == meeting_id:
                    # Update meeting fields
                    for key, value in update_data.items():
                        if key in meeting:
                            meeting[key] = value
                    
                    # Update meeting
                    self.mentorship_sessions[session_id]["meetings"][i] = meeting
                    meeting_found = True
                    break
            
            if not meeting_found:
                print(f"{Fore.YELLOW}Meeting ID {meeting_id} not found in session {session_id}{Style.RESET_ALL}")
                return False
            
            # Update session
            self.mentorship_sessions[session_id]["last_updated"] = datetime.now().isoformat()
            
            # Save to file
            self._save_mentorship_sessions()
            
            return True
            
        except Exception as e:
            print(f"{Fore.YELLOW}Could not update meeting: {e}{Style.RESET_ALL}")
            return False
    
    def add_progress_note(self, session_id, note_data):
        """Add a progress note to a mentorship session.
        
        Args:
            session_id: Session ID
            note_data: Progress note details
            
        Returns:
            True if successful, False otherwise
        """
        if session_id not in self.mentorship_sessions:
            print(f"{Fore.YELLOW}Session ID {session_id} not found{Style.RESET_ALL}")
            return False
        
        try:
            # Create progress note
            progress_note = {
                "timestamp": datetime.now().isoformat(),
                "author": note_data.get("author"),
                "content": note_data.get("content", ""),
                "goals_progress": note_data.get("goals_progress", {}),
                "next_steps": note_data.get("next_steps", [])
            }
            
            # Add to session
            self.mentorship_sessions[session_id]["progress_notes"].append(progress_note)
            self.mentorship_sessions[session_id]["last_updated"] = datetime.now().isoformat()
            
            # Save to file
            self._save_mentorship_sessions()
            
            return True
            
        except Exception as e:
            print(f"{Fore.YELLOW}Could not add progress note: {e}{Style.RESET_ALL}")
            return False
    
    def get_user_sessions(self, user_id):
        """Get mentorship sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of user's mentorship sessions
        """
        user_sessions = {}
        for session_id, session in self.mentorship_sessions.items():
            if session["user_id"] == user_id:
                # Get mentor details
                mentor_info = self.get_mentor_details(session["mentor_id"])
                
                # Create enriched session with mentor details
                enriched_session = session.copy()
                if mentor_info:
                    enriched_session["mentor_name"] = mentor_info.get("name")
                    enriched_session["mentor_position"] = mentor_info.get("current_position")
                    enriched_session["mentor_company"] = mentor_info.get("company")
                
                user_sessions[session_id] = enriched_session
        
        return user_sessions
    
    def get_mentor_sessions(self, mentor_id):
        """Get mentorship sessions for a mentor.
        
        Args:
            mentor_id: Mentor ID
            
        Returns:
            Dictionary of mentor's mentorship sessions
        """
        mentor_sessions = {}
        for session_id, session in self.mentorship_sessions.items():
            if session["mentor_id"] == mentor_id:
                mentor_sessions[session_id] = session
        
        return mentor_sessions
    
    def get_session_details(self, session_id):
        """Get details for a specific mentorship session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session details or None if not found
        """
        if session_id not in self.mentorship_sessions:
            return None
        
        session = self.mentorship_sessions[session_id]
        
        # Get mentor details
        mentor_info = self.get_mentor_details(session["mentor_id"])
        
        # Create enriched session with mentor details
        enriched_session = session.copy()
        if mentor_info:
            enriched_session["mentor_name"] = mentor_info.get("name")
            enriched_session["mentor_position"] = mentor_info.get("current_position")
            enriched_session["mentor_company"] = mentor_info.get("company")
        
        return enriched_session
    
    def generate_mentorship_plan(self, user_data, mentor_id):
        """Generate a mentorship plan for a student.
        
        Args:
            user_data: User information including trade, interests, and goals
            mentor_id: Preferred mentor ID
            
        Returns:
            Structured mentorship plan
        """
        mentor = self.get_mentor_details(mentor_id)
        if not mentor:
            print(f"{Fore.YELLOW}Mentor ID {mentor_id} not found{Style.RESET_ALL}")
            return None
        
        trade = user_data.get("trade", "")
        career_goal = user_data.get("career_goal", "")
        
        # Create mentorship plan
        plan = {
            "mentor": mentor["name"],
            "trade_focus": trade,
            "career_goal": career_goal,
            "duration": "3 months",
            "meeting_frequency": "Biweekly",
            "goals": self._generate_goals(trade, career_goal, mentor),
            "learning_resources": self._suggest_resources(trade, career_goal),
            "milestones": self._generate_milestones(trade, career_goal),
            "success_metrics": self._generate_success_metrics(trade, career_goal)
        }
        
        return plan
    
    def _generate_goals(self, trade, career_goal, mentor):
        """Generate mentorship goals based on trade and career goal.
        
        Args:
            trade: Student's trade
            career_goal: Student's career goal
            mentor: Mentor details
            
        Returns:
            List of mentorship goals
        """
        # Base goals applicable to most trades
        base_goals = [
            "Develop a clear understanding of career pathways in the industry",
            "Identify key technical skills to focus on developing",
            "Build professional network in the industry"
        ]
        
        # Add mentor-specific goals
        mentor_areas = mentor.get("mentorship_areas", [])
        mentor_specific_goals = []
        for area in mentor_areas[:2]:  # Use up to 2 mentorship areas
            mentor_specific_goals.append(f"Learn about {area} from mentor's experience")
        
        # Add trade-specific goals
        trade_specific_goals = []
        if "Fitter" in trade:
            trade_specific_goals = [
                "Master precision measurement techniques",
                "Learn advanced assembly procedures",
                "Develop troubleshooting skills for mechanical systems"
            ]
        elif "Electrician" in trade:
            trade_specific_goals = [
                "Build expertise in electrical safety protocols",
                "Develop skills in reading complex electrical schematics",
                "Learn about modern control systems"
            ]
        elif "Mechanic" in trade:
            trade_specific_goals = [
                "Master diagnostic procedures for vehicle systems",
                "Develop skills in using electronic diagnostic tools",
                "Learn about hybrid/electric vehicle technology"
            ]
        elif "COPA" in trade or "Information Technology" in trade:
            trade_specific_goals = [
                "Develop programming skills relevant to industry needs",
                "Learn about IT infrastructure and networking basics",
                "Build expertise in data management systems"
            ]
        elif "Welder" in trade:
            trade_specific_goals = [
                "Master advanced welding techniques",
                "Learn about material properties and selection",
                "Develop skills in quality inspection of welded joints"
            ]
        else:
            # Generic technical goals
            trade_specific_goals = [
                "Master core technical skills in your trade",
                "Learn about industry standards and best practices",
                "Develop troubleshooting methodology"
            ]
        
        # Add career goal specific goals
        career_specific_goals = []
        if "supervisor" in career_goal.lower() or "management" in career_goal.lower():
            career_specific_goals = [
                "Develop team leadership skills",
                "Learn basics of production planning and management",
                "Build conflict resolution capabilities"
            ]
        elif "entrepreneur" in career_goal.lower() or "business" in career_goal.lower():
            career_specific_goals = [
                "Learn basics of business planning",
                "Understand customer acquisition and service",
                "Develop financial management skills"
            ]
        elif "specialist" in career_goal.lower() or "expert" in career_goal.lower():
            career_specific_goals = [
                "Identify niche specialization areas in your trade",
                "Learn about advanced certification opportunities",
                "Develop deep expertise in specialized techniques"
            ]
        
        # Combine all goals and select a reasonable number
        all_goals = base_goals + mentor_specific_goals + trade_specific_goals + career_specific_goals
        selected_goals = all_goals[:7]  # Limit to 7 goals
        
        return selected_goals
    
    def _suggest_resources(self, trade, career_goal):
        """Suggest learning resources based on trade and career goal.
        
        Args:
            trade: Student's trade
            career_goal: Student's career goal
            
        Returns:
            Dictionary of learning resources
        """
        resources = {
            "books": [],
            "online_courses": [],
            "videos": [],
            "industry_publications": []
        }
        
        # Add trade-specific resources
        if "Fitter" in trade:
            resources["books"].append("Workshop Technology: Volume 1 by W.A.J. Chapman")
            resources["online_courses"].append("Mechanical Measurements and Metrology (NPTEL)")
            resources["videos"].append("Precision Measurement Tools - How to Use Them by Engineering Academy")
        elif "Electrician" in trade:
            resources["books"].append("Electrical Technology by B.L. Theraja")
            resources["online_courses"].append("Electrical Safety in Industrial Plants (NPTEL)")
            resources["videos"].append("Industrial Control Panel Basics by Electrician U")
        elif "Mechanic" in trade:
            resources["books"].append("Automotive Technology: Principles, Diagnosis, and Service by James D. Halderman")
            resources["online_courses"].append("Automotive Engineering (NPTEL)")
            resources["videos"].append("Vehicle Diagnostics Systems Explained by ADPTraining")
        elif "COPA" in trade or "Information Technology" in trade:
            resources["books"].append("Computer Fundamentals by P. K. Sinha")
            resources["online_courses"].append("Introduction to Computer Science and Programming (edX)")
            resources["videos"].append("Web Development Roadmap by Traversy Media")
        elif "Welder" in trade:
            resources["books"].append("Welding Principles and Applications by Larry Jeffus")
            resources["online_courses"].append("Welding Technology (NPTEL)")
            resources["videos"].append("Advanced TIG Welding Techniques by Weld.com")
        
        # Add career-specific resources
        if "supervisor" in career_goal.lower() or "management" in career_goal.lower():
            resources["books"].append("Supervision Today! by Stephen P. Robbins")
            resources["online_courses"].append("First-Time Manager (Coursera)")
        elif "entrepreneur" in career_goal.lower() or "business" in career_goal.lower():
            resources["books"].append("The Lean Startup by Eric Ries")
            resources["online_courses"].append("Starting a Business (Coursera)")
        
        # Add general industry publications
        resources["industry_publications"] = [
            "Indian Journal of Technical Education",
            "Industrial Product Review",
            "Electronics For You (for electrical/electronic trades)",
            "Auto Components India (for automotive trades)"
        ]
        
        return resources
    
    def _generate_milestones(self, trade, career_goal):
        """Generate mentorship milestones based on trade and career goal.
        
        Args:
            trade: Student's trade
            career_goal: Student's career goal
            
        Returns:
            List of mentorship milestones
        """
        # Common milestones for all mentorships
        milestones = [
            {
                "month": 1,
                "milestone": "Complete self-assessment and establish specific goals",
                "activities": [
                    "Initial meeting with mentor",
                    "Identify strengths and areas for improvement",
                    "Finalize mentorship goals and success metrics",
                    "Create learning plan with timeline"
                ]
            },
            {
                "month": 2,
                "milestone": "Develop core technical and professional skills",
                "activities": [
                    "Complete recommended technical training",
                    "Practice communication and workplace skills",
                    "Meet with mentor to review progress",
                    "Adjust goals based on feedback"
                ]
            },
            {
                "month": 3,
                "milestone": "Final assessment and future planning",
                "activities": [
                    "Evaluate progress against initial goals",
                    "Create ongoing professional development plan",
                    "Identify next steps for career advancement",
                    "Final meeting with mentor"
                ]
            }
        ]
        
        return milestones
    
    def _generate_success_metrics(self, trade, career_goal):
        """Generate success metrics for mentorship.
        
        Args:
            trade: Student's trade
            career_goal: Student's career goal
            
        Returns:
            List of success metrics
        """
        # Common success metrics
        metrics = [
            "Completion of all planned mentorship activities",
            "Demonstrated improvement in technical skills",
            "Development of professional network connections",
            "Creation of clear career advancement plan"
        ]
        
        # Add trade-specific metrics
        if "Fitter" in trade or "Mechanic" in trade or "Welder" in trade:
            metrics.append("Completion of a hands-on project demonstrating improved skills")
        elif "Electrician" in trade:
            metrics.append("Ability to troubleshoot complex electrical problems")
        elif "COPA" in trade or "Information Technology" in trade:
            metrics.append("Creation of a portfolio demonstrating programming/IT skills")
        
        # Add career goal-specific metrics
        if "supervisor" in career_goal.lower() or "management" in career_goal.lower():
            metrics.append("Development of basic team leadership capabilities")
        elif "entrepreneur" in career_goal.lower() or "business" in career_goal.lower():
            metrics.append("Creation of a basic business plan for future venture")
        
        return metrics