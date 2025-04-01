"""
Industry connection module for the ITI Assistant.

This module provides functionality to connect ITI students with industry opportunities,
including apprenticeships, job openings, and success stories from ITI graduates.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from colorama import Fore, Style

class IndustryConnectionManager:
    """Manages industry connection functionality."""
    
    def __init__(self, companies_data_path="data/industry_companies.json", 
                 opportunities_data_path="data/industry_opportunities.json",
                 success_stories_path="data/success_stories.json"):
        """Initialize the industry connection manager.
        
        Args:
            companies_data_path: Path to companies data file
            opportunities_data_path: Path to opportunities data file
            success_stories_path: Path to success stories file
        """
        self.companies_data_path = companies_data_path
        self.opportunities_data_path = opportunities_data_path
        self.success_stories_path = success_stories_path
        self.companies = {}
        self.opportunities = {}
        self.success_stories = {}
        self.user_applications = {}
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(companies_data_path), exist_ok=True)
        
        # Load data
        self._load_companies()
        self._load_opportunities()
        self._load_success_stories()
        
        print(f"{Fore.GREEN}✓ Industry connection manager initialized{Style.RESET_ALL}")
    
    def _load_companies(self):
        """Load companies data from file."""
        try:
            if os.path.exists(self.companies_data_path):
                with open(self.companies_data_path, "r", encoding="utf-8") as f:
                    self.companies = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded {len(self.companies)} companies{Style.RESET_ALL}")
            else:
                # Initialize with default companies if file doesn't exist
                self._initialize_default_companies()
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load companies data: {e}{Style.RESET_ALL}")
            self._initialize_default_companies()
    
    def _initialize_default_companies(self):
        """Initialize with default companies data."""
        self.companies = {
            "Tata Motors": {
                "description": "One of India's largest automobile manufacturers with a global presence.",
                "website": "https://www.tatamotors.com",
                "headquarters": "Mumbai, Maharashtra",
                "hiring_trades": ["Fitter", "Electrician", "Mechanic (Motor Vehicle)", "Turner"],
                "industry_sector": "Automobile Manufacturing",
                "contact_info": {
                    "email": "careers@tatamotors.com",
                    "phone": "+91-22-12345678"
                },
                "apprenticeship_program": True,
                "iti_partnerships": ["Government ITI, Mumbai", "Government ITI, Pune"]
            },
            "Larsen & Toubro": {
                "description": "Indian multinational conglomerate with businesses in engineering, construction, manufacturing, and technology.",
                "website": "https://www.larsentoubro.com",
                "headquarters": "Mumbai, Maharashtra",
                "hiring_trades": ["Fitter", "Electrician", "Welder", "Turner", "Machinist"],
                "industry_sector": "Engineering and Construction",
                "contact_info": {
                    "email": "careers@larsentoubro.com",
                    "phone": "+91-22-87654321"
                },
                "apprenticeship_program": True,
                "iti_partnerships": ["Government ITI, Mumbai", "Government ITI, Chennai"]
            },
            "BHEL (Bharat Heavy Electricals Limited)": {
                "description": "India's largest power generation equipment manufacturer.",
                "website": "https://www.bhel.com",
                "headquarters": "New Delhi",
                "hiring_trades": ["Electrician", "Fitter", "Welder", "Turner", "Machinist"],
                "industry_sector": "Power Generation Equipment",
                "contact_info": {
                    "email": "careers@bhel.in",
                    "phone": "+91-11-12345678"
                },
                "apprenticeship_program": True,
                "iti_partnerships": ["Government ITI, Delhi", "Government ITI, Chennai"]
            },
            "Maruti Suzuki": {
                "description": "India's largest passenger car manufacturer.",
                "website": "https://www.marutisuzuki.com",
                "headquarters": "New Delhi",
                "hiring_trades": ["Mechanic (Motor Vehicle)", "Electrician", "Fitter", "Painter"],
                "industry_sector": "Automobile Manufacturing",
                "contact_info": {
                    "email": "careers@maruti.co.in",
                    "phone": "+91-11-87654321"
                },
                "apprenticeship_program": True,
                "iti_partnerships": ["Government ITI, Delhi", "Government ITI, Gurugram"]
            },
            "Tech Mahindra": {
                "description": "Indian multinational provider of information technology and business process outsourcing services.",
                "website": "https://www.techmahindra.com",
                "headquarters": "Pune, Maharashtra",
                "hiring_trades": ["COPA", "Electronics Mechanic", "Information Technology"],
                "industry_sector": "Information Technology",
                "contact_info": {
                    "email": "careers@techmahindra.com",
                    "phone": "+91-20-12345678"
                },
                "apprenticeship_program": False,
                "iti_partnerships": ["Government ITI, Pune"]
            }
        }
        
        # Save the default data
        self._save_companies()
        print(f"{Fore.GREEN}✓ Initialized default companies data{Style.RESET_ALL}")
    
    def _save_companies(self):
        """Save companies data to file."""
        try:
            os.makedirs(os.path.dirname(self.companies_data_path), exist_ok=True)
            with open(self.companies_data_path, "w", encoding="utf-8") as f:
                json.dump(self.companies, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save companies data: {e}{Style.RESET_ALL}")
    
    def _load_opportunities(self):
        """Load opportunities data from file."""
        try:
            if os.path.exists(self.opportunities_data_path):
                with open(self.opportunities_data_path, "r", encoding="utf-8") as f:
                    self.opportunities = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded {len(self.opportunities)} opportunities{Style.RESET_ALL}")
            else:
                # Initialize with default opportunities if file doesn't exist
                self._initialize_default_opportunities()
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load opportunities data: {e}{Style.RESET_ALL}")
            self._initialize_default_opportunities()
    
    def _initialize_default_opportunities(self):
        """Initialize with default opportunities data."""
        self.opportunities = {
            "apprenticeships": [
                {
                    "id": "app-001",
                    "company": "Tata Motors",
                    "title": "Apprentice Fitter",
                    "location": "Pune, Maharashtra",
                    "trade": "Fitter",
                    "duration": "1 year",
                    "stipend": "₹10,000 - ₹12,000 per month",
                    "description": "Learn industrial fitting skills in automobile manufacturing setting.",
                    "requirements": ["ITI certificate in Fitter trade", "Good mechanical aptitude", "Basic computer knowledge"],
                    "application_deadline": "2023-12-31",
                    "start_date": "2024-02-01",
                    "how_to_apply": "Apply online at careers.tatamotors.com or through your ITI placement cell.",
                    "additional_info": "Selected candidates will receive professional certification upon successful completion."
                },
                {
                    "id": "app-002",
                    "company": "BHEL",
                    "title": "Electrician Apprentice",
                    "location": "Trichy, Tamil Nadu",
                    "trade": "Electrician",
                    "duration": "1 year",
                    "stipend": "₹9,000 - ₹11,000 per month",
                    "description": "Comprehensive apprenticeship in industrial electrical systems for power plants.",
                    "requirements": ["ITI certificate in Electrician trade", "Basic understanding of electrical systems", "Willing to work in shifts"],
                    "application_deadline": "2023-11-30",
                    "start_date": "2024-01-15",
                    "how_to_apply": "Submit application through BHEL recruitment portal.",
                    "additional_info": "Possibility of permanent employment for top performers."
                }
            ],
            "job_openings": [
                {
                    "id": "job-001",
                    "company": "Maruti Suzuki",
                    "title": "Automobile Technician",
                    "location": "Gurugram, Haryana",
                    "trade": "Mechanic (Motor Vehicle)",
                    "experience": "0-2 years",
                    "salary": "₹18,000 - ₹25,000 per month",
                    "description": "Perform maintenance and repair of automobiles at Maruti Suzuki service center.",
                    "requirements": ["ITI certificate in Mechanic (Motor Vehicle)", "Knowledge of modern vehicle systems", "Good communication skills"],
                    "application_deadline": "2023-12-15",
                    "how_to_apply": "Email resume to careers@maruti.co.in with subject 'Automobile Technician Application'.",
                    "additional_info": "Training in Maruti Suzuki technologies will be provided."
                },
                {
                    "id": "job-002",
                    "company": "Larsen & Toubro",
                    "title": "Junior Welder",
                    "location": "Mumbai, Maharashtra",
                    "trade": "Welder",
                    "experience": "1-3 years",
                    "salary": "₹20,000 - ₹28,000 per month",
                    "description": "Perform welding operations for construction and manufacturing projects.",
                    "requirements": ["ITI certificate in Welder trade", "Experience with various welding techniques", "Ability to read engineering drawings"],
                    "application_deadline": "2023-11-20",
                    "how_to_apply": "Apply through L&T careers portal.",
                    "additional_info": "Site allowance and accommodation provided for outstation work."
                }
            ],
            "training_programs": [
                {
                    "id": "tp-001",
                    "company": "Tech Mahindra",
                    "title": "IT Skills Enhancement Program for ITI Graduates",
                    "location": "Pune, Maharashtra",
                    "eligible_trades": ["COPA", "Electronics Mechanic"],
                    "duration": "3 months",
                    "stipend": "₹8,000 per month during training",
                    "description": "Intensive training program to bridge the gap between ITI skills and IT industry requirements.",
                    "curriculum": ["Basic programming", "Network fundamentals", "Hardware troubleshooting", "IT support skills"],
                    "application_deadline": "2023-12-10",
                    "start_date": "2024-01-10",
                    "placement_assistance": True,
                    "how_to_apply": "Submit application through Tech Mahindra Foundation website.",
                    "additional_info": "Placement opportunity for successful candidates."
                }
            ]
        }
        
        # Save the default data
        self._save_opportunities()
        print(f"{Fore.GREEN}✓ Initialized default opportunities data{Style.RESET_ALL}")
    
    def _save_opportunities(self):
        """Save opportunities data to file."""
        try:
            os.makedirs(os.path.dirname(self.opportunities_data_path), exist_ok=True)
            with open(self.opportunities_data_path, "w", encoding="utf-8") as f:
                json.dump(self.opportunities, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save opportunities data: {e}{Style.RESET_ALL}")
    
    def _load_success_stories(self):
        """Load success stories from file."""
        try:
            if os.path.exists(self.success_stories_path):
                with open(self.success_stories_path, "r", encoding="utf-8") as f:
                    self.success_stories = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded {len(self.success_stories)} success stories{Style.RESET_ALL}")
            else:
                # Initialize with default success stories if file doesn't exist
                self._initialize_default_success_stories()
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load success stories: {e}{Style.RESET_ALL}")
            self._initialize_default_success_stories()
    
    def _initialize_default_success_stories(self):
        """Initialize with default success stories."""
        self.success_stories = [
            {
                "name": "Rajesh Kumar",
                "trade": "Electrician",
                "graduation_year": "2018",
                "institute": "Government ITI, Delhi",
                "current_position": "Senior Electrician",
                "company": "BHEL",
                "story": "After completing my ITI in Electrician trade, I joined BHEL as an apprentice. I worked hard and learned advanced skills in industrial electrical systems. After my apprenticeship, I was offered a permanent position and have since been promoted to Senior Electrician. I now lead a team of 5 electricians and oversee electrical installations for major power plant projects. My ITI training gave me the strong foundation I needed to succeed in this field.",
                "achievements": ["Promoted to team leader within 3 years", "Received 'Best Performer' award in 2020", "Completed advanced certification in Industrial Automation"],
                "advice_for_students": "Focus on practical skills during your ITI training. Don't hesitate to ask questions and seek additional knowledge. Always be open to learning new technologies as the field is constantly evolving."
            },
            {
                "name": "Priya Sharma",
                "trade": "COPA",
                "graduation_year": "2019",
                "institute": "Government ITI, Pune",
                "current_position": "IT Support Specialist",
                "company": "Tech Mahindra",
                "story": "I joined the ITI COPA program because I was interested in computers but couldn't afford a full engineering degree. After completing my ITI, I enrolled in Tech Mahindra's skills enhancement program specifically designed for ITI graduates. The program helped me bridge the gap between my ITI skills and industry requirements. I was hired by Tech Mahindra after the program and now work as an IT Support Specialist, assisting clients with technical issues and solutions.",
                "achievements": ["Selected for international client project within 1 year", "Completed additional certifications in networking", "Recognized for highest customer satisfaction ratings"],
                "advice_for_students": "Don't limit yourself to just what's taught in the curriculum. Explore additional resources online, practice coding regularly, and stay updated with technology trends. Soft skills like communication are also crucial in the IT industry."
            },
            {
                "name": "Mohammed Farhan",
                "trade": "Mechanic (Motor Vehicle)",
                "graduation_year": "2017",
                "institute": "Government ITI, Gurugram",
                "current_position": "Service Center Manager",
                "company": "Maruti Suzuki",
                "story": "I've always been passionate about automobiles, which led me to pursue ITI in Motor Vehicle Mechanics. After completing my course, I joined Maruti Suzuki as a trainee technician. My technical skills combined with my interest in customer service helped me progress quickly. Within four years, I was promoted to Service Advisor and then to Service Center Manager. I now manage a team of 15 technicians and oversee the entire service operations at our center in Gurugram.",
                "achievements": ["Increased service center efficiency by 30%", "Received 'Excellence in Customer Service' award", "Completed advanced training in hybrid vehicle technology"],
                "advice_for_students": "Technical skills are important, but don't neglect developing your management and communication abilities. These will be crucial as you advance in your career. Always focus on solving customers' problems efficiently."
            },
            {
                "name": "Anita Desai",
                "trade": "Fitter",
                "graduation_year": "2016",
                "institute": "Government ITI, Mumbai",
                "current_position": "Production Supervisor",
                "company": "Tata Motors",
                "story": "Coming from a farming family, ITI was my pathway to industrial employment. I joined Tata Motors as an apprentice after completing my ITI in Fitter trade. My attention to detail and problem-solving abilities were noticed by management, which led to me being offered a permanent position after my apprenticeship. I've since been promoted to Production Supervisor, overseeing critical assembly operations. I've also completed a part-time diploma while working, which has further enhanced my career prospects.",
                "achievements": ["Led team that improved assembly line efficiency by 25%", "Implemented cost-saving measures worth ₹10 lakhs annually", "Pursuing part-time mechanical engineering degree"],
                "advice_for_students": "Be persistent and maintain a positive attitude even when facing challenges. Take initiative and suggest improvements whenever you see an opportunity. Continuous learning is essential for growth in manufacturing."
            }
        ]
        
        # Save the default data
        self._save_success_stories()
        print(f"{Fore.GREEN}✓ Initialized default success stories{Style.RESET_ALL}")
    
    def _save_success_stories(self):
        """Save success stories to file."""
        try:
            os.makedirs(os.path.dirname(self.success_stories_path), exist_ok=True)
            with open(self.success_stories_path, "w", encoding="utf-8") as f:
                json.dump(self.success_stories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save success stories: {e}{Style.RESET_ALL}")
    
    def get_companies(self, trade=None):
        """Get companies information.
        
        Args:
            trade: Filter by trade (optional)
            
        Returns:
            Dictionary of companies
        """
        if not trade:
            return self.companies
        
        # Filter companies by trade
        filtered_companies = {}
        for company_name, company_info in self.companies.items():
            if trade in company_info.get("hiring_trades", []):
                filtered_companies[company_name] = company_info
        
        return filtered_companies
    
    def get_company_details(self, company_name):
        """Get details for a specific company.
        
        Args:
            company_name: Name of the company
            
        Returns:
            Company details or None if not found
        """
        return self.companies.get(company_name)
    
    def get_opportunities(self, category=None, trade=None, location=None):
        """Get industry opportunities.
        
        Args:
            category: Opportunity category (apprenticeships, job_openings, training_programs)
            trade: Filter by trade
            location: Filter by location
            
        Returns:
            List of opportunities
        """
        result = []
        
        # Select categories to include
        if category and category in self.opportunities:
            categories = {category: self.opportunities[category]}
        else:
            categories = self.opportunities
        
        # Gather opportunities
        for category_name, opportunities_list in categories.items():
            for opportunity in opportunities_list:
                # Apply trade filter if specified
                if trade:
                    if category_name == "apprenticeships" or category_name == "job_openings":
                        if opportunity.get("trade") != trade:
                            continue
                    elif category_name == "training_programs":
                        if trade not in opportunity.get("eligible_trades", []):
                            continue
                
                # Apply location filter if specified
                if location and location.lower() not in opportunity.get("location", "").lower():
                    continue
                
                # Add category to the opportunity data
                opportunity_copy = opportunity.copy()
                opportunity_copy["category"] = category_name
                result.append(opportunity_copy)
        
        return result
    
    def get_opportunity_details(self, opportunity_id):
        """Get details for a specific opportunity.
        
        Args:
            opportunity_id: ID of the opportunity
            
        Returns:
            Opportunity details or None if not found
        """
        for category, opportunities_list in self.opportunities.items():
            for opportunity in opportunities_list:
                if opportunity.get("id") == opportunity_id:
                    opportunity_copy = opportunity.copy()
                    opportunity_copy["category"] = category
                    return opportunity_copy
        
        return None
    
    def get_success_stories(self, trade=None, company=None):
        """Get success stories of ITI graduates.
        
        Args:
            trade: Filter by trade (optional)
            company: Filter by company (optional)
            
        Returns:
            List of success stories
        """
        if not trade and not company:
            return self.success_stories
        
        filtered_stories = []
        for story in self.success_stories:
            # Apply trade filter
            if trade and story.get("trade") != trade:
                continue
            
            # Apply company filter
            if company and story.get("company") != company:
                continue
            
            filtered_stories.append(story)
        
        return filtered_stories
    
    def save_application(self, user_id, opportunity_id, application_data):
        """Save user's application for an opportunity.
        
        Args:
            user_id: User identifier
            opportunity_id: Opportunity ID
            application_data: Application details
            
        Returns:
            True if successful, False otherwise
        """
        try:
            application = {
                "user_id": user_id,
                "opportunity_id": opportunity_id,
                "application_date": datetime.now().isoformat(),
                "status": "Submitted",
                "data": application_data
            }
            
            # Load existing applications for this user
            self._load_user_applications(user_id)
            
            if user_id not in self.user_applications:
                self.user_applications[user_id] = []
            
            # Add new application
            self.user_applications[user_id].append(application)
            
            # Save to file
            return self._save_user_applications(user_id)
            
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save application: {e}{Style.RESET_ALL}")
            return False
    
    def _load_user_applications(self, user_id):
        """Load user applications from file.
        
        Args:
            user_id: User identifier
        """
        try:
            applications_path = f"data/user_applications_{user_id}.json"
            
            if os.path.exists(applications_path):
                with open(applications_path, "r", encoding="utf-8") as f:
                    self.user_applications[user_id] = json.load(f)
            else:
                self.user_applications[user_id] = []
                
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load user applications: {e}{Style.RESET_ALL}")
            self.user_applications[user_id] = []
    
    def _save_user_applications(self, user_id):
        """Save user applications to file.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            applications_path = f"data/user_applications_{user_id}.json"
            os.makedirs(os.path.dirname(applications_path), exist_ok=True)
            
            with open(applications_path, "w", encoding="utf-8") as f:
                json.dump(self.user_applications[user_id], f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save user applications: {e}{Style.RESET_ALL}")
            return False
    
    def get_user_applications(self, user_id):
        """Get user's applications.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of user's applications
        """
        self._load_user_applications(user_id)
        
        # Enrich applications with opportunity details
        enriched_applications = []
        for application in self.user_applications.get(user_id, []):
            opportunity_details = self.get_opportunity_details(application.get("opportunity_id"))
            if opportunity_details:
                application_copy = application.copy()
                application_copy["opportunity_details"] = opportunity_details
                enriched_applications.append(application_copy)
            else:
                enriched_applications.append(application)
        
        return enriched_applications
    
    def update_application_status(self, user_id, opportunity_id, new_status):
        """Update the status of a user's application.
        
        Args:
            user_id: User identifier
            opportunity_id: Opportunity ID
            new_status: New application status
            
        Returns:
            True if successful, False otherwise
        """
        self._load_user_applications(user_id)
        
        updated = False
        for application in self.user_applications.get(user_id, []):
            if application.get("opportunity_id") == opportunity_id:
                application["status"] = new_status
                application["last_updated"] = datetime.now().isoformat()
                updated = True
                break
        
        if updated:
            return self._save_user_applications(user_id)
        
        return False
    
    def generate_resume(self, user_data, template="standard"):
        """Generate a resume for ITI students.
        
        Args:
            user_data: User information
            template: Resume template name
            
        Returns:
            Formatted resume text
        """
        if template == "standard":
            return self._generate_standard_resume(user_data)
        elif template == "technical":
            return self._generate_technical_resume(user_data)
        else:
            return self._generate_standard_resume(user_data)
    
    def _generate_standard_resume(self, user_data):
        """Generate a standard resume.
        
        Args:
            user_data: User information
            
        Returns:
            Formatted resume text
        """
        resume = f"# {user_data.get('name', 'Name')}\n\n"
        
        # Contact information
        resume += "## Contact Information\n"
        resume += f"**Email:** {user_data.get('email', '')}\n"
        resume += f"**Phone:** {user_data.get('phone', '')}\n"
        resume += f"**Address:** {user_data.get('address', '')}\n\n"
        
        # Career objective
        resume += "## Career Objective\n"
        resume += f"{user_data.get('objective', '')}\n\n"
        
        # Education
        resume += "## Education\n"
        for education in user_data.get('education', []):
            resume += f"**{education.get('year', '')}** - {education.get('qualification', '')} - {education.get('institution', '')}\n"
            if 'grade' in education:
                resume += f"Grade/Percentage: {education.get('grade', '')}\n"
        resume += "\n"
        
        # Skills
        resume += "## Technical Skills\n"
        for skill in user_data.get('skills', []):
            resume += f"- {skill}\n"
        resume += "\n"
        
        # Experience
        if 'experience' in user_data and user_data['experience']:
            resume += "## Work Experience\n"
            for experience in user_data.get('experience', []):
                resume += f"**{experience.get('position', '')}** - {experience.get('company', '')} ({experience.get('duration', '')})\n"
                resume += f"{experience.get('description', '')}\n\n"
        
        # Projects
        if 'projects' in user_data and user_data['projects']:
            resume += "## Projects\n"
            for project in user_data.get('projects', []):
                resume += f"**{project.get('title', '')}**\n"
                resume += f"{project.get('description', '')}\n\n"
        
        # Certifications
        if 'certifications' in user_data and user_data['certifications']:
            resume += "## Certifications\n"
            for cert in user_data.get('certifications', []):
                resume += f"- {cert}\n"
            resume += "\n"
        
        # References
        resume += "## References\n"
        resume += "Available upon request\n"
        
        return resume
    
    def _generate_technical_resume(self, user_data):
        """Generate a technical resume emphasizing practical skills.
        
        Args:
            user_data: User information
            
        Returns:
            Formatted resume text
        """
        resume = f"# {user_data.get('name', 'Name')} - {user_data.get('trade', '')}\n\n"
        
        # Contact information
        resume += "## Contact Information\n"
        resume += f"**Email:** {user_data.get('email', '')}\n"
        resume += f"**Phone:** {user_data.get('phone', '')}\n"
        resume += f"**Address:** {user_data.get('address', '')}\n\n"
        
        # Technical summary
        resume += "## Technical Summary\n"
        resume += f"{user_data.get('summary', '')}\n\n"
        
        # Technical skills - organized by category
        resume += "## Technical Skills\n"
        
        if 'technical_skills' in user_data:
            for category, skills in user_data.get('technical_skills', {}).items():
                resume += f"**{category}:** {', '.join(skills)}\n"
        else:
            for skill in user_data.get('skills', []):
                resume += f"- {skill}\n"
        resume += "\n"
        
        # Practical experience
        resume += "## Practical Experience\n"
        for experience in user_data.get('experience', []):
            resume += f"**{experience.get('position', '')}** - {experience.get('company', '')} ({experience.get('duration', '')})\n"
            
            # Responsibilities
            if 'responsibilities' in experience:
                resume += "Responsibilities:\n"
                for responsibility in experience.get('responsibilities', []):
                    resume += f"- {responsibility}\n"
            else:
                resume += f"{experience.get('description', '')}\n"
                
            # Achievements
            if 'achievements' in experience:
                resume += "Achievements:\n"
                for achievement in experience.get('achievements', []):
                    resume += f"- {achievement}\n"
            
            resume += "\n"
        
        # Education - simplified for technical resume
        resume += "## Education\n"
        for education in user_data.get('education', []):
            resume += f"**{education.get('qualification', '')}** - {education.get('institution', '')} ({education.get('year', '')})\n"
        resume += "\n"
        
        # Projects with technical details
        if 'projects' in user_data and user_data['projects']:
            resume += "## Technical Projects\n"
            for project in user_data.get('projects', []):
                resume += f"**{project.get('title', '')}**\n"
                resume += f"Description: {project.get('description', '')}\n"
                
                if 'technologies' in project:
                    resume += f"Technologies: {', '.join(project.get('technologies', []))}\n"
                
                if 'outcome' in project:
                    resume += f"Outcome: {project.get('outcome', '')}\n"
                
                resume += "\n"
        
        # Certifications with date and issuing authority
        if 'certifications' in user_data and user_data['certifications']:
            resume += "## Certifications\n"
            for cert in user_data.get('certifications', []):
                if isinstance(cert, dict):
                    resume += f"- {cert.get('name', '')} - {cert.get('issuer', '')} ({cert.get('date', '')})\n"
                else:
                    resume += f"- {cert}\n"
            resume += "\n"
        
        # References
        resume += "## References\n"
        if 'references' in user_data and user_data['references']:
            for reference in user_data.get('references', []):
                resume += f"**{reference.get('name', '')}** - {reference.get('position', '')}, {reference.get('company', '')}\n"
                resume += f"Contact: {reference.get('contact', '')}\n\n"
        else:
            resume += "Available upon request\n"
        
        return resume 