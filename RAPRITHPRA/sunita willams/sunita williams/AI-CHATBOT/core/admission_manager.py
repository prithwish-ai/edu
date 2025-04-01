"""
Admission Process Module for ITI Assistant.

This module provides comprehensive information about the admission process for
ITI courses, including eligibility criteria, application procedures, required documents,
important dates, and entrance examination details.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from colorama import Fore, Style

class AdmissionManager:
    """Manages admission process information for ITI courses."""
    
    def __init__(self, admission_data_path="data/admission_details.json"):
        """Initialize the admission manager.
        
        Args:
            admission_data_path: Path to admission data JSON file
        """
        self.admission_data_path = admission_data_path
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(admission_data_path), exist_ok=True)
        
        # Load admission data
        self.admission_data = self._load_admission_data()
        
        print(f"{Fore.GREEN}✓ Admission manager initialized{Style.RESET_ALL}")
        
    def _load_admission_data(self) -> Dict:
        """Load admission data from file or initialize with default data."""
        try:
            if os.path.exists(self.admission_data_path):
                with open(self.admission_data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Initialize with default admission data
                default_data = self._initialize_default_admission_data()
                self._save_admission_data(default_data)
                return default_data
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load admission data: {e}{Style.RESET_ALL}")
            default_data = self._initialize_default_admission_data()
            return default_data
    
    def _initialize_default_admission_data(self) -> Dict:
        """Initialize default admission data."""
        # Calculate dates for upcoming admission cycle
        today = datetime.now()
        
        # Simulate upcoming admission cycle
        next_cycle_year = today.year if today.month < 6 else today.year + 1
        
        # Admission dates
        form_start_date = datetime(next_cycle_year, 5, 15).strftime("%Y-%m-%d")
        form_end_date = datetime(next_cycle_year, 6, 30).strftime("%Y-%m-%d")
        exam_date = datetime(next_cycle_year, 7, 15).strftime("%Y-%m-%d")
        result_date = datetime(next_cycle_year, 7, 31).strftime("%Y-%m-%d")
        counseling_start_date = datetime(next_cycle_year, 8, 10).strftime("%Y-%m-%d")
        counseling_end_date = datetime(next_cycle_year, 8, 25).strftime("%Y-%m-%d")
        session_start_date = datetime(next_cycle_year, 9, 1).strftime("%Y-%m-%d")
        
        return {
            "general_information": {
                "about_iti": "Industrial Training Institutes (ITIs) are post-secondary schools in India that provide technical education and training across various trades. These institutes are governed by the Directorate General of Training (DGT) under the Ministry of Skill Development and Entrepreneurship. ITIs offer certificate courses in technical, non-technical, and vocational subjects to prepare students for employment in specific trades or for self-employment.",
                "course_duration": "ITI courses typically range from 6 months to 2 years depending on the trade and specialization. Engineering trades usually have a duration of 1-2 years, while non-engineering trades may be 6 months to 1 year.",
                "certification": "Upon successful completion of ITI courses, students receive National Trade Certificates (NTC) recognized by the National Council for Vocational Training (NCVT) or State Trade Certificates (STC) recognized by the State Council for Vocational Training (SCVT)."
            },
            "eligibility_criteria": {
                "minimum_qualification": {
                    "engineering_trades": "10th pass (Secondary School Certificate) with Science and Mathematics",
                    "non_engineering_trades": "8th pass for select trades, 10th pass for most trades",
                    "advanced_courses": "10th or 12th pass depending on the specific course"
                },
                "age_criteria": {
                    "minimum_age": 14,
                    "maximum_age": 40,
                    "relaxation": {
                        "SC_ST": "5 years age relaxation",
                        "OBC": "3 years age relaxation",
                        "PwD": "10 years age relaxation"
                    }
                },
                "physical_fitness": "Candidates must be physically fit and meet specific physical standards depending on the trade. Certain trades may have specific requirements for vision, height, weight, etc.",
                "reservation_policy": {
                    "SC": "15% of seats reserved",
                    "ST": "7.5% of seats reserved",
                    "OBC": "27% of seats reserved",
                    "EWS": "10% of seats reserved",
                    "PwD": "4% of seats reserved",
                    "Women": "30% horizontal reservation in many states"
                }
            },
            "admission_process": {
                "steps": [
                    {
                        "step": 1,
                        "name": "Application Form Submission",
                        "description": "Candidates must fill and submit the application form either online through the official ITI admission portal or offline at designated centers. Application forms are usually available from May to June each year.",
                        "start_date": form_start_date,
                        "end_date": form_end_date,
                        "fees": {
                            "general": "₹300",
                            "sc_st": "₹150"
                        }
                    },
                    {
                        "step": 2,
                        "name": "Entrance Examination/Merit List",
                        "description": "Some states conduct entrance examinations for ITI admissions, while others prepare merit lists based on qualifying examination marks. The entrance exam typically tests basic mathematics, science, reasoning, and general knowledge.",
                        "exam_date": exam_date,
                        "result_date": result_date
                    },
                    {
                        "step": 3,
                        "name": "Counseling and Seat Allotment",
                        "description": "Qualified candidates participate in counseling sessions for trade and institute selection. Seat allotment is based on rank, preference, and seat availability. Multiple rounds of counseling may be conducted.",
                        "start_date": counseling_start_date,
                        "end_date": counseling_end_date
                    },
                    {
                        "step": 4,
                        "name": "Document Verification",
                        "description": "Candidates must present original documents for verification at the allotted ITI. This includes educational certificates, identity proof, address proof, category certificates (if applicable), etc."
                    },
                    {
                        "step": 5,
                        "name": "Fee Payment",
                        "description": "After document verification, candidates must pay the required fees to confirm their admission. Fee structures vary by state, institute, and trade."
                    },
                    {
                        "step": 6,
                        "name": "Admission Confirmation and Course Commencement",
                        "description": "Upon fee payment, admission is confirmed. Classes usually begin in August or September.",
                        "session_start_date": session_start_date
                    }
                ]
            },
            "required_documents": [
                {
                    "name": "Educational Certificates",
                    "description": "Original and photocopies of 8th/10th/12th marksheets and certificates as per the eligibility requirement"
                },
                {
                    "name": "Identity Proof",
                    "description": "Aadhaar Card, Voter ID, Passport, or any other valid government-issued ID"
                },
                {
                    "name": "Address Proof",
                    "description": "Utility bills, Passport, Ration Card, or any other valid address proof"
                },
                {
                    "name": "Passport-sized Photographs",
                    "description": "Recent color photographs (typically 4-6 photos required)"
                },
                {
                    "name": "Category Certificate",
                    "description": "For SC/ST/OBC/EWS candidates, valid category certificate issued by competent authority"
                },
                {
                    "name": "Disability Certificate",
                    "description": "For PwD candidates, disability certificate issued by a recognized medical authority"
                },
                {
                    "name": "Domicile/Residency Certificate",
                    "description": "Proof of residency in the state where admission is sought"
                },
                {
                    "name": "Income Certificate",
                    "description": "For fee concessions or scholarships (if applicable)"
                },
                {
                    "name": "Migration Certificate",
                    "description": "If the candidate has studied outside the state"
                },
                {
                    "name": "Character Certificate",
                    "description": "From the last attended educational institution"
                }
            ],
            "fee_structure": {
                "government_itis": {
                    "admission_fee": "₹500-₹2,000 (varies by state and trade)",
                    "tuition_fee": "₹1,000-₹5,000 per semester/year (varies by state and trade)",
                    "caution_deposit": "₹500-₹1,000 (refundable)",
                    "exam_fee": "₹100-₹500 per semester/year",
                    "other_charges": "Library, development, identity card, etc. (varies by institute)"
                },
                "private_itis": {
                    "fee_range": "₹15,000-₹50,000 per year (varies by institute, location, and trade)"
                },
                "fee_concessions": {
                    "SC_ST": "Full or partial tuition fee waiver in many states",
                    "OBC": "Partial fee concession in some states",
                    "PwD": "Fee concessions as per state policy",
                    "EWS": "Fee concessions as per state policy",
                    "Girls": "Special fee concessions for female candidates in many states"
                }
            },
            "entrance_examination": {
                "exam_pattern": {
                    "sections": [
                        {
                            "name": "Mathematics",
                            "topics": ["Arithmetic", "Algebra", "Geometry", "Mensuration", "Trigonometry"],
                            "questions": 25,
                            "marks": 25
                        },
                        {
                            "name": "Science",
                            "topics": ["Physics", "Chemistry", "Biology (Basic)"],
                            "questions": 25,
                            "marks": 25
                        },
                        {
                            "name": "General Knowledge",
                            "topics": ["Current Affairs", "Static GK", "Logical Reasoning"],
                            "questions": 25,
                            "marks": 25
                        },
                        {
                            "name": "Language Comprehension",
                            "topics": ["Reading Comprehension", "Basic Grammar", "Vocabulary"],
                            "questions": 25,
                            "marks": 25
                        }
                    ],
                    "total_marks": 100,
                    "duration": "2 hours",
                    "negative_marking": "No negative marking in most states"
                },
                "preparation_tips": [
                    "Focus on 10th standard Mathematics and Science syllabus",
                    "Practice previous years' question papers",
                    "Strengthen basic arithmetic, algebra, and geometry concepts",
                    "Study fundamental physics and chemistry concepts",
                    "Stay updated with current affairs and general knowledge",
                    "Improve reading comprehension and language skills",
                    "Manage time effectively during the exam",
                    "Focus on accuracy rather than speed"
                ],
                "sample_questions": {
                    "mathematics": [
                        {
                            "question": "If a = 5 and b = 3, find the value of a² + b².",
                            "options": ["25", "34", "8", "64"],
                            "correct_answer": 1,
                            "explanation": "a² + b² = 5² + 3² = 25 + 9 = 34"
                        },
                        {
                            "question": "Find the area of a circle with radius 7 cm.",
                            "options": ["44 cm²", "49 cm²", "154 cm²", "22 cm²"],
                            "correct_answer": 2,
                            "explanation": "Area = πr² = 3.14 × 7² = 3.14 × 49 ≈ 154 cm²"
                        }
                    ],
                    "science": [
                        {
                            "question": "Which of the following is the unit of electric current?",
                            "options": ["Volt", "Watt", "Ampere", "Ohm"],
                            "correct_answer": 2,
                            "explanation": "Ampere is the SI unit of electric current."
                        },
                        {
                            "question": "What is the chemical formula for water?",
                            "options": ["H₂O", "CO₂", "O₂", "N₂"],
                            "correct_answer": 0,
                            "explanation": "Water has the chemical formula H₂O."
                        }
                    ]
                }
            },
            "trade_selection_guidance": {
                "factors_to_consider": [
                    "Personal interest and aptitude",
                    "Physical fitness and capabilities",
                    "Future job prospects and market demand",
                    "Salary potential in the industry",
                    "Opportunities for further education and career advancement",
                    "Geographical mobility and job locations"
                ],
                "popular_trades": {
                    "engineering": [
                        {
                            "name": "Electrician",
                            "eligibility": "10th pass with Science and Mathematics",
                            "duration": "2 years",
                            "job_prospects": "Excellent",
                            "key_skills_gained": "Electrical wiring, motor repair, basic electronics"
                        },
                        {
                            "name": "Fitter",
                            "eligibility": "10th pass with Science and Mathematics",
                            "duration": "2 years",
                            "job_prospects": "Very good",
                            "key_skills_gained": "Fitting, assembling, reading engineering drawings"
                        },
                        {
                            "name": "Mechanic (Motor Vehicle)",
                            "eligibility": "10th pass with Science and Mathematics",
                            "duration": "2 years",
                            "job_prospects": "Excellent",
                            "key_skills_gained": "Automobile repair, engine maintenance, diagnostics"
                        },
                        {
                            "name": "COPA (Computer Operator & Programming Assistant)",
                            "eligibility": "10th pass",
                            "duration": "1 year",
                            "job_prospects": "Good",
                            "key_skills_gained": "Computer operations, office applications, basic programming"
                        },
                        {
                            "name": "Welder",
                            "eligibility": "8th pass",
                            "duration": "1 year",
                            "job_prospects": "Very good",
                            "key_skills_gained": "Various welding techniques, metal joining"
                        }
                    ],
                    "non_engineering": [
                        {
                            "name": "Dress Making",
                            "eligibility": "8th pass",
                            "duration": "1 year",
                            "job_prospects": "Good",
                            "key_skills_gained": "Cutting, stitching, garment design"
                        },
                        {
                            "name": "Stenography",
                            "eligibility": "10th pass",
                            "duration": "1 year",
                            "job_prospects": "Moderate",
                            "key_skills_gained": "Shorthand, typing, office management"
                        },
                        {
                            "name": "Hair & Skin Care",
                            "eligibility": "8th pass",
                            "duration": "1 year",
                            "job_prospects": "Good",
                            "key_skills_gained": "Hair styling, skincare techniques, salon management"
                        }
                    ]
                }
            },
            "important_dates": {
                "application_start": form_start_date,
                "application_end": form_end_date,
                "entrance_exam": exam_date,
                "result_declaration": result_date,
                "counseling_start": counseling_start_date,
                "counseling_end": counseling_end_date,
                "session_commencement": session_start_date
            },
            "common_faqs": [
                {
                    "question": "What is the minimum educational qualification for ITI courses?",
                    "answer": "The minimum qualification varies by trade. For engineering trades, it's typically 10th pass with Science and Mathematics. For some non-engineering trades, 8th pass may be sufficient."
                },
                {
                    "question": "Can I pursue further education after completing an ITI course?",
                    "answer": "Yes, ITI graduates can pursue further education. They can appear for AITT (All India Trade Test) to get a National Trade Certificate. With this, they can seek admission in diploma courses through lateral entry or prepare for Advanced Vocational Training Scheme (AVTS). They can also appear for Senior Secondary (Class 12) examinations through National Institute of Open Schooling (NIOS)."
                },
                {
                    "question": "Are there any age restrictions for ITI admission?",
                    "answer": "The general age criterion is 14-40 years, though there may be variations across states. Age relaxation is provided for reserved categories: 5 years for SC/ST, 3 years for OBC, and 10 years for PwD candidates."
                },
                {
                    "question": "How can I choose the right trade for ITI admission?",
                    "answer": "Consider your personal interests, aptitude, physical capabilities, market demand for the trade, salary potential, and future career growth opportunities. Also, research the specific skills you'll gain in each trade and their relevance to your career goals."
                },
                {
                    "question": "What is the fee structure for ITI courses?",
                    "answer": "Fees vary significantly between government and private ITIs. Government ITIs have lower fees, ranging from ₹1,000-₹7,000 per year depending on the state and trade. Private ITIs can charge ₹15,000-₹50,000 per year. Many states offer fee concessions for SC/ST, OBC, PwD, EWS, and female candidates."
                },
                {
                    "question": "Is there an entrance exam for ITI admission?",
                    "answer": "It depends on the state. Some states conduct entrance examinations, while others admit students based on merit lists prepared using qualifying examination (8th/10th) marks. When conducted, entrance exams typically test Mathematics, Science, General Knowledge, and Language Comprehension."
                },
                {
                    "question": "What documents are required for ITI admission?",
                    "answer": "Required documents include educational certificates (8th/10th/12th), identity proof, address proof, passport-sized photographs, category certificate (if applicable), disability certificate (if applicable), domicile certificate, income certificate (if applicable), migration certificate (if from another state), and character certificate."
                },
                {
                    "question": "Can I transfer from one ITI to another?",
                    "answer": "Transfer between ITIs is possible but subject to seat availability in the desired ITI and approval from both institutes. The transfer process typically requires an application to the Directorate of Training or similar authority in your state."
                },
                {
                    "question": "What is the difference between NCVT and SCVT certification?",
                    "answer": "NCVT (National Council for Vocational Training) certification is nationally recognized and preferred for central government jobs, interstate employment, and international opportunities. SCVT (State Council for Vocational Training) certification is primarily recognized within the state and for state government jobs."
                },
                {
                    "question": "Are there any scholarships available for ITI students?",
                    "answer": "Yes, several scholarships are available for ITI students, including central government schemes through the National Scholarship Portal, state government scholarships, SC/ST/OBC scholarships, scholarships for women in technical education, and industry-sponsored scholarships. Eligibility varies by scheme."
                }
            ],
            "application_tips": [
                "Apply early to avoid last-minute technical issues or server overloads",
                "Double-check all information before final submission",
                "Keep digital copies of all required documents ready before starting the application",
                "Use a valid, regularly checked email address and phone number",
                "Take note of application number/registration ID for future reference",
                "Pay application fees through recommended payment methods only",
                "Keep printed copies of the completed application form and payment receipt",
                "Regularly check the official website for updates on your application status",
                "Select multiple trade preferences to increase chances of admission",
                "Research institutes thoroughly before selecting preferences"
            ],
            "state_specific_information": {
                "note": "ITI admission processes may vary by state. The information provided here is general. For state-specific details, please visit the official website of the State Directorate of Technical Education or Employment."
            }
        }
    
    def _save_admission_data(self, data: Dict) -> None:
        """Save admission data to file.
        
        Args:
            data: Admission data to save
        """
        try:
            with open(self.admission_data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save admission data: {e}{Style.RESET_ALL}")
    
    def get_general_information(self) -> Dict:
        """Get general information about ITI courses.
        
        Returns:
            Dictionary containing general information
        """
        return self.admission_data.get("general_information", {})
    
    def get_eligibility_criteria(self) -> Dict:
        """Get eligibility criteria for ITI admission.
        
        Returns:
            Dictionary containing eligibility criteria
        """
        return self.admission_data.get("eligibility_criteria", {})
    
    def get_admission_process(self) -> Dict:
        """Get step-by-step admission process details.
        
        Returns:
            Dictionary containing admission process steps
        """
        return self.admission_data.get("admission_process", {})
    
    def get_required_documents(self) -> List[Dict]:
        """Get list of required documents for admission.
        
        Returns:
            List of required documents
        """
        return self.admission_data.get("required_documents", [])
    
    def get_fee_structure(self) -> Dict:
        """Get fee structure details for different ITIs.
        
        Returns:
            Dictionary containing fee structure information
        """
        return self.admission_data.get("fee_structure", {})
    
    def get_entrance_examination_details(self) -> Dict:
        """Get details about the entrance examination.
        
        Returns:
            Dictionary containing entrance examination details
        """
        return self.admission_data.get("entrance_examination", {})
    
    def get_trade_selection_guidance(self) -> Dict:
        """Get guidance for selecting appropriate trades.
        
        Returns:
            Dictionary containing trade selection guidance
        """
        return self.admission_data.get("trade_selection_guidance", {})
    
    def get_important_dates(self) -> Dict:
        """Get important dates for the admission cycle.
        
        Returns:
            Dictionary containing important dates
        """
        return self.admission_data.get("important_dates", {})
    
    def get_common_faqs(self) -> List[Dict]:
        """Get common frequently asked questions about ITI admission.
        
        Returns:
            List of FAQ dictionaries
        """
        return self.admission_data.get("common_faqs", [])
    
    def get_application_tips(self) -> List[str]:
        """Get tips for the application process.
        
        Returns:
            List of application tips
        """
        return self.admission_data.get("application_tips", [])
    
    def get_eligibility_by_trade(self, trade_name: str) -> Optional[str]:
        """Get eligibility requirements for a specific trade.
        
        Args:
            trade_name: Name of the ITI trade
            
        Returns:
            Eligibility requirements as a string, or None if trade not found
        """
        popular_trades = self.admission_data.get("trade_selection_guidance", {}).get("popular_trades", {})
        
        # Search in engineering trades
        engineering_trades = popular_trades.get("engineering", [])
        for trade in engineering_trades:
            if trade.get("name", "").lower() == trade_name.lower():
                return trade.get("eligibility", "Not specified")
        
        # Search in non-engineering trades
        non_engineering_trades = popular_trades.get("non_engineering", [])
        for trade in non_engineering_trades:
            if trade.get("name", "").lower() == trade_name.lower():
                return trade.get("eligibility", "Not specified")
        
        return None
    
    def get_admission_timeline(self) -> List[Dict]:
        """Get timeline of the admission process.
        
        Returns:
            List of timeline events in chronological order
        """
        important_dates = self.admission_data.get("important_dates", {})
        
        timeline = [
            {
                "event": "Application Period",
                "start_date": important_dates.get("application_start"),
                "end_date": important_dates.get("application_end"),
                "description": "Period for submitting application forms"
            },
            {
                "event": "Entrance Examination",
                "date": important_dates.get("entrance_exam"),
                "description": "Date of entrance examination"
            },
            {
                "event": "Result Declaration",
                "date": important_dates.get("result_declaration"),
                "description": "Announcement of entrance examination results"
            },
            {
                "event": "Counseling Period",
                "start_date": important_dates.get("counseling_start"),
                "end_date": important_dates.get("counseling_end"),
                "description": "Period for counseling and seat allocation"
            },
            {
                "event": "Session Commencement",
                "date": important_dates.get("session_commencement"),
                "description": "Beginning of classes for new batch"
            }
        ]
        
        # Sort timeline by dates
        def get_sort_date(event):
            if "start_date" in event:
                return event["start_date"]
            return event.get("date", "9999-12-31")  # Default to far future if no date
            
        timeline.sort(key=get_sort_date)
        
        return timeline
    
    def check_trade_eligibility(self, education_level: str, has_science_math: bool = False) -> List[Dict]:
        """Check which trades a student is eligible for based on education.
        
        Args:
            education_level: Education level ("8th pass", "10th pass", "12th pass")
            has_science_math: Whether the student has studied Science and Mathematics
            
        Returns:
            List of eligible trades with details
        """
        eligible_trades = []
        
        popular_trades = self.admission_data.get("trade_selection_guidance", {}).get("popular_trades", {})
        all_trades = popular_trades.get("engineering", []) + popular_trades.get("non_engineering", [])
        
        for trade in all_trades:
            trade_eligibility = trade.get("eligibility", "").lower()
            
            if "8th pass" in trade_eligibility and education_level in ["8th pass", "10th pass", "12th pass"]:
                eligible_trades.append(trade)
            elif "10th pass" in trade_eligibility and education_level in ["10th pass", "12th pass"]:
                if "science and mathematics" in trade_eligibility and not has_science_math:
                    continue  # Skip this trade if science and math required but student doesn't have it
                eligible_trades.append(trade)
            elif "12th pass" in trade_eligibility and education_level == "12th pass":
                eligible_trades.append(trade)
                
        return eligible_trades
    
    def get_document_checklist(self, category: str = "general") -> List[Dict]:
        """Get document checklist for admission process.
        
        Args:
            category: Category of the student (general, SC, ST, OBC, PwD, etc.)
            
        Returns:
            List of required documents with descriptions
        """
        all_documents = self.admission_data.get("required_documents", [])
        required_documents = []
        
        for document in all_documents:
            # Add common documents for all categories
            if document["name"] in ["Educational Certificates", "Identity Proof", "Address Proof", "Passport-sized Photographs", "Domicile/Residency Certificate", "Character Certificate"]:
                document_copy = document.copy()
                document_copy["priority"] = "Essential"
                required_documents.append(document_copy)
                
        # Add category-specific documents
        if category.lower() in ["sc", "st", "obc", "ews"]:
            for document in all_documents:
                if document["name"] == "Category Certificate":
                    document_copy = document.copy()
                    document_copy["priority"] = "Essential"
                    required_documents.append(document_copy)
                    break
                    
        if category.lower() == "pwd":
            for document in all_documents:
                if document["name"] == "Disability Certificate":
                    document_copy = document.copy()
                    document_copy["priority"] = "Essential"
                    required_documents.append(document_copy)
                    break
                    
        # Add other optional documents
        for document in all_documents:
            if document["name"] in ["Income Certificate", "Migration Certificate"]:
                document_copy = document.copy()
                document_copy["priority"] = "If Applicable"
                required_documents.append(document_copy)
                
        return required_documents
    
    def get_application_form_guide(self) -> Dict:
        """Get guide for filling out the application form.
        
        Returns:
            Dictionary with application form guidelines
        """
        return {
            "sections": [
                {
                    "name": "Personal Information",
                    "fields": ["Name", "Date of Birth", "Gender", "Category", "Nationality", "Aadhaar Number"],
                    "tips": "Ensure name and other details match exactly with your educational certificates."
                },
                {
                    "name": "Contact Information",
                    "fields": ["Address", "Mobile Number", "Email Address", "Guardian's Contact Details"],
                    "tips": "Provide a valid mobile number and email address that you check regularly."
                },
                {
                    "name": "Educational Qualifications",
                    "fields": ["Qualifying Examination", "Board/University", "Year of Passing", "Marks/Percentage", "Subjects Studied"],
                    "tips": "Have your marksheets ready for reference. Be accurate with marks and percentages."
                },
                {
                    "name": "Trade Preferences",
                    "fields": ["First Preference", "Second Preference", "Third Preference"],
                    "tips": "Research trades thoroughly before selecting preferences. Consider multiple backup options."
                },
                {
                    "name": "Institute Preferences",
                    "fields": ["First Preference", "Second Preference", "Third Preference"],
                    "tips": "Consider factors like location, facilities, placement record, and available trades."
                },
                {
                    "name": "Document Upload",
                    "fields": ["Photo", "Signature", "Required Documents"],
                    "tips": "Ensure documents are clearly scanned and within the specified file size (typically 50-300 KB)."
                },
                {
                    "name": "Declaration",
                    "fields": ["Acceptance of Terms and Conditions"],
                    "tips": "Read the declaration carefully before accepting."
                }
            ],
            "common_mistakes": [
                "Leaving mandatory fields blank",
                "Providing incorrect personal details",
                "Uploading unclear or invalid documents",
                "Selecting trades for which the candidate is not eligible",
                "Submitting multiple applications",
                "Making spelling errors in name or other critical fields",
                "Providing incorrect examination details",
                "Not completing the application process after payment"
            ],
            "important_notes": [
                "Fill the application form in one sitting if possible",
                "Keep all required documents and information ready before starting",
                "Double-check all information before final submission",
                "Take printouts of the completed application form and payment receipt",
                "Note down the application number/registration ID for future reference",
                "Check email and SMS regularly for updates on your application"
            ]
        }
    
    def update_important_dates(self, dates: Dict) -> bool:
        """Update important dates for the admission process.
        
        Args:
            dates: Dictionary containing updated dates
            
        Returns:
            Boolean indicating success
        """
        try:
            # Update important dates
            self.admission_data["important_dates"].update(dates)
            
            # Update dates in admission process steps
            steps = self.admission_data.get("admission_process", {}).get("steps", [])
            
            for step in steps:
                if step["name"] == "Application Form Submission" and "application_start" in dates and "application_end" in dates:
                    step["start_date"] = dates["application_start"]
                    step["end_date"] = dates["application_end"]
                elif step["name"] == "Entrance Examination/Merit List" and "entrance_exam" in dates and "result_declaration" in dates:
                    step["exam_date"] = dates["entrance_exam"]
                    step["result_date"] = dates["result_declaration"]
                elif step["name"] == "Counseling and Seat Allotment" and "counseling_start" in dates and "counseling_end" in dates:
                    step["start_date"] = dates["counseling_start"]
                    step["end_date"] = dates["counseling_end"]
                elif step["name"] == "Admission Confirmation and Course Commencement" and "session_commencement" in dates:
                    step["session_start_date"] = dates["session_commencement"]
            
            # Save updated data
            self._save_admission_data(self.admission_data)
            return True
        except Exception as e:
            print(f"{Fore.YELLOW}Could not update important dates: {e}{Style.RESET_ALL}")
            return False
    
    def get_private_vs_government_comparison(self) -> Dict:
        """Get comparison between private and government ITIs.
        
        Returns:
            Dictionary with comparison details
        """
        return {
            "government_itis": {
                "pros": [
                    "Lower fees and affordable education",
                    "Nationally recognized certification (NCVT)",
                    "Better scholarship opportunities",
                    "Established reputation and credibility",
                    "Better industry connections for placements"
                ],
                "cons": [
                    "Limited seats and high competition",
                    "May have older infrastructure in some institutes",
                    "Less flexibility in curriculum",
                    "May lack specialized or niche trades"
                ],
                "average_annual_fee": "₹1,000-₹7,000 (varies by state and trade)"
            },
            "private_itis": {
                "pros": [
                    "More seats and easier admission",
                    "May offer newer trades and specializations",
                    "Often newer infrastructure and facilities",
                    "Flexible timing options in some institutes",
                    "May provide additional training beyond syllabus"
                ],
                "cons": [
                    "Higher fees compared to government ITIs",
                    "Variable quality and reputation",
                    "May not always have NCVT affiliation",
                    "Less established connections with industries"
                ],
                "average_annual_fee": "₹15,000-₹50,000 (varies by institute and trade)"
            },
            "key_factors_to_consider": [
                "Affiliation status (NCVT/SCVT)",
                "Infrastructure and training facilities",
                "Faculty qualifications and experience",
                "Placement record and industry tie-ups",
                "Availability of desired trade",
                "Location and accessibility",
                "Fee structure and affordability",
                "Student reviews and alumni success"
            ]
        }
    
    def search_faqs(self, query: str) -> List[Dict]:
        """Search frequently asked questions for a specific query.
        
        Args:
            query: Search query
            
        Returns:
            List of matching FAQ dictionaries
        """
        query_lower = query.lower()
        results = []
        
        for faq in self.admission_data.get("common_faqs", []):
            question = faq.get("question", "").lower()
            answer = faq.get("answer", "").lower()
            
            if query_lower in question or query_lower in answer:
                results.append(faq)
                
        return results 