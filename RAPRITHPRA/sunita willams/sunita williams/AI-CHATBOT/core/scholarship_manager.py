"""
Scholarship Information Module for ITI Assistant.

This module provides comprehensive information about scholarships, grants, and financial aid
available to ITI students, including eligibility criteria, application processes, and deadlines.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from colorama import Fore, Style

class ScholarshipManager:
    """Manages scholarship information for ITI students."""
    
    def __init__(self, scholarships_path="data/scholarships.json"):
        """Initialize the scholarship manager.
        
        Args:
            scholarships_path: Path to scholarships data JSON file
        """
        self.scholarships_path = scholarships_path
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(scholarships_path), exist_ok=True)
        
        # Load scholarship data
        self.scholarships = self._load_scholarships()
        
        print(f"{Fore.GREEN}✓ Scholarship manager initialized{Style.RESET_ALL}")
        
    def _load_scholarships(self) -> Dict:
        """Load scholarship data from file or initialize with default data."""
        try:
            if os.path.exists(self.scholarships_path):
                with open(self.scholarships_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Initialize with default scholarship data
                default_scholarships = self._initialize_default_scholarships()
                self._save_scholarships(default_scholarships)
                return default_scholarships
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load scholarships: {e}{Style.RESET_ALL}")
            default_scholarships = self._initialize_default_scholarships()
            return default_scholarships
    
    def _initialize_default_scholarships(self) -> Dict:
        """Initialize default scholarship data."""
        today = datetime.now()
        
        # Calculate future dates for application deadlines
        next_month = today + timedelta(days=30)
        two_months = today + timedelta(days=60)
        three_months = today + timedelta(days=90)
        
        return {
            "government_scholarships": [
                {
                    "name": "National Scholarship Portal (NSP) Scholarships",
                    "provider": "Ministry of Skill Development and Entrepreneurship",
                    "description": "Centralized scholarship program for ITI students from economically weaker sections.",
                    "eligibility": [
                        "Family income less than ₹2.5 lakh per annum",
                        "Minimum 75% attendance",
                        "Regular ITI student in government recognized institute"
                    ],
                    "benefits": [
                        "Tuition fee reimbursement",
                        "Monthly stipend of ₹1000-₹1500",
                        "One-time book allowance"
                    ],
                    "documents_required": [
                        "Income certificate",
                        "Caste certificate (if applicable)",
                        "Aadhaar card",
                        "Bank account details",
                        "Latest marksheet",
                        "Admission letter from ITI"
                    ],
                    "application_process": "Apply online through National Scholarship Portal (www.scholarships.gov.in)",
                    "deadline": next_month.strftime("%Y-%m-%d"),
                    "website": "https://scholarships.gov.in",
                    "contact": "helpdesk@nsp.gov.in",
                    "success_rate": "70% of eligible applicants receive the scholarship"
                },
                {
                    "name": "Post-Matric Scholarship for SC/ST Students",
                    "provider": "Ministry of Social Justice and Empowerment",
                    "description": "Financial assistance for SC/ST students pursuing ITI courses.",
                    "eligibility": [
                        "Belong to SC/ST category",
                        "Family income below ₹2.5 lakh per annum",
                        "Enrolled in recognized ITI"
                    ],
                    "benefits": [
                        "Full tuition fee reimbursement",
                        "Monthly maintenance allowance of ₹1200",
                        "Annual book grant of ₹1500"
                    ],
                    "documents_required": [
                        "Caste certificate",
                        "Income certificate",
                        "Previous education certificates",
                        "Bank account linked with Aadhaar",
                        "Passport size photographs",
                        "ITI admission proof"
                    ],
                    "application_process": "Apply through National Scholarship Portal or state-specific portals",
                    "deadline": two_months.strftime("%Y-%m-%d"),
                    "website": "https://scholarships.gov.in",
                    "contact": "scsthelpdesk@gov.in",
                    "success_rate": "85% of eligible applicants receive the scholarship"
                },
                {
                    "name": "Prime Minister's Scholarship Scheme for ITI Students",
                    "provider": "Ministry of Skill Development",
                    "description": "Special scholarship for students pursuing high-demand trades in ITIs.",
                    "eligibility": [
                        "Enrolled in recognized ITI",
                        "Pursuing courses in high-demand sectors (manufacturing, healthcare, automotive)",
                        "Minimum 70% marks in 10th standard",
                        "Family income below ₹8 lakh per annum"
                    ],
                    "benefits": [
                        "Tuition fee waiver up to ₹20,000 per year",
                        "Monthly stipend of ₹2000",
                        "One-time tool kit allowance of ₹5000"
                    ],
                    "documents_required": [
                        "Income certificate",
                        "10th marksheet",
                        "Aadhaar card",
                        "Domicile certificate",
                        "ITI admission letter",
                        "Passport size photographs"
                    ],
                    "application_process": "Apply online through Skill Development Ministry portal or through ITI institute",
                    "deadline": three_months.strftime("%Y-%m-%d"),
                    "website": "https://msde.gov.in/en/schemes-initiatives",
                    "contact": "pmss.helpdesk@gov.in",
                    "success_rate": "60% of eligible applicants receive the scholarship"
                }
            ],
            "state_scholarships": [
                {
                    "name": "State Merit Scholarship for ITI Students",
                    "provider": "State Directorate of Technical Education",
                    "description": "Merit-based scholarship for top-performing ITI students in each state.",
                    "eligibility": [
                        "Top 10% rank in ITI entrance exam or previous semester",
                        "Minimum 85% attendance",
                        "State resident"
                    ],
                    "benefits": [
                        "₹10,000 per semester",
                        "Recognition certificate",
                        "Priority in industry placements"
                    ],
                    "documents_required": [
                        "Domicile certificate",
                        "ITI ID card",
                        "Previous semester marksheet",
                        "Bank account details",
                        "Aadhaar card"
                    ],
                    "application_process": "Apply through respective state technical education portal or through ITI institute",
                    "deadline": "Varies by state, check with local ITI",
                    "website": "Check respective state technical education website",
                    "contact": "Contact state directorate of technical education",
                    "success_rate": "Top 10% students in each trade receive the scholarship"
                },
                {
                    "name": "Minority Community Scholarship for ITI",
                    "provider": "State Minority Welfare Department",
                    "description": "Financial assistance for students from minority communities pursuing ITI courses.",
                    "eligibility": [
                        "Belong to notified minority community",
                        "Family income below ₹3 lakh per annum",
                        "Enrolled in recognized ITI",
                        "Minimum 60% marks in previous education"
                    ],
                    "benefits": [
                        "Tuition fee reimbursement up to ₹15,000",
                        "Maintenance allowance of ₹1000 per month",
                        "Book and stationery allowance"
                    ],
                    "documents_required": [
                        "Minority community certificate",
                        "Income certificate",
                        "Previous education certificates",
                        "Bank account details",
                        "ITI admission proof",
                        "Passport size photographs"
                    ],
                    "application_process": "Apply through State Minority Welfare Department or online state scholarship portal",
                    "deadline": next_month.strftime("%Y-%m-%d"),
                    "website": "Check respective state minority welfare department website",
                    "contact": "Contact state minority welfare department",
                    "success_rate": "75% of eligible applicants receive the scholarship"
                }
            ],
            "industry_scholarships": [
                {
                    "name": "Tata Motors ITI Scholarship Program",
                    "provider": "Tata Motors",
                    "description": "Industry-sponsored scholarship for ITI students pursuing automobile related trades.",
                    "eligibility": [
                        "Enrolled in automobile related trades (Mechanic Motor Vehicle, Electrician, Machinist)",
                        "Minimum 70% marks in 10th standard",
                        "Family income below ₹5 lakh per annum",
                        "ITI located in operational areas of Tata Motors"
                    ],
                    "benefits": [
                        "Full tuition fee coverage",
                        "Monthly stipend of ₹2500",
                        "On-job training opportunities",
                        "Priority in placement"
                    ],
                    "documents_required": [
                        "Income certificate",
                        "10th marksheet",
                        "ITI admission proof",
                        "Aadhaar card",
                        "Recommendation letter from ITI principal",
                        "Bank account details"
                    ],
                    "application_process": "Apply through ITI institution or directly to Tata Motors CSR division",
                    "deadline": two_months.strftime("%Y-%m-%d"),
                    "website": "https://www.tatamotors.com/csr/",
                    "contact": "education.csr@tatamotors.com",
                    "success_rate": "50% of applicants receive the scholarship"
                },
                {
                    "name": "Maruti Suzuki Technical Scholarship",
                    "provider": "Maruti Suzuki India Limited",
                    "description": "Scholarship for promising ITI students in automotive and allied trades.",
                    "eligibility": [
                        "Pursuing Mechanic Motor Vehicle, Mechanic Auto Electrical & Electronics trades",
                        "Minimum 65% marks in 10th standard",
                        "ITIs located near Maruti Suzuki facilities",
                        "Demonstrated technical aptitude"
                    ],
                    "benefits": [
                        "Tuition fee support up to ₹20,000 per year",
                        "Monthly allowance of ₹1500",
                        "Internship opportunity at Maruti Suzuki facilities",
                        "Skill development workshops"
                    ],
                    "documents_required": [
                        "10th marksheet",
                        "ITI admission letter",
                        "Aadhaar card",
                        "Passport size photographs",
                        "Bank account details",
                        "Technical aptitude test score"
                    ],
                    "application_process": "Apply through ITI institution or Maruti Suzuki authorized center",
                    "deadline": three_months.strftime("%Y-%m-%d"),
                    "website": "https://www.marutisuzuki.com/corporate/csr",
                    "contact": "technical.scholarship@maruti.co.in",
                    "success_rate": "40% of applicants receive the scholarship"
                },
                {
                    "name": "Larsen & Toubro Construction Skills Scholarship",
                    "provider": "L&T Construction",
                    "description": "Scholarship for ITI students in construction-related trades.",
                    "eligibility": [
                        "Pursuing Civil, Electrical, or Plumbing trades",
                        "Economically disadvantaged background",
                        "Minimum 60% marks in previous education",
                        "Good physical fitness"
                    ],
                    "benefits": [
                        "Full tuition fee coverage",
                        "Monthly stipend of ₹3000",
                        "On-site training at L&T projects",
                        "Tool kit worth ₹10,000",
                        "Employment opportunity after course completion"
                    ],
                    "documents_required": [
                        "Income certificate",
                        "Previous education certificates",
                        "ITI admission proof",
                        "Aadhaar card",
                        "Medical fitness certificate",
                        "Bank account details"
                    ],
                    "application_process": "Apply through ITI institution or L&T Construction Skills Training Centers",
                    "deadline": next_month.strftime("%Y-%m-%d"),
                    "website": "https://www.lntecc.com/csr-initiatives/",
                    "contact": "skillscholarship@lntecc.com",
                    "success_rate": "60% of eligible applicants receive the scholarship"
                }
            ],
            "special_category_scholarships": [
                {
                    "name": "Scholarship for Women in Technical Education",
                    "provider": "Ministry of Women and Child Development",
                    "description": "Special scholarship to encourage women to pursue ITI courses and technical education.",
                    "eligibility": [
                        "Female candidates enrolled in any ITI trade",
                        "Minimum 60% marks in 10th standard",
                        "Family income below ₹6 lakh per annum"
                    ],
                    "benefits": [
                        "Tuition fee reimbursement up to ₹25,000 per year",
                        "Monthly stipend of ₹2000",
                        "One-time contingency grant of ₹5000",
                        "Career counseling and mentorship"
                    ],
                    "documents_required": [
                        "10th marksheet",
                        "Income certificate",
                        "ITI admission proof",
                        "Aadhaar card",
                        "Bank account details",
                        "Passport size photographs"
                    ],
                    "application_process": "Apply online through National Scholarship Portal or Women and Child Development portal",
                    "deadline": two_months.strftime("%Y-%m-%d"),
                    "website": "https://wcd.nic.in/schemes-listing/2405",
                    "contact": "women.tech@wcd.gov.in",
                    "success_rate": "80% of eligible applicants receive the scholarship"
                },
                {
                    "name": "Divyangjan ITI Scholarship Scheme",
                    "provider": "Department of Empowerment of Persons with Disabilities",
                    "description": "Scholarship for differently-abled students pursuing ITI courses.",
                    "eligibility": [
                        "Persons with disabilities (40% or more disability)",
                        "Enrolled in recognized ITI",
                        "Family income below ₹3 lakh per annum"
                    ],
                    "benefits": [
                        "Full tuition fee waiver",
                        "Monthly maintenance allowance of ₹2500",
                        "Reader allowance for visually impaired students",
                        "Escort allowance for severely disabled students",
                        "Special equipment allowance"
                    ],
                    "documents_required": [
                        "Disability certificate",
                        "Income certificate",
                        "ITI admission proof",
                        "Aadhaar card",
                        "Bank account details",
                        "Previous education certificates"
                    ],
                    "application_process": "Apply through National Scholarship Portal or Department of Empowerment of Persons with Disabilities",
                    "deadline": next_month.strftime("%Y-%m-%d"),
                    "website": "https://disabilityaffairs.gov.in/content/page/scholarships.php",
                    "contact": "scholarship-depwd@gov.in",
                    "success_rate": "90% of eligible applicants receive the scholarship"
                },
                {
                    "name": "Scholarship for Wards of Construction Workers",
                    "provider": "Construction Workers Welfare Board",
                    "description": "Scholarship for children of registered construction workers pursuing ITI courses.",
                    "eligibility": [
                        "Parent/guardian registered with Construction Workers Welfare Board for at least 1 year",
                        "Enrolled in any ITI trade",
                        "Minimum 55% marks in previous education"
                    ],
                    "benefits": [
                        "Tuition fee reimbursement up to ₹20,000 per year",
                        "Monthly stipend of ₹1500",
                        "Tool kit allowance of ₹5000",
                        "One-time travel allowance"
                    ],
                    "documents_required": [
                        "Parent's Construction Worker registration certificate",
                        "Relationship proof with registered worker",
                        "ITI admission proof",
                        "Previous education certificates",
                        "Aadhaar card",
                        "Bank account details"
                    ],
                    "application_process": "Apply through State Construction Workers Welfare Board office or authorized ITI nodal officer",
                    "deadline": "Rolling applications, check with local welfare board",
                    "website": "Check respective state construction workers welfare board website",
                    "contact": "Contact state construction workers welfare board",
                    "success_rate": "75% of eligible applicants receive the scholarship"
                }
            ],
            "international_scholarships": [
                {
                    "name": "Indo-German Dual Vocational Training Scholarship",
                    "provider": "German Society for International Cooperation (GIZ)",
                    "description": "Scholarship for exceptional ITI students to undergo dual training in Germany.",
                    "eligibility": [
                        "Top 5% performers in specified ITI trades",
                        "Excellent communication skills",
                        "Minimum 70% marks in ITI coursework",
                        "Basic understanding of German language (preferred)",
                        "Age between 18-25 years"
                    ],
                    "benefits": [
                        "Fully funded training in Germany for 3-6 months",
                        "International certification",
                        "Monthly stipend of €750",
                        "Return airfare",
                        "Health insurance",
                        "Accommodation assistance"
                    ],
                    "documents_required": [
                        "ITI performance certificate",
                        "Recommendation letter from ITI principal",
                        "Valid passport",
                        "Medical fitness certificate",
                        "Statement of purpose",
                        "Language proficiency certificate (if available)"
                    ],
                    "application_process": "Apply through Ministry of Skill Development or GIZ India office",
                    "deadline": three_months.strftime("%Y-%m-%d"),
                    "website": "https://www.giz.de/en/worldwide/368.html",
                    "contact": "indo-german.scholarship@giz.de",
                    "success_rate": "Highly competitive, approximately 20 scholarships awarded annually"
                }
            ]
        }
    
    def _save_scholarships(self, scholarships: Dict) -> None:
        """Save scholarship data to file.
        
        Args:
            scholarships: Scholarship data to save
        """
        try:
            with open(self.scholarships_path, 'w', encoding='utf-8') as f:
                json.dump(scholarships, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save scholarships: {e}{Style.RESET_ALL}")
    
    def get_all_scholarships(self) -> Dict:
        """Get all available scholarships.
        
        Returns:
            Dictionary containing all scholarship categories and details
        """
        return self.scholarships
    
    def get_scholarships_by_category(self, category: str) -> List[Dict]:
        """Get scholarships filtered by category.
        
        Args:
            category: Scholarship category (e.g., "government_scholarships", "state_scholarships")
            
        Returns:
            List of scholarships in the specified category
        """
        return self.scholarships.get(category, [])
    
    def get_scholarship_by_name(self, name: str) -> Optional[Dict]:
        """Find a specific scholarship by name.
        
        Args:
            name: Name of the scholarship to find
            
        Returns:
            Scholarship details if found, None otherwise
        """
        name_lower = name.lower()
        
        for category, scholarships in self.scholarships.items():
            for scholarship in scholarships:
                if scholarship["name"].lower() == name_lower:
                    return scholarship
        
        return None
    
    def get_eligible_scholarships(self, student_profile: Dict) -> List[Dict]:
        """Find scholarships that a student might be eligible for based on their profile.
        
        Args:
            student_profile: Dictionary containing student information
                - income: Family annual income in rupees
                - caste: SC/ST/OBC/General
                - gender: Male/Female/Other
                - marks_10th: 10th standard marks percentage
                - trade: ITI trade being pursued
                - state: State of residence
                - disability: Disability percentage if applicable
                - minority: True if belongs to minority community
                
        Returns:
            List of potentially eligible scholarships
        """
        eligible_scholarships = []
        
        # Extract profile details with defaults
        income = student_profile.get("income", float("inf"))
        caste = student_profile.get("caste", "").lower()
        gender = student_profile.get("gender", "").lower()
        marks_10th = student_profile.get("marks_10th", 0)
        trade = student_profile.get("trade", "").lower()
        state = student_profile.get("state", "")
        disability = student_profile.get("disability", 0)
        minority = student_profile.get("minority", False)
        
        # Helper function to check basic eligibility
        def is_potentially_eligible(scholarship: Dict) -> bool:
            # Check income criteria if present in eligibility
            for criterion in scholarship.get("eligibility", []):
                if "family income" in criterion.lower():
                    # Extract income limit from the criterion
                    import re
                    income_matches = re.findall(r'₹(\d+(?:\.\d+)?)\s*(?:lakh|lac)', criterion.lower())
                    if income_matches:
                        income_limit = float(income_matches[0]) * 100000  # Convert lakhs to rupees
                        if income > income_limit:
                            return False
            
            # Check caste-specific scholarships
            if "sc/st" in " ".join(scholarship.get("eligibility", [])).lower() and caste not in ["sc", "st"]:
                return False
                
            # Check gender-specific scholarships
            if "female" in " ".join(scholarship.get("eligibility", [])).lower() and gender != "female":
                return False
                
            # Check disability-specific scholarships
            if "disability" in " ".join(scholarship.get("eligibility", [])).lower() and disability < 40:
                return False
                
            # Check minority-specific scholarships
            if "minority" in " ".join(scholarship.get("eligibility", [])).lower() and not minority:
                return False
                
            # Check marks criteria
            for criterion in scholarship.get("eligibility", []):
                if "marks" in criterion.lower():
                    import re
                    marks_matches = re.findall(r'minimum\s+(\d+)%', criterion.lower())
                    if marks_matches:
                        required_marks = int(marks_matches[0])
                        if marks_10th < required_marks:
                            return False
            
            # Check trade-specific scholarships
            trade_mentioned = False
            for criterion in scholarship.get("eligibility", []):
                if "trade" in criterion.lower() or "course" in criterion.lower():
                    trade_mentioned = True
                    # If specific trades are mentioned, check if student's trade is included
                    if trade and any(t.lower() in criterion.lower() for t in ["mechanic", "electrician", "fitter", "plumbing", "civil"]):
                        if trade not in criterion.lower():
                            return False
            
            return True
        
        # Check each scholarship
        for category, scholarships in self.scholarships.items():
            for scholarship in scholarships:
                if is_potentially_eligible(scholarship):
                    scholarship_with_category = scholarship.copy()
                    scholarship_with_category["category"] = category
                    eligible_scholarships.append(scholarship_with_category)
        
        return eligible_scholarships
    
    def get_upcoming_deadlines(self, days: int = 30) -> List[Dict]:
        """Get scholarships with application deadlines in the next specified number of days.
        
        Args:
            days: Number of days to look ahead
            
        Returns:
            List of scholarships with upcoming deadlines
        """
        upcoming = []
        today = datetime.now().date()
        cutoff_date = today + timedelta(days=days)
        
        for category, scholarships in self.scholarships.items():
            for scholarship in scholarships:
                deadline = scholarship.get("deadline", "")
                if deadline and deadline != "Varies by state, check with local ITI" and deadline != "Rolling applications, check with local welfare board":
                    try:
                        deadline_date = datetime.strptime(deadline, "%Y-%m-%d").date()
                        if today <= deadline_date <= cutoff_date:
                            scholarship_info = {
                                "name": scholarship["name"],
                                "provider": scholarship["provider"],
                                "deadline": deadline,
                                "days_remaining": (deadline_date - today).days,
                                "category": category
                            }
                            upcoming.append(scholarship_info)
                    except ValueError:
                        # Skip scholarships with non-standard date formats
                        continue
        
        # Sort by days remaining
        upcoming.sort(key=lambda x: x["days_remaining"])
        
        return upcoming
    
    def get_scholarship_stats(self) -> Dict:
        """Get statistics about available scholarships.
        
        Returns:
            Dictionary with scholarship statistics
        """
        stats = {
            "total_scholarships": 0,
            "by_category": {},
            "by_income_range": {
                "below_3_lakh": 0,
                "3_to_5_lakh": 0,
                "5_to_8_lakh": 0,
                "above_8_lakh": 0,
                "not_specified": 0
            },
            "by_special_category": {
                "women": 0,
                "sc_st": 0,
                "disability": 0,
                "minority": 0
            }
        }
        
        # Count scholarships by category
        for category, scholarships in self.scholarships.items():
            stats["total_scholarships"] += len(scholarships)
            stats["by_category"][category] = len(scholarships)
            
            # Analyze each scholarship
            for scholarship in scholarships:
                eligibility_text = " ".join(scholarship.get("eligibility", [])).lower()
                
                # Count by income range
                if "income below ₹3 lakh" in eligibility_text or "income less than ₹3 lakh" in eligibility_text:
                    stats["by_income_range"]["below_3_lakh"] += 1
                elif "income below ₹5 lakh" in eligibility_text or "income less than ₹5 lakh" in eligibility_text:
                    stats["by_income_range"]["3_to_5_lakh"] += 1
                elif "income below ₹8 lakh" in eligibility_text or "income less than ₹8 lakh" in eligibility_text:
                    stats["by_income_range"]["5_to_8_lakh"] += 1
                elif "income" in eligibility_text and "lakh" in eligibility_text:
                    # Try to detect other income limits
                    import re
                    income_matches = re.findall(r'income\s+(?:below|less than)\s+₹(\d+(?:\.\d+)?)\s*lakh', eligibility_text)
                    if income_matches and float(income_matches[0]) > 8:
                        stats["by_income_range"]["above_8_lakh"] += 1
                    else:
                        stats["by_income_range"]["not_specified"] += 1
                else:
                    stats["by_income_range"]["not_specified"] += 1
                
                # Count by special category
                if "female" in eligibility_text or "women" in eligibility_text:
                    stats["by_special_category"]["women"] += 1
                if "sc" in eligibility_text or "st" in eligibility_text or "scheduled" in eligibility_text:
                    stats["by_special_category"]["sc_st"] += 1
                if "disability" in eligibility_text or "differently-abled" in eligibility_text or "divyang" in eligibility_text:
                    stats["by_special_category"]["disability"] += 1
                if "minority" in eligibility_text:
                    stats["by_special_category"]["minority"] += 1
        
        return stats
    
    def get_application_guide(self, scholarship_name: str = None) -> Dict:
        """Get application guide for a specific scholarship or general application tips.
        
        Args:
            scholarship_name: Optional name of the scholarship
            
        Returns:
            Dictionary with application guide information
        """
        # General application tips
        general_tips = [
            "Start the application process well before the deadline",
            "Prepare all required documents in advance and keep scanned copies ready",
            "Ensure all certificates and documents are valid and not expired",
            "Fill the application form carefully and double-check all information",
            "Follow up on your application if you don't receive confirmation",
            "Keep a copy of the submitted application and acknowledgment",
            "Check the scholarship website regularly for updates",
            "Prepare a personal statement or essay that highlights your achievements and aspirations",
            "Get recommendation letters from teachers or principals if required",
            "Ensure bank account is active and linked to Aadhaar for direct benefit transfer"
        ]
        
        # Common mistakes to avoid
        common_mistakes = [
            "Missing the application deadline",
            "Submitting incomplete applications",
            "Providing incorrect bank account details",
            "Neglecting to update contact information",
            "Submitting low-quality or unclear document scans",
            "Not following up on the application status",
            "Applying for scholarships where you don't meet eligibility criteria",
            "Making errors in personal or academic details",
            "Not keeping copies of submitted documents",
            "Ignoring communication from the scholarship provider"
        ]
        
        if scholarship_name:
            # Get specific scholarship details
            scholarship = self.get_scholarship_by_name(scholarship_name)
            if scholarship:
                return {
                    "scholarship_name": scholarship["name"],
                    "provider": scholarship["provider"],
                    "application_process": scholarship.get("application_process", "Not specified"),
                    "documents_required": scholarship.get("documents_required", []),
                    "deadline": scholarship.get("deadline", "Not specified"),
                    "website": scholarship.get("website", "Not specified"),
                    "contact": scholarship.get("contact", "Not specified"),
                    "general_tips": general_tips,
                    "common_mistakes": common_mistakes
                }
            else:
                return {
                    "error": f"Scholarship '{scholarship_name}' not found",
                    "general_tips": general_tips,
                    "common_mistakes": common_mistakes
                }
        else:
            # Return general application guide
            return {
                "general_tips": general_tips,
                "common_mistakes": common_mistakes,
                "important_documents": [
                    "Income certificate",
                    "Caste certificate (if applicable)",
                    "Domicile certificate",
                    "Previous education marksheets",
                    "ITI admission letter/ID card",
                    "Aadhaar card",
                    "Passport size photographs",
                    "Bank account details/passbook copy",
                    "Disability certificate (if applicable)",
                    "Recommendation letters (if required)"
                ]
            }
    
    def add_or_update_scholarship(self, scholarship_data: Dict, category: str) -> bool:
        """Add a new scholarship or update an existing one.
        
        Args:
            scholarship_data: Scholarship details to add/update
            category: Category of scholarship
            
        Returns:
            Boolean indicating success
        """
        # Validate required fields
        required_fields = ["name", "provider", "description", "eligibility", "benefits"]
        for field in required_fields:
            if field not in scholarship_data:
                print(f"{Fore.YELLOW}Missing required field: {field}{Style.RESET_ALL}")
                return False
        
        # Check if category exists
        if category not in self.scholarships:
            self.scholarships[category] = []
        
        # Check if scholarship already exists
        existing_index = None
        for i, scholarship in enumerate(self.scholarships[category]):
            if scholarship["name"] == scholarship_data["name"]:
                existing_index = i
                break
        
        if existing_index is not None:
            # Update existing scholarship
            self.scholarships[category][existing_index] = scholarship_data
        else:
            # Add new scholarship
            self.scholarships[category].append(scholarship_data)
        
        # Save changes
        self._save_scholarships(self.scholarships)
        return True
    
    def remove_scholarship(self, name: str) -> bool:
        """Remove a scholarship from the database.
        
        Args:
            name: Name of the scholarship to remove
            
        Returns:
            Boolean indicating success
        """
        for category, scholarships in self.scholarships.items():
            for i, scholarship in enumerate(scholarships):
                if scholarship["name"] == name:
                    # Remove scholarship
                    self.scholarships[category].pop(i)
                    self._save_scholarships(self.scholarships)
                    return True
        
        print(f"{Fore.YELLOW}Scholarship '{name}' not found{Style.RESET_ALL}")
        return False
    
    def update_deadlines(self) -> None:
        """Update scholarship deadlines for recurring scholarships.
        
        This function updates deadlines that have passed to the next application cycle.
        """
        today = datetime.now().date()
        updated = False
        
        for category, scholarships in self.scholarships.items():
            for i, scholarship in enumerate(scholarships):
                deadline = scholarship.get("deadline", "")
                if deadline and deadline != "Varies by state, check with local ITI" and deadline != "Rolling applications, check with local welfare board":
                    try:
                        deadline_date = datetime.strptime(deadline, "%Y-%m-%d").date()
                        if deadline_date < today:
                            # Set deadline to next year
                            next_deadline = deadline_date.replace(year=deadline_date.year + 1)
                            self.scholarships[category][i]["deadline"] = next_deadline.strftime("%Y-%m-%d")
                            updated = True
                    except ValueError:
                        # Skip scholarships with non-standard date formats
                        continue
        
        if updated:
            self._save_scholarships(self.scholarships)
            print(f"{Fore.GREEN}✓ Scholarship deadlines updated{Style.RESET_ALL}")
    
    def search_scholarships(self, query: str) -> List[Dict]:
        """Search for scholarships matching the query.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching scholarships
        """
        results = []
        query_lower = query.lower()
        
        for category, scholarships in self.scholarships.items():
            for scholarship in scholarships:
                # Search in name, provider, description, eligibility, benefits
                name = scholarship.get("name", "").lower()
                provider = scholarship.get("provider", "").lower()
                description = scholarship.get("description", "").lower()
                eligibility = " ".join(scholarship.get("eligibility", [])).lower()
                benefits = " ".join(scholarship.get("benefits", [])).lower()
                
                search_text = f"{name} {provider} {description} {eligibility} {benefits}"
                
                if query_lower in search_text:
                    result = scholarship.copy()
                    result["category"] = category
                    results.append(result)
        
        return results 