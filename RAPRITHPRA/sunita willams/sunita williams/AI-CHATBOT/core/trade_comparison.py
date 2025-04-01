"""
Trade comparison module for the ITI Assistant.

This module provides functionality to compare different ITI trades
side by side, showing details like duration, fees, job prospects, and salary potential.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from colorama import Fore, Style

class TradeComparisonManager:
    """Manages trade comparison functionality."""
    
    def __init__(self, trades_data_path="data/iti_trades.json", market_data_path="data/job_market.json"):
        """Initialize the trade comparison manager.
        
        Args:
            trades_data_path: Path to trades data file
            market_data_path: Path to job market data file
        """
        self.trades_data_path = trades_data_path
        self.market_data_path = market_data_path
        self.trades = {}
        self.market_data = {}
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(trades_data_path), exist_ok=True)
        
        # Load data
        self._load_trades()
        self._load_market_data()
        
        print(f"{Fore.GREEN}✓ Trade comparison manager initialized{Style.RESET_ALL}")
    
    def _load_trades(self):
        """Load trades data from file."""
        try:
            if os.path.exists(self.trades_data_path):
                with open(self.trades_data_path, "r", encoding="utf-8") as f:
                    self.trades = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded {len(self.trades)} ITI trades{Style.RESET_ALL}")
            else:
                # Initialize with default trades if file doesn't exist
                self._initialize_default_trades()
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load trades data: {e}{Style.RESET_ALL}")
            self._initialize_default_trades()
    
    def _initialize_default_trades(self):
        """Initialize with default ITI trades data."""
        self.trades = {
            "Fitter": {
                "duration": "1 year",
                "fees": {
                    "government": "₹5,000 - ₹8,000 per year",
                    "private": "₹15,000 - ₹25,000 per year"
                },
                "job_prospects": {
                    "demand_rating": 4.2,
                    "sectors": ["Manufacturing", "Automobile", "Railways", "Construction"],
                    "roles": ["Maintenance Fitter", "Assembly Fitter", "Production Fitter"]
                },
                "salary": {
                    "entry_level": "₹12,000 - ₹18,000 per month",
                    "experienced": "₹20,000 - ₹35,000 per month",
                    "highly_skilled": "₹35,000 - ₹50,000 per month"
                },
                "skills_gained": [
                    "Blueprint reading",
                    "Precision measurement",
                    "Hand tools operation",
                    "Assembly techniques",
                    "Troubleshooting"
                ],
                "certification_exams": [
                    "NCVT (National Council for Vocational Training)",
                    "SCVT (State Council for Vocational Training)"
                ],
                "advancement_paths": [
                    "Supervisor",
                    "Foreman",
                    "Quality Inspector",
                    "Technical Trainer"
                ]
            },
            "Electrician": {
                "duration": "2 years",
                "fees": {
                    "government": "₹5,000 - ₹10,000 per year",
                    "private": "₹20,000 - ₹30,000 per year"
                },
                "job_prospects": {
                    "demand_rating": 4.5,
                    "sectors": ["Power", "Manufacturing", "Construction", "Railways", "Telecom"],
                    "roles": ["Maintenance Electrician", "Wiring Technician", "Panel Operator"]
                },
                "salary": {
                    "entry_level": "₹15,000 - ₹20,000 per month",
                    "experienced": "₹25,000 - ₹40,000 per month",
                    "highly_skilled": "₹40,000 - ₹60,000 per month"
                },
                "skills_gained": [
                    "Electrical wiring",
                    "Circuit design",
                    "Troubleshooting",
                    "Safety protocols",
                    "Power distribution"
                ],
                "certification_exams": [
                    "NCVT (National Council for Vocational Training)",
                    "SCVT (State Council for Vocational Training)",
                    "Electrical License Exam"
                ],
                "advancement_paths": [
                    "Supervisor",
                    "Contractor",
                    "Electrical Inspector",
                    "Technical Trainer"
                ]
            },
            "Mechanic (Motor Vehicle)": {
                "duration": "2 years",
                "fees": {
                    "government": "₹5,000 - ₹10,000 per year",
                    "private": "₹20,000 - ₹35,000 per year"
                },
                "job_prospects": {
                    "demand_rating": 4.3,
                    "sectors": ["Automobile", "Transport", "Service Stations", "Manufacturing"],
                    "roles": ["Automobile Mechanic", "Service Technician", "Diagnostic Technician"]
                },
                "salary": {
                    "entry_level": "₹12,000 - ₹20,000 per month",
                    "experienced": "₹25,000 - ₹40,000 per month",
                    "highly_skilled": "₹40,000 - ₹60,000 per month"
                },
                "skills_gained": [
                    "Engine repair",
                    "Diagnostics",
                    "Electrical systems",
                    "Transmission systems",
                    "Preventive maintenance"
                ],
                "certification_exams": [
                    "NCVT (National Council for Vocational Training)",
                    "SCVT (State Council for Vocational Training)",
                    "ASE (Automotive Service Excellence) equivalent"
                ],
                "advancement_paths": [
                    "Service Manager",
                    "Workshop Owner",
                    "Technical Specialist",
                    "Automotive Trainer"
                ]
            },
            "COPA (Computer Operator & Programming Assistant)": {
                "duration": "1 year",
                "fees": {
                    "government": "₹5,000 - ₹8,000 per year",
                    "private": "₹15,000 - ₹25,000 per year"
                },
                "job_prospects": {
                    "demand_rating": 4.0,
                    "sectors": ["IT", "Office Administration", "Education", "Retail"],
                    "roles": ["Computer Operator", "Data Entry Operator", "Office Assistant"]
                },
                "salary": {
                    "entry_level": "₹10,000 - ₹18,000 per month",
                    "experienced": "₹20,000 - ₹30,000 per month",
                    "highly_skilled": "₹30,000 - ₹45,000 per month"
                },
                "skills_gained": [
                    "Office software",
                    "Basic programming",
                    "Data management",
                    "Computer hardware",
                    "Troubleshooting"
                ],
                "certification_exams": [
                    "NCVT (National Council for Vocational Training)",
                    "SCVT (State Council for Vocational Training)",
                    "Microsoft Office Specialist"
                ],
                "advancement_paths": [
                    "Office Administrator",
                    "Technical Support",
                    "Junior Programmer",
                    "IT Coordinator"
                ]
            },
            "Turner": {
                "duration": "1 year",
                "fees": {
                    "government": "₹5,000 - ₹8,000 per year",
                    "private": "₹15,000 - ₹25,000 per year"
                },
                "job_prospects": {
                    "demand_rating": 4.0,
                    "sectors": ["Manufacturing", "Tool Rooms", "Production Units", "Automobile"],
                    "roles": ["Turner", "Lathe Operator", "CNC Operator"]
                },
                "salary": {
                    "entry_level": "₹12,000 - ₹18,000 per month",
                    "experienced": "₹20,000 - ₹35,000 per month",
                    "highly_skilled": "₹35,000 - ₹50,000 per month"
                },
                "skills_gained": [
                    "Lathe operation",
                    "Precision measurement",
                    "Blueprint reading",
                    "Cutting tools knowledge",
                    "Quality control"
                ],
                "certification_exams": [
                    "NCVT (National Council for Vocational Training)",
                    "SCVT (State Council for Vocational Training)"
                ],
                "advancement_paths": [
                    "Supervisor",
                    "CNC Programmer",
                    "Quality Inspector",
                    "Technical Trainer"
                ]
            }
        }
        
        # Save the default data
        self._save_trades()
        print(f"{Fore.GREEN}✓ Initialized default trades data{Style.RESET_ALL}")
    
    def _save_trades(self):
        """Save trades data to file."""
        try:
            os.makedirs(os.path.dirname(self.trades_data_path), exist_ok=True)
            with open(self.trades_data_path, "w", encoding="utf-8") as f:
                json.dump(self.trades, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save trades data: {e}{Style.RESET_ALL}")
    
    def _load_market_data(self):
        """Load job market data from file."""
        try:
            if os.path.exists(self.market_data_path):
                with open(self.market_data_path, "r", encoding="utf-8") as f:
                    self.market_data = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded job market data{Style.RESET_ALL}")
            else:
                # Initialize with default market data if file doesn't exist
                self._initialize_default_market_data()
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load market data: {e}{Style.RESET_ALL}")
            self._initialize_default_market_data()
    
    def _initialize_default_market_data(self):
        """Initialize with default job market data."""
        self.market_data = {
            "employment_rates": {
                "Fitter": {
                    "overall": 85,
                    "by_region": {
                        "North": 88,
                        "South": 82,
                        "East": 80,
                        "West": 90
                    },
                    "trend": "Increasing"
                },
                "Electrician": {
                    "overall": 92,
                    "by_region": {
                        "North": 90,
                        "South": 94,
                        "East": 88,
                        "West": 95
                    },
                    "trend": "Stable"
                },
                "Mechanic (Motor Vehicle)": {
                    "overall": 88,
                    "by_region": {
                        "North": 85,
                        "South": 90,
                        "East": 82,
                        "West": 92
                    },
                    "trend": "Increasing"
                },
                "COPA": {
                    "overall": 75,
                    "by_region": {
                        "North": 80,
                        "South": 85,
                        "East": 70,
                        "West": 78
                    },
                    "trend": "Stable"
                },
                "Turner": {
                    "overall": 80,
                    "by_region": {
                        "North": 82,
                        "South": 78,
                        "East": 75,
                        "West": 85
                    },
                    "trend": "Stable"
                }
            },
            "industry_demand": {
                "Manufacturing": ["Fitter", "Turner", "Electrician"],
                "Automobile": ["Mechanic (Motor Vehicle)", "Fitter", "Electrician"],
                "IT & Services": ["COPA", "Electronics Mechanic"],
                "Construction": ["Electrician", "Fitter"],
                "Power": ["Electrician", "Electronics Mechanic"]
            },
            "future_outlook": {
                "Fitter": {
                    "growth_potential": "Moderate",
                    "automation_risk": "Medium",
                    "emerging_opportunities": ["Smart Manufacturing", "Precision Engineering"]
                },
                "Electrician": {
                    "growth_potential": "High",
                    "automation_risk": "Low",
                    "emerging_opportunities": ["Renewable Energy", "Smart Homes", "EV Infrastructure"]
                },
                "Mechanic (Motor Vehicle)": {
                    "growth_potential": "High",
                    "automation_risk": "Medium",
                    "emerging_opportunities": ["Electric Vehicles", "Hybrid Technology", "Autonomous Systems"]
                },
                "COPA": {
                    "growth_potential": "Moderate",
                    "automation_risk": "High",
                    "emerging_opportunities": ["Data Analysis", "Digital Marketing", "E-commerce Support"]
                },
                "Turner": {
                    "growth_potential": "Moderate",
                    "automation_risk": "High",
                    "emerging_opportunities": ["CNC Programming", "Precision Components", "Aerospace"]
                }
            }
        }
        
        # Save the default data
        self._save_market_data()
        print(f"{Fore.GREEN}✓ Initialized default market data{Style.RESET_ALL}")
    
    def _save_market_data(self):
        """Save market data to file."""
        try:
            os.makedirs(os.path.dirname(self.market_data_path), exist_ok=True)
            with open(self.market_data_path, "w", encoding="utf-8") as f:
                json.dump(self.market_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save market data: {e}{Style.RESET_ALL}")
    
    def get_all_trades(self):
        """Get all available trades.
        
        Returns:
            Dictionary of all trades
        """
        return self.trades
    
    def get_trade_details(self, trade_name):
        """Get details for a specific trade.
        
        Args:
            trade_name: Name of the trade
            
        Returns:
            Trade details or None if not found
        """
        if trade_name in self.trades:
            return {
                "name": trade_name,
                **self.trades[trade_name]
            }
        
        return None
    
    def compare_trades(self, trade_names):
        """Compare multiple trades side by side.
        
        Args:
            trade_names: List of trade names to compare
            
        Returns:
            Dictionary with comparison data
        """
        comparison = {
            "trades": [],
            "comparison_points": [
                "duration", "fees", "job_prospects", "salary", 
                "skills_gained", "certification_exams", "advancement_paths"
            ]
        }
        
        for trade_name in trade_names:
            if trade_name in self.trades:
                comparison["trades"].append({
                    "name": trade_name,
                    **self.trades[trade_name]
                })
        
        return comparison
    
    def get_employment_rates(self, trade_names=None):
        """Get employment rates for trades.
        
        Args:
            trade_names: List of trade names (optional, if None returns all)
            
        Returns:
            Dictionary with employment rate data
        """
        if not trade_names:
            return self.market_data.get("employment_rates", {})
        
        rates = {}
        for trade in trade_names:
            if trade in self.market_data.get("employment_rates", {}):
                rates[trade] = self.market_data["employment_rates"][trade]
        
        return rates
    
    def get_industry_demand(self, industry=None):
        """Get industry demand for trades.
        
        Args:
            industry: Specific industry (optional, if None returns all)
            
        Returns:
            Dictionary with industry demand data
        """
        if not industry:
            return self.market_data.get("industry_demand", {})
        
        if industry in self.market_data.get("industry_demand", {}):
            return {industry: self.market_data["industry_demand"][industry]}
        
        return {}
    
    def get_future_outlook(self, trade_names=None):
        """Get future outlook for trades.
        
        Args:
            trade_names: List of trade names (optional, if None returns all)
            
        Returns:
            Dictionary with future outlook data
        """
        if not trade_names:
            return self.market_data.get("future_outlook", {})
        
        outlook = {}
        for trade in trade_names:
            if trade in self.market_data.get("future_outlook", {}):
                outlook[trade] = self.market_data["future_outlook"][trade]
        
        return outlook
    
    def get_salary_comparison(self, trade_names):
        """Get salary comparison for trades.
        
        Args:
            trade_names: List of trade names
            
        Returns:
            Dictionary with salary comparison data
        """
        comparison = {
            "entry_level": {},
            "experienced": {},
            "highly_skilled": {}
        }
        
        for trade in trade_names:
            if trade in self.trades and "salary" in self.trades[trade]:
                salary = self.trades[trade]["salary"]
                comparison["entry_level"][trade] = salary.get("entry_level", "N/A")
                comparison["experienced"][trade] = salary.get("experienced", "N/A")
                comparison["highly_skilled"][trade] = salary.get("highly_skilled", "N/A")
        
        return comparison
    
    def get_skills_comparison(self, trade_names):
        """Get skills comparison for trades.
        
        Args:
            trade_names: List of trade names
            
        Returns:
            Dictionary with skills comparison data
        """
        comparison = {}
        
        for trade in trade_names:
            if trade in self.trades and "skills_gained" in self.trades[trade]:
                comparison[trade] = self.trades[trade]["skills_gained"]
        
        return comparison
    
    def recommend_trades_by_interest(self, interests):
        """Recommend trades based on user interests.
        
        Args:
            interests: List of interest keywords
            
        Returns:
            List of recommended trades with scores
        """
        recommendations = []
        
        for trade_name, details in self.trades.items():
            score = 0
            
            # Check job prospects
            for interest in interests:
                interest_lower = interest.lower()
                
                # Check in sectors
                for sector in details.get("job_prospects", {}).get("sectors", []):
                    if interest_lower in sector.lower():
                        score += 2
                
                # Check in roles
                for role in details.get("job_prospects", {}).get("roles", []):
                    if interest_lower in role.lower():
                        score += 2
                
                # Check in skills
                for skill in details.get("skills_gained", []):
                    if interest_lower in skill.lower():
                        score += 1
            
            if score > 0:
                recommendations.append({
                    "name": trade_name,
                    "score": score,
                    "details": details
                })
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return recommendations 