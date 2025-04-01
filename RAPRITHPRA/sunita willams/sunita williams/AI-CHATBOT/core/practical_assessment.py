"""
Practical assessment module for the ITI Assistant.

This module provides practical assessment preparation functionality,
including virtual lab simulations, tool diagrams, and step-by-step guides
for different ITI trades.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from colorama import Fore, Style

class PracticalAssessmentManager:
    """Manages practical assessment preparation functionality."""
    
    def __init__(self, practicals_data_path="data/practical_assessments.json", 
                 tools_data_path="data/trade_tools.json"):
        """Initialize the practical assessment manager.
        
        Args:
            practicals_data_path: Path to practical assessments data file
            tools_data_path: Path to tools reference data file
        """
        self.practicals_data_path = practicals_data_path
        self.tools_data_path = tools_data_path
        self.practicals = {}
        self.tools = {}
        self.user_attempts = {}
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(practicals_data_path), exist_ok=True)
        
        # Load data
        self._load_practicals()
        self._load_tools()
        
        print(f"{Fore.GREEN}✓ Practical assessment manager initialized{Style.RESET_ALL}")
    
    def _load_practicals(self):
        """Load practical assessments data from file."""
        try:
            if os.path.exists(self.practicals_data_path):
                with open(self.practicals_data_path, "r", encoding="utf-8") as f:
                    self.practicals = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded practical assessments data{Style.RESET_ALL}")
            else:
                # Initialize with default practicals if file doesn't exist
                self._initialize_default_practicals()
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load practicals data: {e}{Style.RESET_ALL}")
            self._initialize_default_practicals()
    
    def _initialize_default_practicals(self):
        """Initialize with default practical assessments data."""
        self.practicals = {
            "Electrician": [
                {
                    "title": "House Wiring Installation",
                    "difficulty": "Medium",
                    "duration": "3 hours",
                    "tools_required": ["Screwdriver set", "Wire cutter", "Wire stripper", "Multimeter", "Pliers"],
                    "materials_required": ["Electrical wires", "Switch board", "MCB", "Switch", "Socket"],
                    "safety_equipment": ["Insulated gloves", "Safety goggles"],
                    "steps": [
                        {
                            "step_number": 1,
                            "description": "Prepare the wiring diagram and identify points for switches and sockets",
                            "details": "Use blueprint to mark all connection points on the wall. Ensure proper spacing.",
                            "common_mistakes": ["Incorrect spacing between components", "Not following local electrical codes"]
                        },
                        {
                            "step_number": 2,
                            "description": "Install the distribution board",
                            "details": "Mount the distribution board securely on the wall, ensuring it's level.",
                            "common_mistakes": ["Mounting too low or too high", "Not securing properly"]
                        },
                        {
                            "step_number": 3,
                            "description": "Run conduit or casing for wires",
                            "details": "Install protective conduit for all wiring paths.",
                            "common_mistakes": ["Sharp bends in conduit", "Inadequate support"]
                        },
                        {
                            "step_number": 4,
                            "description": "Pull wires through conduit",
                            "details": "Use pull wire to draw electrical cables through conduit. Label each wire.",
                            "common_mistakes": ["Damaging wire insulation", "Not labeling wires", "Exceeding conduit capacity"]
                        },
                        {
                            "step_number": 5,
                            "description": "Install outlet boxes and switches",
                            "details": "Mount boxes securely and connect appropriate wires.",
                            "common_mistakes": ["Loose connections", "Reversed polarity"]
                        },
                        {
                            "step_number": 6,
                            "description": "Connect to main distribution board",
                            "details": "Connect circuit wires to appropriate MCBs.",
                            "common_mistakes": ["Incorrect circuit rating", "Poor termination"]
                        },
                        {
                            "step_number": 7,
                            "description": "Test the installation",
                            "details": "Use multimeter to check continuity, insulation, and correct voltage.",
                            "common_mistakes": ["Skipping tests", "Not checking for earth continuity"]
                        }
                    ],
                    "assessment_criteria": [
                        "Accuracy of connections",
                        "Neatness of wiring",
                        "Proper labeling",
                        "Functionality of all circuits",
                        "Adherence to safety standards",
                        "Time management"
                    ],
                    "reference_diagrams": [
                        "house_wiring_diagram.png",
                        "distribution_board_connections.png"
                    ],
                    "tips": [
                        "Always double-check polarity before powering the circuit",
                        "Use wire colors according to electrical code standards",
                        "Label all wires at both ends for future maintenance"
                    ]
                },
                {
                    "title": "Motor Starter Installation and Testing",
                    "difficulty": "Hard",
                    "duration": "4 hours",
                    "tools_required": ["Screwdriver set", "Spanner set", "Multimeter", "Megger", "Crimping tool"],
                    "materials_required": ["3-phase motor", "DOL starter", "Control cables", "Power cables", "Terminal connectors"],
                    "safety_equipment": ["Insulated gloves", "Safety goggles", "Insulated footwear"],
                    "steps": [
                        {
                            "step_number": 1,
                            "description": "Prepare the installation plan",
                            "details": "Study the circuit diagram and identify all components.",
                            "common_mistakes": ["Not understanding the control circuit", "Misinterpreting diagram"]
                        },
                        {
                            "step_number": 2,
                            "description": "Mount the starter panel",
                            "details": "Securely fix the starter panel at appropriate height.",
                            "common_mistakes": ["Improper alignment", "Inadequate fixing"]
                        },
                        {
                            "step_number": 3,
                            "description": "Connect power circuit",
                            "details": "Connect 3-phase supply to input terminals and motor connections to output.",
                            "common_mistakes": ["Incorrect phase sequence", "Loose terminations"]
                        },
                        {
                            "step_number": 4,
                            "description": "Wire the control circuit",
                            "details": "Connect start/stop buttons, thermal overload, and indicators.",
                            "common_mistakes": ["Incorrect control voltage", "Wrong contactor coil connections"]
                        },
                        {
                            "step_number": 5,
                            "description": "Set the overload relay",
                            "details": "Adjust thermal overload according to motor rating.",
                            "common_mistakes": ["Setting too high or too low", "Not considering ambient temperature"]
                        },
                        {
                            "step_number": 6,
                            "description": "Test the control circuit",
                            "details": "Check operation of start/stop buttons and indicators without power circuit.",
                            "common_mistakes": ["Testing with power before control check", "Not verifying indicator function"]
                        },
                        {
                            "step_number": 7,
                            "description": "Test full operation",
                            "details": "Connect motor and verify rotation, starting current, and protection features.",
                            "common_mistakes": ["Not checking rotation direction", "Ignoring unusual noises or vibrations"]
                        }
                    ],
                    "assessment_criteria": [
                        "Correct wiring connections",
                        "Proper component mounting",
                        "Accurate overload setting",
                        "Functional operation of all controls",
                        "Motor parameters within normal range",
                        "Safety considerations"
                    ],
                    "reference_diagrams": [
                        "motor_starter_wiring.png",
                        "control_circuit_diagram.png"
                    ],
                    "tips": [
                        "Always disconnect power before making any changes",
                        "Test control circuit operation before connecting to motor",
                        "Verify thermal overload settings match motor nameplate"
                    ]
                }
            ],
            "Fitter": [
                {
                    "title": "V-Block Machining and Filing",
                    "difficulty": "Medium",
                    "duration": "5 hours",
                    "tools_required": ["Files (flat, triangular)", "Try square", "Steel rule", "Vernier caliper", "Scriber", "Center punch", "Hammer"],
                    "materials_required": ["Mild steel block", "Engineer's blue", "Marking table"],
                    "safety_equipment": ["Safety goggles", "Work gloves", "Apron"],
                    "steps": [
                        {
                            "step_number": 1,
                            "description": "Study the drawing and prepare the material",
                            "details": "Understand dimensions and tolerances from drawing. Clean the raw stock.",
                            "common_mistakes": ["Not checking stock dimensions", "Misinterpreting tolerances"]
                        },
                        {
                            "step_number": 2,
                            "description": "Mark out the workpiece",
                            "details": "Apply engineer's blue. Mark dimensions using height gauge and scriber.",
                            "common_mistakes": ["Inaccurate marking", "Not using center lines", "Poor visibility of lines"]
                        },
                        {
                            "step_number": 3,
                            "description": "File the primary datum surface",
                            "details": "File one surface flat and check with try square and surface plate.",
                            "common_mistakes": ["Uneven pressure while filing", "Not checking flatness frequently"]
                        },
                        {
                            "step_number": 4,
                            "description": "Create perpendicular surfaces",
                            "details": "File adjacent surfaces perpendicular to the datum surface.",
                            "common_mistakes": ["Not maintaining perpendicularity", "Rounding edges"]
                        },
                        {
                            "step_number": 5,
                            "description": "Mark and machine the V-groove",
                            "details": "Mark the center line and angle lines for V-groove. File carefully to the lines.",
                            "common_mistakes": ["Uneven V-groove angles", "Not checking symmetry regularly"]
                        },
                        {
                            "step_number": 6,
                            "description": "File to final dimensions",
                            "details": "Use fine file for finishing all surfaces within tolerance.",
                            "common_mistakes": ["Removing too much material", "Poor surface finish"]
                        },
                        {
                            "step_number": 7,
                            "description": "Check final dimensions and finish",
                            "details": "Verify all dimensions with calipers and angle gauges. Check surface finish quality.",
                            "common_mistakes": ["Not checking all dimensions", "Ignoring surface finish requirements"]
                        }
                    ],
                    "assessment_criteria": [
                        "Dimensional accuracy",
                        "Flatness of datum surface",
                        "Perpendicularity of sides",
                        "Accuracy of V-groove angle",
                        "Symmetry of V-groove",
                        "Surface finish quality",
                        "Time management"
                    ],
                    "reference_diagrams": [
                        "v_block_drawing.png",
                        "filing_technique.png"
                    ],
                    "tips": [
                        "Periodically check with try square while filing",
                        "Keep files clean using a file card",
                        "Use chalk on files to prevent clogging when filing soft metals",
                        "Apply even pressure and maintain correct posture while filing"
                    ]
                }
            ],
            "Mechanic (Motor Vehicle)": [
                {
                    "title": "Engine Timing Belt Replacement",
                    "difficulty": "Hard",
                    "duration": "4 hours",
                    "tools_required": ["Socket set", "Torque wrench", "Timing pins", "Crankshaft holding tool", "Breaker bar"],
                    "materials_required": ["Timing belt kit", "Coolant", "Gaskets", "Seals"],
                    "safety_equipment": ["Safety goggles", "Work gloves", "Protective footwear"],
                    "steps": [
                        {
                            "step_number": 1,
                            "description": "Prepare the vehicle",
                            "details": "Position on lift, disconnect battery, drain coolant if required.",
                            "common_mistakes": ["Not disconnecting battery", "Incomplete coolant drainage"]
                        },
                        {
                            "step_number": 2,
                            "description": "Remove accessories for access",
                            "details": "Remove necessary covers, belts, and components to access timing cover.",
                            "common_mistakes": ["Damaging accessories during removal", "Not labeling parts and bolts"]
                        },
                        {
                            "step_number": 3,
                            "description": "Position engine at TDC",
                            "details": "Rotate crankshaft to Top Dead Center for cylinder #1. Verify with timing marks.",
                            "common_mistakes": ["Incorrect TDC identification", "Not verifying camshaft position"]
                        },
                        {
                            "step_number": 4,
                            "description": "Lock engine timing",
                            "details": "Insert timing pins/tools to lock crankshaft and camshaft in correct positions.",
                            "common_mistakes": ["Using wrong timing pins", "Forcing pins into position"]
                        },
                        {
                            "step_number": 5,
                            "description": "Remove timing belt",
                            "details": "Release tensioner and carefully remove old timing belt.",
                            "common_mistakes": ["Forcing belt removal", "Disturbing locked positions"]
                        },
                        {
                            "step_number": 6,
                            "description": "Inspect and replace components",
                            "details": "Check tensioners, pulleys, and water pump for wear. Replace as required.",
                            "common_mistakes": ["Reusing worn components", "Improper installation of new parts"]
                        },
                        {
                            "step_number": 7,
                            "description": "Install new timing belt",
                            "details": "Follow manufacturer's routing diagram, starting at crankshaft.",
                            "common_mistakes": ["Incorrect routing sequence", "Incorrect tension"]
                        },
                        {
                            "step_number": 8,
                            "description": "Set belt tension",
                            "details": "Adjust tensioner to correct specification and secure in position.",
                            "common_mistakes": ["Over or under tensioning", "Not following tensioner procedure"]
                        },
                        {
                            "step_number": 9,
                            "description": "Verify timing and remove locking tools",
                            "details": "Double-check all timing marks align. Remove timing pins/tools.",
                            "common_mistakes": ["Not verifying alignment before finalizing", "Incomplete pin removal"]
                        },
                        {
                            "step_number": 10,
                            "description": "Reinstall accessories and finalize",
                            "details": "Reinstall all components, refill coolant, reconnect battery.",
                            "common_mistakes": ["Missing parts during reassembly", "Incorrect torque specifications"]
                        }
                    ],
                    "assessment_criteria": [
                        "Correct positioning of timing marks",
                        "Proper belt tension",
                        "Proper component installation",
                        "Complete reassembly",
                        "Engine operation after service",
                        "Work area organization",
                        "Time management"
                    ],
                    "reference_diagrams": [
                        "timing_belt_routing.png",
                        "tdc_timing_marks.png"
                    ],
                    "tips": [
                        "Take photos before disassembly for reference",
                        "Always rotate engine by hand for two complete revolutions after installation",
                        "Replace water pump when changing timing belt if it's driven by the timing belt",
                        "Follow manufacturer's specific procedures as they vary by engine model"
                    ]
                }
            ]
        }
        
        # Save the default data
        self._save_practicals()
        print(f"{Fore.GREEN}✓ Initialized default practicals data{Style.RESET_ALL}")
    
    def _save_practicals(self):
        """Save practicals data to file."""
        try:
            os.makedirs(os.path.dirname(self.practicals_data_path), exist_ok=True)
            with open(self.practicals_data_path, "w", encoding="utf-8") as f:
                json.dump(self.practicals, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save practicals data: {e}{Style.RESET_ALL}")
    
    def _load_tools(self):
        """Load tools reference data from file."""
        try:
            if os.path.exists(self.tools_data_path):
                with open(self.tools_data_path, "r", encoding="utf-8") as f:
                    self.tools = json.load(f)
                print(f"{Fore.GREEN}✓ Loaded tools reference data{Style.RESET_ALL}")
            else:
                # Initialize with default tools if file doesn't exist
                self._initialize_default_tools()
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load tools data: {e}{Style.RESET_ALL}")
            self._initialize_default_tools()
    
    def _initialize_default_tools(self):
        """Initialize with default tools reference data."""
        self.tools = {
            "Electrician": {
                "Screwdriver set": {
                    "description": "Set of various sizes and types of screwdrivers for electrical work",
                    "usage": "Tightening and loosening screws on electrical fixtures and panels",
                    "types": ["Flat head", "Phillips head", "Cabinet tip", "Insulated"],
                    "safety_tips": ["Use insulated handles", "Match size to screw", "Don't use as pry bar"],
                    "image": "screwdriver_set.png"
                },
                "Multimeter": {
                    "description": "Device for measuring electrical values such as voltage, current, and resistance",
                    "usage": "Testing circuits, fault finding, measuring electrical parameters",
                    "types": ["Digital", "Analog", "Clamp meter"],
                    "safety_tips": ["Verify meter is working before use", "Use correct measurement range", "Don't exceed rated voltage"],
                    "image": "multimeter.png"
                },
                "Wire stripper": {
                    "description": "Tool for removing insulation from electrical wires",
                    "usage": "Preparing wires for connections by stripping insulation without damaging conductor",
                    "types": ["Manual", "Automatic", "Combination"],
                    "safety_tips": ["Match tool to wire gauge", "Don't cut too deep", "Keep handles clean"],
                    "image": "wire_stripper.png"
                }
            },
            "Fitter": {
                "Files": {
                    "description": "Hand tools used for removing material and finishing surfaces",
                    "usage": "Shaping, smoothing, and deburring metal surfaces",
                    "types": ["Flat", "Round", "Half-round", "Triangular", "Square", "Bastard", "Second cut", "Smooth cut"],
                    "safety_tips": ["Always use with handle", "Clean with file card", "Don't hammer or use as pry bar"],
                    "image": "files.png"
                },
                "Vernier caliper": {
                    "description": "Precision measuring instrument for internal, external and depth measurements",
                    "usage": "Taking accurate measurements of length, diameter, depth, and step dimensions",
                    "types": ["Dial", "Digital", "Vernier"],
                    "safety_tips": ["Handle carefully", "Clean measuring faces before use", "Don't drop"],
                    "image": "vernier_caliper.png"
                },
                "Try square": {
                    "description": "L-shaped measurement tool for checking right angles",
                    "usage": "Verifying perpendicularity and straightness of surfaces",
                    "types": ["Fixed", "Combination", "Engineer's square"],
                    "safety_tips": ["Store carefully to maintain accuracy", "Clean before use", "Don't drop or mishandle"],
                    "image": "try_square.png"
                }
            },
            "Mechanic (Motor Vehicle)": {
                "Socket set": {
                    "description": "Set of various sized sockets with ratchet handles",
                    "usage": "Removing and installing various nuts and bolts on vehicles",
                    "types": ["Standard (SAE)", "Metric", "Deep socket", "Impact socket"],
                    "safety_tips": ["Use correct size socket", "Don't use with power tools unless impact rated", "Keep clean"],
                    "image": "socket_set.png"
                },
                "Torque wrench": {
                    "description": "Precision tool for applying specific amount of torque to fasteners",
                    "usage": "Tightening bolts to manufacturer specifications",
                    "types": ["Click", "Beam", "Digital", "Dial"],
                    "safety_tips": ["Reset to lowest setting after use", "Don't use for loosening", "Calibrate regularly"],
                    "image": "torque_wrench.png"
                },
                "Timing light": {
                    "description": "Stroboscopic tool that flashes in sync with engine firing",
                    "usage": "Checking and adjusting ignition timing",
                    "types": ["Inductive", "Digital", "Advance"],
                    "safety_tips": ["Don't look directly at light", "Keep away from moving parts", "Shield from water"],
                    "image": "timing_light.png"
                }
            }
        }
        
        # Save the default data
        self._save_tools()
        print(f"{Fore.GREEN}✓ Initialized default tools data{Style.RESET_ALL}")
    
    def _save_tools(self):
        """Save tools data to file."""
        try:
            os.makedirs(os.path.dirname(self.tools_data_path), exist_ok=True)
            with open(self.tools_data_path, "w", encoding="utf-8") as f:
                json.dump(self.tools, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save tools data: {e}{Style.RESET_ALL}")
    
    def get_available_practicals(self, trade=None):
        """Get available practical assessments.
        
        Args:
            trade: Specific trade (optional, if None returns all)
            
        Returns:
            Dictionary of practical assessments
        """
        if not trade:
            return self.practicals
        
        if trade in self.practicals:
            return {trade: self.practicals[trade]}
        
        return {}
    
    def get_practical_details(self, trade, title):
        """Get details for a specific practical assessment.
        
        Args:
            trade: Trade name
            title: Practical assessment title
            
        Returns:
            Practical assessment details or None if not found
        """
        if trade in self.practicals:
            for practical in self.practicals[trade]:
                if practical["title"] == title:
                    return practical
        
        return None
    
    def get_tool_info(self, trade, tool_name):
        """Get information about a specific tool.
        
        Args:
            trade: Trade name
            tool_name: Tool name
            
        Returns:
            Tool information or None if not found
        """
        if trade in self.tools and tool_name in self.tools[trade]:
            return {
                "name": tool_name,
                **self.tools[trade][tool_name]
            }
        
        return None
    
    def get_tools_for_trade(self, trade):
        """Get all tools for a specific trade.
        
        Args:
            trade: Trade name
            
        Returns:
            Dictionary of tools for the trade
        """
        if trade in self.tools:
            return self.tools[trade]
        
        return {}
    
    def find_practicals_by_tools(self, tools_list):
        """Find practical assessments that use specific tools.
        
        Args:
            tools_list: List of tool names
            
        Returns:
            List of practical assessments that use the specified tools
        """
        matching_practicals = []
        
        for trade, practicals in self.practicals.items():
            for practical in practicals:
                required_tools = practical.get("tools_required", [])
                
                # Check if any of the specified tools are required
                if any(tool in required_tools for tool in tools_list):
                    matching_practicals.append({
                        "trade": trade,
                        "title": practical["title"],
                        "difficulty": practical.get("difficulty", "Medium"),
                        "matching_tools": [tool for tool in tools_list if tool in required_tools]
                    })
        
        return matching_practicals
    
    def save_user_attempt(self, user_id, trade, practical_title, results):
        """Save user's practical assessment attempt.
        
        Args:
            user_id: User identifier
            trade: Trade name
            practical_title: Practical assessment title
            results: Dictionary with assessment results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            attempt = {
                "user_id": user_id,
                "trade": trade,
                "practical_title": practical_title,
                "date": datetime.now().isoformat(),
                "results": results
            }
            
            if user_id not in self.user_attempts:
                self.user_attempts[user_id] = []
            
            self.user_attempts[user_id].append(attempt)
            
            # Save to file
            attempts_path = f"data/practical_attempts_{user_id}.json"
            os.makedirs(os.path.dirname(attempts_path), exist_ok=True)
            
            with open(attempts_path, "w", encoding="utf-8") as f:
                json.dump(self.user_attempts[user_id], f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"{Fore.YELLOW}Could not save user attempt: {e}{Style.RESET_ALL}")
            return False
    
    def get_user_attempts(self, user_id, trade=None):
        """Get user's practical assessment attempts.
        
        Args:
            user_id: User identifier
            trade: Trade name (optional, if None returns all)
            
        Returns:
            List of user's practical assessment attempts
        """
        try:
            attempts_path = f"data/practical_attempts_{user_id}.json"
            
            if os.path.exists(attempts_path):
                with open(attempts_path, "r", encoding="utf-8") as f:
                    self.user_attempts[user_id] = json.load(f)
            else:
                self.user_attempts[user_id] = []
            
            if not trade:
                return self.user_attempts[user_id]
            else:
                return [attempt for attempt in self.user_attempts[user_id] if attempt["trade"] == trade]
            
        except Exception as e:
            print(f"{Fore.YELLOW}Could not load user attempts: {e}{Style.RESET_ALL}")
            return []
    
    def generate_practical_guide(self, trade, practical_title):
        """Generate a comprehensive guide for a practical assessment.
        
        Args:
            trade: Trade name
            practical_title: Practical assessment title
            
        Returns:
            Formatted guide text or None if not found
        """
        practical = self.get_practical_details(trade, practical_title)
        if not practical:
            return None
        
        guide = f"# {practical_title} - {trade} Trade Practical Guide\n\n"
        
        # Add difficulty and duration
        guide += f"**Difficulty**: {practical.get('difficulty', 'Medium')}\n"
        guide += f"**Duration**: {practical.get('duration', 'Not specified')}\n\n"
        
        # Add tools and materials
        guide += "## Required Tools\n"
        for tool in practical.get("tools_required", []):
            tool_info = self.get_tool_info(trade, tool)
            if tool_info:
                guide += f"- **{tool}**: {tool_info.get('description', '')}\n"
            else:
                guide += f"- {tool}\n"
        
        guide += "\n## Required Materials\n"
        for material in practical.get("materials_required", []):
            guide += f"- {material}\n"
        
        guide += "\n## Safety Equipment\n"
        for equipment in practical.get("safety_equipment", []):
            guide += f"- {equipment}\n"
        
        # Add step-by-step procedure
        guide += "\n## Procedure\n"
        for step in practical.get("steps", []):
            guide += f"### Step {step.get('step_number', '')}: {step.get('description', '')}\n"
            guide += f"{step.get('details', '')}\n\n"
            
            if "common_mistakes" in step:
                guide += "**Common mistakes to avoid:**\n"
                for mistake in step["common_mistakes"]:
                    guide += f"- {mistake}\n"
            guide += "\n"
        
        # Add assessment criteria
        guide += "## Assessment Criteria\n"
        for criterion in practical.get("assessment_criteria", []):
            guide += f"- {criterion}\n"
        
        # Add tips
        guide += "\n## Tips for Success\n"
        for tip in practical.get("tips", []):
            guide += f"- {tip}\n"
        
        return guide 