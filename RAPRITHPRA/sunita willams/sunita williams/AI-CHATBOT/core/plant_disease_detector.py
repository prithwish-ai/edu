"""
Plant Disease Detection Module

This module provides functionality to detect plant diseases using a pre-trained model.
It supports PyTorch models for plant disease detection.
"""

import os
import numpy as np
import cv2
import colorama
from colorama import Fore, Style
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# Initialize colorama for cross-platform colored terminal output
colorama.init()

class PlantDiseaseDetector:
    """Class for detecting plant diseases in images using a pre-trained model."""
    
    def __init__(self, model_path=None):
        """Initialize the plant disease detector.
        
        Args:
            model_path (str, optional): Path to the trained model file.
                                       Defaults to None (will use a default path).
        """
        self.model = None
        self.model_type = 'pytorch'
        
        # Disease class names for PyTorch model
        self.disease_classes = [
            "Healthy", 
            "Powdery Mildew", 
            "Rust", 
            "Scab"
        ]
        
        # Default model path (using PyTorch model)
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "trained_plant_disease_model.pth")
        
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Load the trained model from the specified path.
        
        Args:
            model_path (str): Path to the trained model file.
        """
        try:
            print(f"{Fore.YELLOW}Loading plant disease detection model...{Style.RESET_ALL}")
            
            if model_path.endswith('.pth'):
                # Load PyTorch model
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                
                # Create a model instance - using 4 classes based on the actual model architecture
                self.model = models.resnet50(pretrained=False)
                num_classes = 4  # Using 4 classes to match the actual model checkpoint
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
                
                # Load the weights
                if isinstance(state_dict, dict):
                    self.model.load_state_dict(state_dict)
                else:
                    # If it's already a model object, use it directly
                    self.model = state_dict
                    
                # Set to evaluation mode
                self.model.eval()
                self.model_type = 'pytorch'
                print(f"{Fore.GREEN}✓ Successfully loaded PyTorch plant disease model{Style.RESET_ALL}")
            else:
                raise ValueError(f"Unsupported model format: {model_path}. Only .pth files are supported.")
                
        except Exception as e:
            print(f"{Fore.RED}Error loading plant disease detection model: {e}{Style.RESET_ALL}")
            print(f"Make sure the model file exists at: {model_path}")
            self.model = None
    
    def set_model_type(self, model_type):
        """Set the model type.
        
        Args:
            model_type (str): The model type ('pytorch').
        """
        self.model_type = model_type
    
    def detect_disease(self, image_path, model=None, use_external_model=False):
        """Predict the disease in a plant image.
        
        Args:
            image_path (str): Path to the image file.
            model (object, optional): External model to use for prediction.
            use_external_model (bool): Whether to use the provided external model.
            
        Returns:
            dict: A dictionary containing the prediction results with:
                - disease: The predicted disease name
                - confidence: The confidence score (0-100%)
                - is_healthy: Boolean indicating if the plant is healthy
                - treatment: Treatment recommendations
        """
        # Use external model if provided and requested
        active_model = model if use_external_model and model is not None else self.model
        
        if active_model is None:
            return {
                "status": "error",
                "error": "Model not loaded"
            }
        
        try:
            # Process the image using PyTorch
            # Define transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Load and preprocess the image
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
            
            # Check if model is an OrderedDict (state_dict)
            if isinstance(active_model, dict) or hasattr(active_model, 'items'):
                # Create a proper model and load the state dict
                temp_model = models.resnet50(pretrained=False)
                num_classes = 4  # Using 4 classes to match the actual model checkpoint
                temp_model.fc = torch.nn.Linear(temp_model.fc.in_features, num_classes)
                temp_model.load_state_dict(active_model)
                temp_model.eval()
                
                # Perform inference
                with torch.no_grad():
                    outputs = temp_model(img_tensor)
                    _, predicted = torch.max(outputs, 1)
                    # Get confidence scores
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    confidence_score = probabilities[predicted.item()].item() * 100
            else:
                # Use the model directly
                with torch.no_grad():
                    outputs = active_model(img_tensor)
                    _, predicted = torch.max(outputs, 1)
                    # Get confidence scores
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    confidence_score = probabilities[predicted.item()].item() * 100
            
            # Get the disease name
            prediction = self.disease_classes[predicted.item()]
            formatted_prediction = prediction
            
            # Check if the plant is healthy
            is_healthy = "healthy" in prediction.lower()
            
            # Get treatment recommendation
            treatment_info = self.get_treatment_recommendation(prediction)
            
            return {
                "status": "success",
                "disease": formatted_prediction,
                "confidence": f"{confidence_score:.2f}%",
                "is_healthy": is_healthy,
                "treatment": treatment_info.get('treatment', 'No specific treatment available.'),
                "description": treatment_info.get('description', '')
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_treatment_recommendation(self, disease_name):
        """Get treatment recommendations for a detected disease.
        
        Args:
            disease_name (str): The predicted disease name.
            
        Returns:
            dict: Treatment recommendations with:
                - disease: The disease name
                - description: Description of the disease
                - treatment: Treatment recommendations
                - prevention: Prevention tips
        """
        # Dictionary of treatment recommendations for the PyTorch model diseases
        treatments = {
            "Healthy": {
                "disease": "Healthy Plant",
                "description": "The plant appears to be healthy.",
                "treatment": "No treatment needed. Continue good plant care practices.",
                "prevention": "Maintain regular watering, appropriate fertilization, and monitor for early signs of pests or disease."
            },
            "Powdery Mildew": {
                "disease": "Powdery Mildew",
                "description": "A fungal disease that appears as white powdery spots on leaves and stems.",
                "treatment": "Apply sulfur or potassium bicarbonate-based fungicides. Remove and destroy severely infected parts.",
                "prevention": "Ensure good air circulation, avoid overhead watering, and space plants properly."
            },
            "Rust": {
                "disease": "Rust",
                "description": "A fungal disease that appears as rusty spots or pustules on leaves and stems.",
                "treatment": "Apply fungicides containing mancozeb. Remove and destroy infected leaves to prevent spread.",
                "prevention": "Avoid wetting leaves when watering, ensure good air circulation, and plant resistant varieties."
            },
            "Scab": {
                "disease": "Scab",
                "description": "A fungal disease causing dark, scabby lesions on leaves and fruit.",
                "treatment": "Apply fungicides containing captan or myclobutanil. Remove and destroy fallen leaves.",
                "prevention": "Plant resistant varieties, ensure good air circulation through proper pruning, and clean up fallen leaves."
            }
        }
        
        # Check if we have recommendations for this disease
        if disease_name in treatments:
            return treatments[disease_name]
        
        # For diseases not in our database, return a generic response
        return {
            "disease": disease_name,
            "description": f"A plant disease affecting crops.",
            "treatment": "Consult with a local agricultural extension service for specific treatment recommendations.",
            "prevention": "Practice crop rotation, ensure good air circulation, use disease-free seeds, and maintain plant vigor with proper nutrition and watering."
        }

# Example usage:
# detector = PlantDiseaseDetector()
# result = detector.detect_disease("path/to/image.jpg")
# if result["status"] == "success":
#     print(f"Prediction: {result['disease']}")
#     print(f"Confidence: {result['confidence']}")
#     
#     if not result["is_healthy"]:
#         treatment = detector.get_treatment_recommendation(result["disease"])
#         print(f"\nTreatment for {treatment['disease']}:")
#         print(f"Description: {treatment['description']}")
#         print(f"Treatment: {treatment['treatment']}")
#         print(f"Prevention: {treatment['prevention']}")
# else:
#     print(f"Error: {result['error']}") 