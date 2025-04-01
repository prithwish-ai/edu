"""
Multimodal manager for the AI Chatbot.

This module provides multimodal capabilities, enabling the chatbot
to process image inputs alongside text and generate multimodal responses.
"""

import os
import json
import base64
from datetime import datetime
import tempfile
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
import io
import uuid

class MultimodalManager:
    """Manages multimodal capabilities for the chatbot."""
    
    def __init__(self, storage_path="data/multimodal"):
        """Initialize the multimodal manager.
        
        Args:
            storage_path: Path to store uploaded images
        """
        self.storage_path = storage_path
        self.is_available = self._check_availability()
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        print(f"✓ Multimodal manager initialized")
    
    def _check_availability(self) -> bool:
        """Check if required packages are available."""
        try:
            import PIL
            return True
        except ImportError:
            print(f"Multimodal interaction requires additional packages.")
            print("Install with: pip install Pillow")
            return False
    
    def process_image(self, image_data: str or bytes, source_format: str = "base64") -> Dict:
        """Process an image and prepare it for the model.
        
        Args:
            image_data: Image data either as base64 string or bytes
            source_format: Format of the image data ('base64', 'bytes', 'path')
            
        Returns:
            Dict containing image info and processed data
        """
        if not self.is_available:
            return {"error": "Multimodal processing is not available"}
        
        try:
            # Generate a unique filename
            image_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{image_id}_{timestamp}.jpg"
            filepath = os.path.join(self.storage_path, filename)
            
            # Convert the image data to PIL Image
            if source_format == "base64":
                if isinstance(image_data, str):
                    # Remove data URL prefix if present
                    if "base64," in image_data:
                        image_data = image_data.split("base64,")[1]
                    image_bytes = base64.b64decode(image_data)
                else:
                    image_bytes = image_data
                image = Image.open(io.BytesIO(image_bytes))
            elif source_format == "bytes":
                image = Image.open(io.BytesIO(image_data))
            elif source_format == "path":
                image = Image.open(image_data)
            else:
                return {"error": f"Unsupported source format: {source_format}"}
            
            # Resize if necessary (for consistency)
            max_size = 1024
            if max(image.size) > max_size:
                # Calculate new dimensions maintaining aspect ratio
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)
            
            # Save processed image
            image.save(filepath, "JPEG")
            
            # Get basic image metadata
            width, height = image.size
            
            # Prepare the response
            result = {
                "image_id": image_id,
                "filename": filename,
                "filepath": filepath,
                "size": {
                    "width": width,
                    "height": height
                },
                "timestamp": timestamp
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Image processing error: {str(e)}"}
    
    def extract_image_features(self, image_path: str) -> Dict:
        """Extract features from an image for use in the model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing extracted features and metadata
        """
        if not self.is_available:
            return {"error": "Multimodal processing is not available"}
        
        try:
            # Open and process the image
            image = Image.open(image_path)
            
            # Get basic image metadata
            width, height = image.size
            format_name = image.format
            mode = image.mode
            
            # Extract simple color information
            if mode == "RGB" or mode == "RGBA":
                # Convert to RGB if it's RGBA
                if mode == "RGBA":
                    image = image.convert("RGB")
                
                # Get color histogram
                histogram = image.histogram()
                
                # Calculate average RGB values
                r_values = histogram[0:256]
                g_values = histogram[256:512]
                b_values = histogram[512:768]
                
                total_pixels = width * height
                avg_r = sum(i * count for i, count in enumerate(r_values)) / total_pixels
                avg_g = sum(i * count for i, count in enumerate(g_values)) / total_pixels
                avg_b = sum(i * count for i, count in enumerate(b_values)) / total_pixels
                
                color_info = {
                    "average_rgb": [int(avg_r), int(avg_g), int(avg_b)],
                }
            else:
                color_info = {"mode": mode}
            
            # Build the features dictionary
            features = {
                "metadata": {
                    "width": width,
                    "height": height,
                    "format": format_name,
                    "mode": mode,
                    "aspect_ratio": width / height if height > 0 else 0
                },
                "color_info": color_info,
                # This is where we would add more sophisticated features like
                # object detection, classification results, etc.
            }
            
            return features
            
        except Exception as e:
            return {"error": f"Feature extraction error: {str(e)}"}
    
    def generate_image_description(self, image_path: str) -> str:
        """Generate a text description of an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Text description of the image
        """
        if not self.is_available:
            return "Image description is not available."
        
        # This is a placeholder for actual image description generation
        # In a real implementation, you would use a vision-language model here
        
        features = self.extract_image_features(image_path)
        
        if "error" in features:
            return f"Could not analyze image: {features['error']}"
        
        metadata = features["metadata"]
        
        # Generate a simple description based on extracted features
        orientation = "portrait" if metadata["height"] > metadata["width"] else "landscape"
        resolution = "high" if metadata["width"] * metadata["height"] > 1000000 else "standard"
        
        description = f"This is a {orientation} image in {resolution} resolution. "
        
        if "color_info" in features and "average_rgb" in features["color_info"]:
            avg_rgb = features["color_info"]["average_rgb"]
            # Determine dominant color tone based on RGB values
            r, g, b = avg_rgb
            
            if max(r, g, b) < 80:
                color_tone = "dark"
            elif max(r, g, b) > 200:
                color_tone = "bright"
            else:
                color_tone = "moderate"
                
            if r > g + 30 and r > b + 30:
                color_desc = "red"
            elif g > r + 30 and g > b + 30:
                color_desc = "green"
            elif b > r + 30 and b > g + 30:
                color_desc = "blue"
            elif r > 180 and g > 180 and b < 100:
                color_desc = "yellow"
            elif r > 180 and g < 100 and b > 180:
                color_desc = "purple"
            elif r < 100 and g > 180 and b > 180:
                color_desc = "cyan"
            elif max(r, g, b) - min(r, g, b) < 30:
                if r > 200:
                    color_desc = "white"
                elif r < 50:
                    color_desc = "black"
                else:
                    color_desc = "gray"
            else:
                color_desc = "multi-colored"
            
            description += f"The image has a predominantly {color_tone} {color_desc} tone. "
        
        description += "I'd need to analyze this image further to provide specific details about its content."
        
        return description
    
    def create_multimodal_response(self, text: str, image_path: Optional[str] = None) -> Dict:
        """Create a multimodal response with both text and optional image.
        
        Args:
            text: Text response
            image_path: Optional path to an image to include
            
        Returns:
            Dict with the multimodal response
        """
        response = {
            "type": "text",
            "content": text
        }
        
        if image_path and os.path.exists(image_path):
            try:
                # Convert image to base64 for sending to frontend
                with open(image_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                response = {
                    "type": "multimodal",
                    "text": text,
                    "image": {
                        "data": f"data:image/jpeg;base64,{img_data}",
                        "path": image_path
                    }
                }
            except Exception as e:
                print(f"Error including image in response: {str(e)}")
        
        return response 