#!/usr/bin/env python3
"""
Market Price Prediction Module

This module handles crop price prediction and alerts for farmers.
"""

import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class MarketPredictor:
    """Manages crop price predictions and alerts for farmers."""
    
    def __init__(self, api_key=None, data_path="data/market_data"):
        """Initialize the market predictor.
        
        Args:
            api_key: API key for market data services
            data_path: Path to store market data and models
        """
        # Use provided API key or default to the data.gov.in API key
        self.api_key = api_key or "579b464db66ec23bdd0000013d337248dcc14a4d570c76dc7c0ea216"
        self.data_path = data_path
        self.logger = logging.getLogger("MarketPredictor")
        self.models = {}
        
        # API endpoints
        self.api_endpoint = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        
        # Ensure data directory exists
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(f"{self.data_path}/models", exist_ok=True)
        
        # Load existing data and models
        self._load_market_data()
        self._load_prediction_models()
    
    def _load_market_data(self) -> None:
        """Load saved market data."""
        try:
            market_data_path = f"{self.data_path}/market_data.json"
            if os.path.exists(market_data_path):
                with open(market_data_path, 'r') as f:
                    self.market_data = json.load(f)
                self.logger.info(f"Loaded market data for {len(self.market_data)} crops")
            else:
                self.market_data = {}
                self.logger.info("No existing market data found. Starting fresh.")
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            self.market_data = {}
    
    def _save_market_data(self) -> None:
        """Save market data to disk."""
        try:
            with open(f"{self.data_path}/market_data.json", 'w') as f:
                json.dump(self.market_data, f, indent=2)
            self.logger.info("Market data saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving market data: {e}")
    
    def _load_prediction_models(self) -> None:
        """Load saved prediction models."""
        try:
            models_dir = f"{self.data_path}/models"
            if os.path.exists(models_dir):
                for model_file in os.listdir(models_dir):
                    if model_file.endswith('.joblib'):
                        crop_name = model_file.split('_model.joblib')[0]
                        model_path = os.path.join(models_dir, model_file)
                        self.models[crop_name] = joblib.load(model_path)
                self.logger.info(f"Loaded prediction models for {len(self.models)} crops")
            else:
                self.logger.info("No prediction models found. Models will be trained as data is collected.")
        except Exception as e:
            self.logger.error(f"Error loading prediction models: {e}")
    
    def _save_prediction_model(self, crop: str, model) -> None:
        """Save a prediction model to disk.
        
        Args:
            crop: Crop name
            model: Trained model object
        """
        try:
            model_path = f"{self.data_path}/models/{crop}_model.joblib"
            joblib.dump(model, model_path)
            self.logger.info(f"Saved prediction model for {crop}")
        except Exception as e:
            self.logger.error(f"Error saving prediction model for {crop}: {e}")
    
    def fetch_market_data(self, crop: str, location: str = None) -> Dict:
        """Fetch current market data for a specific crop from data.gov.in API.
        
        Args:
            crop: Type of crop (e.g., "Rice", "Wheat", "Potato")
            location: Optional location to filter results (state or market name)
            
        Returns:
            Dictionary with market data
        """
        try:
            self.logger.info(f"Fetching market data for {crop} in {location if location else 'all locations'}")
            
            # Prepare API request parameters
            params = {
                "api-key": self.api_key,
                "format": "json",
                "limit": 100  # Fetch reasonable number of records
            }
            
            # Add filters for crop and location if provided
            if crop:
                params["filters[commodity]"] = crop
            if location:
                # Try both state and market filters
                params["filters[state]"] = location
            
            # Make the API request
            response = requests.get(self.api_endpoint, params=params)
            
            # Check response status
            if response.status_code != 200:
                self.logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return {
                    "error": f"API request failed with status code {response.status_code}",
                    "status": "error"
                }
            
            # Parse the response
            data = response.json()
            
            # Check if we got any records
            if data.get("count", 0) == 0 or not data.get("records"):
                self.logger.warning(f"No data found for {crop} in {location if location else 'any location'}")
                
                # Try with a more general search if specific crop wasn't found
                if crop:
                    # Extract the first word of the crop name for broader matching
                    general_crop = crop.split()[0]
                    if general_crop != crop:
                        self.logger.info(f"Trying broader search with '{general_crop}'")
                        params["filters[commodity]"] = general_crop
                        response = requests.get(self.api_endpoint, params=params)
                        data = response.json() if response.status_code == 200 else {"count": 0}
                
                # If still no data, return error
                if data.get("count", 0) == 0 or not data.get("records"):
                    return {
                        "error": f"No data found for {crop} in {location if location else 'any location'}",
                        "status": "error"
                    }
            
            # Process the records
            records = data.get("records", [])
            
            # Extract the latest price data
            latest_records = []
            for record in records:
                try:
                    price = float(record.get("modal_price", "0").replace(",", ""))
                    latest_records.append({
                        "crop": record.get("commodity", crop),
                        "location": record.get("market") or record.get("state") or location or "Unknown",
                        "state": record.get("state", "Unknown"),
                        "district": record.get("district", "Unknown"),
                        "market": record.get("market", "Unknown"),
                        "price": price,
                        "min_price": float(record.get("min_price", "0").replace(",", "")),
                        "max_price": float(record.get("max_price", "0").replace(",", "")),
                        "unit": record.get("unit", "Kg"),
                        "variety": record.get("variety", "Common"),
                        "timestamp": datetime.now().isoformat(),
                        "arrival_date": record.get("arrival_date", datetime.now().strftime("%Y-%m-%d")),
                        "source": "data.gov.in"
                    })
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Could not process record: {record}. Error: {e}")
            
            if not latest_records:
                return {
                    "error": "Could not process any records from the API response",
                    "status": "error"
                }
            
            # Calculate average price if multiple records
            if len(latest_records) > 1:
                avg_price = sum(r["price"] for r in latest_records) / len(latest_records)
                price_range = (min(r["price"] for r in latest_records), max(r["price"] for r in latest_records))
                markets = list(set(r["market"] for r in latest_records))
                
                # Create a summary record
                summary_record = {
                    "crop": crop,
                    "location": location or "Multiple",
                    "price": avg_price,
                    "min_market_price": price_range[0],
                    "max_market_price": price_range[1],
                    "unit": latest_records[0]["unit"],
                    "markets": markets,
                    "timestamp": datetime.now().isoformat(),
                    "record_count": len(latest_records),
                    "source": "data.gov.in",
                    "demand": self._estimate_demand(latest_records)
                }
            else:
                # Use the single record
                record = latest_records[0]
                summary_record = {
                    "crop": record["crop"],
                    "location": record["location"],
                    "price": record["price"],
                    "min_price": record["min_price"],
                    "max_price": record["max_price"],
                    "unit": record["unit"],
                    "market": record["market"],
                    "state": record["state"],
                    "timestamp": record["timestamp"],
                    "arrival_date": record["arrival_date"],
                    "source": record["source"],
                    "demand": "Medium"  # Default when we can't estimate
                }
            
            # Store the raw records for historical analysis
            if crop not in self.market_data:
                self.market_data[crop] = []
            
            # Add new records to our historical data
            self.market_data[crop].extend(latest_records)
            self._save_market_data()
            
            return {**summary_record, "status": "success"}
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {crop}: {e}")
            return {"error": str(e), "status": "error"}
    
    def _estimate_demand(self, records: List[Dict]) -> str:
        """Estimate demand based on price trends and other factors.
        
        Args:
            records: List of market data records
            
        Returns:
            Demand level (High, Medium, Low)
        """
        try:
            # This is a simple heuristic - in a real system you'd use more factors
            if len(records) < 2:
                return "Medium"
            
            # Calculate price percentile within historical range
            prices = [r["price"] for r in records]
            avg_price = sum(prices) / len(prices)
            price_range = max(prices) - min(prices)
            
            if price_range == 0:
                return "Medium"
            
            # Normalize to 0-1 range
            normalized_price = (avg_price - min(prices)) / price_range
            
            # Determine demand level
            if normalized_price > 0.7:
                return "High"
            elif normalized_price < 0.3:
                return "Low"
            else:
                return "Medium"
                
        except Exception as e:
            self.logger.warning(f"Error estimating demand: {e}")
            return "Medium"  # Default to medium on error
    
    def update_historical_data(self, crop: str, external_data: List[Dict] = None) -> None:
        """Update historical data with external data or fetch new data.
        
        Args:
            crop: Crop name
            external_data: Optional external data to add
        """
        try:
            if external_data:
                if crop not in self.market_data:
                    self.market_data[crop] = []
                self.market_data[crop].extend(external_data)
                self._save_market_data()
                self.logger.info(f"Added {len(external_data)} historical records for {crop}")
            else:
                # Here you would implement fetching historical data from APIs
                self.logger.info(f"No external data provided for {crop}")
        except Exception as e:
            self.logger.error(f"Error updating historical data for {crop}: {e}")
    
    def train_prediction_model(self, crop: str) -> None:
        """Train a prediction model for a specific crop.
        
        Args:
            crop: Crop name to train model for
        """
        try:
            if crop not in self.market_data or not self.market_data[crop]:
                self.logger.warning(f"No data available to train model for {crop}. Fetching data first.")
                market_data = self.fetch_market_data(crop)
                if "error" in market_data:
                    self.logger.error(f"Could not fetch data for {crop}: {market_data['error']}")
                    return
            
            # Check if we have at least some data points
            if crop not in self.market_data or len(self.market_data[crop]) < 5:
                self.logger.warning(f"Insufficient data to train model for {crop}. Need at least 5 data points.")
                return
            
            # Prepare training data
            df = pd.DataFrame(self.market_data[crop])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # If we have limited data, generate synthetic data points
            if len(df) < 30:
                self.logger.info(f"Limited data for {crop} ({len(df)} points). Augmenting with synthetic data.")
                df = self._augment_training_data(df, crop)
            
            # Feature engineering
            df['price'] = df['price'].astype(float)
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['week_of_year'] = df['timestamp'].dt.isocalendar().week
            
            # Prepare features and target
            X = df[['day_of_week', 'month', 'day', 'week_of_year']].values
            y = df['price'].values
            
            # Scale the target
            scaler = MinMaxScaler()
            y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
            
            # Train a Random Forest model with fewer estimators if data is limited
            n_estimators = min(100, max(10, len(df)))
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(X, y_scaled)
            
            # Save model and scaler
            self.models[crop] = {
                'model': model,
                'scaler': scaler,
                'last_updated': datetime.now().isoformat(),
                'data_points': len(df),
                'synthetic_ratio': (len(df) - len(self.market_data[crop])) / len(df) if len(df) > 0 else 0
            }
            
            self._save_prediction_model(crop, self.models[crop])
            self.logger.info(f"Successfully trained prediction model for {crop} with {len(df)} data points")
            
        except Exception as e:
            self.logger.error(f"Error training prediction model for {crop}: {e}")
    
    def _augment_training_data(self, df, crop):
        """Augment limited training data with synthetic points.
        
        Args:
            df: DataFrame with original data
            crop: Crop name
            
        Returns:
            Augmented DataFrame
        """
        try:
            # Get basic statistics from the available data
            mean_price = df['price'].mean()
            std_price = df['price'].std() if len(df) > 1 else mean_price * 0.1
            
            # Create a date range that extends before and after our data
            first_date = df['timestamp'].min()
            last_date = df['timestamp'].max()
            
            # Create date range covering 60 days, centered around our data
            # This will help fill gaps and extend the date range
            date_range = pd.date_range(
                start=first_date - pd.Timedelta(days=30), 
                end=last_date + pd.Timedelta(days=30), 
                freq='D'
            )
            
            # Create synthetic data frame
            synthetic_df = pd.DataFrame({'timestamp': date_range})
            
            # Add some seasonal pattern - prices tend to vary by month and day of week
            synthetic_df['day_of_week'] = synthetic_df['timestamp'].dt.dayofweek
            synthetic_df['month'] = synthetic_df['timestamp'].dt.month
            synthetic_df['day'] = synthetic_df['timestamp'].dt.day
            
            # Generate synthetic prices based on patterns
            # Monthly seasonality (higher in some months)
            monthly_factor = np.sin(synthetic_df['month'] * (2 * np.pi / 12)) * 0.2 + 1
            # Day of week effect (e.g., higher on weekends)
            dow_factor = np.sin(synthetic_df['day_of_week'] * (2 * np.pi / 7)) * 0.1 + 1
            
            # Generate base price with some random noise
            np.random.seed(42)  # For reproducibility
            random_noise = np.random.normal(0, std_price * 0.5, len(synthetic_df))
            
            # Combine factors to create synthetic prices
            synthetic_df['price'] = mean_price * monthly_factor * dow_factor + random_noise
            
            # Add other required columns with placeholder values
            location = df['location'].iloc[0] if 'location' in df.columns else "Unknown"
            synthetic_df['location'] = location
            
            # For other fields, use dictionaries with defaults
            synthetic_records = []
            for _, row in synthetic_df.iterrows():
                record = {
                    'crop': crop,
                    'timestamp': row['timestamp'].isoformat(),
                    'price': row['price'],
                    'location': location,
                    'source': 'synthetic',
                    'unit': df['unit'].iloc[0] if 'unit' in df.columns else "Kg",
                    'demand': 'Medium',
                    'is_synthetic': True
                }
                synthetic_records.append(record)
            
            # Merge real data with synthetic data
            augmented_df = pd.DataFrame(synthetic_records + self.market_data[crop])
            augmented_df['timestamp'] = pd.to_datetime(augmented_df['timestamp'])
            augmented_df = augmented_df.sort_values('timestamp')
            
            # Remove duplicates (prefer real data over synthetic)
            augmented_df = augmented_df.drop_duplicates(subset=['timestamp', 'crop'], keep='last')
            
            self.logger.info(f"Augmented data for {crop}: {len(df)} real points + {len(augmented_df) - len(df)} synthetic points")
            return augmented_df
            
        except Exception as e:
            self.logger.error(f"Error augmenting training data: {e}")
            return df  # Return original data if augmentation fails
    
    def predict_price(self, crop: str, days_ahead: int = 7) -> Dict:
        """Predict future price for a specific crop.
        
        Args:
            crop: Crop name
            days_ahead: Number of days to predict ahead
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if crop not in self.models:
                self.logger.warning(f"No prediction model available for {crop}. Training now...")
                self.train_prediction_model(crop)
                
                if crop not in self.models:
                    return {
                        "error": "Insufficient data to make prediction",
                        "status": "error"
                    }
            
            # Get future dates
            future_dates = [datetime.now() + timedelta(days=i) for i in range(1, days_ahead+1)]
            
            # Prepare features for prediction
            X_pred = []
            for d in future_dates:
                # Calculate the week of the year
                week_of_year = d.isocalendar()[1]  # Week number
                X_pred.append([d.weekday(), d.month, d.day, week_of_year])
            X_pred = np.array(X_pred)
            
            # Make prediction
            model = self.models[crop]['model']
            scaler = self.models[crop]['scaler']
            
            y_pred_scaled = model.predict(X_pred)
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            # Get current price for comparison
            if crop in self.market_data and self.market_data[crop]:
                # Filter out synthetic data points for current price
                real_data = [d for d in self.market_data[crop] if d.get('source') != 'synthetic']
                if real_data:
                    latest_data = sorted(real_data, key=lambda x: x['timestamp'], reverse=True)[0]
                    current_price = latest_data['price']
                else:
                    # If no real data, use the latest of any type
                    latest_data = sorted(self.market_data[crop], key=lambda x: x['timestamp'], reverse=True)[0]
                    current_price = latest_data['price']
            else:
                # If no data at all, fetch from API
                market_data = self.fetch_market_data(crop)
                if market_data.get("status") == "success":
                    current_price = market_data["price"]
                else:
                    current_price = float(y_pred[0]) * 0.9  # Fallback - use 90% of first prediction
            
            # Calculate confidence based on data quality
            if crop in self.models:
                data_points = self.models[crop].get('data_points', 0)
                synthetic_ratio = self.models[crop].get('synthetic_ratio', 0)
                
                # Higher confidence with more data points and less synthetic data
                base_confidence = min(0.9, max(0.5, data_points / 100))
                synthetic_penalty = synthetic_ratio * 0.3  # Penalize for synthetic data
                confidence = base_confidence - synthetic_penalty
            else:
                confidence = 0.5  # Default confidence
            
            # Format results with dynamic confidence
            predictions = []
            for i, date in enumerate(future_dates):
                # Reduce confidence as we predict further into the future
                day_penalty = i * 0.01  # Decrease confidence by 1% per day
                day_confidence = max(0.3, confidence - day_penalty)
                
                predictions.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "predicted_price": float(y_pred[i]),
                    "confidence": day_confidence
                })
            
            # Determine best selling time
            best_price = max(y_pred)
            best_day_index = np.argmax(y_pred)
            best_day = future_dates[best_day_index].strftime("%Y-%m-%d")
            
            result = {
                "crop": crop,
                "current_price": float(current_price),
                "predictions": predictions,
                "best_selling_date": best_day,
                "best_selling_price": float(best_price),
                "price_trend": "increasing" if best_price > current_price else "decreasing",
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting price for {crop}: {e}")
            return {"error": str(e), "status": "error"}
    
    def generate_farmer_alert(self, crop: str, farmer_id: str, contact_info: Dict) -> Dict:
        """Generate an alert for a farmer about optimal selling time.
        
        Args:
            crop: Crop name
            farmer_id: Farmer's ID
            contact_info: Farmer's contact information
            
        Returns:
            Alert information
        """
        try:
            prediction = self.predict_price(crop)
            
            if prediction.get("status") != "success":
                return {"status": "failed", "error": prediction.get("error", "Prediction failed")}
            
            current_price = prediction["current_price"]
            best_price = prediction["best_selling_price"]
            best_date = prediction["best_selling_date"]
            
            # Calculate price difference percentage
            price_diff_pct = ((best_price - current_price) / current_price) * 100 if current_price else 0
            
            # Determine if alert should be sent (e.g., if predicted price is 10% higher)
            should_alert = price_diff_pct >= 10
            
            if should_alert:
                alert_message = (
                    f"PRICE ALERT: {crop} prices are predicted to {prediction['price_trend']}. "
                    f"Current price: ₹{current_price:.2f}. "
                    f"Best selling price in next week: ₹{best_price:.2f} on {best_date}. "
                    f"This is {price_diff_pct:.1f}% higher than current price."
                )
                
                alert_data = {
                    "farmer_id": farmer_id,
                    "crop": crop,
                    "message": alert_message,
                    "contact": contact_info,
                    "timestamp": datetime.now().isoformat(),
                    "prediction": prediction,
                    "status": "ready_to_send"
                }
                
                # In a real system, you would save this alert to a database
                # and later use n8n to actually send it
                
                return alert_data
            else:
                return {
                    "status": "no_alert_needed",
                    "reason": f"Price difference ({price_diff_pct:.1f}%) is less than threshold (10%)"
                }
            
        except Exception as e:
            self.logger.error(f"Error generating farmer alert for {crop}: {e}")
            return {"status": "failed", "error": str(e)}
    
    def get_n8n_workflow_config(self) -> Dict:
        """Get n8n workflow configuration for price predictions and alerts.
        
        Returns:
            n8n workflow configuration
        """
        # Create a standard n8n workflow configuration
        workflow = {
            "name": "Agricultural Price Prediction & Alerts",
            "nodes": [
                {
                    "id": "scheduler",
                    "type": "n8n-nodes-base.cron",
                    "name": "Daily Data Update",
                    "parameters": {
                        "triggerTimes": {
                            "item": [
                                {
                                    "mode": "everyDay"
                                }
                            ]
                        }
                    }
                },
                {
                    "id": "fetch_data",
                    "type": "n8n-nodes-base.httpRequest",
                    "name": "Fetch Market Data",
                    "parameters": {
                        "url": self.api_endpoint,
                        "method": "GET",
                        "queryParameters": {
                            "api-key": self.api_key,
                            "format": "json",
                            "limit": "100"
                        }
                    }
                },
                {
                    "id": "process_data",
                    "type": "n8n-nodes-base.function",
                    "name": "Process Data",
                    "parameters": {
                        "functionCode": "// Process the market data\nconst data = $input.item.json.records;\nreturn data.map(record => ({\n  crop: record.commodity,\n  price: parseFloat(record.modal_price.replace(',', '')),\n  market: record.market,\n  state: record.state,\n  date: record.arrival_date\n}));"
                    }
                },
                {
                    "id": "predict_prices",
                    "type": "n8n-nodes-base.function",
                    "name": "Predict Prices",
                    "parameters": {
                        "functionCode": "// This would call the price prediction API\nconst crops = ['Rice', 'Wheat', 'Potato', 'Onion', 'Tomato'];\nreturn crops.map(crop => ({\n  crop,\n  prediction: {\n    current_price: Math.random() * 100 + 50,\n    future_price: Math.random() * 100 + 60,\n    trend: Math.random() > 0.5 ? 'up' : 'down'\n  }\n}));"
                    }
                },
                {
                    "id": "check_alerts",
                    "type": "n8n-nodes-base.if",
                    "name": "Check Alert Conditions",
                    "parameters": {
                        "conditions": {
                            "string": [
                                {
                                    "value1": "={{$json[\"prediction\"][\"trend\"]}}",
                                    "operation": "equals",
                                    "value2": "up"
                                }
                            ]
                        }
                    }
                },
                {
                    "id": "send_sms",
                    "type": "n8n-nodes-base.twilioSms",
                    "name": "Send SMS Alert",
                    "parameters": {
                        "accountSid": "TWILIO_ACCOUNT_SID",
                        "authToken": "TWILIO_AUTH_TOKEN",
                        "from": "TWILIO_FROM_NUMBER",
                        "to": "={{$json[\"farmer_phone\"]}}",
                        "message": "Price Alert: {{$json[\"crop\"]}} prices are predicted to {{$json[\"prediction\"][\"trend\"]}} to ₹{{$json[\"prediction\"][\"future_price\"].toFixed(2)}} in the coming week. Consider selling now or waiting based on this forecast."
                    }
                },
                {
                    "id": "save_data",
                    "type": "n8n-nodes-base.function",
                    "name": "Save Historical Data",
                    "parameters": {
                        "functionCode": "// Code to save data to database or file\nconsole.log('Saving data for future analytics');\nreturn $input.item;"
                    }
                }
            ],
            "connections": {
                "scheduler": {
                    "main": [
                        [
                            {
                                "node": "fetch_data",
                                "type": "main",
                                "index": 0
                            }
                        ]
                    ]
                },
                "fetch_data": {
                    "main": [
                        [
                            {
                                "node": "process_data",
                                "type": "main",
                                "index": 0
                            }
                        ]
                    ]
                },
                "process_data": {
                    "main": [
                        [
                            {
                                "node": "predict_prices",
                                "type": "main",
                                "index": 0
                            }
                        ]
                    ]
                },
                "predict_prices": {
                    "main": [
                        [
                            {
                                "node": "check_alerts",
                                "type": "main",
                                "index": 0
                            },
                            {
                                "node": "save_data",
                                "type": "main",
                                "index": 0
                            }
                        ]
                    ]
                },
                "check_alerts": {
                    "true": [
                        [
                            {
                                "node": "send_sms",
                                "type": "main",
                                "index": 0
                            }
                        ]
                    ]
                }
            }
        }
        
        return workflow
        
    def get_market_trends(self) -> Dict:
        """Get current market trends for top crops.
        
        Returns:
            Dictionary mapping crop names to trend information
        """
        trends = {}
        
        # List of common crops to analyze
        common_crops = ["Rice", "Wheat", "Potato", "Onion", "Tomato", "Soybean", 
                        "Cotton", "Sugarcane", "Maize", "Chickpea"]
        
        # Get trends for each crop
        for crop in common_crops:
            if crop in self.market_data and len(self.market_data[crop]) >= 2:
                # Sort data by date
                crop_data = sorted(self.market_data[crop], 
                                 key=lambda x: x.get("timestamp", "0"))
                
                # Calculate trend over the last month
                recent_data = crop_data[-30:] if len(crop_data) > 30 else crop_data
                
                if len(recent_data) >= 2:
                    oldest_price = recent_data[0].get("price", 0)
                    newest_price = recent_data[-1].get("price", 0)
                    
                    if oldest_price > 0:
                        change_pct = ((newest_price - oldest_price) / oldest_price) * 100
                        
                        # Determine trend direction
                        if change_pct > 5:
                            direction = "up"
                            description = "Strong upward trend"
                        elif change_pct > 1:
                            direction = "up"
                            description = "Slight upward trend"
                        elif change_pct < -5:
                            direction = "down"
                            description = "Strong downward trend"
                        elif change_pct < -1:
                            direction = "down"
                            description = "Slight downward trend"
                        else:
                            direction = "stable"
                            description = "Stable prices"
                        
                        trends[crop] = {
                            "direction": direction,
                            "percentage": round(change_pct, 2),
                            "description": description,
                            "current_price": newest_price,
                            "time_period": f"Last {len(recent_data)} days"
                        }
            
            # If no data available, try to fetch current data
            if crop not in trends:
                try:
                    market_data = self.fetch_market_data(crop)
                    if market_data.get("status") == "success":
                        # For newly fetched data without historical context, use a neutral assessment
                        trends[crop] = {
                            "direction": "stable",
                            "percentage": 0.0,
                            "description": "Insufficient historical data for trend analysis",
                            "current_price": market_data.get("price", 0),
                            "time_period": "Current data only"
                        }
                except Exception as e:
                    self.logger.error(f"Error fetching trend data for {crop}: {e}")
        
        # If still no trends (e.g., no data available at all), use some fallback data
        if not trends:
            # Provide simulated trends as fallback
            fallback_trends = {
                "Rice": {"direction": "up", "percentage": 3.5, "description": "Gradually increasing", "time_period": "Estimated"},
                "Wheat": {"direction": "stable", "percentage": 0.2, "description": "Stable market", "time_period": "Estimated"},
                "Potato": {"direction": "down", "percentage": -2.1, "description": "Slight decrease", "time_period": "Estimated"},
                "Onion": {"direction": "up", "percentage": 7.8, "description": "Sharp increase due to lower supply", "time_period": "Estimated"},
                "Tomato": {"direction": "down", "percentage": -4.5, "description": "Decreasing due to good harvest", "time_period": "Estimated"}
            }
            # Add a note that these are estimated
            for crop, data in fallback_trends.items():
                data["note"] = "Estimated data - no real-time data available"
            trends = fallback_trends
        
        return trends
    
    def compare_crops(self, crop_list: List[str]) -> Dict:
        """Compare prices and trends between different crops.
        
        Args:
            crop_list: List of crop names to compare
            
        Returns:
            Dictionary with comparison data
        """
        comparison = {}
        
        for crop in crop_list:
            try:
                # Try to get current market data
                market_data = self.fetch_market_data(crop)
                
                if market_data.get("status") == "success":
                    # Calculate 7-day trend if we have historical data
                    trend = 0
                    if crop in self.market_data and len(self.market_data[crop]) >= 7:
                        recent_data = sorted(self.market_data[crop][-7:], 
                                           key=lambda x: x.get("timestamp", "0"))
                        if len(recent_data) >= 2:
                            oldest = recent_data[0].get("price", 0)
                            newest = recent_data[-1].get("price", 0)
                            if oldest > 0:
                                trend = ((newest - oldest) / oldest) * 100
                    
                    # Estimate profit margin based on current price and production costs
                    # (This is a simplified estimate)
                    production_cost = market_data.get("price", 0) * 0.6  # Assume cost is 60% of market price
                    profit_margin = ((market_data.get("price", 0) - production_cost) / market_data.get("price", 0)) * 100 if market_data.get("price", 0) > 0 else 0
                    
                    comparison[crop] = {
                        "price": market_data.get("price", 0),
                        "unit": market_data.get("unit", "Kg"),
                        "trend": round(trend, 2),
                        "profit_margin": round(profit_margin, 2),
                        "demand": market_data.get("demand", "Medium")
                    }
            except Exception as e:
                self.logger.error(f"Error comparing crop {crop}: {e}")
                comparison[crop] = {
                    "price": 0,
                    "unit": "Kg",
                    "trend": 0,
                    "profit_margin": 0,
                    "demand": "Unknown",
                    "error": str(e)
                }
        
        # Add a recommendation based on the comparison
        if comparison:
            best_crop = max(comparison.items(), key=lambda x: x[1]['profit_margin'] + x[1]['trend'])
            comparison["recommendation"] = f"{best_crop[0]} currently shows the best balance of price stability and profit margin."
        
        return comparison
    
    def get_price_history(self, crop: str, months: int = 6) -> List[Dict]:
        """Get historical price data for a crop.
        
        Args:
            crop: Crop name
            months: Number of months of history to retrieve
            
        Returns:
            List of monthly price data
        """
        history = []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * months)
        
        # Check if we have data for this crop
        if crop in self.market_data and self.market_data[crop]:
            # Filter data by date range
            data_in_range = []
            for record in self.market_data[crop]:
                try:
                    timestamp = record.get("timestamp", "")
                    if timestamp:
                        record_date = datetime.fromisoformat(timestamp)
                        if start_date <= record_date <= end_date:
                            data_in_range.append(record)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Error parsing date in record: {e}")
            
            # If we have data in range
            if data_in_range:
                # Group by month
                monthly_data = {}
                for record in data_in_range:
                    try:
                        timestamp = record.get("timestamp", "")
                        if timestamp:
                            record_date = datetime.fromisoformat(timestamp)
                            month_key = record_date.strftime("%Y-%m")
                            if month_key not in monthly_data:
                                monthly_data[month_key] = []
                            monthly_data[month_key].append(record)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error processing record for monthly grouping: {e}")
                
                # Calculate monthly statistics
                for month, records in monthly_data.items():
                    prices = [r.get("price", 0) for r in records if r.get("price", 0) > 0]
                    if prices:
                        history.append({
                            "month": month,
                            "avg_price": sum(prices) / len(prices),
                            "min_price": min(prices),
                            "max_price": max(prices),
                            "records": len(records)
                        })
                
                # Sort by month
                history.sort(key=lambda x: x["month"])
                
                # Add seasonality insights if we have enough data
                if len(history) >= 3:
                    # Simple seasonality detection
                    price_trend = "increasing" if history[-1]["avg_price"] > history[0]["avg_price"] else "decreasing"
                    seasonality = f"Overall {price_trend} price trend. "
                    
                    high_price_months = sorted(history, key=lambda x: x["avg_price"], reverse=True)[:2]
                    high_price_months_str = ", ".join([h["month"] for h in high_price_months])
                    
                    low_price_months = sorted(history, key=lambda x: x["avg_price"])[:2]
                    low_price_months_str = ", ".join([l["month"] for l in low_price_months])
                    
                    seasonality += f"Highest prices observed in {high_price_months_str}. Lowest prices in {low_price_months_str}."
                    history[0]["seasonality"] = seasonality
        
        # If no historical data found, create simulated data
        if not history:
            self.logger.warning(f"No historical data found for {crop}. Creating simulated data.")
            
            # Generate simulated monthly data
            current_price = 100  # Default starting price
            
            # Try to get actual current price
            try:
                market_data = self.fetch_market_data(crop)
                if market_data.get("status") == "success":
                    current_price = market_data.get("price", 100)
            except Exception:
                pass
            
            # Generate monthly data with some randomness
            base_price = current_price * 0.8  # Start a bit lower than current
            for i in range(months):
                month_date = end_date - timedelta(days=30 * (months - i - 1))
                month_str = month_date.strftime("%Y-%m")
                
                # Add some seasonal variation (higher in middle months)
                seasonal_factor = 1 + 0.1 * np.sin(np.pi * i / months)
                # Add some random noise
                random_factor = 1 + 0.05 * (np.random.random() - 0.5)
                
                month_price = base_price * seasonal_factor * random_factor
                min_price = month_price * 0.9
                max_price = month_price * 1.1
                
                history.append({
                    "month": month_str,
                    "avg_price": month_price,
                    "min_price": min_price,
                    "max_price": max_price,
                    "records": 10,  # Simulated record count
                    "simulated": True  # Flag to indicate this is simulated data
                })
                
                # Gradually trend toward current price
                base_price = base_price * 0.9 + current_price * 0.1
            
            # Add seasonality note
            history[0]["seasonality"] = "Based on simulated data. Historical price patterns suggest buying during harvest season and selling during off-season may maximize profits."
        
        return history
    
    def get_crop_recommendations(self, location: str, season: str) -> Dict:
        """Get personalized crop recommendations based on location and season.
        
        Args:
            location: Farmer's location
            season: Current season (summer/winter/rainy)
            
        Returns:
            Dictionary with crop recommendations
        """
        # Standardize season input
        season = season.lower()
        if season not in ["summer", "winter", "rainy", "monsoon"]:
            season = "summer"  # Default to summer if invalid
        
        # Map "monsoon" to "rainy"
        if season == "monsoon":
            season = "rainy"
        
        # Regional crop recommendations based on location and season
        regional_crops = {
            "north": {
                "summer": ["Millet", "Maize", "Cotton", "Rice"],
                "winter": ["Wheat", "Mustard", "Chickpea", "Potato"],
                "rainy": ["Rice", "Soybean", "Groundnut", "Sugarcane"]
            },
            "south": {
                "summer": ["Rice", "Groundnut", "Sugarcane", "Cotton"],
                "winter": ["Rice", "Pulses", "Vegetables", "Wheat"],
                "rainy": ["Rice", "Millets", "Pulses", "Coconut"]
            },
            "east": {
                "summer": ["Rice", "Jute", "Maize", "Vegetables"],
                "winter": ["Wheat", "Pulses", "Oilseeds", "Potato"],
                "rainy": ["Rice", "Jute", "Sugarcane", "Maize"]
            },
            "west": {
                "summer": ["Cotton", "Groundnut", "Castor", "Vegetables"],
                "winter": ["Wheat", "Mustard", "Gram", "Barley"],
                "rainy": ["Cotton", "Groundnut", "Maize", "Pulses"]
            },
            "central": {
                "summer": ["Cotton", "Soybean", "Vegetables", "Maize"],
                "winter": ["Wheat", "Gram", "Lentil", "Safflower"],
                "rainy": ["Soybean", "Cotton", "Pigeon Pea", "Rice"]
            }
        }
        
        # Determine region based on location
        north_states = ["Punjab", "Haryana", "Uttar Pradesh", "Jammu and Kashmir", "Himachal Pradesh", "Uttarakhand"]
        south_states = ["Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh", "Telangana"]
        east_states = ["West Bengal", "Bihar", "Odisha", "Jharkhand", "Assam"]
        west_states = ["Gujarat", "Maharashtra", "Rajasthan", "Goa"]
        central_states = ["Madhya Pradesh", "Chhattisgarh"]
        
        region = "central"  # Default region
        
        # Check if location matches any state
        location_lower = location.lower()
        for state in north_states:
            if state.lower() in location_lower:
                region = "north"
                break
        for state in south_states:
            if state.lower() in location_lower:
                region = "south"
                break
        for state in east_states:
            if state.lower() in location_lower:
                region = "east"
                break
        for state in west_states:
            if state.lower() in location_lower:
                region = "west"
                break
        for state in central_states:
            if state.lower() in location_lower:
                region = "central"
                break
        
        # Get recommended crops for the region and season
        recommended_crops = regional_crops[region][season]
        
        # Gather market data for the recommended crops
        top_crops = []
        
        for crop in recommended_crops:
            try:
                # Get current market data
                market_data = self.fetch_market_data(crop)
                if market_data.get("status") == "success":
                    price = market_data.get("price", 0)
                    unit = market_data.get("unit", "Kg")
                else:
                    # If API fetch fails, use estimated prices
                    price = {
                        "Rice": 40, "Wheat": 30, "Maize": 25, "Cotton": 80, "Millet": 35,
                        "Potato": 20, "Groundnut": 75, "Soybean": 45, "Sugarcane": 3,
                        "Pulses": 70, "Vegetables": 30, "Coconut": 25, "Jute": 50,
                        "Oilseeds": 60, "Castor": 85, "Mustard": 55, "Gram": 60,
                        "Barley": 35, "Lentil": 75, "Safflower": 70, "Pigeon Pea": 85
                    }.get(crop, 50)
                    unit = "Kg"
                
                # Get prediction for this crop
                prediction = self.predict_price(crop, 30)
                
                # Calculate profit potential
                if prediction.get("status") == "success":
                    profit_potential = ((prediction.get("best_selling_price", price) - price) / price) * 100 if price > 0 else 10
                else:
                    profit_potential = 10  # Default
                
                # Crop-specific growing periods (simplified)
                growing_periods = {
                    "Rice": 120, "Wheat": 100, "Maize": 90, "Cotton": 160, "Millet": 80,
                    "Potato": 70, "Groundnut": 110, "Soybean": 100, "Sugarcane": 365,
                    "Pulses": 90, "Vegetables": 60, "Coconut": 365, "Jute": 120,
                    "Oilseeds": 110, "Castor": 150, "Mustard": 120, "Gram": 100,
                    "Barley": 90, "Lentil": 100, "Safflower": 130, "Pigeon Pea": 150
                }
                
                # Reason for recommendation
                reasons = {
                    "north": {
                        "summer": "suitable for hot and dry climate",
                        "winter": "cold-tolerant and good market demand",
                        "rainy": "requires good rainfall and humidity"
                    },
                    "south": {
                        "summer": "heat resistant and water efficient",
                        "winter": "mild winter suitable for these crops",
                        "rainy": "benefits from monsoon rainfall patterns"
                    },
                    "east": {
                        "summer": "tolerates high humidity and heat",
                        "winter": "performs well in mild eastern winters",
                        "rainy": "thrives in high rainfall conditions"
                    },
                    "west": {
                        "summer": "drought resistant and heat tolerant",
                        "winter": "suitable for dry western climate",
                        "rainy": "efficient use of limited monsoon rainfall"
                    },
                    "central": {
                        "summer": "heat resistant and low water requirement",
                        "winter": "suitable for central India's climate",
                        "rainy": "good yield with moderate rainfall"
                    }
                }
                
                crop_info = {
                    "name": crop,
                    "expected_price": price,
                    "unit": unit,
                    "growing_period": growing_periods.get(crop, 100),
                    "profit_potential": round(profit_potential, 1),
                    "reason": f"Well-suited for {region}ern {location} during {season} season - {reasons[region][season]}"
                }
                
                # Add prediction info if available
                if prediction.get("status") == "success":
                    crop_info["price_trend"] = prediction.get("price_trend", "stable")
                    crop_info["future_price"] = prediction.get("best_selling_price", price)
                
                top_crops.append(crop_info)
            except Exception as e:
                self.logger.error(f"Error processing recommendation for {crop}: {e}")
        
        # Sort crops by profit potential
        top_crops = sorted(top_crops, key=lambda x: x["profit_potential"], reverse=True)
        
        # Generate market insights
        market_insights = f"For {season} season in {location}, consider {top_crops[0]['name']} and {top_crops[1]['name']} for best returns. Current market conditions suggest a {top_crops[0]['profit_potential']}% potential profit margin for {top_crops[0]['name']}."
        
        return {
            "top_crops": top_crops,
            "region": region,
            "season": season,
            "location": location,
            "market_insights": market_insights,
            "timestamp": datetime.now().isoformat()
        } 