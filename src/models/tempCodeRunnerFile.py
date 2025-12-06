import pandas as pd
import numpy as np
import joblib
import logging
import os
from typing import Dict, Any, Union, Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictor:
    """
    Robust Model Predictor for HCC Recurrence Prediction
    Handles model loading, feature preprocessing, and prediction with comprehensive error handling
    """
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        """
        Initialize the predictor with model and preprocessor paths
        """
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self.actual_feature_names = []  # Store actual feature names
        self.model_expected_features = 0
        self.is_loaded = False
        self.feature_mapping = {}  # Map actual names to generic names
        
        if model_path and preprocessor_path:
            self.load_model_and_preprocessor(model_path, preprocessor_path)
    
    def find_model_files(self):
        """Find model and preprocessor files in common locations"""
        possible_paths = [
            "models/trained_models/best_model.pkl",
            "models/trained_models/random_forest.pkl", 
            "../models/trained_models/best_model.pkl",
        ]
        
        possible_preprocessor_paths = [
            "models/preprocessing/preprocessor.pkl",
            "../models/preprocessing/preprocessor.pkl", 
        ]
        
        model_path = None
        preprocessor_path = None
        
        # Find model file
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                logger.info(f"Found model at: {path}")
                break
        
        # Find preprocessor file
        for path in possible_preprocessor_paths:
            if os.path.exists(path):
                preprocessor_path = path
                logger.info(f"Found preprocessor at: {path}")
                break
                
        return model_path, preprocessor_path
    
    def load_model_and_preprocessor(self, model_path: str = None, preprocessor_path: str = None) -> bool:
        """
        Load model and preprocessor with robust error handling
        """
        try:
            logger.info("Loading model and preprocessor...")
            
            # If no paths provided, try to find them
            if not model_path or not preprocessor_path:
                logger.info("Searching for model files...")
                model_path, preprocessor_path = self.find_model_files()
                
            if not model_path or not preprocessor_path:
                logger.error("Could not find model or preprocessor files")
                return False
            
            # Validate paths
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
                
            if not os.path.exists(preprocessor_path):
                logger.error(f"Preprocessor file not found: {preprocessor_path}")
                return False
            
            # Load model
            self.model = joblib.load(model_path)
            logger.info(f"✅ Model loaded successfully: {type(self.model).__name__}")
            
            # Load preprocessor
            preprocessor_data = joblib.load(preprocessor_path)
            
            if isinstance(preprocessor_data, dict):
                self.preprocessor = preprocessor_data.get('pipeline')
                self.feature_names = preprocessor_data.get('feature_names', [])
                # Try to get actual feature names from config
                self.actual_feature_names = preprocessor_data.get('original_features', [])
            else:
                self.preprocessor = preprocessor_data
                # Try to extract feature names
                try:
                    if hasattr(self.preprocessor, 'get_feature_names_out'):
                        self.feature_names = self.preprocessor.get_feature_names_out().tolist()
                except:
                    logger.warning("Could not extract feature names from preprocessor")
            
            # Create feature mapping if we have generic feature names
            self._create_feature_mapping()
            
            # Get expected feature count
            if hasattr(self.model, 'n_features_in_'):
                self.model_expected_features = self.model.n_features_in_
            else:
                self.model_expected_features = len(self.feature_names) if self.feature_names else 0
            
            logger.info(f"✅ Model expects {self.model_expected_features} features")
            logger.info(f"✅ Feature names type: {type(self.feature_names[0]) if self.feature_names else 'None'}")
            if self.feature_names:
                logger.info(f"✅ First few feature names: {self.feature_names[:10]}")
            
            self.is_loaded = True
            logger.info("✅ Model and preprocessor loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model/preprocessor: {e}")
            self.is_loaded = False
            return False
    
    def _create_feature_mapping(self):
        """Create mapping between actual feature names and generic feature names"""
        # Common HCC features in expected order
        expected_features = [
            'Age', 'Gender', 'Symptoms', 'PS', 'Encephalopathy', 'Ascites',
            'AFP', 'ALT', 'AST', 'Albumin', 'Total_Bil', 'INR', 'Platelets', 
            'Creatinine', 'Nodules', 'Major_Dim', 'Alcohol', 'HBsAg', 'HCVAb',
            'Cirrhosis', 'Diabetes', 'Smoking'
        ]
        
        # If we have generic feature names, create mapping
        if self.feature_names and all(isinstance(f, str) and f.startswith('feature_') for f in self.feature_names):
            logger.info("Creating feature mapping for generic feature names...")
            for i, actual_feature in enumerate(expected_features):
                if i < len(self.feature_names):
                    self.feature_mapping[actual_feature] = self.feature_names[i]
                    logger.info(f"  {actual_feature} -> {self.feature_names[i]}")
        
        self.actual_feature_names = expected_features
    
    def validate_input(self, input_data: Union[pd.DataFrame, Dict, np.ndarray]) -> pd.DataFrame:
        """
        Validate and convert input data to proper format
        """
        try:
            if isinstance(input_data, pd.DataFrame):
                df = input_data.copy()
            elif isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, np.ndarray):
                if input_data.ndim == 1:
                    input_data = input_data.reshape(1, -1)
                df = pd.DataFrame(input_data, columns=[f'feature_{i}' for i in range(input_data.shape[1])])
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            # Convert all columns to numeric
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(0.0)
                except Exception as e:
                    logger.warning(f"Could not convert column {col} to numeric: {e}")
                    df[col] = 0.0
            
            logger.info(f"✅ Input validated. Shape: {df.shape}")
            logger.info(f"✅ Input columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Input validation failed: {e}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering - now creates compatible features
        """
        try:
            df_eng = df.copy()
            
            # If we have feature mapping, use it to create compatible features
            if self.feature_mapping:
                # Age Category
                if 'Age' in df_eng.columns:
                    age = df_eng['Age'].iloc[0]
                    age_young = 1.0 if age < 50 else 0.0
                    age_middle = 1.0 if 50 <= age <= 65 else 0.0
                    age_senior = 1.0 if age > 65 else 0.0
                    
                    # Add these as separate columns if they're in our mapping
                    for feature_name, generic_name in self.feature_mapping.items():
                        if 'Age_Category' in feature_name:
                            if 'Young' in feature_name:
                                df_eng[generic_name] = age_young
                            elif 'Middle' in feature_name:
                                df_eng[generic_name] = age_middle
                            elif 'Senior' in feature_name:
                                df_eng[generic_name] = age_senior
                
                # AFP Risk Category
                if 'AFP' in df_eng.columns:
                    afp = df_eng['AFP'].iloc[0]
                    afp_low = 1.0 if afp < 20 else 0.0
                    afp_intermediate = 1.0 if 20 <= afp <= 400 else 0.0
                    afp_high = 1.0 if afp > 400 else 0.0
                    
                    for feature_name, generic_name in self.feature_mapping.items():
                        if 'AFP_Risk' in feature_name:
                            if 'Low' in feature_name:
                                df_eng[generic_name] = afp_low
                            elif 'Intermediate' in feature_name:
                                df_eng[generic_name] = afp_intermediate
                            elif 'High' in feature_name:
                                df_eng[generic_name] = afp_high
            
            logger.info("✅ Feature engineering completed")
            return df_eng
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            return df
    
    def preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess features using the loaded preprocessor
        """
        if self.preprocessor is None:
            logger.warning("No preprocessor loaded, returning raw features")
            return df.values
        
        try:
            # If we have generic feature names, map actual features to generic ones
            if self.feature_mapping:
                logger.info("Mapping actual features to generic feature names...")
                mapped_df = pd.DataFrame()
                
                # Create a dataframe with generic feature names
                for generic_name in self.feature_names:
                    mapped_df[generic_name] = [0.0] * len(df)
                
                # Map actual values to generic columns
                for actual_name, generic_name in self.feature_mapping.items():
                    if actual_name in df.columns:
                        mapped_df[generic_name] = df[actual_name]
                        logger.info(f"Mapped {actual_name} -> {generic_name}")
                
                df = mapped_df
            
            # Apply preprocessing
            processed_features = self.preprocessor.transform(df)
            
            logger.info(f"✅ Features preprocessed. Output shape: {processed_features.shape}")
            return processed_features
            
        except Exception as e:
            logger.error(f"❌ Feature preprocessing failed: {e}")
            # Fallback: direct feature alignment
            logger.info("Attempting fallback prediction with available features...")
            try:
                aligned_features = np.zeros((df.shape[0], self.model_expected_features))
                
                # If we have feature mapping, use it
                if self.feature_mapping:
                    for i, generic_name in enumerate(self.feature_names):
                        for actual_name, mapped_name in self.feature_mapping.items():
                            if mapped_name == generic_name and actual_name in df.columns:
                                aligned_features[:, i] = df[actual_name].values
                                break
                
                return aligned_features
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                # Last resort: return zeros
                return np.zeros((df.shape[0], self.model_expected_features))
    
    def predict(self, input_data: Union[pd.DataFrame, Dict, np.ndarray]) -> np.ndarray:
        """
        Make prediction on input data
        """
        if not self.is_loaded:
            raise Exception("Model not loaded. Call load_model_and_preprocessor first.")
        
        try:
            # 1. Validate input
            df = self.validate_input(input_data)
            
            # 2. Engineer features
            df_eng = self.engineer_features(df)
            
            # 3. Preprocess features
            processed_features = self.preprocess_features(df_eng)
            
            # 4. Make prediction
            predictions = self.model.predict(processed_features)
            
            logger.info(f"✅ Prediction completed. Results: {np.unique(predictions, return_counts=True)}")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise
    
    def predict_proba(self, input_data: Union[pd.DataFrame, Dict, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities
        """
        if not self.is_loaded:
            raise Exception("Model not loaded. Call load_model_and_preprocessor first.")
        
        if not hasattr(self.model, "predict_proba"):
            raise Exception("Model does not support probability predictions")
        
        try:
            # 1. Validate input
            df = self.validate_input(input_data)
            
            # 2. Engineer features
            df_eng = self.engineer_features(df)
            
            # 3. Preprocess features
            processed_features = self.preprocess_features(df_eng)
            
            # 4. Get probabilities
            probabilities = self.model.predict_proba(processed_features)
            
            logger.info("✅ Probability prediction completed")
            return probabilities
            
        except Exception as e:
            logger.error(f"❌ Probability prediction failed: {e}")
            raise
    
    def predict_with_confidence(self, input_data: Union[pd.DataFrame, Dict, np.ndarray]) -> Dict[str, Any]:
        """
        Make prediction with confidence scores and detailed information
        """
        try:
            # Get predictions and probabilities
            predictions = self.predict(input_data)
            probabilities = self.predict_proba(input_data) if hasattr(self.model, "predict_proba") else None
            
            # Prepare results
            results = {
                "success": True,
                "predictions": predictions.tolist(),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            # Add probabilities if available
            if probabilities is not None:
                results["probabilities"] = probabilities.tolist()
                results["confidence"] = np.max(probabilities, axis=1).tolist()
                
                # For binary classification, get probability of positive class
                if probabilities.shape[1] == 2:
                    results["recurrence_probability"] = probabilities[:, 1].tolist()
                    results["risk_level"] = [self._get_risk_level(prob) for prob in probabilities[:, 1]]
            
            # Add explanations
            if 'recurrence_probability' in results:
                results["explanations"] = [
                    self._generate_explanation(pred, prob)
                    for pred, prob in zip(results["predictions"], results["recurrence_probability"])
                ]
            
            logger.info("✅ Comprehensive prediction completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive prediction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "predictions": [],
                "timestamp": pd.Timestamp.now().isoformat()
            }
    
    def _get_risk_level(self, probability: float) -> str:
        """Determine risk level based on probability"""
        if probability >= 0.7:
            return "High"
        elif probability >= 0.3:
            return "Medium"
        else:
            return "Low"
    
    def _generate_explanation(self, prediction: int, probability: float) -> str:
        """Generate clinical explanation for prediction"""
        if prediction == 1:
            return f"High likelihood of HCC recurrence ({probability:.1%}). Recommend close monitoring and consideration of adjuvant therapy."
        else:
            return f"Low likelihood of HCC recurrence ({probability:.1%}). Continue standard surveillance protocols."
    
    def batch_predict(self, input_data: Union[pd.DataFrame, List[Dict]], 
                     return_detailed: bool = True) -> Dict[str, Any]:
        """
        Make predictions on multiple samples
        """
        try:
            if isinstance(input_data, list):
                input_data = pd.DataFrame(input_data)
            
            if return_detailed:
                return self.predict_with_confidence(input_data)
            else:
                predictions = self.predict(input_data)
                return {
                    "success": True,
                    "predictions": predictions.tolist(),
                    "sample_count": len(predictions),
                    "timestamp": pd.Timestamp.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Batch prediction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "predictions": [],
                "sample_count": 0,
                "timestamp": pd.Timestamp.now().isoformat()
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        info = {
            "model_loaded": self.is_loaded,
            "model_type": type(self.model).__name__,
            "expected_features": self.model_expected_features,
            "preprocessor_loaded": self.preprocessor is not None,
            "feature_names_count": len(self.feature_names),
            "can_predict_proba": hasattr(self.model, "predict_proba"),
            "has_feature_mapping": len(self.feature_mapping) > 0
        }
        
        # Add model-specific information
        if hasattr(self.model, 'feature_importances_'):
            info["has_feature_importances"] = True
            info["top_features"] = self._get_top_features(5)
        else:
            info["has_feature_importances"] = False
        
        return info
    
    def _get_top_features(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top features by importance"""
        if not hasattr(self.model, 'feature_importances_'):
            return []
        
        try:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            top_features = []
            for i, idx in enumerate(indices):
                # Try to find actual feature name
                actual_name = "Unknown"
                for feat_name, generic_name in self.feature_mapping.items():
                    if generic_name == self.feature_names[idx]:
                        actual_name = feat_name
                        break
                
                top_features.append({
                    "rank": i + 1,
                    "feature": actual_name,
                    "generic_name": self.feature_names[idx],
                    "importance": float(importances[idx])
                })
            
            return top_features
            
        except Exception as e:
            logger.warning(f"Could not get top features: {e}")
            return []


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the ModelPredictor
    predictor = ModelPredictor()
    
    # Load model and preprocessor (will auto-search for files)
    success = predictor.load_model_and_preprocessor()
    
    if success:
        # Get model info
        print("Model Info:", predictor.get_model_info())
        
        # Example prediction with sample data
        sample_data = {
            "Age": 65,
            "Gender": 1,
            "AFP": 850,
            "Albumin": 3.2,
            "Total_Bil": 2.1,
            "ALT": 78,
            "AST": 95,
            "INR": 1.3,
            "Platelets": 125,
            "Creatinine": 1.1,
            "Nodules": 2,
            "Major_Dim": 5.5,
            "Symptoms": 1,
            "PS": 1,
            "Encephalopathy": 0,
            "Ascites": 0,
            "Alcohol": 1,
            "HBsAg": 0,
            "HCVAb": 1,
            "Cirrhosis": 1,
            "Diabetes": 1,
            "Smoking": 1
        }
        
        # Make prediction
        try:
            result = predictor.predict_with_confidence(sample_data)
            print("Prediction Result:", result)
        except Exception as e:
            print(f"Prediction error: {e}")
    else:
        print("Failed to load model and preprocessor")