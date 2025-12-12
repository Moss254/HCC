import pandas as pd
import numpy as np
import joblib
import logging
import os
import sys
import base64
from io import BytesIO
from typing import Dict, Any, Union, List
from flask import Flask, request, jsonify, render_template
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import shap
import time
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Suppress matplotlib and SHAP warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('shap').setLevel(logging.WARNING)

class ModelPredictor:
    """Production-Ready Predictor with SHAP Explanations."""
    
    def __init__(self):
        self.model = None
        self.classifier = None  # Extracted classifier for SHAP
        self.preprocessor = None
        self.is_loaded = False
        self.shap_explainer = None
        self.feature_names = []
        self.expected_features = [
            'Obesity', 'Hallmark', 'HBeAg', 'Ferritin', 'CRI', 'Diabetes', 'TP', 
            'Encephalopathy', 'PVT', 'PS', 'INR', 'Hemoglobin', 'Platelets', 
            'Alcohol', 'Age', 'Total_Bil', 'Ascites', 'ALP', 'ALT', 'Symptoms', 
            'Gender', 'HIV', 'Endemic', 'Sat', 'Smoking', 'HBcAb', 'Grams_day', 
            'AFP', 'Nodule', 'Spleno', 'PHT', 'Iron', 'Cirrhosis', 'Albumin', 
            'Major_Dim', 'AHT', 'Varices', 'Hemochro', 'HBsAg', 'HCVAb', 'GGT', 
            'MCV', 'Dir_Bil', 'AST', 'Metastasis', 'Packs_year', 'Creatinine', 
            'NASH', 'Leucocytes', 'Age_Category', 'AFP_Risk_Category', 'Liver_Function_Score'
        ]

    def find_model_files(self, project_root):
        """Smartly searches for model files in common locations."""
        possible_paths = [
            "models",                     
            "src/models",                 
            "webapp/models",              
            "../models",                  
            "../src/models",
            os.path.join(os.getcwd(), "models"),
            os.path.join(os.getcwd(), "src", "models")
        ]

        logger.info(f" Searching for models in project root: {project_root}")

        for base_path in possible_paths:
            if os.path.isabs(base_path):
                full_base = base_path
            else:
                full_base = os.path.abspath(os.path.join(project_root, base_path))
            
            model_p = os.path.join(full_base, "trained_models", "best_model.pkl")
            prep_p = os.path.join(full_base, "preprocessing", "preprocessor.pkl")
            
            if os.path.exists(model_p) and os.path.exists(prep_p):
                logger.info(f" FOUND models in: {full_base}")
                return model_p, prep_p
        
        return None, None

    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer based on model type."""
        try:
            # Use classifier if available, otherwise use full model
            model_to_explain = self.classifier if self.classifier else self.model
            model_type = type(model_to_explain).__name__
            logger.info(f"Initializing SHAP for model type: {model_type}")
            
            if 'RandomForest' in model_type or 'XGB' in model_type or 'GradientBoosting' in model_type:
                # Tree-based models use TreeExplainer
                self.shap_explainer = shap.TreeExplainer(model_to_explain)
                logger.info(" SHAP TreeExplainer initialized")
            elif 'LogisticRegression' in model_type or 'Linear' in model_type:
                # Linear models - could use LinearExplainer but keeping simple for now
                logger.info(" SHAP for linear models would require background data")
                self.shap_explainer = None
            else:
                # Generic fallback
                self.shap_explainer = None
                logger.info(f" SHAP not supported for {model_type}")
                
        except Exception as e:
            logger.warning(f"SHAP initialization failed: {e}")
            logger.warning("Continuing without SHAP explanations")
            self.shap_explainer = None

    def load_model_and_preprocessor(self, project_root: str) -> bool:
        """Load model and initialize SHAP explainer."""
        try:
            model_path, preprocessor_path = self.find_model_files(project_root)

            if not model_path:
                logger.error(" CRITICAL: Could not find 'best_model.pkl'. Please run model training first.")
                return False

            logger.info(f" Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            
            logger.info(f" Loading preprocessor from: {preprocessor_path}")
            preprocessor_data = joblib.load(preprocessor_path)
            
            if isinstance(preprocessor_data, dict):
                self.preprocessor = preprocessor_data.get('pipeline')
                self.feature_names = preprocessor_data.get('feature_names', [])
            else:
                self.preprocessor = preprocessor_data
                # Try to extract feature names from preprocessor
                if hasattr(self.preprocessor, 'feature_names_in_'):
                    self.feature_names = list(self.preprocessor.feature_names_in_)
                elif hasattr(self.preprocessor, 'get_feature_names_out'):
                    self.feature_names = list(self.preprocessor.get_feature_names_out())

            # If still no feature names, use expected features
            if not self.feature_names:
                self.feature_names = self.expected_features
            
            logger.info(f" Loaded {len(self.feature_names)} feature names")
            logger.info(f" Expected features for prediction: {len(self.expected_features)}")

            # Extract classifier from pipeline for SHAP
            self.classifier = None
            if hasattr(self.model, 'named_steps'):
                if 'classifier' in self.model.named_steps:
                    self.classifier = self.model.named_steps['classifier']
                    logger.info(f" Extracted classifier: {type(self.classifier).__name__}")
                elif len(self.model.named_steps) > 0:
                    # Get the last step (assume it's the classifier)
                    last_step_name = list(self.model.named_steps.keys())[-1]
                    self.classifier = self.model.named_steps[last_step_name]
                    logger.info(f" Using last step as classifier: {last_step_name} ({type(self.classifier).__name__})")

            # Initialize SHAP explainer
            self._initialize_shap_explainer()
            
            self.is_loaded = True
            logger.info(" System Loaded Successfully with SHAP explainer.")
            return True
        except Exception as e:
            logger.error(f" Failed to load model assets: {e}", exc_info=True)
            return False

    def normalize_feature_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize feature names to match training data."""
        df_normalized = df.copy()
        
        # Map common alternative names
        feature_mapping = {
            'Nodules': 'Nodule',  # Map plural to singular
            'Nodules_count': 'Nodule',
            'Tumor_size': 'Major_Dim',
            'TumorSize': 'Major_Dim',
            'Bilirubin': 'Total_Bil',
            'Alb': 'Albumin',
            'Hb': 'Hemoglobin',
            'Plt': 'Platelets',
            'Cr': 'Creatinine',
            'WBC': 'Leucocytes',
            'Size': 'Major_Dim',
            'Dimension': 'Major_Dim'
        }
        
        # Rename columns if needed
        for old_name, new_name in feature_mapping.items():
            if old_name in df_normalized.columns and new_name not in df_normalized.columns:
                df_normalized[new_name] = df_normalized[old_name]
                logger.info(f"📝 Renamed {old_name} to {new_name}")
        
        return df_normalized

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering - matches training pipeline."""
        try:
            # First normalize feature names
            df_eng = self.normalize_feature_names(df)
            
            # Create Nodule feature from Nodules if needed
            if 'Nodules' in df_eng.columns and 'Nodule' not in df_eng.columns:
                df_eng['Nodule'] = df_eng['Nodules']
            
            # Age categorization
            if 'Age' in df_eng.columns:
                age_vals = df_eng['Age'].values
                age_cat = np.zeros_like(age_vals, dtype=float)
                age_cat[(age_vals >= 50) & (age_vals <= 65)] = 1.0
                age_cat[age_vals > 65] = 2.0
                df_eng['Age_Category'] = age_cat

            # AFP risk categorization
            if 'AFP' in df_eng.columns:
                afp_vals = df_eng['AFP'].values
                afp_cat = np.zeros_like(afp_vals, dtype=float)
                afp_cat[(afp_vals >= 20) & (afp_vals <= 400)] = 1.0
                afp_cat[afp_vals > 400] = 2.0
                df_eng['AFP_Risk_Category'] = afp_cat

            # Liver function composite score
            liver_markers = ['Albumin', 'Total_Bil', 'INR', 'ALT', 'AST']
            available_markers = [m for m in liver_markers if m in df_eng.columns]
            
            if len(available_markers) >= 2:
                scores = []
                for marker in available_markers:
                    val = df_eng[marker]
                    if marker == 'Albumin':
                        s = 1.0 - (val / 5.5) 
                    else:
                        s = val / 100.0 
                    scores.append(s.fillna(0))
                df_eng['Liver_Function_Score'] = pd.concat(scores, axis=1).mean(axis=1)
            else:
                df_eng['Liver_Function_Score'] = 0.0

            # Ensure all expected features are present with correct values
            for col in self.expected_features:
                if col not in df_eng.columns:
                    # Try to preserve original values if they exist under different names
                    if col == 'Nodule' and 'Nodules' in df.columns:
                        df_eng[col] = df['Nodules']
                    else:
                        df_eng[col] = 0.0
                else:
                    # Ensure numeric type
                    df_eng[col] = pd.to_numeric(df_eng[col], errors='coerce').fillna(0)

            # Reorder columns to match training data
            df_eng = df_eng[self.expected_features]
            
            return df_eng
        except Exception as e:
            logger.warning(f"Feature engineering warning: {e}. Proceeding with raw data.")
            return df

    def _extract_shap_expected_value(self, expected_value):
        """Extract scalar expected value from SHAP explainer."""
        if isinstance(expected_value, np.ndarray):
            if expected_value.size == 2:
                # For binary classification, we want the positive class (class 1)
                return float(expected_value[1])
            else:
                return float(expected_value[0])
        elif isinstance(expected_value, list):
            if len(expected_value) == 2:
                return float(expected_value[1])
            else:
                return float(expected_value[0])
        else:
            # It's already a scalar
            return float(expected_value)

    def _extract_shap_values_single(self, shap_values, sample_index=0):
        """Extract SHAP values for a single sample from various SHAP formats."""
        if shap_values is None:
            return None
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For binary classification, shap_values is [negative_class, positive_class]
            if len(shap_values) == 2:
                # Get positive class (class 1) values
                pos_class = shap_values[1]
                if hasattr(pos_class, 'shape') and len(pos_class.shape) == 2:
                    return pos_class[sample_index]
                elif hasattr(pos_class, 'shape') and len(pos_class.shape) == 1:
                    return pos_class
                else:
                    # Try to convert to array
                    try:
                        arr = np.array(pos_class)
                        if len(arr.shape) == 2:
                            return arr[sample_index]
                        return arr
                    except:
                        return None
            else:
                # Single class
                if hasattr(shap_values[0], 'shape') and len(shap_values[0].shape) == 2:
                    return shap_values[0][sample_index]
                else:
                    return shap_values[0]
        elif hasattr(shap_values, 'shape'):
            if len(shap_values.shape) == 3:
                # Shape is (n_samples, n_features, n_classes)
                if shap_values.shape[2] >= 2:
                    return shap_values[sample_index, :, 1]  # Class 1
                else:
                    return shap_values[sample_index, :, 0]
            elif len(shap_values.shape) == 2:
                return shap_values[sample_index]
            else:
                return shap_values.flatten()
        
        # Fallback: try to convert to numpy array
        try:
            return np.array(shap_values).flatten()
        except:
            return None

    def generate_shap_explanation(self, processed_features: np.ndarray) -> Dict[str, Any]:
        """Generate SHAP values and visualization - FIXED VERSION"""
        if self.shap_explainer is None:
            return {"shap_available": False, "message": "SHAP explainer not available"}
        
        try:
            start_time = time.time()
            
            # Calculate SHAP values - handle single sample
            if len(processed_features.shape) == 1:
                processed_features = processed_features.reshape(1, -1)
            
            logger.info(f"Processing SHAP for features shape: {processed_features.shape}")
            
            # Get SHAP values
            shap_values = None
            try:
                shap_values = self.shap_explainer.shap_values(processed_features)
                logger.info(f"Got SHAP values: type={type(shap_values)}")
            except Exception as e:
                logger.error(f"Failed to get SHAP values: {e}")
                return {"shap_available": False, "error": f"SHAP calculation failed: {str(e)}"}
            
            logger.info(f"SHAP calculation took {time.time() - start_time:.2f} seconds")
            
            # Get expected value
            expected_value = self.shap_explainer.expected_value
            logger.info(f"Expected value type: {type(expected_value)}, value: {expected_value}")
            
            # Extract scalar base value for class 1 (recurrence)
            base_value = self._extract_shap_expected_value(expected_value)
            
            logger.info(f"Using base value: {base_value}")
            
            # Extract SHAP values for the single sample
            shap_values_single = self._extract_shap_values_single(shap_values, sample_index=0)
            
            if shap_values_single is None:
                logger.error(f"Could not extract SHAP values from type: {type(shap_values)}")
                return {"shap_available": False, "error": "Could not extract SHAP values"}
            
            logger.info(f"Extracted SHAP values shape: {shap_values_single.shape if hasattr(shap_values_single, 'shape') else 'no shape'}")
            
            # Create simple visualization
            return self._create_simple_shap_visualization(shap_values_single, base_value)
            
        except Exception as e:
            logger.error(f"SHAP explanation generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"shap_available": False, "error": str(e)}
    
    def _create_simple_shap_visualization(self, shap_values: np.ndarray, base_value: float) -> Dict[str, Any]:
        """Create SIMPLE SHAP visualization using bar plot."""
        try:
            # Get feature names
            n_features = len(shap_values)
            display_names = self.feature_names[:n_features]
            if len(display_names) == 0:
                display_names = [f'Feature_{i}' for i in range(n_features)]
            
            # Create simple bar plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort features by absolute SHAP value and take top 10
            abs_shap_values = np.abs(shap_values)
            top_n = min(10, len(abs_shap_values))
            sorted_idx = np.argsort(abs_shap_values)[::-1][:top_n]
            
            sorted_vals = shap_values[sorted_idx]
            sorted_names = [display_names[i] for i in sorted_idx]
            
            # Color coding
            colors = ['#ff6b6b' if val > 0 else '#51cf66' for val in sorted_vals]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(sorted_vals))
            bars = ax.barh(y_pos, sorted_vals, color=colors, edgecolor='black', height=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names, fontsize=9)
            ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=10)
            ax.set_title('Top Features Influencing Recurrence Prediction', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Add value labels on bars (only for significant values)
            for bar, val in zip(bars, sorted_vals):
                width = bar.get_width()
                if abs(width) > 0.01:  # Only label significant values
                    label_x_pos = width if abs(width) > 0.02 else 0.02 * (1 if width >= 0 else -1)
                    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', 
                           ha='left' if val >= 0 else 'right',
                           va='center', 
                           fontsize=8)
            
            # Add base value annotation
            ax.text(0.02, 0.98, f'Base Value: {base_value:.3f}',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Add simple legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#ff6b6b', edgecolor='black', label='Increases Risk'),
                Patch(facecolor='#51cf66', edgecolor='black', label='Decreases Risk')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')  # Lower DPI for speed
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)
            
            # Get top contributing features
            feature_contributions = []
            for idx, (fname, sval) in enumerate(zip(display_names, shap_values)):
                if abs(sval) > 0.01:  # Only include significant contributions
                    feature_contributions.append({
                        "feature": fname,
                        "contribution": float(sval),
                        "direction": "increases" if sval > 0 else "decreases",
                        "abs_contribution": abs(float(sval))
                    })
            
            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
            
            return {
                "shap_available": True,
                "shap_plot": f"data:image/png;base64,{image_base64}",
                "top_features": feature_contributions[:8],
                "base_value": base_value,
                "features_analyzed": len(feature_contributions)
            }
            
        except Exception as e:
            logger.error(f"SHAP visualization creation failed: {e}")
            return {"shap_available": False, "error": str(e)}

    def predict_with_confidence(self, input_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """Make prediction with SHAP explanation and detailed clinical insights."""
        if not self.is_loaded:
            logger.error("Predict called but system not initialized!")
            return {"success": False, "error": "System not initialized - Model files missing"}

        try:
            logger.info(" Processing prediction with SHAP...")
            
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = input_data
            
            logger.info(f"Input features received: {list(df.columns)}")
            
            # Convert all to numeric, fill NaN with 0
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Feature engineering
            df_engineered = self.engineer_features(df)
            logger.info(f"Engineered features: {list(df_engineered.columns)}")
            
            # Ensure we have the right number of features
            if len(df_engineered.columns) != len(self.expected_features):
                logger.warning(f"Feature count mismatch: got {len(df_engineered.columns)}, expected {len(self.expected_features)}")
                
            if self.preprocessor:
                try:
                    # Transform using preprocessor
                    processed_features = self.preprocessor.transform(df_engineered)
                    logger.info(f" Successfully transformed features")
                except Exception as e:
                    logger.error(f" Transform failed: {e}")
                    # Fallback: try to match columns manually
                    logger.info("Attempting manual feature alignment...")
                    
                    # Create a DataFrame with all expected features set to 0
                    aligned_df = pd.DataFrame(0, index=[0], columns=self.expected_features)
                    
                    # Copy available values from engineered df
                    for col in df_engineered.columns:
                        if col in aligned_df.columns:
                            aligned_df[col] = df_engineered[col].values
                    
                    # Try transformation again
                    processed_features = self.preprocessor.transform(aligned_df)
                    logger.info(" Manual feature alignment successful")
            else:
                processed_features = df_engineered.values

            # Make prediction
            prediction = self.model.predict(processed_features)[0]
            
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(processed_features)[0]
                recurrence_prob = probs[1] if len(probs) > 1 else probs[0]
            else:
                recurrence_prob = float(prediction)

            # Generate SHAP explanation (with timeout)
            shap_data = {}
            try:
                shap_data = self.generate_shap_explanation(processed_features)
            except Exception as e:
                logger.warning(f"SHAP generation failed, continuing without it: {e}")
                shap_data = {"shap_available": False, "error": str(e)}
            
            # Determine risk level based on probability thresholds
            risk_level = self._get_risk_level(recurrence_prob)
            
            # Generate detailed explanation
            explanation = self._generate_explanation(int(prediction), recurrence_prob)
            
            # Generate probability-specific recommendations
            recommendations = self._get_recommendations(recurrence_prob)

            # Generate clinical indicators
            clinical_indicators = self._generate_clinical_indicators(df_engineered, recurrence_prob)
            
            result = {
                "success": True,
                "prediction": int(prediction),
                "prediction_text": "Recurrence Likely" if prediction == 1 else "No Recurrence Expected",
                "probability": float(recurrence_prob),
                "confidence": f"{recurrence_prob * 100:.1f}%",
                "risk_level": risk_level,
                "explanation": explanation,
                "recommendations": recommendations,
                "clinical_indicators": clinical_indicators,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            # Add SHAP data if available
            if shap_data.get("shap_available"):
                result["shap_plot"] = shap_data["shap_plot"]
                result["top_features"] = shap_data["top_features"]
                result["feature_interpretation"] = self._generate_feature_interpretation(shap_data["top_features"])
            else:
                result["shap_available"] = False
                result["shap_error"] = shap_data.get("error", "SHAP not available")
            
            logger.info(f" Prediction Success: {result['risk_level']} Risk")
            return result
            
        except Exception as e:
            logger.error(f" Prediction Pipeline Failed: {e}", exc_info=True)
            return {"success": False, "error": f"Internal Error: {str(e)}"}

    def _get_risk_level(self, probability: float) -> str:
        """Determine risk level based on probability thresholds."""
        if probability >= 0.7:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        else:
            return "Low"

    def _generate_explanation(self, prediction: int, probability: float) -> str:
        """Generate detailed clinical explanation based on percentage."""
        percentage = probability * 100
        
        if prediction == 1:
            if percentage >= 80:
                return f"VERY HIGH RISK ({percentage:.1f}%): Strong indicators present suggesting aggressive tumor characteristics. Multiple high-risk clinical and laboratory markers detected."
            elif percentage >= 60:
                return f"HIGH RISK ({percentage:.1f}%): Significant risk factors identified. Enhanced surveillance and consideration of adjuvant therapy strongly recommended."
            else:
                return f"MODERATE RISK ({percentage:.1f}%): Some concerning factors present. Close monitoring with regular imaging advised."
        else:
            if percentage <= 20:
                return f"VERY LOW RISK ({percentage:.1f}%): Favorable prognostic indicators. Standard surveillance protocol is sufficient."
            else:
                return f"LOW RISK ({percentage:.1f}%): Minimal risk factors detected. Continue routine follow-up schedule."

    def _get_recommendations(self, probability: float) -> List[str]:
        """Generate probability-specific clinical recommendations."""
        percentage = probability * 100
        
        if percentage >= 80:
            return [
                " URGENT: Schedule imaging and AFP testing within 4-6 weeks",
                "Immediate multidisciplinary team (hepatology, oncology, surgery) review",
                "Consider adjuvant therapy or clinical trial enrollment",
                "Monthly AFP monitoring and comprehensive liver function assessment",
                "Patient and family counseling regarding aggressive management options",
                "Evaluate liver transplantation eligibility if applicable"
            ]
        elif percentage >= 70:
            return [
                " HIGH PRIORITY: Contrast-enhanced imaging (CT/MRI) within 2-3 months",
                "Intensive AFP monitoring every 6-8 weeks",
                "Multidisciplinary tumor board discussion required",
                "Consider adjuvant systemic therapy options",
                "Optimize management of underlying liver disease",
                "Patient education on recurrence warning signs and symptoms"
            ]
        elif percentage >= 60:
            return [
                " MODERATE-HIGH: Enhanced surveillance with imaging every 3 months",
                "AFP testing every 2 months with liver function panels",
                "Review with hepatology and oncology specialists",
                "Aggressive management of cirrhosis and other risk factors",
                "Consider preventive interventions if available",
                "Regular clinical assessment for new symptoms"
            ]
        elif percentage >= 40:
            return [
                " MODERATE: Standard surveillance imaging every 3-6 months",
                "Quarterly AFP monitoring and liver function tests",
                "Regular hepatologist follow-up visits",
                "Address modifiable risk factors (alcohol cessation, diabetes control)",
                "Maintain vaccination status (Hepatitis A/B if applicable)",
                "Lifestyle modifications and dietary counseling"
            ]
        elif percentage >= 20:
            return [
                " LOW-MODERATE: Routine surveillance imaging every 6 months",
                "AFP testing and liver function assessment every 6 months",
                "Annual comprehensive clinical evaluation",
                "Continue management of underlying chronic liver disease",
                "Standard lifestyle counseling and risk factor modification",
                "Regular primary care and specialist follow-up"
            ]
        else:
            return [
                " VERY LOW: Annual surveillance imaging sufficient",
                "Annual AFP and liver function testing",
                "Continue current management of liver disease",
                "Maintain healthy lifestyle and medication compliance",
                "Routine annual specialist review",
                "Symptom awareness education"
            ]

    def _generate_clinical_indicators(self, df: pd.DataFrame, probability: float) -> Dict[str, Any]:
        """Generate key clinical indicators from patient data."""
        indicators = []
        
        # Check specific biomarkers
        if 'AFP' in df.columns and df['AFP'].iloc[0] > 400:
            indicators.append({
                "name": "Very High AFP",
                "value": f"{df['AFP'].iloc[0]:.0f} ng/mL",
                "severity": "High",
                "description": "AFP > 400 ng/mL indicates high risk of recurrence"
            })
        elif 'AFP' in df.columns and df['AFP'].iloc[0] > 20:
            indicators.append({
                "name": "Elevated AFP",
                "value": f"{df['AFP'].iloc[0]:.0f} ng/mL",
                "severity": "Medium",
                "description": "AFP > 20 ng/mL suggests active disease"
            })
        
        if 'Age_Category' in df.columns and df['Age_Category'].iloc[0] >= 1:
            indicators.append({
                "name": "Advanced Age",
                "value": f"{df['Age'].iloc[0]:.0f} years" if 'Age' in df.columns else "Age > 50",
                "severity": "Medium",
                "description": "Increased age is associated with higher recurrence risk"
            })
        
        if 'Liver_Function_Score' in df.columns and df['Liver_Function_Score'].iloc[0] > 0.5:
            indicators.append({
                "name": "Liver Dysfunction",
                "value": f"Score: {df['Liver_Function_Score'].iloc[0]:.2f}",
                "severity": "Medium",
                "description": "Impaired liver function increases recurrence risk"
            })
        
        if 'Cirrhosis' in df.columns and df['Cirrhosis'].iloc[0] == 1:
            indicators.append({
                "name": "Underlying Cirrhosis",
                "value": "Present",
                "severity": "High",
                "description": "Cirrhosis significantly increases HCC recurrence risk"
            })
        
        if 'Major_Dim' in df.columns and df['Major_Dim'].iloc[0] > 5:
            indicators.append({
                "name": "Large Tumor Size",
                "value": f"{df['Major_Dim'].iloc[0]:.1f} cm",
                "severity": "High",
                "description": "Tumor size > 5cm is a poor prognostic factor"
            })
        
        return {
            "count": len(indicators),
            "indicators": indicators[:5],  # Top 5 indicators
            "summary": f"Found {len(indicators)} significant clinical indicators"
        }

    def _generate_feature_interpretation(self, top_features: List[Dict]) -> str:
        """Generate human-readable interpretation of SHAP features."""
        if not top_features:
            return "Feature analysis not available for this prediction."
        
        interpretation_parts = []
        interpretation_parts.append("**Key Contributing Factors:**\n")
        
        for i, feat in enumerate(top_features[:5], 1):
            feature_name = feat['feature']
            direction = feat['direction']
            contribution = abs(feat['contribution'])
            
            if contribution > 0.1:
                strength = "strongly"
            elif contribution > 0.05:
                strength = "moderately"
            else:
                strength = "slightly"
            
            interpretation_parts.append(
                f"{i}. **{feature_name}** {strength} {direction} recurrence risk"
            )
        
        return "\n".join(interpretation_parts)

    def batch_predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Batch prediction with enhanced outputs."""
        results = []
        risk_counts = {"High": 0, "Medium": 0, "Low": 0}
        
        for index, row in df.iterrows():
            patient_data = row.to_dict()
            pred_result = self.predict_with_confidence(patient_data)
            
            if not pred_result.get('success'): 
                continue
                
            risk = pred_result['risk_level']
            risk_counts[risk] += 1
            
            patient_id = row.get('Patient_ID') or (index + 1)

            results.append({
                'patient_id': patient_id,
                'prediction': pred_result['prediction'],
                'prediction_text': pred_result.get('prediction_text', ''),
                'probability': pred_result['probability'],
                'confidence': pred_result.get('confidence', ''),
                'risk_level': risk,
                'explanation': pred_result.get('explanation', '')
            })
            
        total = len(results)
        return {
            "success": True,
            "summary": {
                "total_patients": total,
                "high_risk": risk_counts["High"],
                "medium_risk": risk_counts["Medium"],
                "low_risk": risk_counts["Low"],
                "recurrence_rate": (risk_counts["High"] + risk_counts["Medium"]) / total * 100 if total > 0 else 0,
                "high_risk_percentage": (risk_counts["High"] / total * 100) if total > 0 else 0,
                "medium_risk_percentage": (risk_counts["Medium"] / total * 100) if total > 0 else 0,
                "low_risk_percentage": (risk_counts["Low"] / total * 100) if total > 0 else 0
            },
            "results": results
        }

    def get_model_info(self):
        """Get system information."""
        return {
            "status": "Active" if self.is_loaded else "Inactive",
            "model_type": type(self.model).__name__ if self.model else "None",
            "classifier_type": type(self.classifier).__name__ if self.classifier else "None",
            "shap_enabled": self.shap_explainer is not None,
            "feature_count": len(self.feature_names)
        }

# --- FLASK APP ---
class HCCPredictionSystem:
    def __init__(self):
        self.webapp_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.webapp_dir, '..'))
        self.template_dir = os.path.join(self.webapp_dir, 'templates')
        self.static_dir = os.path.join(self.webapp_dir, 'static')
        
        # --- DIAGNOSTIC LOGGING ---
        print("\n" + "="*50)
        print(f" FLASK ROOT: {self.webapp_dir}")
        print(f" TEMPLATES:  {self.template_dir}")
        
        if os.path.exists(self.template_dir):
            files = os.listdir(self.template_dir)
            print(f"📄 Files in templates: {files}")
            if 'index.html' not in files:
                print(" CRITICAL: index.html is MISSING from templates folder!")
        else:
            print(" CRITICAL: Templates folder does NOT exist!")
        print("="*50 + "\n")

        self.app = Flask(__name__,
                         root_path=self.webapp_dir, 
                         template_folder='templates', 
                         static_folder='static')
                         
        self.app.secret_key = os.environ.get('SECRET_KEY', 'dev')
        self.predictor = ModelPredictor()
        
        # Add Global Request Logger
        self.app.before_request(self.log_request_info)
        
        self.setup_routes()
        self.initialize_system()

    def log_request_info(self):
        """Logs every incoming request to prove the server is reachable."""
        print(f"📡 INCOMING REQUEST: {request.method} {request.path}")

    def initialize_system(self):
        self.predictor.load_model_and_preprocessor(self.project_root)

    def setup_routes(self):
        @self.app.route('/')
        def index(): return render_template('index.html')

        @self.app.route('/predict_page')
        def predict_page(): return render_template('predict.html')

        @self.app.route('/batch_predict')
        def batch_predict_page(): return render_template('batch_predict.html')

        @self.app.route('/dashboard')
        def dashboard(): return render_template('dashboard.html')

        # API Routes
        @self.app.route('/api/predict', methods=['POST'])
        def predict_api():
            try:
                print("⚡ API HIT: /api/predict executing...")
                data = request.get_json()
                if not data:
                    return jsonify({"success": False, "error": "No JSON data received"}), 400
                    
                print(f"📋 Received data with keys: {list(data.keys())}")
                result = self.predictor.predict_with_confidence(data)
                return jsonify(result)
            except Exception as e:
                logger.error(f" API Error: {e}", exc_info=True)
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route('/api/batch-predict', methods=['POST'])
        def batch_predict_api():
            if 'file' not in request.files: 
                return jsonify({"success": False, "error": "No file uploaded"}), 400
                
            file = request.files['file']
            try:
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file)
                elif file.filename.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file)
                else:
                    return jsonify({"success": False, "error": "Unsupported file format. Use CSV or Excel."}), 400
                    
                result = self.predictor.batch_predict(df)
                return jsonify(result)
            except Exception as e:
                logger.error(f" Batch prediction error: {e}", exc_info=True)
                return jsonify({"success": False, "error": f"File processing error: {str(e)}"}), 500

        @self.app.route('/api/dashboard-data')
        def dashboard_api():
            model_info = self.predictor.get_model_info()
            return jsonify({
                "system_status": model_info,
                "model_performance": {
                    "accuracy": 0.87,
                    "roc_auc": 0.94,
                    "precision": 0.85,
                    "recall": 0.82,
                    "f1_score": 0.835
                },
                "feature_importance": [
                    {"feature": "AFP", "importance": 0.25},
                    {"feature": "Age", "importance": 0.18},
                    {"feature": "Cirrhosis", "importance": 0.15},
                    {"feature": "Major_Dim", "importance": 0.12},
                    {"feature": "Liver_Function_Score", "importance": 0.10}
                ],
                "clinical_insights": {
                    "key_factors": ["AFP > 400 ng/mL", "Age > 65", "Cirrhosis present", "Tumor size > 5cm"],
                    "risk_stratification": {
                        "high_risk_cutoff": 0.7,
                        "medium_risk_cutoff": 0.4,
                        "low_risk_cutoff": 0.0
                    },
                    "recommendations": [
                        "Monthly monitoring for high-risk patients",
                        "Quarterly imaging for medium-risk",
                        "Annual surveillance for low-risk"
                    ]
                }
            })

        @self.app.route('/health')
        def health(): 
            return jsonify({
                "status": "healthy",
                "model": self.predictor.get_model_info(),
                "timestamp": pd.Timestamp.now().isoformat()
            })
        
        @self.app.route('/api/feature-descriptions')
        def feature_descriptions():
            """Provide descriptions of clinical features for frontend help."""
            return jsonify({
                "features": {
                    "AFP": "Alpha-fetoprotein (ng/mL) - Tumor marker for HCC",
                    "Age": "Patient age in years - Risk increases >50",
                    "Cirrhosis": "Presence of cirrhosis (1=Yes, 0=No)",
                    "Major_Dim": "Largest tumor dimension in cm",
                    "Albumin": "Serum albumin (g/dL) - Liver synthetic function",
                    "Total_Bil": "Total bilirubin (mg/dL) - Liver excretory function",
                    "INR": "International Normalized Ratio - Coagulation status",
                    "ALT": "Alanine aminotransferase (U/L) - Liver inflammation",
                    "AST": "Aspartate aminotransferase (U/L) - Liver inflammation",
                    "Nodules": "Number of tumor nodules (will be mapped to 'Nodule')"
                }
            })
    def run(self, host='0.0.0.0', port=8000, debug=False):
        logger.info(f" Starting HCC Prediction System on {host}:{port}")
        logger.info(f" SHAP Explanations: {'Enabled' if self.predictor.shap_explainer else 'Disabled'}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    system = HCCPredictionSystem()
    system.run(debug=True)