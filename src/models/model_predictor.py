import pandas as pd
import numpy as np
import joblib
import logging
import os
import sys
import base64
from io import BytesIO
from typing import Dict, Any, Union, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import shap

# --- CONFIGURATION & LOGGING ----
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ModelPredictor:
    """Production-Ready Predictor with SHAP Explanations."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        self.shap_explainer = None
        self.feature_names = []

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
            model_type = type(self.model).__name__
            logger.info(f"Initializing SHAP for model type: {model_type}")
            
            if 'RandomForest' in model_type or 'XGB' in model_type or 'GradientBoosting' in model_type:
                # Tree-based models use TreeExplainer
                self.shap_explainer = shap.TreeExplainer(self.model)
                logger.info(" SHAP TreeExplainer initialized")
            elif 'Linear' in model_type or 'Logistic' in model_type:
                # For linear models - use KernelExplainer as fallback
                background = shap.sample(np.zeros((50, len(self.feature_names))), 50)
                self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background)
                logger.info(" SHAP KernelExplainer initialized for linear model")
            else:
                # Generic fallback
                background = shap.sample(np.zeros((50, len(self.feature_names))), 50)
                self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background)
                logger.info(" SHAP KernelExplainer initialized")
                
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

            # If still no feature names, create generic ones
            if not self.feature_names and hasattr(self.preprocessor, 'n_features_in_'):
                self.feature_names = [f'Feature_{i}' for i in range(self.preprocessor.n_features_in_)]
            
            logger.info(f" Loaded {len(self.feature_names)} feature names")

            # Initialize SHAP explainer
            self._initialize_shap_explainer()
            
            self.is_loaded = True
            logger.info(" System Loaded Successfully with SHAP explainer.")
            return True
        except Exception as e:
            logger.error(f" Failed to load model assets: {e}", exc_info=True)
            return False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering - matches training pipeline with only 17 base features."""
        try:
            df_eng = df.copy()
            
            if 'Nodules' in df_eng.columns:
                df_eng['Nodule'] = df_eng['Nodules']
            
            if 'Age' in df_eng.columns:
                age_vals = df_eng['Age'].values
                age_cat = np.zeros_like(age_vals, dtype=float)
                age_cat[(age_vals >= 50) & (age_vals <= 65)] = 1.0
                age_cat[age_vals > 65] = 2.0
                df_eng['Age_Category'] = age_cat

            if 'AFP' in df_eng.columns:
                afp_vals = df_eng['AFP'].values
                afp_cat = np.zeros_like(afp_vals, dtype=float)
                afp_cat[(afp_vals >= 20) & (afp_vals <= 400)] = 1.0
                afp_cat[afp_vals > 400] = 2.0
                df_eng['AFP_Risk_Category'] = afp_cat

            # Liver function composite score - using only available markers from form
            liver_markers = ['Albumin', 'Total_Bil', 'ALT', 'AST']
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

            # Expected columns based on the 17 form features + 3 engineered features
            expected_columns = [
                'Age', 'Gender', 'Symptoms', 'PS',
                'AFP', 'Albumin', 'Total_Bil', 'ALT', 'AST',
                'Major_Dim', 'Nodule',
                'Alcohol', 'HBsAg', 'HCVAb', 'Cirrhosis', 'Diabetes', 'Smoking',
                'Age_Category', 'AFP_Risk_Category', 'Liver_Function_Score'
            ]

            for col in expected_columns:
                if col not in df_eng.columns:
                    df_eng[col] = 0.0

            return df_eng
        except Exception as e:
            logger.warning(f"Feature engineering warning: {e}. Proceeding with raw data.")
            return df

    def generate_shap_explanation(self, processed_features: np.ndarray) -> Dict[str, Any]:
        """Generate SHAP values and visualization - FIXED VERSION"""
        if self.shap_explainer is None:
            return {"shap_available": False, "message": "SHAP explainer not available"}
        
        try:
            # Calculate SHAP values - handle single sample
            if len(processed_features.shape) == 1:
                processed_features = processed_features.reshape(1, -1)
            
            shap_values = self.shap_explainer.shap_values(processed_features)
            
            # Log SHAP values shape for debugging
            logger.debug(f"SHAP values type: {type(shap_values)}")
            if isinstance(shap_values, list):
                logger.debug(f"SHAP values list length: {len(shap_values)}")
                for i, val in enumerate(shap_values):
                    logger.debug(f"  shap_values[{i}].shape: {val.shape}")
            else:
                logger.debug(f"SHAP values shape: {shap_values.shape}")
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Binary classification - we have shap_values[0] and shap_values[1]
                if len(shap_values) == 2:
                    shap_values_positive = shap_values[1]  # Class 1 (Recurrence)
                    if len(shap_values_positive.shape) == 2:
                        shap_values_positive = shap_values_positive[0]  # Take first sample
                    else:
                        shap_values_positive = shap_values_positive.flatten()
                else:
                    shap_values_positive = shap_values[0].flatten()
            else:
                # Single array
                if len(shap_values.shape) == 2:
                    shap_values_positive = shap_values[0]
                else:
                    shap_values_positive = shap_values.flatten()
            
            # Get expected value - handle array vs scalar
            expected_value = self.shap_explainer.expected_value
            logger.debug(f"Expected value type: {type(expected_value)}, value: {expected_value}")
            
            if isinstance(expected_value, np.ndarray) or isinstance(expected_value, list):
                if len(expected_value) == 2:
                    base_value = float(expected_value[1])  # Expected value for class 1
                else:
                    base_value = float(expected_value[0])
            else:
                base_value = float(expected_value)
            
            logger.debug(f"Using base value: {base_value}")
            
            # Get feature names
            n_features = len(shap_values_positive)
            display_names = self.feature_names[:n_features]
            if len(display_names) == 0:
                display_names = [f"Feature_{i}" for i in range(n_features)]
            
            # Create waterfall plot - SIMPLIFIED VERSION
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create horizontal bar plot (simpler alternative to waterfall)
            sorted_idx = np.argsort(np.abs(shap_values_positive))[::-1][:15]
            sorted_vals = shap_values_positive[sorted_idx]
            sorted_names = [display_names[i] for i in sorted_idx]
            
            colors = ['#ff6b6b' if val > 0 else '#51cf66' for val in sorted_vals]
            
            y_pos = np.arange(len(sorted_vals))
            bars = ax.barh(y_pos, sorted_vals, color=colors, edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
            ax.set_title('Feature Impact on Recurrence Prediction', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add value labels
            for bar, val in zip(bars, sorted_vals):
                width = bar.get_width()
                label_x_pos = width if abs(width) > 0.01 else 0.01 * (1 if width >= 0 else -1)
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', ha='left' if val >= 0 else 'right',
                       va='center', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # Add base value annotation
            ax.text(0.02, 0.98, f'Base Value: {base_value:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#ff6b6b', edgecolor='black', label='Increases Recurrence Risk'),
                Patch(facecolor='#51cf66', edgecolor='black', label='Decreases Recurrence Risk')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            # Get top contributing features
            feature_contributions = []
            for idx, (fname, sval) in enumerate(zip(display_names, shap_values_positive)):
                if abs(sval) > 0.001:  # Include more features
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
                "top_features": feature_contributions[:10],
                "base_value": base_value,
                "features_analyzed": len(feature_contributions),
                "total_impact": float(np.sum(np.abs(shap_values_positive)))
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation generation failed: {e}")
            logger.exception(e)  # Add full traceback for debugging
            return {"shap_available": False, "error": str(e)}

    def predict_with_confidence(self, input_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """Make prediction with SHAP explanation and detailed clinical insights."""
        if not self.is_loaded:
            logger.error("Predict called but system not initialized!")
            return {"success": False, "error": "System not initialized - Model files missing"}

        try:
            logger.info("⚡ Processing prediction with SHAP...")
            
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = input_data
            
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            df_engineered = self.engineer_features(df)

            if self.preprocessor:
                try:
                    processed_features = self.preprocessor.transform(df_engineered)
                except ValueError as ve:
                    logger.warning(f"Transform warning: {ve}")
                    processed_features = self.preprocessor.transform(df_engineered)
            else:
                processed_features = df_engineered.values

            prediction = self.model.predict(processed_features)[0]
            
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(processed_features)[0]
                recurrence_prob = probs[1] if len(probs) > 1 else probs[0]
            else:
                recurrence_prob = float(prediction)

            # Generate SHAP explanation
            shap_data = self.generate_shap_explanation(processed_features)
            
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
                result["shap_debug"] = {
                    "base_value": shap_data.get("base_value"),
                    "features_analyzed": shap_data.get("features_analyzed"),
                    "total_impact": shap_data.get("total_impact")
                }
            
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
            "shap_enabled": self.shap_explainer is not None,
            "feature_count": len(self.feature_names)
        }