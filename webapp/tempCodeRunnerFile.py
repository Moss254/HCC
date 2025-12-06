import pandas as pd
import numpy as np
import joblib
import logging
import os
import sys
from typing import Dict, Any, Union, List
from flask import Flask, request, jsonify, render_template

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ModelPredictor:
    """
    Production-Ready Predictor Class.
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        self.feature_names = []
        self.model_expected_features = 0

    def load_model_and_preprocessor(self, model_path: str, preprocessor_path: str) -> bool:
        """Load artifacts with validation."""
        try:
            if not os.path.exists(model_path):
                logger.error(f"CRITICAL: Model not found at {model_path}")
                return False
            if not os.path.exists(preprocessor_path):
                logger.error(f"CRITICAL: Preprocessor not found at {preprocessor_path}")
                return False

            # Load Model
            self.model = joblib.load(model_path)
            
            # Load Preprocessor
            preprocessor_data = joblib.load(preprocessor_path)
            if isinstance(preprocessor_data, dict):
                self.preprocessor = preprocessor_data.get('pipeline')
            else:
                self.preprocessor = preprocessor_data

            self.is_loaded = True
            logger.info("System Loaded Successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model assets: {e}")
            return False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adapts the input to match the Model's strict requirements.
        1. Fixes singular/plural naming (Nodules -> Nodule).
        2. Fills all missing training columns with 0.0.
        """
        try:
            df_eng = df.copy()

            # --- FIX 1: Naming Mismatches ---
            # The form sends 'Nodules', but the model wants 'Nodule'
            if 'Nodules' in df_eng.columns:
                df_eng['Nodule'] = df_eng['Nodules']
            
            # --- FIX 2: Ensure Clinical Derived Features Exist ---
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

            # Liver Function Score
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

            # --- FIX 3: The Schema Adapter (Crucial) ---
            # These are the 50+ columns the model expects.
            # We ensure EVERY SINGLE ONE exists. If missing, we fill with 0.0.
            expected_columns = [
                'Obesity', 'Hallmark', 'HBeAg', 'Ferritin', 'CRI', 'Diabetes', 'TP', 
                'Encephalopathy', 'PVT', 'PS', 'INR', 'Hemoglobin', 'Platelets', 
                'Alcohol', 'Age', 'Total_Bil', 'Ascites', 'ALP', 'ALT', 'Symptoms', 
                'Gender', 'HIV', 'Endemic', 'Sat', 'Smoking', 'HBcAb', 'Grams_day', 
                'AFP', 'Nodule', 'Spleno', 'PHT', 'Iron', 'Cirrhosis', 'Albumin', 
                'Major_Dim', 'AHT', 'Varices', 'Hemochro', 'HBsAg', 'HCVAb', 'GGT', 
                'MCV', 'Dir_Bil', 'AST', 'Metastasis', 'Packs_year', 'Creatinine', 
                'NASH', 'Leucocytes', 'Age_Category', 'AFP_Risk_Category', 'Liver_Function_Score'
            ]

            for col in expected_columns:
                if col not in df_eng.columns:
                    df_eng[col] = 0.0

            return df_eng

        except Exception as e:
            logger.warning(f"Feature engineering warning: {e}. Proceeding with raw data.")
            return df

    def predict_with_confidence(self, input_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "System not initialized"}

        try:
            # 1. Data Validation
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = input_data
            
            # Ensure numeric types
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

            # 2. Feature Engineering (Fills missing columns)
            df_engineered = self.engineer_features(df)

            # 3. Preprocessing (Transformation)
            # IMPORTANT: We do NOT rename columns to feature_0 here. 
            # We pass them exactly as the preprocessor expects (e.g., 'Age', 'AFP').
            if self.preprocessor:
                try:
                    processed_features = self.preprocessor.transform(df_engineered)
                except ValueError as ve:
                    # Fallback if strict column matching fails
                    logger.warning(f"Transform warning: {ve}")
                    processed_features = self.preprocessor.transform(df_engineered)
            else:
                processed_features = df_engineered.values

            # 4. Prediction
            prediction = self.model.predict(processed_features)[0]
            
            # 5. Probability
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(processed_features)[0]
                recurrence_prob = probs[1] if len(probs) > 1 else probs[0]
            else:
                recurrence_prob = float(prediction)

            # 6. Results
            risk_level = self._get_risk_level(recurrence_prob)
            explanation = self._generate_explanation(int(prediction), recurrence_prob)
            recommendations = self._get_recommendations(risk_level)

            return {
                "success": True,
                "prediction": int(prediction),
                "probability": float(recurrence_prob),
                "risk_level": risk_level,
                "explanation": explanation,
                "recommendations": recommendations,
                "timestamp": pd.Timestamp.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Prediction Pipeline Failed: {e}")
            return {"success": False, "error": f"Internal Error: {str(e)}"}

    def _get_risk_level(self, probability: float) -> str:
        if probability >= 0.7: return "High"
        if probability >= 0.3: return "Medium"
        return "Low"

    def _generate_explanation(self, prediction: int, probability: float) -> str:
        perc = probability * 100
        if prediction == 1:
            return f"High Risk of Recurrence ({perc:.1f}%). Clinical markers suggest aggressive characteristics."
        return f"Low Risk of Recurrence ({perc:.1f}%). Standard surveillance recommended."

    def _get_recommendations(self, risk_level: str) -> List[str]:
        if risk_level == "High":
            return ["Schedule imaging in 3 months.", "Consider adjuvant therapy.", "Monitor AFP monthly."]
        elif risk_level == "Medium":
            return ["Surveillance every 6 months.", "Monitor liver function."]
        return ["Routine annual follow-up."]

    def get_model_info(self):
        return {"status": "Active", "type": type(self.model).__name__ if self.model else "None"}

# --- FLASK APP ---
class HCCPredictionSystem:
    def __init__(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.template_dir = os.path.join(self.base_dir, 'templates')
        self.static_dir = os.path.join(self.base_dir, 'static')

        self.app = Flask(__name__, template_folder=self.template_dir, static_folder=self.static_dir)
        self.app.secret_key = os.environ.get('SECRET_KEY', 'dev')
        self.predictor = ModelPredictor()
        self.setup_routes()
        self.initialize_system()

    def initialize_system(self):
        model_path = os.path.join(self.base_dir, "models", "trained_models", "best_model.pkl")
        prep_path = os.path.join(self.base_dir, "models", "preprocessing", "preprocessor.pkl")
        logger.info(f"Loading model from: {model_path}")
        self.predictor.load_model_and_preprocessor(model_path, prep_path)

    def setup_routes(self):
        @self.app.route('/')
        def index(): return render_template('index.html')

        @self.app.route('/predict_page')
        def predict_page(): return render_template('predict.html')

        @self.app.route('/batch_predict')
        def batch_predict_page(): return render_template('batch_predict.html')

        @self.app.route('/dashboard')
        def dashboard(): return render_template('dashboard.html')

        @self.app.route('/predict', methods=['POST'])
        def predict_api():
            data = request.get_json()
            return jsonify(self.predictor.predict_with_confidence(data))

        @self.app.route('/health')
        def health(): return jsonify(self.predictor.get_model_info())

    def run(self, host='0.0.0.0', port=8000, debug=False):
        self.app.run(host=host, port=port, debug=debug)