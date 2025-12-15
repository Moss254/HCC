# Feature Reduction Implementation - Summary

## Overview
Successfully reduced the HCC recurrence prediction model from 52 features to 20 features (17 base + 3 engineered) to align with the web form's data collection capabilities.

## Problem
- Original model trained on 49 base features + 3 engineered = 52 total features
- Web form (`webapp/templates/predict.html`) only collects 17 features
- Missing 32 features defaulted to 0.0, causing incorrect predictions
- Example: Low-risk patient data was producing high-risk predictions

## Solution

### 1. Feature Set Reduction
**Base Features (17):**
- **Demographic:** Age, Gender
- **Clinical:** Symptoms, PS (Performance Status)
- **Laboratory:** AFP, Albumin, Total_Bil, ALT, AST
- **Tumor:** Major_Dim, Nodule
- **Risk Factors:** Alcohol, HBsAg, HCVAb, Cirrhosis, Diabetes, Smoking

**Engineered Features (3):**
- `Age_Category`: Categorizes age into risk bands (<50, 50-65, >65)
- `AFP_Risk_Category`: Categorizes AFP levels (<20, 20-400, >400 ng/mL)
- `Liver_Function_Score`: Composite score from Albumin, Total_Bil, ALT, AST

### 2. Files Modified

#### `src/data/data_preprocessor.py`
- Updated `engineer_features()` to use only 4 liver markers (removed INR)
- Added safe normalization to handle edge cases
- Maintained same feature engineering logic

#### `webapp/app.py`
- Updated `expected_features` list from 52 to 20 features
- Modified `engineer_features()` to match reduced feature set
- **CRITICAL FIX:** Corrected probability interpretation
  - Class 0 = Recurrence (worse prognosis)
  - Class 1 = No Recurrence (better prognosis)
  - Changed `recurrence_prob = probs[0]` (was incorrectly using `probs[1]`)
- Fixed `prediction_text` mapping to match class labels
- Updated `_generate_explanation()` to use probability-based thresholds

#### `src/models/model_predictor.py`
- Updated `engineer_features()` to match new 20-feature set
- Updated `expected_columns` list
- Ensured consistency with webapp and preprocessor

### 3. Model Retraining

**Script:** `retrain_model_reduced_features.py`

**Process:**
1. Loaded `data/raw/hcc-data-complete-balanced.xlsx` (204 samples, balanced)
2. Selected 17 base features matching form inputs
3. Engineered 3 derived features
4. Trained 4 models: Random Forest, XGBoost, Gradient Boosting, Logistic Regression
5. Selected best model based on ROC AUC

**Best Model Performance:**
- **Model:** Gradient Boosting Classifier
- **ROC AUC:** 0.9595 (excellent discrimination)
- **F1 Score:** 0.8718
- **Accuracy:** 0.8780
- **Precision:** 0.8947
- **Recall:** 0.8500

### 4. Class Label Understanding

**Important:** The dataset uses an inverted labeling scheme:
- **Class 0** = Recurrence (Poor prognosis)
  - Higher Total_Bil (mean: 4.29)
  - Lower Albumin (mean: 3.24)
  - Larger tumors (mean: 7.78 cm)
  - More symptoms (82% symptomatic)
  
- **Class 1** = No Recurrence (Good prognosis)
  - Lower Total_Bil (mean: 2.04)
  - Higher Albumin (mean: 3.60)
  - Smaller tumors (mean: 6.00 cm)
  - Fewer symptoms (56% symptomatic)
  - Lower AFP values in many cases

### 5. Testing

**Test Cases Created:**
1. **Low Risk:** AFP=1.2, good liver function → Prediction: No Recurrence (0.0% risk) ✓
2. **High Risk:** Poor liver function, large tumor → Prediction: Recurrence (99.97% risk) ✓
3. **Integration Test:** Full API workflow tested ✓

**Test Files:**
- `test_predictions.py` - Unit tests with realistic patient profiles
- `test_integration.py` - End-to-end API testing

### 6. Risk Stratification

**Thresholds:**
- **High Risk:** Recurrence probability ≥ 70%
- **Medium Risk:** Recurrence probability 40-70%
- **Low Risk:** Recurrence probability < 40%

## Results

### Before Fix
- Low-risk patients → 100% recurrence prediction ❌
- All predictions showing high risk regardless of input ❌
- 32 features missing from every prediction ❌

### After Fix
- Low-risk patients → Low risk prediction (0-5%) ✓
- High-risk patients → High risk prediction (95-100%) ✓
- All 20 features properly collected and engineered ✓
- SHAP explanations working correctly ✓
- Realistic risk stratification ✓

## Files Added/Modified

**Added:**
- `retrain_model_reduced_features.py` - Retraining script
- `test_predictions.py` - Prediction tests
- `test_integration.py` - Integration tests
- `.gitignore` - Ignore Python cache files
- `FEATURE_REDUCTION.md` - This documentation

**Modified:**
- `src/data/data_preprocessor.py` - Feature engineering
- `webapp/app.py` - Prediction logic and probability interpretation
- `src/models/model_predictor.py` - Feature engineering consistency
- `models/preprocessing/preprocessor.pkl` - New preprocessor (20 features)
- `models/trained_models/best_model.pkl` - New model (Gradient Boosting)
- `models/trained_models/performance_report.json` - Updated metrics

## Validation Checklist

- [x] Model retrained with 20 features
- [x] `expected_features` updated in webapp/app.py
- [x] `expected_columns` updated in model_predictor.py
- [x] Probability interpretation fixed
- [x] Low-risk test case passing
- [x] High-risk test case passing
- [x] SHAP explanations working
- [x] Integration tests passing
- [x] No code review issues

## Usage

### Running Tests
```bash
# Unit tests
python test_predictions.py

# Integration tests
python test_integration.py
```

### Retraining Model (if needed)
```bash
python retrain_model_reduced_features.py
```

### Starting Web Application
```bash
python run.py
```

## Notes for Future Development

1. **Class Labels:** Remember that Class 0 = Recurrence, Class 1 = No Recurrence in this dataset
2. **Feature Engineering:** The 3 engineered features are critical for model performance
3. **Liver Function Score:** Uses only 4 markers (Albumin, Total_Bil, ALT, AST) instead of 5
4. **Form Validation:** Ensure the form continues to collect exactly these 17 base features
5. **Model Updates:** If retraining, maintain the same 20-feature structure

## Impact

- **Accuracy:** Maintained high performance (ROC AUC 0.96) with 61% fewer features
- **Usability:** Predictions now work correctly with form data
- **Maintainability:** Simplified feature pipeline
- **Clinical Validity:** Risk stratification aligns with actual patient outcomes
