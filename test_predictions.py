#!/usr/bin/env python3
"""
Test the prediction system with low, medium, and high-risk test cases
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from webapp.app import ModelPredictor
import json

def test_prediction_system():
    """Test the prediction system with various risk profiles"""
    
    print("=" * 80)
    print("TESTING HCC PREDICTION SYSTEM WITH REDUCED FEATURES")
    print("=" * 80)
    
    # Initialize predictor
    predictor = ModelPredictor()
    success = predictor.load_model_and_preprocessor('.')
    
    if not success:
        print("❌ Failed to load model and preprocessor")
        return
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Expected features: {len(predictor.expected_features)}")
    print(f"✓ Features: {predictor.expected_features}\n")
    
    # Test Case 1: Low Risk Patient (Actual Class 1 sample from training data)
    print("\n" + "=" * 80)
    print("TEST CASE 1: LOW RISK PATIENT (Actual training sample - Class 1)")
    print("=" * 80)
    
    low_risk = {
        'Age': 74,
        'Gender': 1,
        'Symptoms': 0,
        'PS': 0,
        'AFP': 1.2,  # Very low AFP - good prognosis
        'Albumin': 4.7,  # High albumin - good liver function
        'Total_Bil': 0.5,  # Low bilirubin - good
        'ALT': 35,
        'AST': 17,
        'Major_Dim': 8.3,  # Note: tumor size can vary
        'Nodule': 1,
        'Alcohol': 0,
        'HBsAg': 0,
        'HCVAb': 0,
        'Cirrhosis': 0,  # No cirrhosis
        'Diabetes': 0,
        'Smoking': 0
    }
    
    print("Input data:")
    for key, value in low_risk.items():
        print(f"  {key}: {value}")
    
    result = predictor.predict_with_confidence(low_risk)
    
    if result.get('success'):
        print(f"\n✓ Prediction: {result['prediction_text']}")
        print(f"✓ Risk Level: {result['risk_level']}")
        print(f"✓ Probability: {result['probability']:.4f} ({result['probability']*100:.1f}%)")
        print(f"✓ Explanation: {result['explanation']}")
        
        if result['risk_level'] == 'Low':
            print(f"✓ PASS: Correctly predicted Low risk")
        else:
            print(f"⚠️  Expected Low risk, got {result['risk_level']}")
    else:
        print(f"❌ Error: {result.get('error')}")
    
    # Test Case 2: Medium Risk Patient
    print("\n" + "=" * 80)
    print("TEST CASE 2: MEDIUM RISK PATIENT (Borderline profile)")
    print("=" * 80)
    
    medium_risk = {
        'Age': 63,
        'Gender': 1,  # Male
        'Symptoms': 1,
        'PS': 1,
        'AFP': 5000,  # Moderate AFP between class means
        'Albumin': 3.4,  # Between class means
        'Total_Bil': 3.0,  # Between class means
        'ALT': 66,
        'AST': 95,
        'Major_Dim': 7.0,  # Between class means
        'Nodule': 3,
        'Alcohol': 1,
        'HBsAg': 0,
        'HCVAb': 0,
        'Cirrhosis': 1,
        'Diabetes': 0,
        'Smoking': 1
    }
    
    print("Input data:")
    for key, value in medium_risk.items():
        print(f"  {key}: {value}")
    
    result = predictor.predict_with_confidence(medium_risk)
    
    if result.get('success'):
        print(f"\n✓ Prediction: {result['prediction_text']}")
        print(f"✓ Risk Level: {result['risk_level']}")
        print(f"✓ Probability: {result['probability']:.4f} ({result['probability']*100:.1f}%)")
        print(f"✓ Explanation: {result['explanation']}")
    else:
        print(f"❌ Error: {result.get('error')}")
    
    # Test Case 3: High Risk Patient (Class 0 profile - Worse prognosis, high bilirubin)
    print("\n" + "=" * 80)
    print("TEST CASE 3: HIGH RISK PATIENT (Worse prognosis)")
    print("=" * 80)
    
    high_risk = {
        'Age': 68,
        'Gender': 1,
        'Symptoms': 1,  # Symptomatic - worse
        'PS': 2,  # Limited activity - worse
        'AFP': 5000,  # Moderate-high AFP
        'Albumin': 2.9,  # LOW albumin - Class 0 characteristic
        'Total_Bil': 5.5,  # HIGH bilirubin - Class 0 characteristic
        'ALT': 95,
        'AST': 140,
        'Major_Dim': 9.5,  # Large tumor - Class 0
        'Nodule': 3,
        'Alcohol': 1,
        'HBsAg': 0,
        'HCVAb': 0,
        'Cirrhosis': 1,
        'Diabetes': 1,
        'Smoking': 1
    }
    
    print("Input data:")
    for key, value in high_risk.items():
        print(f"  {key}: {value}")
    
    result = predictor.predict_with_confidence(high_risk)
    
    if result.get('success'):
        print(f"\n✓ Prediction: {result['prediction_text']}")
        print(f"✓ Risk Level: {result['risk_level']}")
        print(f"✓ Probability: {result['probability']:.4f} ({result['probability']*100:.1f}%)")
        print(f"✓ Explanation: {result['explanation']}")
        print(f"✓ SHAP available: {result.get('shap_available', False)}")
        
        if result['risk_level'] == 'High':
            print(f"✓ PASS: Correctly predicted High risk")
        else:
            print(f"⚠️  Expected High risk, got {result['risk_level']}")
    else:
        print(f"❌ Error: {result.get('error')}")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    test_prediction_system()
