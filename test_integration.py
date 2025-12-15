#!/usr/bin/env python3
"""
Integration test for the web application API
"""

import sys
sys.path.append('.')

import json
from webapp.app import HCCPredictionSystem

def test_api():
    """Test the full API workflow"""
    
    print("=" * 80)
    print("INTEGRATION TEST - Web Application API")
    print("=" * 80)
    
    # Initialize the system
    system = HCCPredictionSystem()
    
    # Test prediction API
    print("\n1. Testing /api/predict endpoint")
    print("-" * 80)
    
    # Low risk test case (from form sample data format)
    test_data = {
        'Age': 74,
        'Gender': 1,
        'Symptoms': 0,
        'PS': 0,
        'AFP': 1.2,
        'Albumin': 4.7,
        'Total_Bil': 0.5,
        'ALT': 35,
        'AST': 17,
        'Major_Dim': 8.3,
        'Nodules': 1,  # Note: form sends 'Nodules' not 'Nodule'
        'Alcohol': 0,
        'HBsAg': 0,
        'HCVAb': 0,
        'Cirrhosis': 0,
        'Diabetes': 0,
        'Smoking': 0
    }
    
    print(f"Input data: {json.dumps(test_data, indent=2)}")
    
    result = system.predictor.predict_with_confidence(test_data)
    
    if result.get('success'):
        print(f"\n✓ API Response:")
        print(f"  - Prediction: {result['prediction_text']}")
        print(f"  - Risk Level: {result['risk_level']}")
        print(f"  - Probability: {result['probability']:.4f} ({result['confidence']})")
        print(f"  - Explanation: {result['explanation']}")
        print(f"  - SHAP Available: {result.get('shap_available', result.get('shap_plot') is not None)}")
        
        if result.get('recommendations'):
            print(f"  - Recommendations: {len(result['recommendations'])} provided")
        
        # Verify expected response structure
        required_fields = ['success', 'prediction', 'prediction_text', 'probability', 
                          'risk_level', 'explanation', 'recommendations']
        missing = [f for f in required_fields if f not in result]
        
        if missing:
            print(f"\n⚠️  Missing fields: {missing}")
        else:
            print(f"\n✓ All required fields present")
        
        # Test with high risk data
        print("\n2. Testing high-risk prediction")
        print("-" * 80)
        
        high_risk_data = {
            'Age': 68,
            'Gender': 1,
            'Symptoms': 1,
            'PS': 2,
            'AFP': 5000,
            'Albumin': 2.9,
            'Total_Bil': 5.5,
            'ALT': 95,
            'AST': 140,
            'Major_Dim': 9.5,
            'Nodules': 3,
            'Alcohol': 1,
            'HBsAg': 0,
            'HCVAb': 0,
            'Cirrhosis': 1,
            'Diabetes': 1,
            'Smoking': 1
        }
        
        result2 = system.predictor.predict_with_confidence(high_risk_data)
        
        if result2.get('success'):
            print(f"✓ High-risk prediction:")
            print(f"  - Risk Level: {result2['risk_level']}")
            print(f"  - Probability: {result2['probability']:.4f}")
            
            if result2['risk_level'] in ['High', 'Medium']:
                print(f"✓ PASS: Correctly identified as {result2['risk_level']} risk")
            else:
                print(f"⚠️  Expected High/Medium risk, got {result2['risk_level']}")
    else:
        print(f"❌ Error: {result.get('error')}")
    
    print("\n" + "=" * 80)
    print("INTEGRATION TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    test_api()
