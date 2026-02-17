#!/usr/bin/env python3
"""
Test to verify SHAP base value is correct (~0.507) after the fix
"""

import sys
sys.path.append('.')

from webapp.app import ModelPredictor
from src.models.model_predictor import ModelPredictor as SrcModelPredictor


def test_shap_base_value():
    """Test that SHAP base value matches the expected value from balanced dataset"""
    
    print("=" * 80)
    print("TESTING SHAP BASE VALUE FIX")
    print("=" * 80)
    
    # Test webapp ModelPredictor
    print("\n1. Testing webapp/app.py ModelPredictor")
    print("-" * 80)
    
    webapp_predictor = ModelPredictor()
    webapp_success = webapp_predictor.load_model_and_preprocessor('.')
    
    if not webapp_success:
        print("❌ Failed to load webapp model and preprocessor")
        return False
    
    if webapp_predictor.shap_explainer is None:
        print("❌ SHAP explainer not initialized for webapp")
        return False
    
    webapp_base_value = webapp_predictor.shap_explainer.expected_value
    
    # Handle case where expected_value is an array
    if hasattr(webapp_base_value, '__len__') and not isinstance(webapp_base_value, str):
        if len(webapp_base_value) == 2:
            webapp_base_value = float(webapp_base_value[1])  # Class 1 expected value
        else:
            webapp_base_value = float(webapp_base_value[0])
    else:
        webapp_base_value = float(webapp_base_value)
    
    print(f"✓ Webapp SHAP base value: {webapp_base_value:.4f}")
    
    # Expected value should be around 0.507 (balanced dataset: 102/204 = 0.5)
    expected_value = 0.507
    tolerance = 0.05  # Allow 5% tolerance
    
    if abs(webapp_base_value - expected_value) <= tolerance:
        print(f"✓ PASS: Webapp base value {webapp_base_value:.4f} is within tolerance of {expected_value:.4f}")
        webapp_pass = True
    else:
        print(f"❌ FAIL: Webapp base value {webapp_base_value:.4f} is NOT within tolerance of {expected_value:.4f}")
        webapp_pass = False
    
    # Test src/models ModelPredictor
    print("\n2. Testing src/models/model_predictor.py ModelPredictor")
    print("-" * 80)
    
    src_predictor = SrcModelPredictor()
    src_success = src_predictor.load_model_and_preprocessor('.')
    
    if not src_success:
        print("❌ Failed to load src model and preprocessor")
        return False
    
    if src_predictor.shap_explainer is None:
        print("❌ SHAP explainer not initialized for src")
        return False
    
    src_base_value = src_predictor.shap_explainer.expected_value
    
    # Handle case where expected_value is an array
    if hasattr(src_base_value, '__len__') and not isinstance(src_base_value, str):
        if len(src_base_value) == 2:
            src_base_value = float(src_base_value[1])  # Class 1 expected value
        else:
            src_base_value = float(src_base_value[0])
    else:
        src_base_value = float(src_base_value)
    
    print(f"✓ Src SHAP base value: {src_base_value:.4f}")
    
    if abs(src_base_value - expected_value) <= tolerance:
        print(f"✓ PASS: Src base value {src_base_value:.4f} is within tolerance of {expected_value:.4f}")
        src_pass = True
    else:
        print(f"❌ FAIL: Src base value {src_base_value:.4f} is NOT within tolerance of {expected_value:.4f}")
        src_pass = False
    
    # Compare both values
    print("\n3. Comparing both implementations")
    print("-" * 80)
    
    if abs(webapp_base_value - src_base_value) < 0.01:
        print(f"✓ PASS: Both implementations have consistent base values")
        consistency_pass = True
    else:
        print(f"⚠️  WARNING: Base values differ between implementations")
        print(f"   Webapp: {webapp_base_value:.4f}, Src: {src_base_value:.4f}")
        consistency_pass = False
    
    # Overall result
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Webapp base value test: {'✓ PASS' if webapp_pass else '❌ FAIL'}")
    print(f"Src base value test: {'✓ PASS' if src_pass else '❌ FAIL'}")
    print(f"Consistency test: {'✓ PASS' if consistency_pass else '⚠️  WARNING'}")
    
    all_pass = webapp_pass and src_pass
    
    if all_pass:
        print("\n✓ ALL TESTS PASSED - SHAP base value is correct!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED - Please review")
        return False


if __name__ == "__main__":
    success = test_shap_base_value()
    sys.exit(0 if success else 1)
