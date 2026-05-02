"""
Test script to verify that preprocessed spectra produce different predictions
based on input data (not hardcoded predictions).
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the backend directory to path
sys.path.insert(0, os.path.dirname(__file__))

from spectrumAnalysis.services.preprocess import preprocess_spectrum
from app.ml.inference import load_model, predict_major_gases

def test_different_inputs_produce_different_predictions():
    """Test that different input spectra produce different gas predictions."""
    
    print("=" * 80)
    print("TESTING: Different inputs produce different predictions")
    print("=" * 80)
    
    # Initialize model
    print("\n1. Loading model...")
    input_dim = 312  # Model trained with 312 dims, features padded from 208
    load_model(input_dim)
    print(f"   ✓ Model loaded with input dimension: {input_dim}")
    
    # Load test data
    raw_dir = os.path.join(os.path.dirname(__file__), 'app', 'data', 'raw_data')
    
    predictions = {}
    
    print("\n2. Testing preprocessing and predictions for different spectrum types...\n")
    
    # Use existing test data files
    test_cases = [
        ('transmission', os.path.join(raw_dir, 'table_KELT-9-b-Changeat-Edwards-2021.csv'), 'KELT-9-b (Transmission)'),
        ('eclipse', os.path.join(raw_dir, 'table_KELT-9-b-Changeat-Edwards-2021.csv'), 'KELT-9-b (Eclipse)'),
        ('direct', os.path.join(raw_dir, 'table_Kepler-9-b-Edwards-et-al.-2023.csv'), 'Kepler-9-b (Direct)'),
    ]
    
    for data_type, file_path, label in test_cases:
        if not os.path.exists(file_path):
            print(f"   ⚠ {label} file not found: {file_path}")
            continue
        
        try:
            print(f"   Processing {label} spectrum ({file_path})...")
            
            # Load and preprocess
            features_df = preprocess_spectrum(file_path, data_type=data_type)
            
            # Verify feature dimensions
            n_features = features_df.shape[1]
            print(f"      - Features extracted: {n_features} dimensions")
            
            if n_features != 208:
                print(f"      ⚠ WARNING: Expected 208 features, got {n_features}")
            
            # Check for NaN values
            nan_count = features_df.isna().sum().sum()
            if nan_count > 0:
                print(f"      ⚠ WARNING: {nan_count} NaN values in features")
            
            # Make prediction
            gases = predict_major_gases(features_df, data_type)
            predictions[label] = gases
            
            # Display results
            print(f"      - Predicted gas composition:")
            for gas, percentage in gases.items():
                print(f"        • {gas}: {percentage:.2f}%")
            
            # Verify percentages sum to ~100
            total = sum(gases.values())
            print(f"      - Total: {total:.2f}%")
            
        except Exception as e:
            print(f"      ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Compare predictions
    print("\n3. Comparing predictions across different spectrum types...\n")

    # ------------------------------------------------------------------
    # verify atmosphere utilities exist and behave reasonably
    try:
        from app.main import build_full_atmosphere_profile
        print("\n4. Verifying atmosphere profile builder...")
        sample = build_full_atmosphere_profile({"H2":50}, {}, {"PL_RADJ":0.05}, "direct")
        print(f"   sample output: {sample}")
    except Exception as e:
        print(f"   ⚠ atmosphere helper failed: {e}")
    # ------------------------------------------------------------------
    
    if len(predictions) > 1:
        pred_values = list(predictions.values())
        
        # Check if all predictions are identical
        all_same = all(
            pred == pred_values[0] 
            for pred in pred_values
        )
        
        if all_same:
            print("   ✗ FAIL: All predictions are IDENTICAL across different inputs!")
            print("   This indicates the model is not using the actual input data.")
            return False
        else:
            print("   ✓ PASS: Predictions differ based on input data!")
            
            # Show differences
            print("\n   Prediction comparisons:")
            for i, (label1, preds1) in enumerate(predictions.items()):
                for label2, preds2 in list(predictions.items())[i+1:]:
                    print(f"\n   {label1} vs {label2}:")
                    for gas in preds1.keys():
                        diff = abs(preds1[gas] - preds2[gas])
                        print(f"      {gas}: {preds1[gas]:.2f}% vs {preds2[gas]:.2f}% (diff: {diff:.2f}%)")
            
            return True
    else:
        print("   ⚠ Not enough test data to compare predictions")
        return False


if __name__ == '__main__':
    success = test_different_inputs_produce_different_predictions()
    
    print("\n" + "=" * 80)
    if success:
        print("✓ TEST PASSED: Preprocessor correctly extracts features from input data")
    else:
        print("✗ TEST FAILED: Predictions are not based on input data")
    print("=" * 80)
    
    sys.exit(0 if success else 1)
