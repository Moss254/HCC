# train_model.py

import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data.data_preprocessor import HCCDataPreprocessor
from src.models.model_trainer import HCCModelTrainer
from src.models.simple_interpreter import SimpleModelInterpreter

def main():
    print("=" * 80)
    print("HCC RECURRENCE PREDICTION - MODEL TRAINING PHASE")
    print("=" * 80)

    # Load and preprocess data
    print(" Loading and preprocessing data...")

    preprocessor = HCCDataPreprocessor()
    X_train, X_test, y_train, y_test, df_processed = preprocessor.full_preprocessing_pipeline(
        file_path='data/raw/hcc-data-complete-balanced.xlsx',
        save_path='models/preprocessing/'
    )

    print(f" Data preprocessing completed!")
    print(f" Training set: {X_train.shape}")
    print(f" Test set: {X_test.shape}")
    print(f" Class distribution - Train: {pd.Series(y_train).value_counts().to_dict()}")
    print(f" Class distribution - Test: {pd.Series(y_test).value_counts().to_dict()}")

    # Initialize model trainer
    print("\n Initializing model trainer...")
    model_trainer = HCCModelTrainer()

    # Execute complete training pipeline
    print("\n Starting complete model training pipeline...")

    best_model_name, best_model = model_trainer.full_training_pipeline(
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test,
        save_path='models/trained_models/'
    )

    # Display comprehensive result
    print("\n" + "=" * 60)
    print(" COMPREHENSIVE MODEL PERFORMANCE RESULTS")
    print("=" * 60)

    performance_results = model_trainer.model_performance

    # Create performance comparison table
    performance_df = pd.DataFrame()

    for model_name, metrics in performance_results.items():
        performance_df[model_name] = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            metrics['roc_auc'],
            metrics['pr_auc']
        ]

    performance_df.index = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'PR AUC']
    performance_df = performance_df.T.round(4)

    print("\nPerformance Metrics Comparison:")
    print(performance_df.sort_values('ROC AUC', ascending=False))

    # Best model details
    print("\n" + "=" * 50)
    print(" BEST MODEL IDENTIFIED")
    print("=" * 50)
    print(f"Best Model: {model_trainer.best_model_name}")
    print(f"ROC AUC Score: {performance_results[model_trainer.best_model_name]['roc_auc']:.4f}")
    print(f"F1-Score: {performance_results[model_trainer.best_model_name]['f1_score']:.4f}")

    # Model Interpretation
    print("\n" + "=" * 50)
    print(" MODEL INTERPRETATION & FEATURE ANALYSIS")
    print("=" * 50)

    # Initialize interpreter
    interpreter = SimpleModelInterpreter(
        model=best_model,
        feature_names=preprocessor.get_feature_names(),
        class_names=['No Recurrence', 'Recurrence']
    )

    # Generate interpretation report
    print("Generating model interpretation report...")
    interpretation_fig = interpreter.generate_interpretation_report(
        X_test, y_test,
        save_path='reports/feature_importance_report.png'
    )

    # Feature Importance Analysis
    print("\n Top 10 Most Important Features:")

    # Get permutation importance
    perm_importance = interpreter.get_feature_importance(X_test, y_test)
    if perm_importance is not None:
        print(perm_importance.head(10).round(4))

    # Clinical Insights
    print("\n" + "=" * 50)
    print(" CLINICAL INSIGHTS & RECOMMENDATIONS")
    print("=" * 50)

    if perm_importance is not None:
        top_features = perm_importance.head(5)['feature'].tolist()
        print("Top 5 clinically significant features for recurrence prediction:")
        for i, feature in enumerate(top_features, 1):
            print(f"{i}. {feature}")

    # Final Summary
    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETED: MODEL TRAINING & COMPARISON")
    print("=" * 80)
    print(" Key Achievements:")
    print(f"• Trained and compared {len(model_trainer.models)} different algorithms")
    print(f"• Best model: {model_trainer.best_model_name} with AUC: {performance_results[model_trainer.best_model_name]['roc_auc']:.4f}")
    print(f"• Comprehensive model interpretation provided")
    print(f"• All models saved for deployment")
    print(" Next: Web Application Development & Deployment")

if __name__ == "__main__":
    main()