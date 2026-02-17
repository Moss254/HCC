#!/usr/bin/env python3
"""
Retrain HCC Model with Reduced Feature Set
Uses only the 17 base features collected by the web form
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define the 17 base features collected by the form
BASE_FEATURES = [
    'Age', 'Gender', 'Symptoms', 'PS',
    'AFP', 'Albumin', 'Total_Bil', 'ALT', 'AST',
    'Major_Dim', 'Nodule',
    'Alcohol', 'HBsAg', 'HCVAb', 'Cirrhosis', 'Diabetes', 'Smoking'
]

def engineer_features(df):
    """Engineer the 3 derived features"""
    df_eng = df.copy()
    
    # Age categorization
    age_vals = df_eng['Age'].values
    age_cat = np.zeros_like(age_vals, dtype=float)
    age_cat[(age_vals >= 50) & (age_vals <= 65)] = 1.0
    age_cat[age_vals > 65] = 2.0
    df_eng['Age_Category'] = age_cat
    
    # AFP risk categorization
    afp_vals = df_eng['AFP'].values
    afp_cat = np.zeros_like(afp_vals, dtype=float)
    afp_cat[(afp_vals >= 20) & (afp_vals <= 400)] = 1.0
    afp_cat[afp_vals > 400] = 2.0
    df_eng['AFP_Risk_Category'] = afp_cat
    
    # Liver function composite score
    liver_markers = ['Albumin', 'Total_Bil', 'ALT', 'AST']
    liver_scores = []
    liver_norm_stats = {}
    
    for marker in liver_markers:
        if marker == 'Albumin':
            # Higher albumin is better
            marker_min = df_eng[marker].min()
            marker_max = df_eng[marker].max()
            marker_range = marker_max - marker_min
            liver_norm_stats[marker] = {
                'min': float(marker_min),
                'max': float(marker_max),
                'range': float(marker_range)
            }
            if marker_range > 0:
                score = (df_eng[marker] - marker_min) / marker_range
            else:
                score = pd.Series([0.5] * len(df_eng))
        else:
            # Lower values are better for other markers
            marker_min = df_eng[marker].min()
            marker_max = df_eng[marker].max()
            marker_range = marker_max - marker_min
            liver_norm_stats[marker] = {
                'min': float(marker_min),
                'max': float(marker_max),
                'range': float(marker_range)
            }
            if marker_range > 0:
                score = 1 - ((df_eng[marker] - marker_min) / marker_range)
            else:
                score = pd.Series([0.5] * len(df_eng))
        liver_scores.append(score)
    
    df_eng['Liver_Function_Score'] = pd.DataFrame(liver_scores).T.mean(axis=1)
    
    return df_eng, liver_norm_stats

def main():
    print("=" * 80)
    print("HCC MODEL RETRAINING WITH REDUCED FEATURE SET")
    print("=" * 80)
    print(f"Using {len(BASE_FEATURES)} base features + 3 engineered features = 20 total")
    
    # Load data
    logger.info("Loading data...")
    data_path = 'data/raw/hcc-data-complete-balanced.xlsx'
    df = pd.read_excel(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    
    # Select only the base features we need + target
    available_features = [f for f in BASE_FEATURES if f in df.columns]
    missing_features = [f for f in BASE_FEATURES if f not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing features in dataset: {missing_features}")
        logger.error("Cannot proceed without all required features")
        return
    
    logger.info(f"Selected {len(available_features)} base features from dataset")
    
    # Extract features and target
    X = df[BASE_FEATURES].copy()
    y = df['Class'].copy()
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Handle missing values (use median for numerical)
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
            logger.info(f"Filled {X[col].isnull().sum()} missing values in {col}")
    
    # Engineer features
    logger.info("Engineering features...")
    X_eng, liver_norm_stats = engineer_features(X)
    logger.info(f"Saved liver normalization stats: {liver_norm_stats}")
    
    # Get all feature names (base + engineered)
    all_features = BASE_FEATURES + ['Age_Category', 'AFP_Risk_Category', 'Liver_Function_Score']
    X_final = X_eng[all_features]
    
    logger.info(f"Final feature matrix shape: {X_final.shape}")
    logger.info(f"Features: {list(X_final.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    # Scale features
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            random_state=42, class_weight='balanced', n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, eval_metric='logloss', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0, random_state=42, class_weight='balanced', max_iter=1000
        )
    }
    
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        results[name] = metrics
        trained_models[name] = model
        
        logger.info(f"{name} - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    best_model = trained_models[best_model_name]
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    print(f"\nBest Model: {best_model_name}")
    print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    print(f"F1 Score: {results[best_model_name]['f1_score']:.4f}")
    
    # Save model and preprocessor
    logger.info("Saving model and preprocessor...")
    
    os.makedirs('models/trained_models', exist_ok=True)
    os.makedirs('models/preprocessing', exist_ok=True)
    
    # Save best model
    joblib.dump(best_model, 'models/trained_models/best_model.pkl')
    logger.info("Saved best model")
    
    # Save preprocessor (scaler with metadata)
    preprocessor_data = {
        'pipeline': scaler,
        'feature_names': all_features,
        'feature_count': len(all_features),
        'fitted': True,
        'liver_norm_stats': liver_norm_stats,
        'config': {
            'base_features': BASE_FEATURES,
            'engineered_features': ['Age_Category', 'AFP_Risk_Category', 'Liver_Function_Score'],
            'training_date': datetime.now().isoformat(),
            'model_type': type(best_model).__name__
        }
    }
    
    joblib.dump(preprocessor_data, 'models/preprocessing/preprocessor.pkl')
    logger.info("Saved preprocessor")
    
    # Save feature list
    feature_df = pd.DataFrame({'feature_name': all_features})
    feature_df.to_csv('models/preprocessing/feature_list.csv', index=False)
    logger.info("Saved feature list")
    
    # Save performance report
    performance_report = {
        'best_model': best_model_name,
        'performance_metrics': results,
        'training_date': datetime.now().isoformat(),
        'feature_count': len(all_features),
        'base_features': BASE_FEATURES,
        'engineered_features': ['Age_Category', 'AFP_Risk_Category', 'Liver_Function_Score']
    }
    
    import json
    with open('models/trained_models/performance_report.json', 'w') as f:
        json.dump(performance_report, f, indent=4)
    logger.info("Saved performance report")
    
    print("\n" + "=" * 80)
    print("MODEL RETRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"• Features: {len(all_features)} (17 base + 3 engineered)")
    print(f"• Best Model: {best_model_name}")
    print(f"• ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    print(f"• Files saved in models/trained_models/ and models/preprocessing/")
    print("=" * 80)

if __name__ == "__main__":
    main()
