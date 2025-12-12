import numpy as np
import pandas as pd
import json
import os
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HCCModelTrainer:
    """
    Robust model training with comprehensive evaluation
    Aligned with notebook analysis and preprocessing
    """
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.best_model = None
        self.best_model_name = None
        self.cv_results = {}
        self.trained_models = {}
        self.X_test = None
        self.y_test = None
        
    def initialize_models(self):
        """Initialize models with optimized parameters"""
        self.models = {
            "Random Forest": {
                "model": RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=1
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                }
            },
            "XGBoost": {
                "model": XGBClassifier(
                    random_state=42,
                    eval_metric="logloss",
                    n_jobs=1,
                    use_label_encoder=False
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1],
                    "subsample": [0.8, 1.0]
                }
            },
            "Logistic Regression": {
                "model": LogisticRegression(
                    random_state=42,
                    class_weight="balanced",
                    max_iter=1000
                ),
                "params": {
                    "C": [0.1, 1, 10],
                    "solver": ["liblinear", "saga"],
                    "penalty": ["l1", "l2"]
                }
            },
            "Gradient Boosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5],
                    "subsample": [0.8, 1.0]
                }
            },
            "SVM": {
                "model": SVC(random_state=42, class_weight="balanced", probability=True),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"]
                }
            }
        }
        logger.info(f"Initialized {len(self.models)} models")
        
    def perform_cross_validation(self, X_train, y_train, cv_folds=5):
        """Perform cross-validation with reduced parallelism"""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        results = {}
        
        for name, info in self.models.items():
            try:
                model = info["model"]
                # Use single job to avoid multiprocessing issues
                auc_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=cv, 
                    scoring="roc_auc",
                    n_jobs=1
                )
                results[name] = {
                    "auc_mean": auc_scores.mean(),
                    "auc_std": auc_scores.std()
                }
                logger.info(f"CV {name}: AUC = {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
            except Exception as e:
                logger.error(f"CV failed for {name}: {e}")
                results[name] = {"auc_mean": 0, "auc_std": 0}
                
        self.cv_results = results
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, cv_folds=3):
        """Hyperparameter tuning with reduced parallelism"""
        tuned_models = {}
        
        for name, info in self.models.items():
            logger.info(f"Tuning {name}...")
            
            try:
                grid = GridSearchCV(
                    estimator=info["model"],
                    param_grid=info["params"],
                    scoring="roc_auc",
                    cv=cv_folds,
                    n_jobs=1,
                    error_score="raise",
                    return_train_score=True
                )
                
                grid.fit(X_train, y_train)
                
                tuned_models[name] = {
                    "model": grid.best_estimator_,
                    "best_params": grid.best_params_,
                    "best_score": grid.best_score_
                }
                
                logger.info(f"Best params for {name}: {grid.best_params_}")
                logger.info(f"Best CV score: {grid.best_score_:.4f}")
                
            except Exception as e:
                logger.error(f"Grid search failed for {name}: {e}")
                # Use default model as fallback
                tuned_models[name] = {
                    "model": info["model"],
                    "best_params": {},
                    "best_score": 0
                }
        
        self.trained_models = tuned_models
        return tuned_models
    
    def evaluate_models(self, X_test, y_test):
        """Comprehensive model evaluation"""
        # Store test data for plotting
        self.X_test = X_test
        self.y_test = y_test
        
        results = {}
        
        for name, info in self.trained_models.items():
            try:
                model = info["model"]
                
                # Predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1_score": f1_score(y_test, y_pred, zero_division=0),
                    "roc_auc": roc_auc_score(y_test, y_prob),
                }
                
                # Additional metrics
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_prob)
                metrics["pr_auc"] = auc(pr_recall, pr_precision)
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                metrics["confusion_matrix"] = cm.tolist()
                
                # Classification report
                report = classification_report(y_test, y_pred, output_dict=True)
                metrics["classification_report"] = report
                
                results[name] = metrics
                
                logger.info(f"Evaluation - {name}: AUC = {metrics['roc_auc']:.4f}, F1 = {metrics['f1_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Evaluation failed for {name}: {e}")
                results[name] = {
                    "accuracy": 0, "precision": 0, "recall": 0, 
                    "f1_score": 0, "roc_auc": 0, "pr_auc": 0
                }
        
        self.model_performance = results
        return results
    
    def select_best_model(self):
        """Select best model based on ROC AUC"""
        if not self.model_performance:
            raise ValueError("No model performance data available")
            
        best_model_name = max(
            self.model_performance.keys(),
            key=lambda x: self.model_performance[x]["roc_auc"]
        )
        
        self.best_model_name = best_model_name
        self.best_model = self.trained_models[best_model_name]["model"]
        
        logger.info(f"Best model: {best_model_name} with AUC: {self.model_performance[best_model_name]['roc_auc']:.4f}")
        
        return best_model_name, self.best_model
    
    def plot_performance_comparison(self, save_path=None):
        """Create performance comparison visualization"""
        if not self.model_performance:
            logger.warning("No performance data to plot")
            return
            
        models = list(self.model_performance.keys())
        metrics = ['roc_auc', 'f1_score', 'accuracy', 'precision', 'recall']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [self.model_performance[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color='skyblue', alpha=0.8)
            axes[i].set_title(f'{metric.upper()} Comparison', fontweight='bold')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # ROC Curve comparison
        axes[5].set_title('ROC Curves Comparison', fontweight='bold')
        if self.X_test is not None and self.y_test is not None:
            for name in models:
                if name in self.trained_models:
                    model = self.trained_models[name]["model"]
                    try:
                        y_prob = model.predict_proba(self.X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                        auc_score = roc_auc_score(self.y_test, y_prob)
                        axes[5].plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
                    except Exception as e:
                        logger.warning(f"Could not plot ROC for {name}: {e}")
        
        axes[5].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[5].set_xlabel('False Positive Rate')
        axes[5].set_ylabel('True Positive Rate')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def full_training_pipeline(self, X_train, y_train, X_test, y_test, 
                             save_path=None, cv_folds=3):
        """
        Complete training pipeline
        """
        logger.info("Starting complete training pipeline...")
        
        try:
            # Store test data for plotting
            self.X_test = X_test
            self.y_test = y_test
            
            # 1. Initialize models
            self.initialize_models()
            
            # 2. Cross-validation
            self.perform_cross_validation(X_train, y_train, cv_folds)
            
            # 3. Hyperparameter tuning
            self.hyperparameter_tuning(X_train, y_train, cv_folds)
            
            # 4. Refit best models on full training data
            for name, info in self.trained_models.items():
                info["model"].fit(X_train, y_train)
            
            # 5. Evaluate models
            self.evaluate_models(X_test, y_test)
            
            # 6. Select best model
            best_name, best_model = self.select_best_model()
            
            # 7. Save models and results
            if save_path:
                self.save_models_and_results(save_path, best_name, best_model)
            
            logger.info(" Training pipeline completed successfully!")
            
            return best_name, best_model
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def save_models_and_results(self, save_path: str, best_name: str, best_model):
        """Save all models and performance results"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save best model
        joblib.dump(best_model, os.path.join(save_path, "best_model.pkl"))
        
        # Save all models
        for name, info in self.trained_models.items():
            model_path = os.path.join(save_path, f"{name.lower().replace(' ', '_')}.pkl")
            joblib.dump(info["model"], model_path)
        
        # Save performance report
        performance_report = {
            'best_model': best_name,
            'performance_metrics': self.model_performance,
            'cv_results': self.cv_results,
            'training_date': datetime.now().isoformat(),
            'model_parameters': {
                name: info.get('best_params', {}) 
                for name, info in self.trained_models.items()
            }
        }
        
        with open(os.path.join(save_path, "performance_report.json"), 'w') as f:
            json.dump(performance_report, f, indent=4)
        
        # Create performance visualization
        self.plot_performance_comparison(
            os.path.join(save_path, "performance_comparison.png")
        )
        
        logger.info(f" Models and results saved to {save_path}")


# Example usage
if __name__ == "__main__":
    # Example of how to use the trainer
    trainer = HCCModelTrainer()
    
    print("HCC Model Trainer initialized successfully!")