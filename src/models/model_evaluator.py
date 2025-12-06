import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report
import logging
from typing import Dict, Any, Optional
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HCCModelInterpreter:
    """
    Model interpretation and evaluation
    """
    
    def __init__(self, model, feature_names, class_names=None):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ['No Recurrence', 'Recurrence']
        
    def get_feature_importance(self, X_test, y_test, n_repeats=10):
        """Get permutation importance"""
        try:
            result = permutation_importance(
                self.model, X_test, y_test,
                n_repeats=n_repeats,
                random_state=42,
                n_jobs=1
            )
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return None
    
    def generate_interpretation_report(self, X_train, X_test, y_test, save_path=None):
        """Generate comprehensive interpretation report"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Feature Importance
        importance_df = self.get_feature_importance(X_test, y_test)
        if importance_df is not None:
            top_features = importance_df.head(10)
            axes[0, 0].barh(range(len(top_features)), top_features['importance_mean'])
            axes[0, 0].set_yticks(range(len(top_features)))
            axes[0, 0].set_yticklabels(top_features['feature'])
            axes[0, 0].set_title('Top 10 Feature Importance')
            axes[0, 0].set_xlabel('Permutation Importance')
        
        # 2. Confusion Matrix
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 3. ROC Curve
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_test)[:, 1]
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            axes[1, 0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[1, 0].plot([0, 1], [0, 1], 'k--')
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('ROC Curve')
            axes[1, 0].legend()
        
        # 4. Precision-Recall Curve
        if hasattr(self.model, 'predict_proba'):
            from sklearn.metrics import precision_recall_curve, average_precision_score
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            ap_score = average_precision_score(y_test, y_prob)
            axes[1, 1].plot(recall, precision, label=f'PR curve (AP = {ap_score:.2f})')
            axes[1, 1].set_xlabel('Recall')
            axes[1, 1].set_ylabel('Precision')
            axes[1, 1].set_title('Precision-Recall Curve')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Interpretation report saved to {save_path}")
        
        plt.show()
        return fig