import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleModelInterpreter:
    """
    Robust model interpreter without complex dependencies
    """
    
    def __init__(self, model, feature_names, class_names=None):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ['No Recurrence', 'Recurrence']
        logger.info(f"Initialized interpreter with {len(feature_names)} features")
    
    def get_feature_importance(self, X_test, y_test, n_repeats=5):
        """
        Calculate permutation importance with reduced repetitions
        """
        logger.info("Calculating permutation importance.....")
        
        try:
            perm_importance = permutation_importance(
                self.model, X_test, y_test,
                n_repeats=n_repeats,
                random_state=42,
                scoring='roc_auc',
                n_jobs=1  # Single job to avoid multiprocessing issues
            )
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            logger.info(" Permutation importance calculated successfully")
            return importance_df
            
        except Exception as e:
            logger.error(f" Error calculating permutation importance: {e}")
            return None
    
    def plot_feature_importance(self, X_test, y_test, top_n=15, save_path=None):
        """
        Plot feature importance from multiple methods
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Model-specific feature importance
        if hasattr(self.model, 'feature_importances_'):
            try:
                feature_imp = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False).head(top_n)
                
                axes[0].barh(range(len(feature_imp)), feature_imp['importance'])
                axes[0].set_yticks(range(len(feature_imp)))
                axes[0].set_yticklabels(feature_imp['feature'])
                axes[0].set_xlabel('Feature Importance')
                axes[0].set_title('Model Feature Importance', fontsize=14, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
            except Exception as e:
                logger.warning(f"Could not plot model feature importance: {e}")
        
        # Permutation importance
        perm_importance_df = self.get_feature_importance(X_test, y_test)
        if perm_importance_df is not None:
            top_perm = perm_importance_df.head(top_n)
            
            axes[1].barh(range(len(top_perm)), top_perm['importance_mean'],
                       xerr=top_perm['importance_std'], alpha=0.7, color='orange')
            axes[1].set_yticks(range(len(top_perm)))
            axes[1].set_yticklabels(top_perm['feature'])
            axes[1].set_xlabel('Permutation Importance')
            axes[1].set_title('Permutation Feature Importance', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f" Feature importance plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def generate_interpretation_report(self, X_test, y_test, save_path=None):
        """
        Generate comprehensive interpretation report
        """
        logger.info("Generating interpretation report...")
        return self.plot_feature_importance(X_test, y_test, save_path=save_path)
    
    def get_top_features(self, X_test, y_test, top_n=10):
        """
        Get top features for clinical insights
        """
        perm_importance_df = self.get_feature_importance(X_test, y_test)
        if perm_importance_df is not None:
            return perm_importance_df.head(top_n)
        return None