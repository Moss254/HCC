import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from typing import Tuple, Dict, Any, List
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HCCDataPreprocessor:
    """
    FIXED Data Preprocessor - Robust feature engineering and preprocessing
    Aligned with notebook analysis
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feature_names = []
        self.final_pipeline = None
        self.fitted = False
        self.categorical_features = []
        self.numerical_features = []
        
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate dataset"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
                
            logger.info(f"Loaded data with shape: {df.shape}")
            
            # Validate required columns
            required_columns = ['Class', 'Age', 'AFP', 'Major_Dim']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Robust missing value imputation"""
        df_clean = df.copy()
        
        # Numeric: Median imputation
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                logger.info(f"Imputed missing values in {col}")
        
        # Categorical: Mode imputation
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
                
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer clinically relevant derived features
        UPDATED: Uses only the 17 base features collected by the form
        """
        df_eng = df.copy()
        
        # Age Categories (clinically relevant) - Aligned with notebook
        if 'Age' in df_eng.columns:
            age_vals = df_eng['Age'].values
            age_category = np.zeros_like(age_vals, dtype=float)
            age_category[(age_vals >= 50) & (age_vals <= 65)] = 1.0
            age_category[age_vals > 65] = 2.0
            df_eng['Age_Category'] = age_category
        
        # AFP Risk Categories (clinical guidelines) - Aligned with notebook
        if 'AFP' in df_eng.columns:
            afp_vals = df_eng['AFP'].values
            afp_category = np.zeros_like(afp_vals, dtype=float)
            afp_category[(afp_vals >= 20) & (afp_vals <= 400)] = 1.0
            afp_category[afp_vals > 400] = 2.0
            df_eng['AFP_Risk_Category'] = afp_category
        
        # Liver function composite score - using only available markers from form
        liver_markers = ['Albumin', 'Total_Bil', 'ALT', 'AST']
        available_markers = [marker for marker in liver_markers if marker in df_eng.columns]
        
        if len(available_markers) >= 2:
            # Normalize and create composite score
            liver_scores = []
            for marker in available_markers:
                if marker == 'Albumin':
                    # Higher albumin is better - use safe normalization
                    marker_range = df_eng[marker].max() - df_eng[marker].min()
                    if marker_range > 0:
                        score = (df_eng[marker] - df_eng[marker].min()) / marker_range
                    else:
                        score = 0.5  # Default to neutral if no variation
                else:
                    # Lower values are better for other markers
                    marker_range = df_eng[marker].max() - df_eng[marker].min()
                    if marker_range > 0:
                        score = 1 - ((df_eng[marker] - df_eng[marker].min()) / marker_range)
                    else:
                        score = 0.5  # Default to neutral if no variation
                liver_scores.append(score)
            
            df_eng['Liver_Function_Score'] = np.mean(liver_scores, axis=0)
        else:
            # Default liver function score if not enough markers
            df_eng['Liver_Function_Score'] = 0.0
        
        logger.info(f"Engineered features. Final shape: {df_eng.shape}")
        return df_eng
    
    def identify_feature_types(self, df: pd.DataFrame) -> Tuple[list, list]:
        """Identify categorical and numerical features"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target variable and engineered features that should be numeric
        if 'Class' in numerical_cols:
            numerical_cols.remove('Class')
            
        # Ensure engineered features are in numerical
        engineered_features = ['Age_Category', 'AFP_Risk_Category', 'Liver_Function_Score']
        for feature in engineered_features:
            if feature in df.columns and feature not in numerical_cols:
                numerical_cols.append(feature)
            if feature in categorical_cols:
                categorical_cols.remove(feature)
                
        return categorical_cols, numerical_cols
    
    def create_preprocessing_pipeline(self, categorical_cols: list, numerical_cols: list):
        """Create robust preprocessing pipeline"""
        
        # Numeric pipeline with robust scaling
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
        ])
        
        # Combined preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop',  # Remove any unexpected columns
            n_jobs=1  # Avoid multiprocessing issues in containers
        )
        
        return preprocessor
    
    def encode_and_scale(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """Encode categorical variables and scale numerical ones
        Returns processed features and target"""
        logger.info("Starting feature encoding and scaling...")
    
        # Separate target
        if 'Class' not in df.columns:
         raise ValueError("Target column 'Class' not found in data")
        
        X = df.drop(columns=['Class'])
        y = df['Class']
    
        # Identify feature types
        categorical_cols, numerical_cols = self. identify_feature_types(X)
    
        # Store for later use
        self.categorical_features = categorical_cols
        self. numerical_features = numerical_cols
    
        logger.info(f"Categorical features: {len(categorical_cols)}")
        logger.info(f"Numerical features: {len(numerical_cols)}")
    
        # Create and fit preprocessor
        self.final_pipeline = self.create_preprocessing_pipeline(categorical_cols, numerical_cols)
        X_processed = self.final_pipeline.fit_transform(X)
    
        # Get feature names AFTER fitting (this is the fix!)
        self._extract_feature_names(categorical_cols, numerical_cols)
    
        self.fitted = True
        logger.info(f" Final feature count: {len(self.feature_names)}")
        logger.info(f" Processed data shape: {X_processed.shape}")
    
        return X_processed, y

    def _extract_feature_names(self, categorical_cols: list, numerical_cols: list):
        """Extract feature names from the fitted pipeline"""
        try:
            # Check if pipeline is fitted
            if not hasattr(self. final_pipeline, 'transformers_'):
               logger.warning("Pipeline not fitted yet, using basic feature names")
               self.feature_names = numerical_cols + categorical_cols
               return
            
            # Get one-hot encoded feature names from the fitted transformer
            if 'cat' in self.final_pipeline.named_transformers_: 
                cat_transformer = self.final_pipeline.named_transformers_['cat']
            
                 # Check if the categorical transformer has the onehot step and it's fitted
                if hasattr(cat_transformer, 'named_steps') and 'onehot' in cat_transformer.named_steps:
                    ohe = cat_transformer.named_steps['onehot']
                
                    # Only get feature names if the encoder is fitted
                    if hasattr(ohe, 'categories_'):
                        ohe_features = ohe.get_feature_names_out(categorical_cols).tolist()
                        self.feature_names = numerical_cols + ohe_features
                        logger.info(f" Extracted {len(ohe_features)} one-hot encoded feature names")
                        return
        
            # Fallback:  try to get feature names from the pipeline directly
            if hasattr(self. final_pipeline, 'get_feature_names_out'):
                self.feature_names = list(self.final_pipeline.get_feature_names_out())
                logger.info(f" Extracted feature names from pipeline: {len(self.feature_names)}")
                return
            
            # Final fallback: use numerical + categorical
            self.feature_names = numerical_cols + categorical_cols
            logger.info(f" Using basic feature names:  {len(self.feature_names)}")
        
        except Exception as e:
            logger.warning(f"Could not extract feature names properly: {e}")
            # Fallback: create generic feature names based on processed data shape
            self.feature_names = numerical_cols + [f"cat_{i}" for i in range(len(categorical_cols) * 2)]
            logger.info(f" Using fallback feature names: {len(self.feature_names)}")

    def prepare_training_data(self, X: np.ndarray, y: pd.Series, test_size: float = 0.2):
        """Split data into train and test sets"""
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
    
    def save_preprocessor(self, save_path: str):
        """Save preprocessor with all necessary metadata"""
        os.makedirs(save_path, exist_ok=True)
        
        if not self.fitted:
            raise Exception("Preprocessor must be fitted before saving")
        
        # Save as dictionary with all necessary components
        preprocessor_data = {
            'pipeline': self.final_pipeline,
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'fitted': self.fitted,
            'config': self.config
        }
        
        joblib.dump(preprocessor_data, os.path.join(save_path, 'preprocessor.pkl'))
        
        # Save feature list for reference
        feature_df = pd.DataFrame({'feature_name': self.feature_names})
        feature_df.to_csv(os.path.join(save_path, 'feature_list.csv'), index=False)
        
        logger.info(f" Preprocessor saved with {len(self.feature_names)} features")
    
    def get_feature_names(self):
        """Get feature names after fitting"""
        if not self.fitted:
            raise Exception("Preprocessor not fitted yet")
        return self.feature_names
    
    def full_preprocessing_pipeline(self, file_path: str, save_path: str = None) -> Tuple:
        """
        Complete preprocessing pipeline
        Returns: X_train, X_test, y_train, y_test, processed_df
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        try:
            # 1. Load data
            df = self.load_and_validate_data(file_path)
            logger.info(f"Original data shape: {df.shape}")
            
            # 2. Handle missing values
            df_clean = self.handle_missing_values(df)
            logger.info(f"After missing value handling: {df_clean.shape}")
            
            # 3. Engineer features
            df_eng = self.engineer_features(df_clean)
            logger.info(f"After feature engineering: {df_eng.shape}")
            
            # 4. Encode and scale
            X_processed, y = self.encode_and_scale(df_eng)
            
            # 5. Split data
            X_train, X_test, y_train, y_test = self.prepare_training_data(X_processed, y)
            
            logger.info(f"Training set: {X_train.shape}")
            logger.info(f"Test set: {X_test.shape}")
            logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
            logger.info(f"Class distribution - Test: {y_test.value_counts().to_dict()}")
            
            # 6. Save preprocessor
            if save_path:
                self.save_preprocessor(save_path)
            
            return X_train, X_test, y_train, y_test, df_eng
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            raise