import os

# Flask Configuration
SECRET_KEY = os.environ.get('SECRET_KEY') or 'hcc-recurrence-prediction-secret-key-2024'

# File Upload Configuration
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Application Settings
DEBUG = True
TESTING = False

# Model Paths
MODEL_PATH = '../models/trained_models/best_model.pkl'
PREPROCESSOR_PATH = '../models/preprocessing/preprocessor.pkl'

# Feature Configuration
FEATURE_CATEGORIES = {
    'demographic': ['Age', 'Gender'],
    'clinical': ['Symptoms', 'PS', 'Encephalopathy', 'Ascites'],
    'laboratory': ['AFP', 'ALT', 'AST', 'Albumin', 'Total_Bil', 'INR', 'Platelets'],
    'tumor': ['Nodules', 'Major_Dim'],
    'comorbidities': ['Alcohol', 'HBsAg', 'HCVAb', 'Cirrhosis', 'Diabetes', 'Smoking']
}