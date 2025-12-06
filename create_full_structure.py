import os

# Folders
folders = [
    "data/raw",
    "data/processed",
    "data/external",
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/visualization",
    "src/utils",
    "webapp/templates",
    "webapp/static/css",
    "webapp/static/js",
    "webapp/static/images",
    "models/trained_models",
    "models/preprocessing",
    "models/model_performance",
    "tests",
    "docs"
]

# Files to create
files = [
    "requirements.txt",
    "setup.py",
    "run.py",
    "config.yaml",
    "README.md",
    
    # src files
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/data_loader.py",
    "src/data/data_preprocessor.py",
    "src/features/__init__.py",
    "src/features/feature_engineering.py",
    "src/features/feature_selection.py",
    "src/models/__init__.py",
    "src/models/model_trainer.py",
    "src/models/model_evaluator.py",
    "src/models/model_predictor.py",
    "src/visualization/__init__.py",
    "src/visualization/plotter.py",
    "src/utils/__init__.py",
    "src/utils/helpers.py",
    
    # notebooks
    "notebooks/01_data_analysis_and_preprocessing.ipynb",
    "notebooks/02_exploratory_data_analysis.ipynb",
    "notebooks/03_feature_engineering.ipynb",
    "notebooks/04_model_training_comparison.ipynb",
    "notebooks/05_model_evaluation_interpretation.ipynb",
    "notebooks/06_web_application_development.ipynb",
    
    # webapp files
    "webapp/app.py",
    "webapp/config.py",
    "webapp/forms.py",
    "webapp/templates/base.html",
    "webapp/templates/index.html",
    "webapp/templates/predict.html",
    "webapp/templates/batch_predict.html",
    "webapp/templates/results.html",
    "webapp/templates/dashboard.html",
    "webapp/static/css/style.css",
    "webapp/static/css/dashboard.css",
    "webapp/static/js/main.js",
    "webapp/static/js/charts.js",
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create empty files
for file in files:
    if not os.path.exists(file):
        open(file, "w").close()

print("✔ Full project structure created successfully!")
