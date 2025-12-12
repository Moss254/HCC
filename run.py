"""
HCC Recurrence Prediction System - Main Runner
"""
import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    print(f"\n {description}...")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f" Error: {e}")

def main():
    print(" HCC RECURRENCE PREDICTION SYSTEM")
    print("Starting up...")
    
    # 1. Setup Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_dir)
    
    plots_dir = os.path.join(base_dir, "webapp", "static", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f" Verified directories: {plots_dir}")

    # 2. Check Data
    if not os.path.exists("data/raw/hcc-data-complete-balanced.xlsx"):
        print("  Warning: Data file 'hcc-data-complete-balanced.xlsx' not found in data/raw/")

    # 3. Smart Training Check
    model_exists = os.path.exists("models/trained_models/best_model.pkl")
    if not model_exists:
        print("  Model not found. Training now....")
        run_command("python src/data/preprocessing.py", "Preprocessing")
        run_command("python src/models/train_model.py", "Training")
    else:
        print(" Model found. Starting App...")

    # 4. Start App
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    print(f" Server starting on http://localhost:{port}")
    
    try:
        from webapp.app import HCCPredictionSystem
        hcc_system = HCCPredictionSystem()
        hcc_system.run(host=host, port=port, debug=True)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        # Fallback
        run_command("python webapp/app.py", "Fallback Start")

if __name__ == "__main__":
    main()