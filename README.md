#  HCC Recurrence Prediction System

An AI-powered clinical decision support system for predicting Hepatocellular Carcinoma (HCC) recurrence risk using machine learning and explainable AI (SHAP).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Model Information](#model-information)
- [Understanding SHAP Explanations](#understanding-shap-explanations)
- [Clinical Interpretation Guide](#clinical-interpretation-guide)
- [Contributing](#contributing)
- [License](#license)

##  Overview

This system uses machine learning to predict the likelihood of HCC recurrence after treatment.  It provides:

- **Risk Probability**: A percentage indicating recurrence likelihood
- **Risk Stratification**: Low, Medium, or High risk classification
- **Explainable AI**: SHAP-based feature importance visualization
- **Clinical Recommendations**: Evidence-based follow-up suggestions

### Why This Matters

Hepatocellular Carcinoma has a high recurrence rate (50-70% within 5 years). Early identification of high-risk patients enables:
- Intensified surveillance protocols
- Earlier detection of recurrence
- Improved patient outcomes
- Optimized healthcare resource allocation

##  Features

| Feature | Description |
|---------|-------------|
|  **Single Prediction** | Individual patient risk assessment |
|  **Batch Processing** | CSV/Excel upload for multiple patients |
|  **SHAP Explanations** | Visual feature importance analysis |
|  **Dashboard** | Model performance metrics and analytics |
|  **Clinical Recommendations** | Risk-stratified follow-up guidelines |
|  **Privacy-First** | All processing done locally |

##  Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) [Visual Studio Code](https://code.visualstudio.com/) - Recommended IDE

### Quick Start with VS Code (Recommended)

For the best development experience, we recommend using Visual Studio Code:

1. **Clone and Open in VS Code**
   ```bash
   git clone https://github.com/Moss254/HCC.git
   cd HCC
   code .
   ```

2. **Install Recommended Extensions**
   - VS Code will prompt you to install recommended extensions
   - Click "Install All" to get Python, debugging, and other helpful tools

3. **Set Up Python Environment**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Python: Create Environment" and select it
   - Choose "Venv" and select your Python interpreter
   - VS Code will automatically install dependencies from `requirements.txt`

4. **Run the Application**
   - Press `F5` to start debugging, or
   - Press `Ctrl+Shift+P` → "Tasks: Run Task" → "Run Flask Server"
   - Open browser to `http://localhost:8000`

📖 **See [VSCODE_SETUP.md](VSCODE_SETUP.md) for detailed VS Code setup instructions**

### Manual Installation (Alternative)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Moss254/HCC.git
cd HCC
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add Your Data

Place your HCC dataset in the `data/raw/` directory:
```
data/raw/hcc-data-complete-balanced.xlsx
```

**Required columns**:  `Class`, `Age`, `AFP`, `Major_Dim` (plus other clinical features)

### Step 5: Run the Application

```bash
python run.py
```

The system will:
1. Check for existing trained models
2. If not found, automatically train models on your data
3. Start the web server at `http://localhost:8000`

##  Project Structure

```
moss/
├── config. yaml                 # Configuration settings
├── requirements.txt            # Python dependencies
├── run. py                      # Application entry point
├── data/
│   └── raw/                    # Place your dataset here
├── models/
│   ├── trained_models/         # Saved ML models
│   └── preprocessing/          # Data preprocessors
├── src/
│   ├── data/
│   │   └── data_preprocessor.py
│   └── models/
│       ├── model_trainer.py
│       └── simple_interpreter.py
└── webapp/
    ├── app.py                  # Flask application
    ├── templates/              # HTML templates
    └── static/                 # CSS, JS, images
```

## Usage Guide

### Single Patient Prediction

1. Navigate to **Start Prediction** from the homepage
2. Enter patient data across all sections: 
   - **Demographics**: Age, Gender
   - **Clinical Status**:  Symptoms, Performance Status
   - **Laboratory Values**: AFP, Albumin, Bilirubin, ALT, AST
   - **Tumor Characteristics**:  Size, Number of nodules
   - **Comorbidities**: Cirrhosis, Hepatitis status, etc.
3. Click **Predict Recurrence Risk**
4. Review the results including:
   - Risk probability and classification
   - SHAP feature importance chart
   - Clinical recommendations

### Batch Processing

1. Navigate to **Batch Predict** from the homepage
2. Upload a CSV or Excel file with patient data
3. Download the results with predictions for all patients

### Dashboard

View model performance metrics, feature importance rankings, and system status. 

##  API Documentation

### POST /api/predict

Single patient prediction endpoint.

**Request:**
```json
{
  "Age": 65,
  "Gender": 1,
  "AFP": 850,
  "Major_Dim": 6.5,
  "Cirrhosis": 1,
  "Albumin": 3.2,
  "Total_Bil": 2.1,
  "ALT": 75,
  "AST": 82
}
```

**Response:**
```json
{
  "success": true,
  "prediction":  1,
  "prediction_text": "Recurrence Likely",
  "probability": 0.72,
  "confidence": "72. 0%",
  "risk_level": "High",
  "explanation": "HIGH RISK (72.0%): Significant risk factors identified.. .",
  "recommendations": ["... "],
  "shap_plot": "data: image/png;base64,... ",
  "top_features": [...]
}
```

### POST /api/batch-predict

Batch prediction endpoint.  Accepts multipart/form-data with a CSV or Excel file.

### GET /health

Health check endpoint returning system status.

##  Model Information

### Algorithms Used

| Model | Description |
|-------|-------------|
| Random Forest | Ensemble of decision trees with class balancing |
| XGBoost | Gradient boosting with optimized hyperparameters |
| Logistic Regression | Interpretable linear classifier |
| Gradient Boosting | Sequential ensemble learning |
| SVM | Maximum margin classifier with RBF kernel |

### Feature Engineering

The system automatically creates clinically relevant features: 

- **Age_Category**: 0 (<50), 1 (50-65), 2 (>65)
- **AFP_Risk_Category**: 0 (<20), 1 (20-400), 2 (>400 ng/mL)
- **Liver_Function_Score**:  Composite score from liver markers

### Model Selection

The best model is automatically selected based on ROC-AUC score during cross-validation.

##  Understanding SHAP Explanations

### What is SHAP?

SHAP (SHapley Additive exPlanations) is a method to explain individual predictions by showing how each feature contributes to the outcome. 

### Reading the SHAP Chart

```
┌─────────────────────────────────────────────────────┐
│     Top Features Influencing Prediction             │
├─────────────────────────────────────────────────────┤
│  AFP          ████████████████  +0.24  (↑ Risk)     │
│  Cirrhosis    ██████████████    +0.18  (↑ Risk)     │
│  Age_Category ██████████        +0.12  (↑ Risk)     │
│  Albumin      ████████          -0.10  (↓ Risk)     │
│                    ↑                                 │
│               Base Value:  0.35                       │
└─────────────────────────────────────────────────────┘
```

| Element | Meaning |
|---------|---------|
| **Red bars** | Feature increases recurrence risk |
| **Green bars** | Feature decreases recurrence risk |
| **Bar length** | Magnitude of impact |
| **Base value** | Average prediction (starting point) |

### Calculation

```
Final Probability = Base Value + Sum of SHAP values
                  = 0.35 + 0.24 + 0.18 + 0.12 - 0.10 + ... 
                  = 0.72 (72%)
```

##  Clinical Interpretation Guide

### Risk Stratification

| Risk Level | Probability | Recommended Action |
|------------|-------------|-------------------|
| **Low** | <40% | Annual surveillance |
| **Medium** | 40-70% | Quarterly imaging, enhanced monitoring |
| **High** | >70% | Urgent review, monthly AFP, consider adjuvant therapy |

### Key Risk Factors

| Factor | High Risk Threshold | Clinical Significance |
|--------|---------------------|----------------------|
| AFP | >400 ng/mL | Aggressive tumor biology |
| Tumor Size | >5 cm | Increased vascular invasion risk |
| Cirrhosis | Present | Field effect in liver |
| Age | >65 years | Reduced hepatic reserve |

### Important Disclaimers

 **This tool is for decision support only.** Clinical judgment must guide all patient care decisions.

 **Not a diagnostic tool.** Predictions should be considered alongside all available clinical information.

 **Model limitations.** Performance depends on data quality and may vary across patient populations.

##  Contributing

Contributions are welcome! Please: 

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Contact

moseshuds@gmail.com |
For questions or support, please open an issue on GitHub.

---

**Developed for clinical decision support in HCC patient management.**