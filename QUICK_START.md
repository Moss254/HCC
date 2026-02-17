# Quick Start Guide

This guide will help you get the HCC Recurrence Prediction System up and running quickly.

## For VS Code Users (Recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/Moss254/HCC.git
cd HCC
code .
```

### 2. Install Extensions
When VS Code opens, it will prompt you to install recommended extensions. Click **"Install All"**.

### 3. Set Up Python Environment
- Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
- Type "Python: Create Environment" and select it
- Choose "Venv"
- Select your Python 3.8+ interpreter
- Check "Install requirements.txt"
- Wait for installation to complete

### 4. Run the Application
**Option A - Using Debug (Recommended for Development)**
1. Press `F5` or click the "Run and Debug" icon
2. Select "Flask: webapp" configuration
3. The app will start at http://localhost:8000

**Option B - Using Tasks**
1. Press `Ctrl+Shift+P` and type "Tasks: Run Task"
2. Select "Run Flask Server"
3. The app will start at http://localhost:8000

### 5. Open in Browser
Navigate to http://localhost:8000 in your web browser.

## For Command Line Users

### 1. Clone and Navigate
```bash
git clone https://github.com/Moss254/HCC.git
cd HCC
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python run.py
```

### 5. Open in Browser
Navigate to http://localhost:8000 in your web browser.

## First Time Use

### Training the Model
The first time you run the application, it will automatically:
1. Check for existing trained models
2. If not found, train a new model using the data in `data/raw/`
3. Save the trained model for future use

**Note**: Model training can take a few minutes. The console will show progress.

### Making Your First Prediction

1. Click **"Start Prediction"** on the homepage
2. Fill in the patient data form:
   - **Demographics**: Age, Gender
   - **Clinical Status**: Symptoms, Performance Status (PS)
   - **Laboratory Values**: AFP, Albumin, Bilirubin, ALT, AST
   - **Tumor Characteristics**: Major Dimension, Number of Nodules
   - **Risk Factors**: Cirrhosis, Hepatitis status, Diabetes, etc.

3. Click **"Predict Recurrence Risk"**

4. Review the results:
   - **Risk Probability**: Percentage likelihood of recurrence
   - **Risk Level**: Low, Medium, or High
   - **SHAP Plot**: Visual explanation of which features contributed most
   - **Clinical Recommendations**: Evidence-based follow-up guidelines

## Common Issues

### Port Already in Use
If you see "Port 8000 is already in use":
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```
Or change the port in `run.py` or use `config.yaml`.

### ModuleNotFoundError
Make sure your virtual environment is activated and dependencies are installed:
```bash
# Activate venv first, then:
pip install -r requirements.txt
```

### No Training Data Found
Place your dataset in `data/raw/hcc-data-complete-balanced.xlsx` or update the path in `config.yaml`.

## Next Steps

- **Learn about features**: See [README.md](README.md) for detailed feature descriptions
- **VS Code setup**: Read [VSCODE_SETUP.md](VSCODE_SETUP.md) for development tips
- **API usage**: Check [README.md](README.md#api-documentation) for API endpoints
- **Batch predictions**: Upload a CSV/Excel file with multiple patients

## Getting Help

- 📖 Full documentation: [README.md](README.md)
- 🔧 VS Code setup: [VSCODE_SETUP.md](VSCODE_SETUP.md)
- 🐛 Report issues: [GitHub Issues](https://github.com/Moss254/HCC/issues)
- 📧 Email: moseshuds@gmail.com

---

**Ready to predict? Let's go! 🚀**
