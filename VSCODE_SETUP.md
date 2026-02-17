# VS Code Setup Guide for HCC Recurrence Prediction System

This guide will help you set up and work with this project in Visual Studio Code.

## Prerequisites

Before starting, make sure you have:
- [Visual Studio Code](https://code.visualstudio.com/) installed
- [Git](https://git-scm.com/) installed
- [Python 3.8+](https://www.python.org/) installed

## Step 1: Clone the Repository

### Option A: Using VS Code
1. Open VS Code
2. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
3. Type "Git: Clone" and select it
4. Enter the repository URL: `https://github.com/Moss254/HCC.git`
5. Choose a local folder where you want to save the project
6. Click "Open" when prompted to open the cloned repository

### Option B: Using Command Line
```bash
# Clone the repository
git clone https://github.com/Moss254/HCC.git

# Navigate to the project directory
cd HCC

# Open in VS Code
code .
```

## Step 2: Install Recommended Extensions

When you open the project, VS Code will prompt you to install recommended extensions. Click "Install All" or install them individually:

### Essential Extensions
- **Python** (ms-python.python) - Python language support
- **Pylance** (ms-python.vscode-pylance) - Fast Python language server
- **Jupyter** (ms-toolsai.jupyter) - Jupyter notebook support
- **Python Debugger** (ms-python.debugpy) - Python debugging

### Recommended Extensions
- **GitLens** (eamodio.gitlens) - Enhanced Git capabilities
- **Better Comments** (aaron-bond.better-comments) - Improve code comments
- **Error Lens** (usernamehw.errorlens) - Inline error highlighting
- **Markdown All in One** (yzhang.markdown-all-in-one) - Markdown editing
- **Thunder Client** (rangav.vscode-thunder-client) - API testing

## Step 3: Set Up Python Environment

### Create Virtual Environment
1. Open the integrated terminal in VS Code: `Ctrl+`` (backtick)
2. Create a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Select Python Interpreter
1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from your `venv` folder

## Step 4: Configure VS Code Settings

The project includes pre-configured VS Code settings in `.vscode/`:

- **settings.json** - Python linting, formatting, and testing configuration
- **launch.json** - Debug configurations for Flask app and tests
- **tasks.json** - Common development tasks (run server, tests, etc.)
- **extensions.json** - Recommended extensions list

## Step 5: Run the Application

### Using VS Code Tasks (Recommended)
1. Press `Ctrl+Shift+P` and type "Tasks: Run Task"
2. Select one of:
   - **Run Flask Server** - Start the web application
   - **Run Tests** - Execute all tests
   - **Run Specific Test** - Run a single test file
   - **Install Dependencies** - Install/update requirements

### Using Terminal
```bash
# Activate virtual environment first
python run.py
```

The application will be available at `http://localhost:8000`

## Step 6: Debugging

### Debug Flask Application
1. Open `run.py` or any file in `webapp/`
2. Press `F5` or go to Run and Debug (Ctrl+Shift+D)
3. Select **"Flask: webapp"** configuration
4. Set breakpoints by clicking left of line numbers
5. The debugger will start and stop at your breakpoints

### Debug Tests
1. Open a test file (e.g., `test_predictions.py`)
2. Press `F5`
3. Select **"Python: Current File"** or **"Python: Debug Tests"**

### Available Debug Configurations
- **Flask: webapp** - Debug the Flask web application
- **Python: Current File** - Debug the currently open Python file
- **Python: Debug Tests** - Debug all tests with pytest
- **Python: Debug Current Test** - Debug the currently selected test

## Step 7: Common Development Tasks

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python test_predictions.py

# Run with coverage
python -m pytest --cov=src --cov=webapp
```

### Linting and Formatting
The project is configured with:
- **Pylint** - Code analysis
- **Black** - Code formatter (if installed)

To format a file:
1. Right-click in the editor
2. Select "Format Document" or press `Shift+Alt+F`

### Git Operations in VS Code
- **View Changes**: Click the Source Control icon (Ctrl+Shift+G)
- **Stage Changes**: Click the + icon next to files
- **Commit**: Type a message and click ✓
- **Push/Pull**: Click the ⋯ menu in Source Control
- **View History**: Install GitLens for enhanced Git features

## VS Code Keyboard Shortcuts

### Essential Shortcuts
| Action | Windows/Linux | macOS |
|--------|---------------|-------|
| Command Palette | `Ctrl+Shift+P` | `Cmd+Shift+P` |
| Quick Open | `Ctrl+P` | `Cmd+P` |
| Terminal | `Ctrl+`` | `Ctrl+`` |
| Run/Debug | `F5` | `F5` |
| Stop Debug | `Shift+F5` | `Shift+F5` |
| Toggle Breakpoint | `F9` | `F9` |
| Find in Files | `Ctrl+Shift+F` | `Cmd+Shift+F` |
| Format Document | `Shift+Alt+F` | `Shift+Option+F` |

### Python-Specific Shortcuts
| Action | Shortcut |
|--------|----------|
| Run Python File | `Ctrl+F5` |
| Select Interpreter | `Ctrl+Shift+P` → "Python: Select Interpreter" |
| Run Tests | Test icon in sidebar |
| Debug Tests | Right-click test → Debug Test |

## Project Structure in VS Code

```
HCC/
├── .vscode/              # VS Code configuration
│   ├── settings.json     # Editor and extension settings
│   ├── launch.json       # Debug configurations
│   ├── tasks.json        # Automated tasks
│   └── extensions.json   # Recommended extensions
├── data/                 # Data files
├── models/               # Trained ML models
├── src/                  # Source code
│   ├── data/            # Data processing
│   └── models/          # Model training/prediction
├── webapp/              # Flask web application
│   ├── app.py          # Main Flask app
│   ├── templates/      # HTML templates
│   └── static/         # CSS, JS, images
├── test_*.py           # Test files
├── requirements.txt    # Python dependencies
└── run.py             # Application entry point
```

## Troubleshooting

### Python Interpreter Not Found
- Make sure you've created and activated the virtual environment
- Use `Ctrl+Shift+P` → "Python: Select Interpreter" to choose the correct one

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that the Python interpreter is set to your virtual environment

### Debugger Not Stopping at Breakpoints
- Make sure you're using the correct debug configuration
- Check that the file is saved before debugging
- Verify breakpoints are on executable lines (not comments/blank lines)

### Tests Not Discovered
- Make sure pytest is installed: `pip install pytest`
- Check the Python interpreter is correctly selected
- Refresh the test explorer

### Port Already in Use
If port 8000 is in use:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```

## Tips and Best Practices

### Code Navigation
- **Go to Definition**: `F12` or `Ctrl+Click`
- **Go to References**: `Shift+F12`
- **Go to Symbol**: `Ctrl+Shift+O`
- **Go to File**: `Ctrl+P`

### Multi-Cursor Editing
- Add cursor: `Alt+Click`
- Select all occurrences: `Ctrl+Shift+L`
- Select next occurrence: `Ctrl+D`

### Integrated Terminal Tips
- Split terminal: Click the split icon
- New terminal: `Ctrl+Shift+``
- Switch terminals: Dropdown in terminal panel

### Git Integration
- View changes: Click file in Source Control
- Stage/unstage: Click + or - icons
- Commit: Enter message and press `Ctrl+Enter`
- Push: Click ⋯ → Push

## Additional Resources

- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [VS Code Debugging](https://code.visualstudio.com/docs/editor/debugging)
- [Flask in VS Code](https://code.visualstudio.com/docs/python/tutorial-flask)
- [VS Code Tips and Tricks](https://code.visualstudio.com/docs/getstarted/tips-and-tricks)

## Getting Help

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review VS Code's [Python documentation](https://code.visualstudio.com/docs/languages/python)
3. Open an issue on the [GitHub repository](https://github.com/Moss254/HCC/issues)
4. Contact: moseshuds@gmail.com

---

**Happy Coding! 🚀**
