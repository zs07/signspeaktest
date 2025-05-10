# Simple setup script for Turkish Sign Language Interpreter

# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Check for Python
if (-not (Test-Command python)) {
    Write-Host "Python is not installed. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Check for Git
if (-not (Test-Command git)) {
    Write-Host "Git is not installed. Please install Git." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
.\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Green
pip install -r requirements.txt

# Initialize git
Write-Host "Initializing git repository..." -ForegroundColor Green
git init

# Create .gitignore
Write-Host "Creating .gitignore..." -ForegroundColor Green
@"
# Python
__pycache__/
*.py[cod]
*.so
.Python
env/
venv/
ENV/
*.egg-info/
dist/
build/

# IDE
.idea/
.vscode/

# Model files
*.keras
*.h5
*.pkl

# Logs
*.log
"@ | Out-File -FilePath .gitignore -Encoding utf8

# Add files
Write-Host "Adding files to git..." -ForegroundColor Green
git add .

# Initial commit
Write-Host "Creating initial commit..." -ForegroundColor Green
git commit -m "Initial commit"

# Ask for GitHub repository URL
$repoUrl = Read-Host "Enter your GitHub repository URL (e.g., https://github.com/username/repo.git)"

# Add remote and push
Write-Host "Setting up remote and pushing to GitHub..." -ForegroundColor Green
git remote add origin $repoUrl
git branch -M main
git push -u origin main

Write-Host "`nSetup completed!" -ForegroundColor Green
Write-Host "`nTo run the application:" -ForegroundColor Yellow
Write-Host "1. Make sure you're in the virtual environment (venv)"
Write-Host "2. Run: streamlit run streamlit_app.py"
Write-Host "3. Open your browser at http://localhost:8501" 