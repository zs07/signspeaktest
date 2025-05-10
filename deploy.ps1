# PowerShell script for deploying Turkish Sign Language Interpreter

Write-Host "Starting deployment process..." -ForegroundColor Yellow

# Check if git is installed
try {
    git --version | Out-Null
} catch {
    Write-Host "Git is not installed. Please install Git first." -ForegroundColor Red
    exit 1
}

# Initialize git repository if not already initialized
if (-not (Test-Path .git)) {
    Write-Host "Initializing git repository..." -ForegroundColor Green
    git init
}

# Create .gitignore file
Write-Host "Creating .gitignore file..." -ForegroundColor Green
@"
# Python
__pycache__/
*.py[cod]
*`$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Model files
*.keras
*.h5
*.pkl

# Logs
*.log

# OS
.DS_Store
Thumbs.db
"@ | Out-File -FilePath .gitignore -Encoding utf8

# Add files to git
Write-Host "Adding files to git..." -ForegroundColor Green
git add .

# Create initial commit if no commits exist
if (-not (git rev-parse HEAD 2>$null)) {
    Write-Host "Creating initial commit..." -ForegroundColor Green
    git commit -m "Initial commit: Turkish Sign Language Interpreter"
}

# Check if remote exists
$remoteExists = git remote | Select-String -Pattern "origin" -Quiet
if (-not $remoteExists) {
    Write-Host "No remote repository found." -ForegroundColor Yellow
    $repoUrl = Read-Host "Enter your GitHub repository URL (e.g., https://github.com/username/repo.git)"
    git remote add origin $repoUrl
}

# Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Green
git push -u origin main

# Create and activate virtual environment
Write-Host "Setting up Python environment..." -ForegroundColor Green
if (-not (Test-Path venv)) {
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
.\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host "Deployment completed!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Run the application: streamlit run streamlit_app.py"
Write-Host "2. Make sure your model file (tsl_simple_model_v14.keras) is in the same directory"
Write-Host "3. Open your browser at http://localhost:8501 when the app starts" 