#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting deployment process...${NC}"

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    echo -e "${GREEN}Initializing git repository...${NC}"
    git init
fi

# Create .gitignore file
echo -e "${GREEN}Creating .gitignore file...${NC}"
cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
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
EOL

# Add files to git
echo -e "${GREEN}Adding files to git...${NC}"
git add .

# Initial commit if no commits exist
if ! git rev-parse HEAD >/dev/null 2>&1; then
    echo -e "${GREEN}Creating initial commit...${NC}"
    git commit -m "Initial commit: Turkish Sign Language Interpreter"
fi

# Check if remote exists
if ! git remote | grep -q "origin"; then
    echo -e "${YELLOW}No remote repository found.${NC}"
    echo -e "${YELLOW}Please create a new repository on GitHub and run:${NC}"
    echo -e "${GREEN}git remote add origin <your-repo-url>${NC}"
    echo -e "${GREEN}git push -u origin main${NC}"
else
    echo -e "${GREEN}Pushing to GitHub...${NC}"
    git push origin main
fi

echo -e "${GREEN}Deployment script completed!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Create a new repository on GitHub if you haven't already"
echo "2. Add the remote repository using: git remote add origin <your-repo-url>"
echo "3. Push your code using: git push -u origin main"
echo "4. Install dependencies: pip install -r requirements.txt"
echo "5. Run the application: streamlit run streamlit_app.py" 