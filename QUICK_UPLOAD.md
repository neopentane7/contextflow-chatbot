# Quick GitHub Upload Commands

## Run these commands in PowerShell (in order):

### 1. Navigate to project directory (if not already there)
```powershell
cd "c:\Users\Abhik Kumar Mohanta\Desktop\okkkk"
```

### 2. Configure Git (ONLY if first time using Git)
```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 3. Add all files to Git
```powershell
git add .
```

### 4. Create first commit
```powershell
git commit -m "Initial commit: ContextFlow Hybrid Chatbot with LSTM-Transformer architecture"
```

### 5. Create GitHub repository
- Go to: https://github.com/new
- Repository name: `contextflow-chatbot` (or your choice)
- Description: "Hybrid LSTM-Transformer chatbot with advanced inference strategies"
- Choose Public or Private
- DO NOT check "Initialize with README"
- Click "Create repository"

### 6. Connect to GitHub (replace YOUR_USERNAME and REPO_NAME)
```powershell
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

### 7. When prompted for password
- Use a Personal Access Token (NOT your GitHub password)
- Get token from: https://github.com/settings/tokens
- Click "Generate new token (classic)"
- Select "repo" scope
- Copy the token and paste it as password

---

## OR Use GitHub Desktop (Easier!)

1. Download: https://desktop.github.com/
2. Install and sign in
3. File → Add Local Repository
4. Browse to: `c:\Users\Abhik Kumar Mohanta\Desktop\okkkk`
5. Click "Publish repository"
6. Done! ✅

---

## Files Ready for Upload:
✅ .gitignore (excludes unnecessary files)
✅ LICENSE (MIT License)
✅ README.md (project documentation)
✅ ContextFlow_Project.ipynb (Jupyter notebook)
✅ All source code files

## Files Excluded (by .gitignore):
❌ venv/ (virtual environment)
❌ __pycache__/ (Python cache)
❌ *.log (log files)
❌ Large model checkpoints
❌ Data files

Your project is clean and ready to upload! 🚀
