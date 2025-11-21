# GitHub Upload Guide for ContextFlow Chatbot

## Prerequisites
- Git installed on your computer ([Download Git](https://git-scm.com/downloads))
- GitHub account ([Sign up](https://github.com/join))

---

## Step-by-Step Instructions

### 1. Initialize Git Repository

Open PowerShell/Command Prompt in your project directory and run:

```powershell
cd "c:\Users\Abhik Kumar Mohanta\Desktop\okkkk"
git init
```

### 2. Configure Git (First Time Only)

If you haven't configured Git before, set your username and email:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 3. Add Files to Git

```powershell
git add .
```

### 4. Create Initial Commit

```powershell
git commit -m "Initial commit: ContextFlow Hybrid Chatbot"
```

### 5. Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click the **"+"** icon in the top right
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: `contextflow-chatbot` (or your preferred name)
   - **Description**: "Hybrid LSTM-Transformer chatbot with advanced inference strategies"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README (you already have one)
5. Click **"Create repository"**

### 6. Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

**Replace** `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and repository name.

### 7. Enter Credentials

When prompted, enter your GitHub credentials:
- **Username**: Your GitHub username
- **Password**: Use a **Personal Access Token** (not your password)

#### How to Create a Personal Access Token:
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name (e.g., "ContextFlow Upload")
4. Select scopes: Check **"repo"** (full control of private repositories)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)
7. Use this token as your password when pushing

---

## Alternative: Using GitHub Desktop (Easier)

If you prefer a GUI:

1. Download [GitHub Desktop](https://desktop.github.com/)
2. Install and sign in to your GitHub account
3. Click **"Add"** → **"Add existing repository"**
4. Browse to `c:\Users\Abhik Kumar Mohanta\Desktop\okkkk`
5. Click **"Publish repository"**
6. Choose repository name and visibility
7. Click **"Publish repository"**

---

## Updating Your Repository Later

After making changes to your code:

```powershell
git add .
git commit -m "Description of changes"
git push
```

---

## Important Notes

### Files Excluded from Git
The `.gitignore` file excludes:
- Virtual environment (`venv/`)
- Cache files (`__pycache__/`)
- Large model checkpoints (`.pt`, `.pth` files)
- Log files
- Data files (to avoid uploading large datasets)

### Handling Large Model Files
If you want to share trained models, consider:
1. **Git LFS** (Large File Storage) - for files up to 2GB
2. **Google Drive/Dropbox** - share link in README
3. **Hugging Face Hub** - for ML models

---

## Quick Reference Commands

| Command | Description |
|---------|-------------|
| `git status` | Check current status |
| `git add .` | Stage all changes |
| `git commit -m "message"` | Commit changes |
| `git push` | Upload to GitHub |
| `git pull` | Download latest changes |
| `git log` | View commit history |

---

## Troubleshooting

### Error: "fatal: not a git repository"
- Make sure you're in the correct directory
- Run `git init` first

### Error: "failed to push some refs"
- Run `git pull origin main --rebase` first
- Then `git push`

### Authentication Failed
- Use Personal Access Token instead of password
- Or use GitHub Desktop for easier authentication

---

## Next Steps After Upload

1. Add a **LICENSE** file (MIT, Apache, etc.)
2. Add **badges** to README (build status, license, etc.)
3. Create **GitHub Pages** for documentation
4. Set up **GitHub Actions** for CI/CD
5. Add **CONTRIBUTING.md** for collaboration guidelines

---

## Need Help?

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [GitHub Desktop Documentation](https://docs.github.com/en/desktop)
