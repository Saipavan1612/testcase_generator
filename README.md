```bash
git clone <repo_url>
cd <repo_folder>

#create virtual environment
python -m venv venv

# Activate venv

# Windows (CMD)
venv\Scripts\activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python run_generator.py
```