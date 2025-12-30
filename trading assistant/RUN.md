# How to Run the Trading Assistant

## Quick Start

### Step 1: Open Terminal
Open Terminal on your Mac (Applications > Utilities > Terminal, or press `Cmd + Space` and type "Terminal")

### Step 2: Navigate to the Project Directory
```bash
cd "/Users/deepamraval/Downloads/trading assistant/python"
```

### Step 3: Install Dependencies (First Time Only)
```bash
pip3 install -r ../requirements.txt
```

### Step 4: Run the Flask Server
```bash
python3 backend.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000
 * Running on http://0.0.0.0:5000
```

### Step 5: Open in Browser
Open your web browser and go to:
- **Main Page**: http://localhost:5000
- **Chat Assistant**: http://localhost:5000/chat
- **Analysis Page**: http://localhost:5000/analysis

## Complete Command Sequence

Copy and paste these commands one by one:

```bash
# Navigate to project directory
cd "/Users/deepamraval/Downloads/trading assistant/python"

# Install dependencies (only needed first time)
pip3 install -r ../requirements.txt

# Run the server
python3 backend.py
```

## To Stop the Server
Press `Ctrl + C` in the terminal

## Troubleshooting

If you get an error about missing modules:
```bash
pip3 install flask firebase-admin numpy yfinance google-generativeai flask-cors
```

If you get a port already in use error:
- Change the port in `backend.py` (last line) from `port=5000` to `port=5001`
- Then access at http://localhost:5001

