# Trading Assistant - Integrated Application

A comprehensive stock analysis application with AI-powered chat assistant, integrated with Firebase, C++ engine, and Gemini AI.

## Features

- **Stock Analysis**: Technical indicators (EMA, SMA, RSI, Volatility, Support) using C++ engine
- **AI Chat Assistant**: Powered by Google Gemini AI for trading questions
- **Firebase Integration**: Stores trading data and analysis history
- **Modern UI**: Beautiful, responsive web interface

## Project Structure

```
trading assistant/
├── python/
│   ├── backend.py          # Main Flask backend with all integrations
│   ├── geminiModel.py       # Gemini AI model (reference)
│   ├── serviceKey.json      # Firebase service account key
│   └── templates/
│       ├── index.html       # Main input form
│       ├── result.html      # Analysis results page
│       └── chat.html        # AI chat interface
├── cpp/
│   ├── engine.cpp           # C++ technical analysis engine
│   └── engine.so            # Compiled C++ library
└── requirements.txt        # Python dependencies
```

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Compile C++ Engine (if needed)

If `engine.so` doesn't exist, compile it:

```bash
cd cpp
g++ -shared -fPIC -o engine.so engine.cpp
```

### 3. Configure Firebase

Ensure `python/serviceKey.json` contains your Firebase service account credentials.

### 4. Configure Gemini API Key

Update the Gemini API key in `python/backend.py` (line 21) if needed:
```python
gemini_client = genai.Client(api_key="YOUR_API_KEY")
```

### 5. Run the Application

```bash
cd python
python backend.py
```

The application will be available at `http://localhost:5000`

## API Endpoints

### Frontend Routes
- `GET /` - Main input form
- `GET /analysis` - Analysis results page
- `GET /chat` - AI chat interface
- `POST /add_expense` - Submit stock analysis request

### API Endpoints
- `GET/POST /api/analyze?stock=SYMBOL&date_from=YYYY-MM-DD&date_to=YYYY-MM-DD` - Get stock analysis
- `POST /api/chat` - Chat with AI assistant (JSON: `{"message": "your question"}`)

## How It Works

1. **Stock Analysis Flow**:
   - User enters stock symbol and date range on homepage
   - Data is saved to Firebase
   - Backend fetches stock data using yfinance
   - C++ engine calculates technical indicators
   - Results displayed on analysis page

2. **Chat Assistant Flow**:
   - User asks questions in chat interface
   - Backend sends query to Gemini AI
   - Response displayed in chat interface

## Technical Stack

- **Backend**: Flask (Python)
- **Database**: Firebase Firestore
- **AI**: Google Gemini AI
- **Stock Data**: yfinance
- **Technical Analysis**: C++ engine (via ctypes)
- **Frontend**: HTML, CSS, JavaScript

## Notes

- For Indian stocks, use symbols like "TCS", "RELIANCE" (backend adds .NS suffix automatically)
- For US stocks, use symbols like "AAPL", "GOOGL"
- Ensure date ranges are valid and contain trading data
- The C++ engine must be compiled before running the application

## Troubleshooting

1. **C++ engine not found**: Compile `engine.cpp` to create `engine.so`
2. **Firebase errors**: Check `serviceKey.json` credentials
3. **Gemini API errors**: Verify API key is valid
4. **Stock data errors**: Check stock symbol and date range validity

