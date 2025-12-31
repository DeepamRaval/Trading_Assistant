from datetime import datetime
import os
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Try to import optional dependencies
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("Warning: flask-cors not available, CORS disabled")

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except ImportError as e:
    print(f"CRITICAL: Required packages not installed: {e}")
    raise

# Try to import Firebase (optional)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("Warning: firebase-admin not available, Firebase features disabled")
    firebase_admin = None
    credentials = None
    firestore = None

# Try to import Gemini AI (optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not available, AI features disabled")
    genai = None

# Import Python engine (required)
try:
    from engine import (
        calculate_volatility,
        calculate_sma,
        calculate_ema,
        calculate_rsi,
        find_support_resistance
    )
except ImportError as e:
    print(f"CRITICAL: Could not import engine module: {e}")
    raise

# Initialize Firebase (optional - don't crash if it fails)
db = None
if FIREBASE_AVAILABLE:
    try:
        # Try relative path first (for local), then try absolute path
        service_key_path = "serviceKey.json"
        if not os.path.exists(service_key_path):
            service_key_path = "python/serviceKey.json"
        
        if os.path.exists(service_key_path):
            cred = credentials.Certificate(service_key_path)
            try:
                firebase_admin.initialize_app(cred)
            except ValueError:
                # App already initialized
                pass
            db = firestore.client()
            print("Firebase initialized successfully")
        else:
            print("Warning: serviceKey.json not found, Firebase disabled")
    except Exception as e:
        print(f"Warning: Could not initialize Firebase: {e}")
        db = None
else:
    print("Firebase not available (package not installed)")

# Initialize Gemini AI (optional - don't crash if it fails)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini AI configured successfully")
    except Exception as e:
        print(f"Warning: Could not configure Gemini AI: {e}")
elif not GEMINI_AVAILABLE:
    print("Gemini AI not available (package not installed)")
elif not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set, AI features will be disabled")

# Initialize Flask with explicit template folder
app = Flask(__name__, template_folder='templates')
if CORS_AVAILABLE:
    CORS(app)  # Enable CORS for API endpoints

# Add error handler for 500 errors
@app.errorhandler(500)
def internal_error(error):
    print(f"500 Error: {error}")
    import traceback
    traceback.print_exc()
    try:
        return render_template('index.html', error="An internal server error occurred. Please try again."), 500
    except:
        return "Internal Server Error", 500


def analyze_stock(stock_symbol, date_from, date_to):
    """Analyze stock using Python engine and yfinance"""
    try:
        print(f"=== Starting analysis for {stock_symbol} from {date_from} to {date_to} ===")
        
        # Validate dates are not in the future
        from datetime import date as date_class, timedelta
        try:
            date_from_obj = datetime.strptime(date_from, "%Y-%m-%d").date()
            date_to_obj = datetime.strptime(date_to, "%Y-%m-%d").date()
            today = date_class.today()
            # Allow up to 1 day in future (for timezone differences)
            max_future_date = today + timedelta(days=1)
            
            if date_from_obj > max_future_date:
                error_msg = f"Start date {date_from} is in the future. Please use a past date (today is {today.strftime('%Y-%m-%d')})"
                print(f"ERROR: {error_msg}")
                return None
            
            if date_to_obj > max_future_date:
                print(f"WARNING: End date {date_to} is in the future, using today's date ({today.strftime('%Y-%m-%d')})")
                date_to = today.strftime("%Y-%m-%d")
                date_to_obj = today
            
            # Also check that from date is before to date
            if date_from_obj > date_to_obj:
                error_msg = f"Start date {date_from} is after end date {date_to}"
                print(f"ERROR: {error_msg}")
                return None
            
            print(f"Date validation passed: {date_from} to {date_to}")
        except ValueError as ve:
            error_msg = f"Invalid date format: {ve}. Please use YYYY-MM-DD format"
            print(f"ERROR: {error_msg}")
            return None
        
        # Download stock data with error handling and timeout
        print(f"Downloading data for {stock_symbol}...")
        try:
            data = yf.download(stock_symbol, start=date_from, end=date_to, progress=False, timeout=30)
            print(f"Download complete. Data shape: {data.shape}")
        except Exception as e:
            print(f"ERROR downloading data for {stock_symbol}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        if data.empty:
            print(f"ERROR: No data available for {stock_symbol} in date range {date_from} to {date_to}")
            print("This could mean:")
            print("  - Stock symbol is incorrect")
            print("  - Date range has no trading days")
            print("  - Market data is unavailable")
            return None
        
        print(f"Data downloaded successfully: {len(data)} rows")
        
        # Handle different column structures from yfinance
        # yfinance returns MultiIndex columns when downloading single symbol sometimes
        if isinstance(data.columns, pd.MultiIndex):
            # MultiIndex columns - yfinance structure is typically: (Price, Ticker)
            # e.g., ('Close', 'TCS.NS'), ('High', 'TCS.NS'), etc.
            try:
                # Method 1: Try direct tuple access if we know the structure
                try:
                    prices = data[('Close', stock_symbol)].dropna().to_numpy(dtype=np.float64)
                except (KeyError, IndexError):
                    # Method 2: Use xs to extract Close from level 0 (Price level)
                    try:
                        close_data = data.xs('Close', level=0, axis=1)
                        # Get first column (should be the stock symbol)
                        prices = close_data.iloc[:, 0].dropna().to_numpy(dtype=np.float64)
                    except (KeyError, IndexError):
                        # Method 3: Search for Close column manually
                        close_col = None
                        for col in data.columns:
                            if isinstance(col, tuple):
                                # Check if 'Close' is in the tuple
                                if 'Close' in col or any('Close' == str(level) for level in col):
                                    close_col = col
                                    break
                            elif str(col) == 'Close' or 'Close' in str(col):
                                close_col = col
                                break
                        
                        if close_col:
                            prices = data[close_col].dropna().to_numpy(dtype=np.float64)
                        else:
                            # Last resort: use first column (Close is usually first)
                            prices = data.iloc[:, 0].dropna().to_numpy(dtype=np.float64)
            except Exception as e:
                print(f"Error extracting prices from MultiIndex: {e}")
                # Last resort: try first column
                try:
                    prices = data.iloc[:, 0].dropna().to_numpy(dtype=np.float64)
                except:
                    return None
        elif 'Close' in data.columns:
            # Simple column structure
            prices = data['Close'].dropna().to_numpy(dtype=np.float64)
        else:
            # Try to find any column with 'Close' in the name
            close_cols = [col for col in data.columns if 'Close' in str(col)]
            if close_cols:
                prices = data[close_cols[0]].dropna().to_numpy(dtype=np.float64)
            else:
                print(f"Could not find Close prices for {stock_symbol}")
                print(f"Available columns: {data.columns.tolist()}")
                return None
        
        if len(prices) == 0:
            print(f"ERROR: No price data extracted for {stock_symbol}")
            print(f"Data columns: {data.columns.tolist()}")
            return None
        
        print(f"Price data extracted: {len(prices)} data points")
        print(f"Price range: {prices.min():.2f} to {prices.max():.2f}")
        
        # Calculate technical indicators using Python engine
        vol = calculate_volatility(prices)
        sma = calculate_sma(prices)
        ema = calculate_ema(prices, alpha=0.1)
        rsi = calculate_rsi(prices)
        
        supports, resistances = find_support_resistance(prices)
        
        support_level = supports.min() if len(supports) > 0 else prices.min()
        resistance_level = resistances.max() if len(resistances) > 0 else prices.max()
        
        # Calculate trading signals
        trend = "bullish" if ema > 1.01 * sma else ("bearish" if ema < 0.99 * sma else "neutral")
        momentum_status = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "neutral")
        risk_ratio = vol / sma if sma > 0 else 0
        risk = "High Risk" if risk_ratio > 0.08 else ("Medium Risk" if risk_ratio > 0.04 else "Low Risk")
        
        final_signal = "BUY" if trend == "bullish" and rsi < 60 and risk != "High Risk" else \
                      ("SELL" if trend == "bearish" and rsi > 40 else "HOLD")

        # Generate AI explanation using Gemini (non-blocking - don't fail if it times out)
        ai_explanation = None
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            try:
                explanation_prompt = f"""Analyze this stock and provide a brief, professional trading explanation:

Stock Analysis Summary:
- Current Price: ₹{round(prices[-1], 2)}
- EMA: ₹{round(ema, 2)}
- SMA: ₹{round(sma, 2)}
- RSI: {round(rsi, 2)}
- Volatility: {round((vol / sma * 100) if sma > 0 else 0, 2)}%
- Support Level: ₹{round(support_level, 2)}
- Resistance Level: ₹{round(resistance_level, 2)}
- Trend: {trend}
- Momentum: {momentum_status}
- Risk Level: {risk}
- Trading Signal: {final_signal}

Provide a concise 2-3 sentence explanation of why the signal is {final_signal}, considering the technical indicators. Be professional and educational."""
                
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(explanation_prompt)
                ai_explanation = response.text if hasattr(response, 'text') else str(response)
            except Exception as e:
                # Don't fail the whole request if AI explanation fails
                print(f"Error generating AI explanation (non-critical): {e}")
                ai_explanation = None

        # Prepare chart data with OHLC (Open, High, Low, Close) for candlesticks
        window = min(90, len(prices))
        
        # Extract OHLC data
        if isinstance(data.columns, pd.MultiIndex):
            try:
                open_data = data[('Open', stock_symbol)].dropna().to_numpy(dtype=np.float64)
                high_data = data[('High', stock_symbol)].dropna().to_numpy(dtype=np.float64)
                low_data = data[('Low', stock_symbol)].dropna().to_numpy(dtype=np.float64)
                close_data = data[('Close', stock_symbol)].dropna().to_numpy(dtype=np.float64)
            except:
                # Fallback: use xs method
                open_data = data.xs('Open', level=0, axis=1).iloc[:, 0].dropna().to_numpy(dtype=np.float64)
                high_data = data.xs('High', level=0, axis=1).iloc[:, 0].dropna().to_numpy(dtype=np.float64)
                low_data = data.xs('Low', level=0, axis=1).iloc[:, 0].dropna().to_numpy(dtype=np.float64)
                close_data = data.xs('Close', level=0, axis=1).iloc[:, 0].dropna().to_numpy(dtype=np.float64)
        else:
            open_data = data['Open'].dropna().to_numpy(dtype=np.float64)
            high_data = data['High'].dropna().to_numpy(dtype=np.float64)
            low_data = data['Low'].dropna().to_numpy(dtype=np.float64)
            close_data = data['Close'].dropna().to_numpy(dtype=np.float64)
        
        # Ensure all arrays have same length
        min_len = min(len(open_data), len(high_data), len(low_data), len(close_data))
        open_data = open_data[-window:][-min_len:]
        high_data = high_data[-window:][-min_len:]
        low_data = low_data[-window:][-min_len:]
        close_data = close_data[-window:][-min_len:]
        
        # Prepare OHLC data for candlesticks
        chart_dates = [d.strftime("%Y-%m-%d") for d in data.index[-window:][-min_len:]]
        chart_ohlc = []
        for i in range(len(open_data)):
            chart_ohlc.append({
                "x": chart_dates[i],
                "o": float(open_data[i]),
                "h": float(high_data[i]),
                "l": float(low_data[i]),
                "c": float(close_data[i])
            })

        return {
            "volatility": round(vol, 2),
            "volatility_percent": round((vol / sma * 100) if sma > 0 else 0, 2),
            "sma": round(sma, 2),
            "ema": round(ema, 2),
            "rsi": round(rsi, 2),
            "support": round(support_level, 2),
            "resistance": round(resistance_level, 2),
            "trend": trend,
            "momentum": momentum_status,
            "risk": risk,
            "signal": final_signal,
            "current_price": round(prices[-1], 2),
            "ai_explanation": ai_explanation,
            "chart": {
                "dates": chart_dates,
                "ohlc": chart_ohlc
            }
        }
    except Exception as e:
        print(f"CRITICAL ERROR analyzing stock {stock_symbol}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "firebase": FIREBASE_AVAILABLE and db is not None,
        "gemini": GEMINI_AVAILABLE and GEMINI_API_KEY is not None
    }), 200


@app.route('/chat')
def chat():
    return render_template('chat.html')


@app.route('/analysis')
def analysis():
    return render_template('result.html')


@app.route('/add_expense', methods=['POST'])
def add_expense():
    """Handle form submission and redirect to analysis page - SIMPLIFIED to prevent timeout"""
    # Get form data - minimal processing
    stock_name = request.form.get('stockName', '').strip()
    date_from_str = request.form.get("dateFrom", '').strip()
    date_to_str = request.form.get("dateTo", '').strip()
    
    # Quick validation
    if not stock_name or not date_from_str or not date_to_str:
        return render_template('index.html', error="Please fill in all fields"), 200
    
    # Build redirect URL immediately - no other processing
    from urllib.parse import quote
    redirect_url = f"/analysis?stock={quote(stock_name)}&date_from={quote(date_from_str)}&date_to={quote(date_to_str)}"
    
    # Return redirect IMMEDIATELY - no Firebase, no date parsing, nothing else
    # Use Flask's redirect but with minimal overhead
    response = redirect(redirect_url, code=302)
    
    # Skip Firebase entirely to prevent any blocking
    # Firebase can be added back later if needed, but it's causing timeouts
    
    return response


@app.route('/api/analyze', methods=['GET', 'POST'])
def api_analyze():
    """API endpoint for stock analysis"""
    try:
        if request.method == 'POST':
            data = request.get_json() or {}
            stock_symbol = data.get('stock') or request.form.get('stock')
            date_from = data.get('date_from') or request.form.get('dateFrom')
            date_to = data.get('date_to') or request.form.get('dateTo')
        else:
            stock_symbol = request.args.get('stock')
            date_from = request.args.get('date_from')
            date_to = request.args.get('date_to')
        
        if not stock_symbol or not date_from or not date_to:
            return jsonify({"error": "Missing required parameters: stock, date_from, and date_to are required"}), 400
        
        print(f"=== API Analyze Request ===")
        print(f"Stock: {stock_symbol}, From: {date_from}, To: {date_to}")
        
        # Handle Indian stocks (add .NS suffix if not present)
        original_symbol = stock_symbol
        if not '.' in stock_symbol:
            stock_symbol = f"{stock_symbol}.NS"
            print(f"Converted {original_symbol} to {stock_symbol} (Indian stock)")
        
        result = analyze_stock(stock_symbol, date_from, date_to)
        
            if result is None:
                # Try without .NS suffix if it failed (might be US stock)
                if stock_symbol.endswith('.NS') and original_symbol != stock_symbol:
                    print(f"Retrying with original symbol: {original_symbol}")
                    result = analyze_stock(original_symbol, date_from, date_to)
                
                if result is None:
                    # Check if dates are in the future
                    from datetime import date as date_class
                    try:
                        date_from_obj = datetime.strptime(date_from, "%Y-%m-%d").date()
                        date_to_obj = datetime.strptime(date_to, "%Y-%m-%d").date()
                        today = date_class.today()
                        
                        if date_from_obj > today or date_to_obj > today:
                            error_msg = f"Dates are in the future! Today is {today.strftime('%Y-%m-%d')}. "
                            error_msg += f"You requested {date_from} to {date_to}. "
                            error_msg += "Please use past dates (e.g., 2024-11-01 to 2024-12-01)"
                            return jsonify({"error": error_msg}), 400
                    except:
                        pass
                    
                    error_msg = f"Failed to analyze stock '{original_symbol}'. "
                    error_msg += "Possible reasons: "
                    error_msg += "1) Stock symbol is incorrect (try 'RELIANCE.NS' or 'TCS.NS' for Indian stocks), "
                    error_msg += "2) Date range has no trading data, "
                    error_msg += "3) Market data unavailable. "
                    error_msg += f"Tried symbols: {original_symbol}, {stock_symbol}"
                    return jsonify({"error": error_msg}), 400
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in api_analyze: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chat with Gemini AI"""
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return jsonify({"error": "AI chat is not available. Gemini AI is not configured."}), 503
    
    data = request.get_json()
    message = data.get('message', '')
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(message)
        response_text = response.text
        if not response_text:
            return jsonify({"error": "Empty response from AI model"}), 500
        
        return jsonify({"response": response_text})
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Gemini API Error Details:\n{error_details}")
        error_msg = str(e)
        # Provide more helpful error messages
        if "API key" in error_msg.lower() or "authentication" in error_msg.lower():
            error_msg = "API key authentication failed. Please check your Gemini API key."
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            error_msg = "API quota exceeded. Please try again later."
        return jsonify({"error": f"Error generating response: {error_msg}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
