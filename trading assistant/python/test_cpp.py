import numpy as np
import pandas as pd
import yfinance as yf
from google import genai
import os
import time
from engine import (
    calculate_volatility,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    find_support_resistance
)

# Download stock data with retry logic and fallback symbols
# Try Indian stocks first, then US stocks as fallback
stock_symbols = ["DCXINDIA.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS", "AAPL", "GOOGL", "MSFT", "TSLA"]  # Try these in order
max_retries = 2
retry_delay = 2  # seconds
data = None
prices = None
successful_symbol = None

for stock_symbol in stock_symbols:
    print(f"\nTrying stock symbol: {stock_symbol}")
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}...")
            data = yf.download(stock_symbol, start="2025-01-01", end="2025-12-27", progress=False)
            
            if data.empty:
                if attempt < max_retries - 1:
                    print(f"  No data received. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"  ✗ No data for {stock_symbol}")
                    break  # Try next symbol
            
            # Extract Close prices
            if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
                # MultiIndex columns
                if stock_symbol in data.columns.get_level_values(0):
                    prices = data[stock_symbol]['Close'].dropna().to_numpy(dtype=np.float64)
                else:
                    close_cols = [col for col in data.columns if len(col) > 1 and col[1] == 'Close']
                    if close_cols:
                        prices = data[close_cols[0]].dropna().to_numpy(dtype=np.float64)
                    else:
                        prices = data.iloc[:, 0].dropna().to_numpy(dtype=np.float64)
            else:
                prices = data['Close'].dropna().to_numpy(dtype=np.float64)
            
            if len(prices) == 0:
                print(f"  ✗ No price data available")
                break  # Try next symbol
            
            successful_symbol = stock_symbol
            print(f"  ✓ Successfully downloaded {len(prices)} data points for {stock_symbol}")
            break  # Success!
            
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                print(f"  Error: {error_msg[:80]}...")
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"  ✗ Failed after {max_retries} attempts: {error_msg[:80]}...")
                break  # Try next symbol
    
    if prices is not None and len(prices) > 0:
        break  # Found working symbol

if prices is None or len(prices) == 0:
    print("\n" + "="*60)
    print("✗ Could not download stock data from any symbol")
    print("="*60)
    print("\n⚠️  Network issue detected. Using DEMO MODE with sample data.")
    print("   This will allow you to test the C++ engine calculations.")
    print("="*60)
    
    # Use sample/demo data for testing
    print("\nGenerating sample stock price data...")
    np.random.seed(42)  # For reproducible results
    base_price = 150.0
    days = 100
    # Generate realistic-looking stock prices with trend
    trend = np.linspace(0, 10, days)  # Upward trend
    noise = np.random.normal(0, 2, days)  # Random fluctuations
    prices = base_price + trend + np.cumsum(noise)
    prices = np.maximum(prices, 50.0)  # Ensure prices don't go negative
    successful_symbol = "DEMO_DATA"
    
    print(f"✓ Generated {len(prices)} sample data points")
    print(f"  Price range: ₹{prices.min():.2f} - ₹{prices.max():.2f}")
    print(f"  Current price: ₹{prices[-1]:.2f}")
    print("\nNote: Results are based on sample data, not real market data.")
else:
    print(f"\nUsing stock data from: {successful_symbol}")

# Calculate indicators with error handling
try:
    vol = calculate_volatility(prices)
    sma = calculate_sma(prices)
    ema = calculate_ema(prices, alpha=0.1)
    rsi = calculate_rsi(prices)
    
    # Check for invalid values
    if np.isnan(vol) or np.isnan(sma) or len(prices) < 2:
        print("Warning: Insufficient data for accurate calculations")
        if np.isnan(vol):
            vol = 0.0
        if np.isnan(sma):
            sma = prices.mean() if len(prices) > 0 else 0.0
except Exception as e:
    print(f"Error calculating indicators: {e}")
    exit(1)

supports, resistances = find_support_resistance(prices)
num_supports = len(supports)
num_resistances = len(resistances)

print("Volatility:", vol)
print("SMA:", sma)
print("EMA:", ema)
print("RSI:", rsi)
print("Supports:", supports)
print("Resistances:", resistances)

def trendsignals(ema,sma):
    if ema > 1.01*sma:
        return "Bullish"
    elif ema<0.99*sma:
        return "Bearish"
    else:
        return "neutral"
def momentum(rsi):
    if(rsi > 70):
        return "Overbought"
    elif(rsi < 30):
        return "oversold"
    else:
        return "neutral"
def risksignal(volatility, sma):
    risk_ratio = volatility / sma
    if risk_ratio > 0.08:
        return "High Risk"
    elif risk_ratio > 0.04:
        return "Medium Risk"
    else:
        return "Low Risk" 
def finalsignal(trend, rsi, risk):
    if trend == "Bullish" and rsi < 60 and risk != "High Risk":
        return "BUY"
    elif trend == "Bearish" and rsi > 40:
        return "SELL"
    else:
        return "HOLD"


trend = trendsignals(ema, sma)
momentum = momentum(rsi)
risk = risksignal(vol, sma)
decision = finalsignal(trend, rsi, risk)

print("\n--- Trading Assistant Summary ---")
print("Trend:", trend)
print("Momentum:", momentum)
print("Risk Level:", risk)
print("Final Signal:", decision)


# Only call Gemini if we have real data (not demo)
if successful_symbol != "DEMO_DATA":
    # Initialize Gemini client with API key
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAVyNZR_h6RL8rVFJNY7FbqB0xIq6NxNzQ")
    if not GEMINI_API_KEY:
        print("Warning: GOOGLE_API_KEY environment variable not set. Using default key.")
        GEMINI_API_KEY = "AIzaSyAVyNZR_h6RL8rVFJNY7FbqB0xIq6NxNzQ"
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
I have analyzed the stock {successful_symbol} and calculated the following indicators:
- EMA: {ema:.2f}
- SMA: {sma:.2f}
- RSI: {rsi:.2f}
- Volatility: {vol:.4f}
- Trend: {trend}
- Momentum: {momentum}
- Risk Level: {risk}

Based on these indicators, I plan to {decision} the stock. 

Please provide a probability (as a percentage) that this trading decision will succeed in the next 5 trading days, along with a brief explanation.
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",  
            contents=prompt,
        )
        
        print("\n" + "="*60)
        print("AI Analysis:")
        print("="*60)
        print(response.text)
    except Exception as e:
        print("\n" + "="*60)
        print("AI Analysis Error:")
        print("="*60)
        print(f"Could not get AI analysis: {e}")
        print("Technical indicators calculated successfully above.")
else:
    print("\n" + "="*60)
    print("AI Analysis skipped (using demo data)")
    print("="*60)
    print("To get AI analysis, please ensure network connectivity and try again.")

# response = client.models.generate_content(
#     model="gemini-1.5-flash",
#     contents="Explain how AI works in a few words",
# )

# print(response.text)