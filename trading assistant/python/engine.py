"""
Python implementation of trading engine functions.
Replaces the C++ engine for compatibility with cloud hosting platforms like Render.
"""
import numpy as np
import math


def calculate_volatility(prices):
    """
    Calculate the volatility (standard deviation) of prices.
    
    Args:
        prices: Array-like of price values
        
    Returns:
        float: Volatility (standard deviation)
    """
    if len(prices) == 0:
        return 0.0
    
    prices_array = np.asarray(prices, dtype=np.float64)
    mean = np.mean(prices_array)
    
    variance = np.mean((prices_array - mean) ** 2)
    return math.sqrt(variance)


def calculate_sma(prices):
    """
    Calculate Simple Moving Average.
    
    Args:
        prices: Array-like of price values
        
    Returns:
        float: Simple Moving Average
    """
    if len(prices) == 0:
        return 0.0
    
    prices_array = np.asarray(prices, dtype=np.float64)
    return np.mean(prices_array)


def calculate_ema(prices, alpha=0.1):
    """
    Calculate Exponential Moving Average.
    
    Args:
        prices: Array-like of price values
        alpha: Smoothing factor (default: 0.1)
        
    Returns:
        float: Exponential Moving Average
    """
    if len(prices) == 0:
        return 0.0
    
    prices_array = np.asarray(prices, dtype=np.float64)
    ema = prices_array[0]
    
    for i in range(1, len(prices_array)):
        ema = alpha * prices_array[i] + (1 - alpha) * ema
    
    return ema


def calculate_rsi(prices):
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: Array-like of price values
        
    Returns:
        float: RSI value (0-100)
    """
    if len(prices) < 2:
        return 50.0  # Neutral RSI if insufficient data
    
    prices_array = np.asarray(prices, dtype=np.float64)
    gain = 0.0
    loss = 0.0
    
    for i in range(1, len(prices_array)):
        diff = prices_array[i] - prices_array[i - 1]
        if diff > 0:
            gain += diff
        else:
            loss -= diff
    
    if loss == 0:
        return 100.0
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def find_support_resistance(prices):
    """
    Find support and resistance levels in price data.
    
    Args:
        prices: Array-like of price values
        
    Returns:
        tuple: (supports, resistances) where each is a numpy array
    """
    if len(prices) < 3:
        return np.array([]), np.array([])
    
    prices_array = np.asarray(prices, dtype=np.float64)
    supports = []
    resistances = []
    
    for i in range(1, len(prices_array) - 1):
        if prices_array[i] < prices_array[i - 1] and prices_array[i] < prices_array[i + 1]:
            supports.append(prices_array[i])
        if prices_array[i] > prices_array[i - 1] and prices_array[i] > prices_array[i + 1]:
            resistances.append(prices_array[i])
    
    return np.array(supports, dtype=np.float64), np.array(resistances, dtype=np.float64)

