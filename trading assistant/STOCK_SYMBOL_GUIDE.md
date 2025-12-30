# Stock Symbol Syntax Guide

## How to Enter Stock Names

### For Web Interface (http://localhost:5001)

When using the web form, enter stock symbols in the following format:

#### Indian Stocks (NSE - National Stock Exchange)
- **Format**: `SYMBOL` (without .NS suffix - backend adds it automatically)
- **Examples**:
  - `RELIANCE` (for Reliance Industries)
  - `TCS` (for Tata Consultancy Services)
  - `INFY` (for Infosys)
  - `HDFCBANK` (for HDFC Bank)
  - `ICICIBANK` (for ICICI Bank)
  - `WIPRO` (for Wipro)
  - `BHARTIARTL` (for Bharti Airtel)

#### US Stocks
- **Format**: `SYMBOL` (as-is)
- **Examples**:
  - `AAPL` (for Apple)
  - `GOOGL` (for Google/Alphabet)
  - `MSFT` (for Microsoft)
  - `TSLA` (for Tesla)
  - `AMZN` (for Amazon)
  - `META` (for Meta/Facebook)

### For test_cpp.py Script

Edit the `stock_symbols` list in `test_cpp.py`:

```python
stock_symbols = ["DCXINDIA.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS", "AAPL", "GOOGL", "MSFT", "TSLA"]
```

#### Indian Stocks in Script
- **Format**: `SYMBOL.NS` (include .NS suffix)
- **Examples**:
  - `RELIANCE.NS`
  - `TCS.NS`
  - `INFY.NS`
  - `HDFCBANK.NS`

#### US Stocks in Script
- **Format**: `SYMBOL` (no suffix needed)
- **Examples**:
  - `AAPL`
  - `GOOGL`
  - `MSFT`

## Quick Reference

| Stock Type | Web Form | test_cpp.py Script |
|------------|----------|-------------------|
| Indian (NSE) | `RELIANCE` | `RELIANCE.NS` |
| US (NYSE/NASDAQ) | `AAPL` | `AAPL` |

## Common Indian Stock Symbols

- `RELIANCE` - Reliance Industries
- `TCS` - Tata Consultancy Services
- `INFY` - Infosys
- `HDFCBANK` - HDFC Bank
- `ICICIBANK` - ICICI Bank
- `WIPRO` - Wipro
- `BHARTIARTL` - Bharti Airtel
- `ITC` - ITC Limited
- `SBIN` - State Bank of India
- `LT` - Larsen & Toubro

## Common US Stock Symbols

- `AAPL` - Apple Inc.
- `GOOGL` - Alphabet Inc. (Google)
- `MSFT` - Microsoft Corporation
- `TSLA` - Tesla Inc.
- `AMZN` - Amazon.com Inc.
- `META` - Meta Platforms Inc. (Facebook)
- `NVDA` - NVIDIA Corporation
- `JPM` - JPMorgan Chase & Co.

## Notes

1. **Case Sensitivity**: Stock symbols are usually case-insensitive, but it's best to use UPPERCASE
2. **Indian Stocks**: The web interface automatically adds `.NS` suffix, so don't include it
3. **US Stocks**: No suffix needed for either interface
4. **Date Format**: Use YYYY-MM-DD format (e.g., `2025-01-01`)

## Troubleshooting

If a stock symbol doesn't work:
1. Verify the symbol is correct on Yahoo Finance
2. Check if the stock is listed/traded
3. Try a different symbol from the examples above
4. For Indian stocks, ensure you're using the NSE symbol (not BSE)

