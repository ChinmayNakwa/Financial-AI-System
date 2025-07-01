# backend/core/data_sources/polygon.py

import pandas as pd
import pandas_ta as ta
from polygon import RESTClient
from datetime import date, timedelta
from backend.config import settings
import google.generativeai as genai
import json

# Configure the Gemini client
try:
    genai.configure(api_key=settings.GOOGLE_API_KEY)
except Exception as e:
    print(f"Warning: Gemini could not be configured. Extraction will fail. Error: {e}")

def extract_financial_entities(query: str) -> dict:
    """
    Uses Gemini to extract all relevant financial entities from a query, including
    tickers, metrics, and data types specifically for Polygon.io API capabilities.
    
    Returns a dictionary: {"tickers": ["..."], "metrics": ["..."], "data_types": ["..."]}
    """
    if not settings.GOOGLE_API_KEY:
        return {"tickers": [], "metrics": [], "data_types": ["aggregates"]}
        
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Polygon.io specific metrics and data types
        polygon_metrics = [
            # OHLCV data
            "open", "high", "low", "close", "volume", "vwap",
            
            # Technical indicators (calculated from OHLCV)
            "rsi", "macd", "sma", "ema", "bollinger_bands", "stochastic",
            "atr", "adx", "cci", "williams_r", "momentum", "roc",
            
            # Market data
            "price", "daily_change", "percent_change", "trading_volume",
            "market_cap", "shares_outstanding",
            
            # Options data (if available)
            "implied_volatility", "delta", "gamma", "theta", "vega",
            
            # Forex specific (if currency pairs)
            "bid", "ask", "spread", "exchange_rate"
        ]
        
        # Polygon.io data types
        polygon_data_types = [
            "aggregates",      # Historical OHLCV data (bars)
            "trades",          # Individual trade data
            "quotes",          # Bid/ask quote data
            "ticker_details",  # Company information
            "ticker_news",     # News articles
            "market_status",   # Market open/close status
            "splits",          # Stock splits
            "dividends",       # Dividend payments
            "financials",      # Financial statements
            "options_chain",   # Options contracts
            "crypto",          # Cryptocurrency data
            "forex",           # Foreign exchange data
            "indices",         # Market indices
            "snapshots"        # Real-time market snapshots
        ]

        prompt = f"""
        You are an expert financial entity and data extractor specialized in Polygon.io API capabilities. 
        Analyze the user's query and perform three tasks:

        1. **Extract Tickers**: Identify all stock tickers, crypto symbols, or forex pairs.
           - Stock tickers: "AAPL", "TSLA", "MSFT"
           - Crypto: "BTC", "ETH", "X:BTCUSD" (use X: prefix for crypto pairs)
           - Forex: "C:EURUSD", "C:GBPUSD" (use C: prefix for currency pairs)
           - Expand groups: "FAANG" -> ["AAPL", "AMZN", "META", "NFLX", "GOOGL"]
           - Normalize company names: "Apple" -> "AAPL"

        2. **Extract Metrics**: Identify specific financial metrics or technical indicators.
           - Select from: {polygon_metrics}
           - Map natural language: "moving average" -> "sma", "relative strength" -> "rsi"
           - Default if none specified: ["close", "volume"]

        3. **Extract Data Types**: Determine what type of Polygon.io data is needed.
           - Select from: {polygon_data_types}
           - "aggregates" for price/volume history and technical analysis
           - "ticker_news" for company news
           - "ticker_details" for company information
           - "trades" for detailed trade data
           - "quotes" for bid/ask data
           - "snapshots" for real-time data
           - Default: ["aggregates"]

        **Return as valid JSON: {{"tickers": ["..."], "metrics": ["..."], "data_types": ["..."]}}**

        **Examples:**
        - "Apple stock RSI" -> {{"tickers": ["AAPL"], "metrics": ["rsi"], "data_types": ["aggregates"]}}
        - "Tesla news and price" -> {{"tickers": ["TSLA"], "metrics": ["close"], "data_types": ["aggregates", "ticker_news"]}}
        - "Bitcoin price history" -> {{"tickers": ["X:BTCUSD"], "metrics": ["close", "volume"], "data_types": ["aggregates"]}}
        - "EUR/USD forex rates" -> {{"tickers": ["C:EURUSD"], "metrics": ["close"], "data_types": ["aggregates"]}}
        - "Microsoft technical indicators" -> {{"tickers": ["MSFT"], "metrics": ["rsi", "macd", "sma"], "data_types": ["aggregates"]}}
        - "NVDA real-time quotes" -> {{"tickers": ["NVDA"], "metrics": ["bid", "ask"], "data_types": ["snapshots", "quotes"]}}
        - "Apple company info and dividends" -> {{"tickers": ["AAPL"], "metrics": [], "data_types": ["ticker_details", "dividends"]}}

        **User Query:** "{query}"
        """
        
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json\n", "").replace("```", "")
        entities = json.loads(cleaned_response)
        
        # Validation and defaults
        if "tickers" not in entities:
            entities["tickers"] = []
        if "metrics" not in entities:
            entities["metrics"] = []
        if "data_types" not in entities:
            entities["data_types"] = ["aggregates"]
            
        # Validate tickers format for Polygon.io
        validated_tickers = []
        for ticker in entities["tickers"]:
            # Ensure proper prefixes for crypto and forex
            if any(crypto in ticker.upper() for crypto in ["BTC", "ETH", "LTC", "ADA", "DOT"]) and not ticker.startswith("X:"):
                if len(ticker) <= 4:  # Likely a crypto symbol
                    validated_tickers.append(f"X:{ticker.upper()}USD")
                else:
                    validated_tickers.append(ticker.upper())
            elif any(forex in ticker.upper() for forex in ["USD", "EUR", "GBP", "JPY", "CAD"]) and not ticker.startswith("C:"):
                if len(ticker) == 6:  # Likely a forex pair like EURUSD
                    validated_tickers.append(f"C:{ticker.upper()}")
                else:
                    validated_tickers.append(ticker.upper())
            else:
                validated_tickers.append(ticker.upper())
        
        entities["tickers"] = validated_tickers

        return entities

    except Exception as e:
        print(f"Error in Gemini entity extraction for Polygon.io: {e}")
        return {"tickers": [], "metrics": [], "data_types": ["aggregates"]}

# Initialize the Polygon client
try:
    client = RESTClient(api_key=settings.POLYGON_API_KEY)
except Exception as e:
    print(f"Warning: Polygon.io client could not be initialized. Error: {e}")
    client = None

def get_technical_indicators(query: str) -> str:
    """
    Fetches historical stock data from Polygon.io and calculates technical indicators (RSI, MACD)
    for all tickers identified in the query by the central AI extractor.
    """
    if not client:
        return "Polygon.io client is not configured. Please check your API key."

    # 2. Use the central extractor to get the tickers
    entities = extract_financial_entities(query)
    ticker_symbols = entities.get("tickers", [])
    
    if not ticker_symbols:
        return "Could not identify any specific stock tickers for technical analysis."

    print(f"[Polygon.io] Identified tickers: {ticker_symbols}")

    # Define the date range for fetching data
    today = date.today()
    start_date = today - timedelta(days=100)

    all_summaries = []
    # 3. Loop through each identified ticker
    for ticker in ticker_symbols:
        try:
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime("%Y-%m-%d"),
                to=today.strftime("%Y-%m-%d"),
            )
            
            if not aggs:
                all_summaries.append(f"No historical data found for {ticker} on Polygon.io.")
                continue

            df = pd.DataFrame(aggs)
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('date', inplace=True)

            if df.empty:
                all_summaries.append(f"Could not process historical data for {ticker}.")
                continue

            # Calculate technical indicators
            df.ta.rsi(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            
            latest_rsi = df['RSI_14'].iloc[-1]
            latest_macd = df['MACD_12_26_9'].iloc[-1]
            
            summary = (
                f"Polygon.io Technical Indicators for {ticker}:\n"
                f"- RSI (14-day): {latest_rsi:.2f}\n"
                f"- MACD (12, 26, 9): {latest_macd:.2f}"
            )
            all_summaries.append(summary)

        except Exception as e:
            all_summaries.append(f"Error processing technical indicators for {ticker} with Polygon.io: {e}")

    # 4. Aggregate the results
    return "\n\n---\n\n".join(all_summaries)