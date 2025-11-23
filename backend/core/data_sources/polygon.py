# backend/core/data_sources/polygon.py

from backend.config import settings
from backend.core.data_sources.yahoo_finance import extract_financial_entities  # Reuse your smart ticker finder
import pandas as pd
# import pandas_ta as ta
from polygon import RESTClient
import json
from datetime import datetime, timedelta

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

# --- Pydantic Model for Structured LLM Output ---
class IndicatorRequest(BaseModel):
    indicator_name: str = Field(description="The name of the indicator, e.g., 'sma', 'ema', 'rsi', 'macd'.")
    window: Optional[int] = Field(default=None, description="The time window or period for the indicator, e.g., 50 for a 50-day moving average.")

# --- Initialize APIs ---
polygon_client = RESTClient(api_key=settings.POLYGON_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=settings.GOOGLE_API_KEY, temperature=0)

# --- Create a Structured LLM Chain for Parsing ---
parser = JsonOutputParser(pydantic_object=IndicatorRequest)
indicator_prompt = ChatPromptTemplate.from_template(
    """You are an expert at parsing financial queries for technical indicators.
    From the user's query, extract the specific indicator and its time window.

    - "moving average" or "SMA" should map to "sma".
    - "exponential moving average" or "EMA" should map to "ema".
    - "RSI" should map to "rsi".
    - "MACD" should map to "macd".
    
    Default windows:
    - sma/ema: 50
    - rsi: 14
    - macd: standard (12, 26, 9) - window can be null.

    User Query: "{query}"

    {format_instructions}
    """
).partial(format_instructions=parser.get_format_instructions())

indicator_chain = indicator_prompt | llm | parser

# --- Main Tool Function ---
def get_technical_indicators(query: str) -> str:
    """
    Calculates a specific technical indicator (like SMA, EMA, RSI, MACD) for a stock ticker
    based on the user's query.
    """
    ticker_info = extract_financial_entities(query)
    if not ticker_info or not ticker_info.get('tickers'):
        return "Could not identify a stock ticker for technical analysis."
    
    # Extract the first ticker from the list
    ticker = ticker_info['tickers'][0]
    print(f"[Polygon.io] Identified ticker: {ticker}")

    try:
        # Use the LLM to parse the specific indicator request
        request = indicator_chain.invoke({"query": query})
        indicator_name = request.get('indicator_name', 'sma')
        window = request.get('window')

        print(f"[Polygon.io] Parsed request: indicator='{indicator_name}', window={window}")

        # Calculate date range (get enough historical data for indicators)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Get 1 year of data
        
        # Format dates for Polygon API (YYYY-MM-DD format)
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')

        # Fetch historical data to calculate the indicator
        agg_bars = polygon_client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=from_date,
            to=to_date  # Fixed: removed underscore, use 'to' instead of 'to_'
        )
        
        if not agg_bars:
            return f"No historical data found for {ticker} on Polygon.io."

        # Convert to DataFrame
        df = pd.DataFrame([{
            'time': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        } for bar in agg_bars])
        
        if df.empty:
            return f"No historical data available for {ticker}."
        
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        # Get current price for context
        current_price = df['close'].iloc[-1]
        
        # Use pandas_ta to calculate the requested indicator
        if indicator_name == 'sma':
            window = window or 50  # Default to 50 if not specified
            df.ta.sma(length=window, append=True)
            indicator_col = f'SMA_{window}'
            
            if indicator_col not in df.columns:
                return f"Unable to calculate SMA for {ticker}. Not enough data points."
            
            latest_value = df[indicator_col].iloc[-1]
            if pd.isna(latest_value):
                return f"Unable to calculate {window}-day SMA for {ticker}. Not enough historical data."
            
            return f"Polygon.io Data for {ticker}:\n- Current Price: ${current_price:.2f}\n- Simple Moving Average ({window}-day): ${latest_value:.2f}"

        elif indicator_name == 'ema':
            window = window or 50  # Default to 50 if not specified
            df.ta.ema(length=window, append=True)
            indicator_col = f'EMA_{window}'
            
            if indicator_col not in df.columns:
                return f"Unable to calculate EMA for {ticker}. Not enough data points."
            
            latest_value = df[indicator_col].iloc[-1]
            if pd.isna(latest_value):
                return f"Unable to calculate {window}-day EMA for {ticker}. Not enough historical data."
            
            return f"Polygon.io Data for {ticker}:\n- Current Price: ${current_price:.2f}\n- Exponential Moving Average ({window}-day): ${latest_value:.2f}"

        elif indicator_name == 'rsi':
            window = window or 14  # Default to 14
            df.ta.rsi(length=window, append=True)
            indicator_col = f'RSI_{window}'
            
            if indicator_col not in df.columns:
                return f"Unable to calculate RSI for {ticker}. Not enough data points."
            
            latest_value = df[indicator_col].iloc[-1]
            if pd.isna(latest_value):
                return f"Unable to calculate {window}-day RSI for {ticker}. Not enough historical data."
            
            # RSI interpretation
            rsi_signal = "Overbought" if latest_value > 70 else "Oversold" if latest_value < 30 else "Neutral"
            
            return f"Polygon.io Data for {ticker}:\n- Current Price: ${current_price:.2f}\n- RSI ({window}-day): {latest_value:.2f} ({rsi_signal})"
            
        elif indicator_name == 'macd':
            df.ta.macd(append=True)
            # MACD results in multiple columns
            macd_col = 'MACD_12_26_9'
            signal_col = 'MACDs_12_26_9'
            histogram_col = 'MACDh_12_26_9'
            
            if macd_col not in df.columns:
                return f"Unable to calculate MACD for {ticker}. Not enough data points."
            
            latest_macd_line = df[macd_col].iloc[-1]
            latest_signal_line = df[signal_col].iloc[-1]
            latest_histogram = df[histogram_col].iloc[-1]
            
            if pd.isna(latest_macd_line) or pd.isna(latest_signal_line):
                return f"Unable to calculate MACD for {ticker}. Not enough historical data."
            
            # MACD signal interpretation
            macd_signal = "Bullish" if latest_macd_line > latest_signal_line else "Bearish"
            
            return f"Polygon.io Data for {ticker}:\n- Current Price: ${current_price:.2f}\n- MACD (12, 26, 9): {latest_macd_line:.4f}\n- Signal Line: {latest_signal_line:.4f}\n- Histogram: {latest_histogram:.4f}\n- Signal: {macd_signal}"

        else:
            return f"The requested indicator '{indicator_name}' is not supported. Available indicators: SMA, EMA, RSI, MACD."

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"An error occurred while getting technical indicators for {ticker}: {str(e)}"