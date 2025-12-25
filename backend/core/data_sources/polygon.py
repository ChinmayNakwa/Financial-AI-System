from backend.config import settings
from backend.core.data_sources.yahoo_finance import extract_financial_entities
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

try:
    from polygon import RESTClient
except (ImportError, AttributeError):
    import polygon
    RESTClient = polygon.RESTClient

class IndicatorRequest(BaseModel):
    indicator_name: str = Field(description="The name of the indicator, e.g., 'sma', 'ema', 'rsi', 'macd'.")
    window: Optional[int] = Field(default=None, description="The time window or period for the indicator.")

polygon_client = RESTClient(api_key=settings.POLYGON_API_KEY)
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=settings.GOOGLE_API_KEY, temperature=0)

parser = JsonOutputParser(pydantic_object=IndicatorRequest)
indicator_prompt = ChatPromptTemplate.from_template(
    """Extract the indicator and window from the query.
    - "moving average" or "SMA" -> "sma"
    - "exponential moving average" or "EMA" -> "ema"
    - "RSI" -> "rsi"
    - "MACD" -> "macd"
    
    Defaults: sma/ema: 50, rsi: 14, macd: null
    Query: "{query}"
    {format_instructions}
    """
).partial(format_instructions=parser.get_format_instructions())


def extract_ticker_fallback(query: str, api_key: str) -> Optional[str]:
    ticker_map = {
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 
        'alphabet': 'GOOGL', 'amazon': 'AMZN', 'tesla': 'TSLA',
        'meta': 'META', 'facebook': 'META', 'nvidia': 'NVDA',
        'netflix': 'NFLX', 'amd': 'AMD', 'intel': 'INTC'
    }
    
    query_lower = query.lower()
    for key, ticker in ticker_map.items():
        if key in query_lower:
            return ticker
    
    matches = re.findall(r'\b([A-Z]{1,5})\b', query)
    return matches[0] if matches else None

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def get_technical_indicators(query: str, api_key: str) -> str:
    ticker = None
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)
    try:
        ticker_info = extract_financial_entities(query, api_key)
        if ticker_info and ticker_info.get('tickers'):
            ticker = ticker_info['tickers'][0]
    except Exception as e:
        print(f"[Polygon.io] LLM extraction failed: {e}")
    
    if not ticker:
        ticker = extract_ticker_fallback(query)
        if not ticker:
            return "Could not identify a stock ticker. Please specify a company name or ticker symbol."
    
    print(f"[Polygon.io] Using ticker: {ticker}")

    try:
        try:
            
            indicator_chain = indicator_prompt | llm | parser
            request = indicator_chain.invoke({"query": query})
            indicator_name = request.get('indicator_name', 'sma')
            window = request.get('window')
        except Exception as e:
            print(f"[Polygon.io] LLM parsing failed: {e}")
            query_lower = query.lower()
            if 'rsi' in query_lower:
                indicator_name, window = 'rsi', 14
            elif 'macd' in query_lower:
                indicator_name, window = 'macd', None
            elif 'ema' in query_lower or 'exponential' in query_lower:
                indicator_name = 'ema'
                numbers = re.findall(r'\d+', query)
                window = int(numbers[0]) if numbers else 50
            else:
                indicator_name = 'sma'
                numbers = re.findall(r'\d+', query)
                window = int(numbers[0]) if numbers else 50

        print(f"[Polygon.io] indicator='{indicator_name}', window={window}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        agg_bars = polygon_client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d')
        )
        
        if not agg_bars:
            return f"No historical data found for {ticker}."

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
        
        current_price = df['close'].iloc[-1]
        
        if indicator_name == 'sma':
            window = window or 50
            df[f'SMA_{window}'] = calculate_sma(df['close'], window)
            latest_value = df[f'SMA_{window}'].iloc[-1]
            
            if pd.isna(latest_value):
                return f"Not enough data to calculate {window}-day SMA for {ticker}."
            
            return f"Polygon.io Data for {ticker}:\n- Current Price: ${current_price:.2f}\n- Simple Moving Average ({window}-day): ${latest_value:.2f}"

        elif indicator_name == 'ema':
            window = window or 50
            df[f'EMA_{window}'] = calculate_ema(df['close'], window)
            latest_value = df[f'EMA_{window}'].iloc[-1]
            
            if pd.isna(latest_value):
                return f"Not enough data to calculate {window}-day EMA for {ticker}."
            
            return f"Polygon.io Data for {ticker}:\n- Current Price: ${current_price:.2f}\n- Exponential Moving Average ({window}-day): ${latest_value:.2f}"

        elif indicator_name == 'rsi':
            window = window or 14
            df[f'RSI_{window}'] = calculate_rsi(df['close'], window)
            latest_value = df[f'RSI_{window}'].iloc[-1]
            
            if pd.isna(latest_value):
                return f"Not enough data to calculate {window}-day RSI for {ticker}."
            
            rsi_signal = "Overbought" if latest_value > 70 else "Oversold" if latest_value < 30 else "Neutral"
            return f"Polygon.io Data for {ticker}:\n- Current Price: ${current_price:.2f}\n- RSI ({window}-day): {latest_value:.2f} ({rsi_signal})"
            
        elif indicator_name == 'macd':
            macd_line, signal_line, histogram = calculate_macd(df['close'])
            df['MACD'] = macd_line
            df['MACD_signal'] = signal_line
            df['MACD_hist'] = histogram
            
            latest_macd = df['MACD'].iloc[-1]
            latest_signal = df['MACD_signal'].iloc[-1]
            latest_hist = df['MACD_hist'].iloc[-1]
            
            if pd.isna(latest_macd) or pd.isna(latest_signal):
                return f"Not enough data to calculate MACD for {ticker}."
            
            macd_signal = "Bullish" if latest_macd > latest_signal else "Bearish"
            return f"Polygon.io Data for {ticker}:\n- Current Price: ${current_price:.2f}\n- MACD: {latest_macd:.4f}\n- Signal Line: {latest_signal:.4f}\n- Histogram: {latest_hist:.4f}\n- Signal: {macd_signal}"

        else:
            return f"Indicator '{indicator_name}' not supported. Available: SMA, EMA, RSI, MACD."

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error getting technical indicators for {ticker}: {str(e)}"