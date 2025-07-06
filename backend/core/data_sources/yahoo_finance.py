# backend/core/data_sources/yahoo_finance.py

import yfinance as yf
import re
import json
import google.generativeai as genai
from backend.config import settings
import pandas as pd
from datetime import datetime, timedelta

# Configure the Gemini client
try:
    genai.configure(api_key=settings.GOOGLE_API_KEY)
except Exception as e:
    print(f"Warning: Gemini could not be configured. Extraction will fail. Error: {e}")

def extract_financial_entities(query: str) -> dict:
    """
    Uses Gemini to extract all relevant financial entities from a query, including
    tickers, metrics, and data types (current, historical, news, etc.).
    
    Returns a dictionary: {"tickers": ["..."], "metrics": ["..."], "data_types": ["..."]}
    """
    if not settings.GOOGLE_API_KEY:
        return {"tickers": [], "metrics": [], "data_types": ["info"]}
        
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Expanded list of possible metrics
        possible_metrics = [
            # Price metrics
            "previousClose", "open", "dayLow", "dayHigh", "regularMarketPrice", "currentPrice",
            "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "volume", "averageVolume",
            
            # Valuation metrics
            "marketCap", "enterpriseValue", "trailingPE", "forwardPE", "priceToSalesTrailing12Months", 
            "enterpriseToRevenue", "bookValue", "priceToBook",
            
            # Financial performance
            "profitMargins", "operatingMargins", "returnOnEquity", "returnOnAssets", "debtToEquity",
            "totalRevenue", "revenueGrowth", "earningsGrowth", "ebitda", "totalCash", "totalDebt",
            
            # Dividend info
            "dividendYield", "dividendRate", "payoutRatio", "exDividendDate",
            
            # Company info
            "sector", "industry", "fullTimeEmployees", "website", "businessSummary"
        ]
        
        # Types of data we can fetch
        data_types = [
            "info",        # Basic company info and metrics
            "history",     # Historical price data
            "news",        # Recent news
            "financials",  # Income statement
            "balance_sheet", # Balance sheet
            "cashflow",    # Cash flow statement
            "earnings",    # Earnings data
            "calendar"     # Earnings calendar
        ]

        prompt = f"""
        You are an expert financial entity and data extractor. Analyze the user's query and perform three tasks:

        1. **Extract Tickers**: Identify all stock tickers or company names.
            **Crucially, append the correct exchange suffix for non-US stocks.** This is vital for correct data retrieval.
                -   For Indian stocks on the National Stock Exchange, use the `.NS` suffix (e.g., "Tata Motors" -> "TATAMOTORS.NS", "Reliance" -> "RELIANCE.NS").
                -   For London Stock Exchange, use `.L` (e.g., "BP" -> "BP.L").
                -   For US stocks (like Apple, Microsoft), no suffix is needed.
            -   Normalize company names to their primary ticker (e.g., "Apple" -> "AAPL").
            -   Expand acronyms where appropriate (e.g., "FAANG" -> "META", "AMZN", "AAPL", "NFLX", "GOOGL").


        2. **Extract Metrics**: Identify specific financial metrics requested.
           - Select from: {possible_metrics}
           - Map natural language: "P/E ratio" -> "trailingPE", "market cap" -> "marketCap"
           - Default if none specified: ["currentPrice", "trailingPE", "marketCap"]

        3. **Extract Data Types**: Determine what type of data is needed.
           - Select from: {data_types}
           - "info" for current metrics, "history" for price trends, "news" for recent news
           - "financials" for income statement, "balance_sheet", "cashflow" for financial statements
           - Default: ["info"]

        **Return as valid JSON: {{"tickers": ["..."], "metrics": ["..."], "data_types": ["..."]}}**

        **Examples:**
        - "Apple stock price" -> {{"tickers": ["AAPL"], "metrics": ["currentPrice"], "data_types": ["info"]}}
        - "Tesla news and earnings" -> {{"tickers": ["TSLA"], "metrics": [], "data_types": ["news", "earnings"]}}
        - "Microsoft financial statements" -> {{"tickers": ["MSFT"], "metrics": [], "data_types": ["financials", "balance_sheet", "cashflow"]}}
        - "NVDA price history last month" -> {{"tickers": ["NVDA"], "metrics": [], "data_types": ["history"]}}

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
            entities["data_types"] = ["info"]

        return entities

    except Exception as e:
        print(f"Error in Gemini entity extraction: {e}")
        return {"tickers": [], "metrics": [], "data_types": ["info"]}

# --- HELPER FUNCTIONS FOR CURRENCY ---
def get_currency_symbol(info):
    """Get currency symbol from ticker info"""
    currency_map = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'INR': '₹',
        'CAD': 'C$',
        'AUD': 'A$',
        'CHF': 'CHF ',
        'CNY': '¥',
        'HKD': 'HK$',
        'SGD': 'S$',
        'KRW': '₩',
        'BRL': 'R$',
        'MXN': '$',
        'RUB': '₽',
        'ZAR': 'R',
        'SEK': 'kr',
        'NOK': 'kr',
        'DKK': 'kr',
        'PLN': 'zł',
        'CZK': 'Kč',
        'HUF': 'Ft',
        'ILS': '₪',
        'TRY': '₺',
        'THB': '฿',
        'MYR': 'RM',
        'IDR': 'Rp',
        'PHP': '₱',
        'VND': '₫',
        'TWD': 'NT$',
        'NZD': 'NZ$'
    }
    
    # Try different currency fields
    currency_code = info.get('financialCurrency') or info.get('currency') or 'USD'
    return currency_map.get(currency_code, currency_code + ' ')

def format_currency_value(value, currency_symbol, metric):
    """Format currency values with appropriate currency symbol"""
    if isinstance(value, (int, float)):
        if metric in ['marketCap', 'enterpriseValue', 'totalRevenue', 'ebitda', 'totalCash', 'totalDebt']:
            return f"{currency_symbol}{value:,.0f}" if value > 1000000 else f"{currency_symbol}{value:,.2f}"
        elif metric in ['currentPrice', 'previousClose', 'open', 'dayHigh', 'dayLow', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'bookValue', 'dividendRate']:
            return f"{currency_symbol}{value:.2f}"
        elif metric in ['dividendYield', 'profitMargins', 'operatingMargins', 'returnOnEquity', 'returnOnAssets', 'revenueGrowth', 'earningsGrowth', 'payoutRatio']:
            return f"{value:.2%}" if abs(value) <= 1 else f"{value:.2f}%"
        elif metric in ['volume', 'averageVolume', 'fullTimeEmployees']:
            return f"{value:,.0f}"
        else:
            return f"{value:,.2f}"
    else:
        return str(value)

# --- HELPER FUNCTIONS FOR DIFFERENT DATA TYPES ---
def get_basic_info(ticker, metrics_to_fetch):
    """Get basic company info and metrics"""
    try:
        info = ticker.info
        if not info:
            return "No basic info available."
        
        company_name = info.get('shortName', ticker.ticker)
        result = [f"=== {company_name} ({ticker.ticker}) ==="]
        
        # Get the proper currency symbol
        currency_symbol = get_currency_symbol(info)

        # Create a mapping for better labels
        metric_labels = {
            "previousClose": "Previous Close",
            "open": "Open",
            "dayLow": "Day Low",
            "dayHigh": "Day High",
            "regularMarketPrice": "Market Price",
            "currentPrice": "Current Price",
            "fiftyTwoWeekHigh": "52-Week High",
            "fiftyTwoWeekLow": "52-Week Low",
            "volume": "Volume",
            "averageVolume": "Avg Volume",
            "marketCap": "Market Cap",
            "enterpriseValue": "Enterprise Value",
            "trailingPE": "P/E Ratio",
            "forwardPE": "Forward P/E",
            "priceToSalesTrailing12Months": "P/S Ratio",
            "enterpriseToRevenue": "EV/Revenue",
            "bookValue": "Book Value",
            "priceToBook": "P/B Ratio",
            "profitMargins": "Profit Margin",
            "operatingMargins": "Operating Margin",
            "returnOnEquity": "ROE",
            "returnOnAssets": "ROA",
            "debtToEquity": "Debt/Equity",
            "totalRevenue": "Total Revenue",
            "revenueGrowth": "Revenue Growth",
            "earningsGrowth": "Earnings Growth",
            "ebitda": "EBITDA",
            "totalCash": "Total Cash",
            "totalDebt": "Total Debt",
            "dividendYield": "Dividend Yield",
            "dividendRate": "Dividend Rate",
            "payoutRatio": "Payout Ratio",
            "exDividendDate": "Ex-Dividend Date",
            "sector": "Sector",
            "industry": "Industry",
            "fullTimeEmployees": "Employees",
            "website": "Website",
            "businessSummary": "Business Summary"
        }
        
        if metrics_to_fetch:
            for metric in metrics_to_fetch:
                value = info.get(metric, 'N/A')
                if value == 'N/A' or value is None:
                    continue
                
                # Format based on metric type
                if metric in ['sector', 'industry', 'website', 'businessSummary', 'exDividendDate']:
                    value_str = str(value)
                else:
                    value_str = format_currency_value(value, currency_symbol, metric)
                
                label = metric_labels.get(metric, metric.replace('_', ' ').title())
                result.append(f"• {label}: {value_str}")
        else:
            # Default overview
            current_price = info.get('currentPrice', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
            trailing_pe = info.get('trailingPE', 'N/A')
            
            if current_price != 'N/A':
                result.append(f"• Current Price: {currency_symbol}{current_price}")
            else:
                result.append("• Current Price: N/A")
                
            if market_cap != 'N/A':
                result.append(f"• Market Cap: {currency_symbol}{market_cap:,}")
            else:
                result.append("• Market Cap: N/A")
                
            result.extend([
                f"• P/E Ratio: {trailing_pe}",
                f"• Sector: {info.get('sector', 'N/A')}",
                f"• Industry: {info.get('industry', 'N/A')}"
            ])
        
        return "\n".join(result)
    except Exception as e:
        return f"Error getting basic info: {e}"

def get_price_history(ticker, period="1mo"):
    """Get historical price data"""
    try:
        hist = ticker.history(period=period)
        if hist.empty:
            return "No historical data available."
        
        # Get currency symbol from ticker info
        info = ticker.info
        currency_symbol = get_currency_symbol(info)
        
        latest = hist.iloc[-1]
        first = hist.iloc[0]
        change = ((latest['Close'] - first['Close']) / first['Close']) * 100
        
        result = [f"=== Price History ({period}) ==="]
        result.append(f"• Start Price: {currency_symbol}{first['Close']:.2f}")
        result.append(f"• End Price: {currency_symbol}{latest['Close']:.2f}")
        result.append(f"• Change: {change:+.2f}%")
        result.append(f"• High: {currency_symbol}{hist['High'].max():.2f}")
        result.append(f"• Low: {currency_symbol}{hist['Low'].min():.2f}")
        result.append(f"• Avg Volume: {hist['Volume'].mean():,.0f}")
        
        return "\n".join(result)
    except Exception as e:
        return f"Error getting price history: {e}"

def get_recent_news(ticker, max_items=5):
    """Get recent news"""
    try:
        news = ticker.news
        if not news:
            return "No recent news available."
        
        result = [f"=== Recent News ==="]
        for item in news[:max_items]:
            title = item.get('title', 'No title')
            publisher = item.get('publisher', 'Unknown')
            # Convert timestamp if available
            pub_time = ""
            if 'providerPublishTime' in item:
                pub_time = datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d')
                pub_time = f" ({pub_time})"
            
            result.append(f"• {title} - {publisher}{pub_time}")
        
        return "\n".join(result)
    except Exception as e:
        return f"Error getting news: {e}"

def get_financials(ticker, statement_type="financials"):
    """Get financial statements"""
    try:
        if statement_type == "financials":
            data = ticker.financials
            title = "Income Statement"
        elif statement_type == "balance_sheet":
            data = ticker.balance_sheet
            title = "Balance Sheet"
        elif statement_type == "cashflow":
            data = ticker.cashflow
            title = "Cash Flow Statement"
        else:
            return "Invalid financial statement type."
        
        if data.empty:
            return f"No {title.lower()} data available."
        
        # Get currency symbol from ticker info
        info = ticker.info
        currency_symbol = get_currency_symbol(info)
        
        result = [f"=== {title} (Most Recent Year) ==="]
        
        # Get the most recent column (year)
        latest_col = data.columns[0]
        latest_data = data[latest_col]
        
        # Show top 10 items
        for idx, (item, value) in enumerate(latest_data.head(10).items()):
            if pd.notna(value):
                value_str = f"{currency_symbol}{value:,.0f}" if abs(value) > 1000 else f"{currency_symbol}{value:,.2f}"
                result.append(f"• {item}: {value_str}")
        
        return "\n".join(result)
    except Exception as e:
        return f"Error getting {statement_type}: {e}"

def get_earnings_info(ticker):
    """Get earnings information"""
    try:
        calendar = ticker.calendar
        earnings = ticker.earnings
        
        # Get currency symbol from ticker info
        info = ticker.info
        currency_symbol = get_currency_symbol(info)
        
        result = [f"=== Earnings Information ==="]
        
        if calendar is not None and not calendar.empty:
            next_earnings = calendar.index[0].strftime('%Y-%m-%d')
            result.append(f"• Next Earnings Date: {next_earnings}")
        
        if earnings is not None and not earnings.empty:
            latest_year = earnings.index[-1]
            latest_earnings = earnings.loc[latest_year, 'Earnings']
            latest_revenue = earnings.loc[latest_year, 'Revenue']
            result.append(f"• Latest Earnings ({latest_year}): {currency_symbol}{latest_earnings:.2f}")
            result.append(f"• Latest Revenue ({latest_year}): {currency_symbol}{latest_revenue:,.0f}")
        
        return "\n".join(result) if len(result) > 1 else "No earnings information available."
    except Exception as e:
        return f"Error getting earnings info: {e}"

# --- MAIN FUNCTION ---
def get_stock_data(query: str) -> str:
    """
    Dynamically fetches comprehensive financial data based on user query.
    """
    print(f"[Yahoo Finance] Processing query: '{query}'")
    entities = extract_financial_entities(query)
    
    ticker_symbols = entities.get("tickers", [])
    metrics_to_fetch = entities.get("metrics", [])
    data_types = entities.get("data_types", ["info"])
    
    if not ticker_symbols:
        return "Could not identify any stock tickers in the query."

    print(f"[Yahoo Finance] Tickers: {ticker_symbols}")
    print(f"[Yahoo Finance] Metrics: {metrics_to_fetch}")
    print(f"[Yahoo Finance] Data Types: {data_types}")

    all_results = []
    
    for ticker_symbol in ticker_symbols:
        try:
            ticker = yf.Ticker(ticker_symbol)
            ticker_results = []
            
            for data_type in data_types:
                if data_type == "info":
                    ticker_results.append(get_basic_info(ticker, metrics_to_fetch))
                elif data_type == "history":
                    ticker_results.append(get_price_history(ticker))
                elif data_type == "news":
                    ticker_results.append(get_recent_news(ticker))
                elif data_type in ["financials", "balance_sheet", "cashflow"]:
                    ticker_results.append(get_financials(ticker, data_type))
                elif data_type == "earnings":
                    ticker_results.append(get_earnings_info(ticker))
                elif data_type == "calendar":
                    ticker_results.append(get_earnings_info(ticker))
            
            if ticker_results:
                all_results.append("\n\n".join(ticker_results))
            
        except Exception as e:
            all_results.append(f"Error processing {ticker_symbol}: {e}")
    
    return "\n\n---\n\n".join(all_results) if all_results else "No data could be retrieved."