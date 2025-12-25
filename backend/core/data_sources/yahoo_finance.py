# backend/core/data_sources/yahoo_finance.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import yfinance as yf
import re
import json
from backend.config import settings
import pandas as pd
from datetime import datetime, timedelta

# google_client = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite", api_key=settings.GOOGLE_API_KEY
# )


def format_news_safely(news_items):
    """Safely format news items with fallbacks for missing data"""
    if not news_items:
        return "• No recent news available"
    
    formatted_news = []
    for item in news_items[:5]:  # Limit to 5 items
        try:
            # Yahoo Finance news structure can vary
            # Try different possible keys
            title = (
                item.get('title') or 
                item.get('headline') or 
                item.get('text') or 
                'No title available'
            )
            
            publisher = (
                item.get('publisher') or 
                item.get('source') or 
                item.get('providerName') or 
                'Unknown source'
            )
            
            # Skip if we couldn't get meaningful data
            if title == 'No title available' and publisher == 'Unknown source':
                continue
            
            formatted_news.append(f"• {title} - {publisher}")
        except Exception as e:
            print(f"[Yahoo Finance] Error formatting news item: {e}")
            continue
    
    return "\n".join(formatted_news) if formatted_news else "• No recent news available"

def extract_financial_entities(query: str, api_key: str) -> dict:
    """
    Uses Gemini (via LangChain wrapper) to extract tickers, metrics, and data types.
    Returns: {"tickers": [...], "metrics": [...], "data_types": [...]}
    """

    if not api_key:
        return {"tickers": [], "metrics": [], "data_types": ["info"]}

    google_client = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", api_key=api_key
    )

    try:
        possible_metrics = [
            "previousClose", "open", "dayLow", "dayHigh", "regularMarketPrice", "currentPrice",
            "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "volume", "averageVolume",
            "marketCap", "enterpriseValue", "trailingPE", "forwardPE", "priceToSalesTrailing12Months",
            "enterpriseToRevenue", "bookValue", "priceToBook",
            "profitMargins", "operatingMargins", "returnOnEquity", "returnOnAssets", "debtToEquity",
            "totalRevenue", "revenueGrowth", "earningsGrowth", "ebitda", "totalCash", "totalDebt",
            "dividendYield", "dividendRate", "payoutRatio", "exDividendDate",
            "sector", "industry", "fullTimeEmployees", "website", "businessSummary"
        ]

        data_types = [
            "info", "history", "news", "financials", "balance_sheet", "cashflow", "earnings", "calendar"
        ]

        # Build the prompt as a regular string (no f-string to avoid conflicts with LangChain)
        prompt_text = """
        You are an expert financial entity and data extractor. Analyze the user's query and perform three tasks:

        1. Extract Tickers: Identify all stock tickers or company names.
        - Append correct exchange suffix for non-US stocks:
            - Indian (NSE): .NS
            - London (LSE): .L
            - US: no suffix
        - Normalize company names to primary ticker and expand acronyms where appropriate.

        2. Extract Metrics: Identify specific financial metrics requested.
        - Select from: """ + str(possible_metrics) + """
        - Map natural language (e.g., "P/E ratio" -> "trailingPE")
        - Default if none specified: ["currentPrice", "trailingPE", "marketCap"]

        3. Extract Data Types: Determine what type of data is needed.
        - Select from: """ + str(data_types) + """
        - Default: ["info"]

        Return ONLY valid JSON with no additional text or markdown formatting. Use this exact format:
        {{"tickers": ["TICKER1", "TICKER2"], "metrics": ["metric1", "metric2"], "data_types": ["type1"]}}

        Examples:
        - "Apple stock price" -> {{"tickers": ["AAPL"], "metrics": ["currentPrice"], "data_types": ["info"]}}
        - "Tesla news and earnings" -> {{"tickers": ["TSLA"], "metrics": [], "data_types": ["news", "earnings"]}}
        - "Microsoft financial statements" -> {{"tickers": ["MSFT"], "metrics": [], "data_types": ["financials", "balance_sheet", "cashflow"]}}
        - "NVDA price history last month" -> {{"tickers": ["NVDA"], "metrics": [], "data_types": ["history"]}}
        - "Compare Apple and Microsoft P/E" -> {{"tickers": ["AAPL", "MSFT"], "metrics": ["trailingPE"], "data_types": ["info"]}}

        User Query: {query}
        """
        
        chat_prompt = ChatPromptTemplate.from_template(prompt_text)
        response = google_client.invoke(chat_prompt.format_messages(query=query))
        content = response.content
        
        # Debug: Print raw response
        print(f"[DEBUG] Raw Gemini response: {content}")

        # Try to extract JSON from markdown code blocks first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # No JSON found
                print(f"[WARN] No JSON found in response, using defaults")
                return {"tickers": [], "metrics": [], "data_types": ["info"]}
        
        # Clean up the JSON string
        json_str = json_str.strip()
        
        # Debug: Print extracted JSON
        print(f"[DEBUG] Extracted JSON: {json_str}")
        
        # Parse JSON
        try:
            entities = json.loads(json_str)
        except json.JSONDecodeError as je:
            print(f"[ERROR] JSON decode error: {je}")
            print(f"[ERROR] Attempted to parse: {json_str}")
            return {"tickers": [], "metrics": [], "data_types": ["info"]}

        # Standardize keys and types
        result = {
            "tickers": [t.upper() for t in entities.get("tickers", []) if isinstance(t, str)],
            "metrics": entities.get("metrics", []),
            "data_types": entities.get("data_types", ["info"])
        }
        
        print(f"[DEBUG] Parsed entities: {result}")
        return result

    except Exception as e:
        print(f"🚨 Error in Gemini entity extraction: {e}")
        import traceback
        print(traceback.format_exc())
        return {"tickers": [], "metrics": [], "data_types": ["info"]}

def get_currency_symbol(info):
    currency_map = {
        'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥', 'INR': '₹', 'CAD': 'C$',
        'AUD': 'A$', 'CHF': 'CHF ', 'CNY': '¥', 'HKD': 'HK$', 'SGD': 'S$',
        'KRW': '₩', 'BRL': 'R$', 'MXN': '$', 'RUB': '₽', 'ZAR': 'R', 'SEK': 'kr',
        'NOK': 'kr', 'DKK': 'kr', 'PLN': 'zł', 'CZK': 'Kč', 'HUF': 'Ft', 'ILS': '₪',
        'TRY': '₺', 'THB': '฿', 'MYR': 'RM', 'IDR': 'Rp', 'PHP': '₱', 'VND': '₫',
        'TWD': 'NT$', 'NZD': 'NZ$'
    }
    currency_code = info.get('financialCurrency') or info.get('currency') or 'USD'
    return currency_map.get(currency_code, currency_code + ' ')


def format_currency_value(value, currency_symbol, metric):
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


def get_basic_info(ticker, metrics_to_fetch):
    try:
        info = ticker.info
        if not info:
            return "No basic info available."

        company_name = info.get('shortName', ticker.ticker)
        result = [f"=== {company_name} ({ticker.ticker}) ==="]
        currency_symbol = get_currency_symbol(info)

        metric_labels = {
            "previousClose": "Previous Close", "open": "Open", "dayLow": "Day Low", "dayHigh": "Day High",
            "regularMarketPrice": "Market Price", "currentPrice": "Current Price", "fiftyTwoWeekHigh": "52-Week High",
            "fiftyTwoWeekLow": "52-Week Low", "volume": "Volume", "averageVolume": "Avg Volume", "marketCap": "Market Cap",
            "enterpriseValue": "Enterprise Value", "trailingPE": "P/E Ratio", "forwardPE": "Forward P/E", "priceToSalesTrailing12Months": "P/S Ratio",
            "enterpriseToRevenue": "EV/Revenue", "bookValue": "Book Value", "priceToBook": "P/B Ratio", "profitMargins": "Profit Margin",
            "operatingMargins": "Operating Margin", "returnOnEquity": "ROE", "returnOnAssets": "ROA", "debtToEquity": "Debt/Equity",
            "totalRevenue": "Total Revenue", "revenueGrowth": "Revenue Growth", "earningsGrowth": "Earnings Growth", "ebitda": "EBITDA",
            "totalCash": "Total Cash", "totalDebt": "Total Debt", "dividendYield": "Dividend Yield", "dividendRate": "Dividend Rate",
            "payoutRatio": "Payout Ratio", "exDividendDate": "Ex-Dividend Date", "sector": "Sector", "industry": "Industry",
            "fullTimeEmployees": "Employees", "website": "Website", "businessSummary": "Business Summary"
        }

        if metrics_to_fetch:
            for metric in metrics_to_fetch:
                value = info.get(metric, 'N/A')
                if value == 'N/A' or value is None:
                    continue
                if metric in ['sector', 'industry', 'website', 'businessSummary', 'exDividendDate']:
                    value_str = str(value)
                else:
                    value_str = format_currency_value(value, currency_symbol, metric)
                label = metric_labels.get(metric, metric.replace('_', ' ').title())
                result.append(f"• {label}: {value_str}")
        else:
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
    try:
        hist = ticker.history(period=period)
        if hist.empty:
            return "No historical data available."
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
    try:
        news = ticker.news
        if not news:
            return "No recent news available."
        result = [f"=== Recent News ==="]
        for item in news[:max_items]:
            title = item.get('title', 'No title')
            publisher = item.get('publisher', 'Unknown')
            pub_time = ""
            if 'providerPublishTime' in item:
                pub_time = datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d')
                pub_time = f" ({pub_time})"
            result.append(f"• {title} - {publisher}{pub_time}")
        return "\n".join(result)
    except Exception as e:
        return f"Error getting news: {e}"


def get_financials(ticker, statement_type="financials"):
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
        info = ticker.info
        currency_symbol = get_currency_symbol(info)
        result = [f"=== {title} (Most Recent Year) ==="]
        latest_col = data.columns[0]
        latest_data = data[latest_col]
        for idx, (item, value) in enumerate(latest_data.head(10).items()):
            if pd.notna(value):
                value_str = f"{currency_symbol}{value:,.0f}" if abs(value) > 1000 else f"{currency_symbol}{value:,.2f}"
                result.append(f"• {item}: {value_str}")
        return "\n".join(result)
    except Exception as e:
        return f"Error getting {statement_type}: {e}"


def get_earnings_info(ticker):
    try:
        calendar = ticker.calendar
        earnings = ticker.earnings
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


def get_stock_data(query: str, api_key: str) -> str:
    """
    Main entry point for Yahoo Finance data retrieval.
    Uses Gemini to extract tickers, metrics, and data types from natural language query.
    """
    print(f"[Yahoo Finance] Processing query: '{query}'")
    
    entities = extract_financial_entities(query, api_key)
    tickers = entities.get("tickers", [])
    metrics = entities.get("metrics", [])
    data_types = entities.get("data_types", ["info"])
    
    print(f"[Yahoo Finance] Tickers: {tickers}")
    print(f"[Yahoo Finance] Metrics: {metrics}")
    print(f"[Yahoo Finance] Data Types: {data_types}")
    
    if not tickers:
        return "Could not identify any stock tickers in the query."
    
    all_results = []
    
    for ticker_symbol in tickers:
        try:
            ticker = yf.Ticker(ticker_symbol)
            ticker_results = []
            
            # Process each requested data type
            for data_type in data_types:
                if data_type == "info":
                    info = ticker.info
                    if not info or 'symbol' not in info:
                        ticker_results.append(f"Unable to retrieve info for {ticker_symbol}")
                        continue
                    
                    company_name = info.get('longName', ticker_symbol)
                    ticker_results.append(f"=== {company_name} ({ticker_symbol}) ===")
                    
                    # If specific metrics requested, show only those
                    if metrics:
                        for metric in metrics:
                            value = info.get(metric, "N/A")
                            # Format metric name nicely
                            metric_display = metric.replace('_', ' ').title()
                            if metric == "trailingPE":
                                metric_display = "P/E Ratio"
                            elif metric == "forwardPE":
                                metric_display = "Forward P/E"
                            elif metric == "marketCap":
                                metric_display = "Market Cap"
                                if value != "N/A":
                                    value = f"${value:,.0f}"
                            
                            ticker_results.append(f"• {metric_display}: {value}")
                    else:
                        # Default metrics if none specified
                        ticker_results.append(f"• Current Price: ${info.get('currentPrice', 'N/A')}")
                        ticker_results.append(f"• P/E Ratio: {info.get('trailingPE', 'N/A')}")
                        ticker_results.append(f"• Market Cap: ${info.get('marketCap', 'N/A'):,}")
                
                elif data_type == "news":
                    try:
                        news = ticker.news
                        ticker_results.append("\n=== Recent News ===")
                        ticker_results.append(format_news_safely(news))
                    except Exception as e:
                        ticker_results.append(f"\n=== Recent News ===")
                        ticker_results.append(f"• Unable to retrieve news: {str(e)}")
                
                elif data_type == "history":
                    try:
                        hist = ticker.history(period="1mo")
                        if not hist.empty:
                            ticker_results.append("\n=== Recent Price History (Last 5 Days) ===")
                            for date, row in hist.tail(5).iterrows():
                                ticker_results.append(f"• {date.strftime('%Y-%m-%d')}: Close=${row['Close']:.2f}, Volume={row['Volume']:,.0f}")
                        else:
                            ticker_results.append("\n=== Recent Price History ===")
                            ticker_results.append("• No historical data available")
                    except Exception as e:
                        ticker_results.append(f"\n=== Recent Price History ===")
                        ticker_results.append(f"• Unable to retrieve history: {str(e)}")
                
                # Add other data types as needed (financials, balance_sheet, etc.)
            
            all_results.append("\n".join(ticker_results))
            
        except Exception as e:
            all_results.append(f"Error retrieving data for {ticker_symbol}: {str(e)}")
    
    return "\n\n---\n\n".join(all_results)