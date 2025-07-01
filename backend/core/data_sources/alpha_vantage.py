# # backend/core/data_sources/alpha_vantage.py

# from alpha_vantage.techindicators import TechIndicators
# from backend.config import settings
# # 1. Import the new multi-ticker extraction function
# from .yahoo_finance import _get_tickers_from_gemini

# def get_technical_indicators(query: str) -> str:
#     """
#     Fetches technical indicators for ALL tickers identified in the query.
#     """
#     # 2. Use the new function to get a list of tickers
#     ticker_symbols = _get_tickers_from_gemini(query)
    
#     if not ticker_symbols:
#         return "Could not identify any specific stock tickers for technical analysis."

#     print(f"[Alpha Vantage] Gemini identified tickers: {ticker_symbols}")
    
#     all_summaries = []
#     # 3. Loop through each identified ticker
#     for ticker_symbol in ticker_symbols:
#         try:
#             ti = TechIndicators(key=settings.ALPHA_VANTAGE_API_KEY, output_format='pandas')
#             summary_parts = [f"Alpha Vantage Technical Indicators for {ticker_symbol}:"]

#             # Fetch RSI
#             data_rsi, _ = ti.get_rsi(symbol=ticker_symbol, interval='daily', time_period=14)
#             latest_rsi = data_rsi['RSI'].iloc[-1]
#             summary_parts.append(f"- RSI (14-day): {latest_rsi:.2f}")

#             # Fetch MACD
#             data_macd, _ = ti.get_macd(symbol=ticker_symbol, interval='daily')
#             latest_macd = data_macd['MACD'].iloc[-1]
#             summary_parts.append(f"- MACD: {latest_macd:.2f}")
            
#             all_summaries.append("\n".join(summary_parts))
#         except Exception as e:
#             # Add specific error messages for each ticker that fails
#             all_summaries.append(f"Error fetching technical indicators for {ticker_symbol}: {e}. The Alpha Vantage free tier is very limited.")

#     # 4. Join all the individual summaries into one final string
#     return "\n\n---\n\n".join(all_summaries)