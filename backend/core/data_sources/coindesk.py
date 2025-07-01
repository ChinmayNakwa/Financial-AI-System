import requests 
from backend.config import settings
import google.generativeai as genai
from datetime import datetime

coindesk_api_key = settings.COINDESK_API_KEY
genai.configure(api_key=settings.GOOGLE_API_KEY)


# VALID_MARKETS = {
#     "usdli": "USD Index",
#     "cadli": "CAD Index",
#     "euli": "Euro Index",
#     "gbpli": "GBP Index",
#     "jpyli": "JPY Index",
# }

# def get_market_from_prompt(prompt: str) -> str:
#     market_prompt = f"""
#     The user is requesting cryptocurrency data. From the sentence below, identify the most appropriate market code from this list:
#     {', '.join(VALID_MARKETS.keys())}

#     Only return the **single most relevant** market code from this list. If the user mentions USD, return 'usdli'. If JPY is mentioned, return 'jpyli', and so on.

#     User sentence:
#     {prompt}
#     """
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     response = model.generate_content(market_prompt)
#     market = response.text.strip().lower()

#     if market not in VALID_MARKETS:
#         return "usdli"
    
#     return market

def get_instruments(prompt:str) -> list[str]:
    """Uses Gemini to extract cryptocurrency instrument tickers from a user query."""
    instrument_prompt = f"""From the user's financial question below, extract the cryptocurrency instrument tickers.
- The format must be 'BASE-QUOTE' (e.g., BTC-USD, ETH-EUR).
- If no specific quote currency (like EUR, CAD) is mentioned, default to USD.
- If the question is general (e.g., "what's the crypto market like?"), return the top 5 most common instruments: BTC-USD,ETH-USD,SOL-USD,XRP-USD,DOGE-USD.
- For a specific query like "price of ethereum", return 'ETH-USD'.

User Question: "{prompt}"

Return only a comma-separated list of the instrument tickers.
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(instrument_prompt)
    instruments_text = response.text.strip()
    return [i.strip().upper() for i in instruments_text.split(",") if i.strip()]

def get_latest_tick_data(prompt:str) -> str:
    """
    Fetches the latest tick data for crypto instruments from CoinDesk/CryptoCompare
    and formats the result into a human-readable string.
    """
    try:
        instruments = get_instruments(prompt)
        if not instruments:
            return "Could not identify any cryptocurrency instruments in the query."

        instruments_query = ",".join(instruments)
        
        # Use the CryptoCompare API endpoint as it's more direct
        # Note: The URL is slightly different from before. We'll use the one that works with this JSON structure.
        api_url = "https://min-api.cryptocompare.com/data/pricemultifull"
        
        # This endpoint expects tickers as 'fsyms' and quote currencies as 'tsyms'
        base_symbols = list(set([i.split('-')[0] for i in instruments]))
        quote_symbols = list(set([i.split('-')[1] for i in instruments]))
        
        params = {
            "fsyms": ",".join(base_symbols),
            "tsyms": ",".join(quote_symbols),
            "api_key": coindesk_api_key # CryptoCompare uses 'api_key' param
        }
        
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        
        json_data = response.json()

        # Check for errors in the API response itself
        if 'Response' in json_data and json_data['Response'] == 'Error':
            return f"CryptoCompare API returned an error: {json_data.get('Message', 'Unknown error')}"
        
        # --- THIS IS THE NEW, CORRECT PARSING LOGIC ---
        raw_data = json_data.get("RAW", {})
        if not raw_data:
            return f"Could not find RAW data for instruments {instruments_query}."
        
        formatted_ticks = []
        # The data is nested: RAW -> BTC -> USD -> {data}
        for base_sym, quote_map in raw_data.items():
            for quote_sym, data in quote_map.items():
                instrument_name = f"{base_sym}-{quote_sym}"
                
                # Extract the correct fields
                price = data.get('PRICE', 'N/A')
                last_update_ts = data.get('LASTUPDATE', 0)
                
                # Format them
                dt_object = datetime.fromtimestamp(last_update_ts)
                time_str = dt_object.strftime('%Y-%m-%d %H:%M:%S UTC')
                formatted_price = f"${price:,.2f}" if isinstance(price, (float, int)) else "N/A"

                formatted_ticks.append(
                    f"Instrument: {instrument_name}\n"
                    f"- Price: {formatted_price}\n"
                    f"- Last Updated: {time_str}"
                )

        if not formatted_ticks:
            return f"No valid data parsed for instruments: {instruments_query}"

        return "CryptoCompare Data:\n\n" + "\n\n---\n\n".join(formatted_ticks)

    except requests.exceptions.HTTPError as http_err:
        return f"CryptoCompare API request failed: {http_err}. Response: {response.text}"
    except Exception as e:
        return f"An error occurred in the CryptoCompare tool: {e}"