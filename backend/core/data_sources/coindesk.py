from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

import requests 
from backend.config import settings
from datetime import datetime

coindesk_api_key = settings.COINDESK_API_KEY
google_client = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", api_key=settings.GOOGLE_API_KEY
)


def get_instruments(prompt: str) -> list[str]:
    instrument_prompt = f"""From the user's financial question below, extract the cryptocurrency instrument tickers.
- Format: BASE-QUOTE (BTC-USD, ETH-EUR).
- Default quote currency: USD.
- If general query, return: BTC-USD,ETH-USD,SOL-USD,XRP-USD,DOGE-USD.
- For "price of ethereum", return ETH-USD.

User Question: "{prompt}"

Return only a comma-separated list of tickers.
"""
    chat_prompt = ChatPromptTemplate.from_template(instrument_prompt)
    response = google_client.generate(
        [{"role": "user", "content": chat_prompt.format()}]
    )

    try:
        gen = response.generations
        if isinstance(gen, list) and gen and isinstance(gen[0], list):
            instruments_text = gen[0][0].text
        else:
            instruments_text = gen[0].text
    except Exception:
        instruments_text = str(response)

    return [i.strip().upper() for i in instruments_text.split(",") if i.strip()]


def get_latest_tick_data(prompt: str) -> str:
    try:
        instruments = get_instruments(prompt)
        if not instruments:
            return "Could not identify any cryptocurrency instruments in the query."

        instruments_query = ",".join(instruments)

        api_url = "https://min-api.cryptocompare.com/data/pricemultifull"
        base_symbols = list(set([i.split('-')[0] for i in instruments]))
        quote_symbols = list(set([i.split('-')[1] for i in instruments]))

        params = {
            "fsyms": ",".join(base_symbols),
            "tsyms": ",".join(quote_symbols),
            "api_key": coindesk_api_key
        }
        
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        json_data = response.json()

        if json_data.get("Response") == "Error":
            return f"CryptoCompare API returned an error: {json_data.get('Message')}"

        raw_data = json_data.get("RAW", {})
        if not raw_data:
            return f"Could not find RAW data for instruments {instruments_query}."

        formatted = []
        for base_sym, quote_map in raw_data.items():
            for quote_sym, data in quote_map.items():
                instrument_name = f"{base_sym}-{quote_sym}"
                price = data.get("PRICE", "N/A")
                ts = data.get("LASTUPDATE", 0)
                dt_object = datetime.fromtimestamp(ts)
                time_str = dt_object.strftime('%Y-%m-%d %H:%M:%S UTC')
                price_str = f"${price:,.2f}" if isinstance(price, (float, int)) else "N/A"

                formatted.append(
                    f"Instrument: {instrument_name}\n"
                    f"- Price: {price_str}\n"
                    f"- Last Updated: {time_str}"
                )

        return "CryptoCompare Data:\n\n" + "\n\n---\n\n".join(formatted)

    except requests.exceptions.HTTPError as http_err:
        return f"CryptoCompare API request failed: {http_err}. Response: {response.text}"
    except Exception as e:
        return f"An error occurred in the CryptoCompare tool: {e}"
