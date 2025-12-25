# backend/core/data_sources/sec_edgar.py

from sec_api import QueryApi
from backend.config import settings
# 1. Import the flexible entity extraction function
from .yahoo_finance import extract_financial_entities

query_api = QueryApi(api_key=settings.SEC_API_KEY)

def get_sec_filings(query: str, api_key: str) -> str:
    """
    Fetches recent SEC filings for ALL tickers identified in the query by the central AI extractor.
    """
    # 2. Use the central extractor to get the tickers
    entities = extract_financial_entities(query, api_key)
    ticker_symbols = entities.get("tickers", [])
    
    if not ticker_symbols:
        return "Could not identify any stock tickers to search for SEC filings."
        
    print(f"[SEC Edgar] Identified tickers: {ticker_symbols}")

    # Determine filing type from query using simple keyword matching
    form_type = "\"10-K\" OR \"10-Q\""
    query_lower = query.lower()
    if "10-k" in query_lower or "annual report" in query_lower:
        form_type = "\"10-K\""
    elif "10-q" in query_lower or "quarterly report" in query_lower:
        form_type = "\"10-Q\""
    elif "8-k" in query_lower:
        form_type = "\"8-K\""

    all_filing_summaries = []
    # 3. Loop through each identified ticker
    for ticker in ticker_symbols:
        try:
            api_query = {
                "query": {"query_string": {"query": f"ticker:{ticker} AND formType:({form_type})"}},
                "from": "0", "size": "3",
                "sort": [{"filedAt": {"order": "desc"}}]
            }

            response = query_api.get_filings(api_query)
            filings = response.get('filings', [])

            if not filings:
                all_filing_summaries.append(f"No recent {form_type.replace('\"', '')} filings found for {ticker} via SEC API.")
                continue

            ticker_header = f"Latest SEC filings for {ticker}:\n"
            filing_details = [
                f"Filing: {filing['formType']} for {filing['companyName']}\n"
                f"Filed On: {filing['filedAt'][:10]}\n"
                f"Link: {filing['linkToFilingDetails']}"
                for filing in filings
            ]
            all_filing_summaries.append(ticker_header + "\n\n".join(filing_details))

        except Exception as e:
            all_filing_summaries.append(f"Error fetching SEC filings for {ticker}: {e}")

    # 4. Aggregate the results
    return "\n\n---\n\n".join(all_filing_summaries)