# specific to the US market
# backend/core/data_sources/fred.py

"""
FRED Data Source Module
-----------------------
This module provides an interface to the Federal Reserve Economic Data (FRED) API.
It utilizes a hybrid approach for data retrieval:
1. Static Mapping: Direct lookup for common economic indicators (CPI, GDP, etc.)
2. Agentic Search: Uses a LLM to search the FRED database when a query is ambiguous.

The LLM is initialized dynamically per-request to support user-provided API keys,
ensuring that the backend does not rely on a hardcoded system key for model inference.
"""

from fredapi import Fred
from backend.config import settings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging

# Configure logging for the FRED module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FREDDataSource:
    """
    Enhanced FRED data source with improved error handling, flexibility, and reliability.
    
    This class manages the connection to the FRED API and provides methods to
    retrieve, calculate, and format economic data. It supports dynamic LLM
    initialization to allow users to provide their own Gemini API keys.
    """
    
    def __init__(self):
        """
        Initializes the FRED API client using the system-level FRED API key.
        The LLM is not initialized here as it requires a dynamic user key.
        """
        # Initialize FRED API using the key from the server configuration
        try:
            self.fred = Fred(api_key=settings.FRED_API_KEY)
            logger.info("FRED API initialized successfully using system settings.")
        except Exception as e:
            logger.error(f"Failed to initialize FRED API: {e}")
            raise
            
        # Enhanced primary series mapping with aliases for common economic terms.
        # This mapping allows for instant retrieval without LLM overhead for 
        # standard queries.
        self.PRIMARY_SERIES_MAP = {
            # Inflation indicators
            "inflation": "CPIAUCSL",
            "cpi": "CPIAUCSL",
            "consumer price index": "CPIAUCSL",
            "core inflation": "CPILFESL",
            "pce": "PCEPI",
            "core pce": "PCEPILFE",
            
            # Employment indicators
            "unemployment": "UNRATE",
            "unemployment rate": "UNRATE",
            "jobless rate": "UNRATE",
            "employment": "PAYEMS",
            "jobs": "PAYEMS",
            "nonfarm payrolls": "PAYEMS",
            
            # Interest rates
            "interest rate": "DFF",
            "fed funds": "DFF",
            "federal funds rate": "DFF",
            "fed rate": "DFF",
            "10 year treasury": "DGS10",
            "10y treasury": "DGS10",
            "30 year mortgage": "MORTGAGE30US",
            
            # GDP and growth
            "gdp": "GDP",
            "gross domestic product": "GDP",
            "economic growth": "GDP",
            "real gdp": "GDPC1",
            
            # Other key indicators
            "housing starts": "HOUST",
            "retail sales": "RSAFS",
            "industrial production": "INDPRO",
            "consumer confidence": "UMCSENT",
            "durable goods": "DGORDER"
        }
        
        # Special handling configurations for indicators that require 
        # year-over-year (YoY) calculations rather than just the latest value.
        self.SPECIAL_CALCULATIONS = {
            "CPIAUCSL": self._calculate_inflation_rate,
            "CPILFESL": self._calculate_inflation_rate,
            "PCEPI": self._calculate_inflation_rate,
            "PCEPILFE": self._calculate_inflation_rate
        }
        
        # LLM chooser prompt used to select the best FRED series ID from 
        # a list of search results.
        self.chooser_prompt = ChatPromptTemplate.from_template(
            """You are an expert economist selecting the best FRED data series ID.
            
USER QUESTION: "{query}"

AVAILABLE SERIES FROM FRED SEARCH:
{search_results}

SELECTION CRITERIA:
1. Choose the most commonly used, headline economic indicator.
2. Prioritize seasonally adjusted data when available.
3. Avoid regional, experimental, or highly specialized series.
4. For inflation: prefer CPIAUCSL (Consumer Price Index).
5. For unemployment: prefer UNRATE (Unemployment Rate).
6. For interest rates: prefer DFF (Federal Funds Rate).
7. For GDP: prefer GDP (Gross Domestic Product).

Return ONLY the series ID (e.g., UNRATE) with no explanation:"""
        )

    def get_economic_data(self, query: str, google_api_key: str, limit: int = 10) -> str:
        """
        Main method to fetch economic data based on user query.
        
        Args:
            query: User's economic data request.
            google_api_key: The user-provided Google Gemini API key.
            limit: Maximum number of search results to consider if LLM is used.
            
        Returns:
            Formatted string with economic data or an error message.
        """
        logger.info(f"Processing FRED query: '{query}'")
        
        try:
            # Step 1: Check for direct keyword matches in the static map
            # This is the fastest path and avoids LLM usage.
            series_id = self._find_primary_series(query)
            
            # Step 2: If no direct match is found, use LLM-powered search.
            # We pass the google_api_key to initialize the LLM locally.
            if not series_id:
                logger.info("No static match found. Proceeding to LLM-powered FRED search.")
                series_id = self._llm_search_and_select(query, google_api_key, limit)
            
            # Step 3: Fetch the data for the identified series and format it.
            if series_id:
                return self._fetch_and_format_data(series_id, query)
            else:
                return f"Could not find relevant economic data for: '{query}'"
                
        except Exception as e:
            logger.error(f"Error processing FRED query '{query}': {e}")
            return f"An error occurred while fetching economic data: {str(e)}"

    def _find_primary_series(self, query: str) -> Optional[str]:
        """
        Performs a keyword-based search in the PRIMARY_SERIES_MAP.
        
        Args:
            query: The user's query string.
            
        Returns:
            A FRED series ID if a match is found, else None.
        """
        query_lower = query.lower().strip()
        
        # Direct exact matches first
        if query_lower in self.PRIMARY_SERIES_MAP:
            series_id = self.PRIMARY_SERIES_MAP[query_lower]
            logger.info(f"Direct match found: '{query_lower}' -> {series_id}")
            return series_id
        
        # Partial keyword matches
        for keyword, series_id in self.PRIMARY_SERIES_MAP.items():
            if keyword in query_lower:
                logger.info(f"Partial match found: '{keyword}' in '{query_lower}' -> {series_id}")
                return series_id
                
        return None

    def _llm_search_and_select(self, query: str, google_api_key: str, limit: int) -> Optional[str]:
        """
        Uses a locally initialized LLM to search FRED and select the best series.
        
        Args:
            query: The user's query.
            google_api_key: User's API key for Gemini.
            limit: Search result limit.
            
        Returns:
            The selected series ID or None.
        """
        if not google_api_key:
            logger.warning("Google API Key missing. Skipping LLM-powered search.")
            return None
            
        try:
            # Initialize the LLM locally for this specific request.
            # This ensures we use the user's provided key.
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                google_api_key=google_api_key, 
                temperature=0
            )
            
            # Perform a search using the FRED API
            logger.info(f"Searching FRED API for: '{query}'")
            search_results = self.fred.search(query, limit=limit)
            
            if search_results is None or search_results.empty:
                logger.warning(f"No FRED search results returned for: '{query}'")
                return None
            
            # Format the search results into a text block for the LLM
            results_text = self._format_search_results(search_results)
            
            # Use the LLM to choose the best series from the results
            logger.info("Invoking LLM to select the most relevant economic series.")
            chain = self.chooser_prompt | llm
            response = chain.invoke({
                "query": query, 
                "search_results": results_text
            })
            
            chosen_id = response.content.strip()
            
            # Validate that the LLM's choice exists in our search results
            if chosen_id in search_results['id'].values:
                logger.info(f"LLM successfully selected series: {chosen_id}")
                return chosen_id
            else:
                logger.warning(f"LLM selected an ID not in search results: {chosen_id}. Falling back to top result.")
                # Fallback to the first (most relevant) result from FRED search
                return search_results.iloc[0]['id']
                
        except Exception as e:
            logger.error(f"LLM-powered FRED search failed: {e}")
            return None

    def _format_search_results(self, search_results: pd.DataFrame) -> str:
        """
        Formats search results into a readable string for LLM processing.
        
        Args:
            search_results: DataFrame containing FRED search results.
            
        Returns:
            A formatted string of IDs and Titles.
        """
        formatted_results = []
        for _, row in search_results.iterrows():
            formatted_results.append(
                f"ID: {row['id']} | Title: {row['title']} | Units: {row.get('units', 'N/A')}"
            )
        return "\n".join(formatted_results)

    def _fetch_and_format_data(self, series_id: str, original_query: str) -> str:
        """
        Fetches the numerical data for a series and applies formatting.
        
        Args:
            series_id: The FRED ID to fetch.
            original_query: The user's original question.
            
        Returns:
            A formatted data string.
        """
        try:
            logger.info(f"Fetching data for series: {series_id}")
            
            # Retrieve metadata for the series
            series_info = self._get_series_info(series_id)
            
            # Check if this series requires a special calculation (like YoY Inflation)
            if series_id in self.SPECIAL_CALCULATIONS:
                return self.SPECIAL_CALCULATIONS[series_id](series_id, series_info, original_query)
            else:
                # Otherwise, perform a standard data fetch
                return self._standard_data_fetch(series_id, series_info)
                
        except Exception as e:
            logger.error(f"Failed to fetch or format data for {series_id}: {e}")
            return f"Error retrieving data for series {series_id}: {str(e)}"

    def _get_series_info(self, series_id: str) -> Dict[str, Any]:
        """
        Fetches metadata for a specific series ID.
        
        Args:
            series_id: The FRED ID.
            
        Returns:
            A dictionary of series metadata.
        """
        try:
            return self.fred.get_series_info(series_id)
        except Exception as e:
            logger.warning(f"Could not retrieve metadata for {series_id}: {e}")
            return {"title": series_id, "units": "Unknown", "units_short": ""}

    def _standard_data_fetch(self, series_id: str, series_info: Dict[str, Any]) -> str:
        """
        Performs a standard data fetch for the latest value and recent change.
        
        Args:
            series_id: The FRED ID.
            series_info: Metadata dictionary.
            
        Returns:
            A formatted string with the latest economic data.
        """
        try:
            # Retrieve the last 2 years of data to provide context and calculate change
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            data = self.fred.get_series(
                series_id, 
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=end_date.strftime('%Y-%m-%d')
            )
            
            if data.empty:
                return f"No recent data points available for series {series_id}."
            
            # Extract the most recent value and its date
            latest_value = data.iloc[-1]
            latest_date = data.index[-1].strftime('%Y-%m-%d')
            
            # Calculate the change from the previous observation if available
            change_info = ""
            if len(data) >= 2:
                previous_value = data.iloc[-2]
                change = latest_value - previous_value
                change_pct = (change / previous_value) * 100 if previous_value != 0 else 0
                change_info = f"\n- Change from previous: {change:+,.2f} ({change_pct:+.2f}%)"
            
            # Prepare the final display strings
            title = series_info.get('title', series_id)
            units = series_info.get('units_short', series_info.get('units', ''))
            
            return (
                f"FRED Data for '{title}' ({series_id}):\n"
                f"- Latest Value: {latest_value:,.2f} {units}\n"
                f"- As of: {latest_date}"
                f"{change_info}"
            )
            
        except Exception as e:
            logger.error(f"Standard data fetch failed for {series_id}: {e}")
            raise Exception(f"Standard fetch failed for {series_id}: {e}")

    def _calculate_inflation_rate(self, series_id: str, series_info: Dict[str, Any], original_query: str) -> str:
        """
        Calculates the Year-over-Year (YoY) inflation rate for price indices.
        
        Args:
            series_id: The FRED ID (e.g., CPIAUCSL).
            series_info: Metadata dictionary.
            original_query: The user's query.
            
        Returns:
            A formatted string showing the calculated inflation rate.
        """
        try:
            # We need at least 13 months of data to calculate a 12-month YoY change
            end_date = datetime.now()
            start_date = end_date - timedelta(days=450)
            
            data = self.fred.get_series(
                series_id,
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=end_date.strftime('%Y-%m-%d')
            )
            
            if len(data) < 13:
                return f"Insufficient data for inflation calculation (need 13+ observations, found {len(data)})."
            
            # Calculate YoY inflation: ((Current / YearAgo) - 1) * 100
            latest_value = data.iloc[-1]
            year_ago_value = data.iloc[-13]
            
            inflation_rate = ((latest_value - year_ago_value) / year_ago_value) * 100
            latest_date = data.index[-1].strftime('%Y-%m-%d')
            
            title = series_info.get('title', 'Consumer Price Index')
            
            return (
                f"FRED Data for US Inflation Rate (derived from {title}):\n"
                f"- Year-over-Year Inflation: {inflation_rate:.2f}%\n"
                f"- Current Index Level: {latest_value:.2f}\n"
                f"- As of: {latest_date}\n"
                f"- Series ID: {series_id}"
            )
            
        except Exception as e:
            logger.error(f"Inflation calculation failed for {series_id}: {e}")
            raise Exception(f"Inflation calculation failed for {series_id}: {e}")


# Initialize the global FRED data source instance.
# This instance is shared across the application.
fred_data_source = FREDDataSource()

def get_economic_data(query: str, google_api_key: str) -> str:
    """
    Public interface function for the LangGraph workflow.
    
    Args:
        query: The user's economic data request.
        google_api_key: The user-provided Google Gemini API key.
        
    Returns:
        A formatted string containing the requested economic data.
    """
    # Delegate the request to the global FREDDataSource instance.
    return fred_data_source.get_economic_data(query, google_api_key)

