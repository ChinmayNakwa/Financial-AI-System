# specific to the US market
# backend/core/data_sources/fred.py
from fredapi import Fred
from backend.config import settings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FREDDataSource:
    """
    Enhanced FRED data source with improved error handling, flexibility, and reliability.
    """
    
    def __init__(self):
        # Initialize FRED API
        try:
            self.fred = Fred(api_key=settings.FRED_API_KEY)
            logger.info("FRED API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FRED API: {e}")
            raise
            
        # Initialize LLM (optional for fallback)
        self.llm = None
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite", 
                google_api_key=settings.GOOGLE_API_KEY, 
                temperature=0
            )
            logger.info("Gemini LLM initialized successfully")
        except Exception as e:
            logger.warning(f"Gemini LLM could not be configured: {e}")
        
        # Enhanced primary series mapping with aliases
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
        
        # Special handling configurations
        self.SPECIAL_CALCULATIONS = {
            "CPIAUCSL": self._calculate_inflation_rate,
            "CPILFESL": self._calculate_inflation_rate,
            "PCEPI": self._calculate_inflation_rate,
            "PCEPILFE": self._calculate_inflation_rate
        }
        
        # LLM chooser prompt
        self.chooser_prompt = ChatPromptTemplate.from_template(
            """You are an expert economist selecting the best FRED data series.
            
USER QUESTION: "{query}"

AVAILABLE SERIES:
{search_results}

SELECTION CRITERIA:
1. Choose the most commonly used, headline economic indicator
2. Prioritize seasonally adjusted data when available
3. Avoid regional, experimental, or highly specialized series
4. For inflation: prefer CPIAUCSL (Consumer Price Index)
5. For unemployment: prefer UNRATE (Unemployment Rate)
6. For interest rates: prefer DFF (Federal Funds Rate)
7. For GDP: prefer GDP (Gross Domestic Product)

Return ONLY the series ID (no explanation):"""
        )

    def get_economic_data(self, query: str, limit: int = 10) -> str:
        """
        Main method to fetch economic data based on user query.
        
        Args:
            query: User's economic data request
            limit: Maximum number of search results to consider
            
        Returns:
            Formatted string with economic data
        """
        logger.info(f"Processing query: '{query}'")
        
        try:
            # Step 1: Check for direct keyword matches
            series_id = self._find_primary_series(query)
            
            # Step 2: If no direct match, use LLM-powered search
            if not series_id:
                series_id = self._llm_search_and_select(query, limit)
            
            # Step 3: Fetch and format data
            if series_id:
                return self._fetch_and_format_data(series_id, query)
            else:
                return f"Could not find relevant economic data for: '{query}'"
                
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return f"An error occurred while fetching economic data: {str(e)}"

    def _find_primary_series(self, query: str) -> Optional[str]:
        """Find series ID from primary mapping."""
        query_lower = query.lower().strip()
        
        # Direct exact matches first
        if query_lower in self.PRIMARY_SERIES_MAP:
            series_id = self.PRIMARY_SERIES_MAP[query_lower]
            logger.info(f"Direct match found: '{query_lower}' -> {series_id}")
            return series_id
        
        # Partial matches
        for keyword, series_id in self.PRIMARY_SERIES_MAP.items():
            if keyword in query_lower:
                logger.info(f"Partial match found: '{keyword}' in '{query_lower}' -> {series_id}")
                return series_id
                
        return None

    def _llm_search_and_select(self, query: str, limit: int) -> Optional[str]:
        """Use LLM to search and select best series."""
        if not self.llm:
            logger.warning("LLM not available for search")
            return None
            
        try:
            # Search FRED
            logger.info(f"Searching FRED for: '{query}'")
            search_results = self.fred.search(query, limit=limit)
            
            if search_results is None or search_results.empty:
                logger.warning(f"No FRED search results for: '{query}'")
                return None
            
            # Prepare results for LLM
            results_text = self._format_search_results(search_results)
            
            # Use LLM to choose
            logger.info("Using LLM to select best series")
            response = (self.chooser_prompt | self.llm).invoke({
                "query": query, 
                "search_results": results_text
            })
            
            chosen_id = response.content.strip()
            
            # Validate choice
            if chosen_id in search_results['id'].values:
                logger.info(f"LLM selected valid series: {chosen_id}")
                return chosen_id
            else:
                logger.warning(f"LLM selected invalid series: {chosen_id}")
                # Fallback to first result
                return search_results.iloc[0]['id']
                
        except Exception as e:
            logger.error(f"LLM search failed: {e}")
            return None

    def _format_search_results(self, search_results: pd.DataFrame) -> str:
        """Format search results for LLM consumption."""
        formatted_results = []
        for _, row in search_results.iterrows():
            formatted_results.append(
                f"ID: {row['id']} | Title: {row['title']} | Units: {row.get('units', 'N/A')}"
            )
        return "\n".join(formatted_results)

    def _fetch_and_format_data(self, series_id: str, original_query: str) -> str:
        """Fetch data and format output."""
        try:
            logger.info(f"Fetching data for series: {series_id}")
            
            # Get series information
            series_info = self._get_series_info(series_id)
            
            # Check if special calculation is needed
            if series_id in self.SPECIAL_CALCULATIONS:
                return self.SPECIAL_CALCULATIONS[series_id](series_id, series_info, original_query)
            else:
                return self._standard_data_fetch(series_id, series_info)
                
        except Exception as e:
            logger.error(f"Failed to fetch data for {series_id}: {e}")
            return f"Error fetching data for series {series_id}: {str(e)}"

    def _get_series_info(self, series_id: str) -> Dict[str, Any]:
        """Get series metadata with error handling."""
        try:
            return self.fred.get_series_info(series_id)
        except Exception as e:
            logger.warning(f"Could not get series info for {series_id}: {e}")
            return {"title": series_id, "units": "Unknown", "units_short": ""}

    def _standard_data_fetch(self, series_id: str, series_info: Dict[str, Any]) -> str:
        """Standard data fetching and formatting."""
        try:
            # Get latest data (last 2 years for context)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            data = self.fred.get_series(
                series_id, 
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=end_date.strftime('%Y-%m-%d')
            )
            
            if data.empty:
                return f"No recent data available for series {series_id}"
            
            # Get latest value and date
            latest_value = data.iloc[-1]
            latest_date = data.index[-1].strftime('%Y-%m-%d')
            
            # Calculate change if possible
            change_info = ""
            if len(data) >= 2:
                previous_value = data.iloc[-2]
                change = latest_value - previous_value
                change_pct = (change / previous_value) * 100 if previous_value != 0 else 0
                change_info = f"\n- Change from previous: {change:+,.2f} ({change_pct:+.2f}%)"
            
            # Format output
            title = series_info.get('title', series_id)
            units = series_info.get('units_short', series_info.get('units', ''))
            
            return (
                f"FRED Data for '{title}' ({series_id}):\n"
                f"- Latest Value: {latest_value:,.2f} {units}\n"
                f"- As of: {latest_date}"
                f"{change_info}"
            )
            
        except Exception as e:
            raise Exception(f"Standard fetch failed for {series_id}: {e}")

    def _calculate_inflation_rate(self, series_id: str, series_info: Dict[str, Any], original_query: str) -> str:
        """Calculate year-over-year inflation rate."""
        try:
            # Get 14 months of data to ensure we have year-over-year comparison
            end_date = datetime.now()
            start_date = end_date - timedelta(days=450)  # ~15 months
            
            data = self.fred.get_series(
                series_id,
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=end_date.strftime('%Y-%m-%d')
            )
            
            if len(data) < 13:
                return f"Insufficient data for inflation calculation (need 13+ months, got {len(data)})"
            
            # Calculate year-over-year inflation
            latest_value = data.iloc[-1]
            year_ago_value = data.iloc[-13]  # 12 months ago
            
            inflation_rate = ((latest_value - year_ago_value) / year_ago_value) * 100
            latest_date = data.index[-1].strftime('%Y-%m-%d')
            
            # Get series name for output
            title = series_info.get('title', 'Consumer Price Index')
            
            return (
                f"FRED Data for US Inflation Rate (from {title}):\n"
                f"- Year-over-Year Inflation: {inflation_rate:.2f}%\n"
                f"- Current Index Level: {latest_value:.1f}\n"
                f"- As of: {latest_date}\n"
                f"- Series ID: {series_id}"
            )
            
        except Exception as e:
            raise Exception(f"Inflation calculation failed for {series_id}: {e}")


# Initialize global instance
fred_data_source = FREDDataSource()

def get_economic_data(query: str) -> str:
    """
    Public interface function for getting economic data.
    
    Args:
        query: User's economic data request
        
    Returns:
        Formatted string with economic data
    """
    return fred_data_source.get_economic_data(query)