#backend\core\rag\adaptive_rag.py

# from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import settings
mistral_model = "mistral-large-latest" 
# llm = ChatMistralAI(model=mistral_model, temperature=0, api_key=settings.MISTRAL_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.5, api_key=settings.GOOGLE_API_KEY)
    
import sys
import sys
from pathlib import Path

# Add the root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Literal, List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

class RouteQuery(BaseModel):
    """Enhanced routing for specific financial data sources"""
    
    primary_datasource: Literal[
        "yahoo_finance",      # Stock prices, company info, historical data
        "polygon_io",      # Technical indicators, intraday data
        "fred",               # Economic indicators, interest rates
        "newsapi",            # Financial news and sentiment
        "tavily",             # General web search and research
        "sec_edgar",          # Official company filings (recommended addition)
        "coindesk"            # Cryptocurrency data
    ] = Field(
        ...,
        description="Choose the PRIMARY data source based on the user's question type"
    )
    
    secondary_sources: List[Literal[
        "yahoo_finance", "polygon_io", "fred", "newsapi", "tavily", "sec_edgar", "coindesk"
    ]] = Field(
        default=[],
        description="Additional sources for cross-validation and completeness"
    )
    
    query_type: Literal[
        "stock_price",        # Current/historical stock prices
        "company_analysis",   # Company fundamentals, earnings, ratios
        "technical_analysis", # Charts, indicators, patterns
        "economic_data",      # GDP, inflation, interest rates
        "market_news",        # Breaking news, earnings announcements
        "sector_analysis",    # Industry trends, comparisons
        "general_research"    # Broad financial topics
    ] = Field(
        ...,
        description="Categorize the type of financial query"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Confidence in routing decision (0.0 to 1.0)"
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

router_instructions = """
You are an expert financial data routing engine. Your primary function is to analyze a user's financial question and create a structured routing plan. You must reason about the user's intent and select the optimal primary and secondary data sources to provide a complete and accurate answer.

## Your Reasoning Process (Chain of Thought):
1.  **Deconstruct the Query:** What is the core question? What specific entities (stocks, economic indicators, cryptocurrencies) are mentioned?
2.  **Identify Information Needs:** What specific data points are required to answer this question (e.g., price data, news articles, official filings, technical indicators)?
3.  **Select the Primary Source:** Based on the most critical information need, choose the single best data source from the list below.
4.  **Select Secondary Sources:** What other sources could enrich, validate, or provide necessary context to the primary data? For example, if the primary source is price data, a secondary source might be news to explain price movements.
5.  **Categorize the Query:** Assign a `query_type` that best represents the user's overall goal.
6.  **Assign Confidence:** Score your confidence in the routing plan based on the query's clarity and how well it maps to the available data sources.

## Data Source Capabilities:

### yahoo_finance 🏢
- **Use for:** Stock prices (real-time & historical), basic company fundamentals (Market Cap, P/E), dividend data.
- **Best for:** "What's Apple's stock price?", "Show me Tesla's P/E ratio."

### polygon_io 📊 
- **Use for:** Advanced technical indicators (RSI, MACD), intraday trading data, and foreign exchange (FX) rates.
- **Best for:** "What's Apple's 50-day moving average?", "Technical analysis of TSLA."

### fred 🏛️
- **Use for:** Macroeconomic data from the Federal Reserve.
- **Best for:** "What is the current US inflation rate?", "Show me historical GDP growth."

### sec_edgar 📋
- **Use for:** Official, detailed financial reports filed with the SEC (10-K, 10-Q, 8-K).
- **Best for:** "Pull the revenue numbers from Apple's latest 10-Q filing.", "Get Tesla's annual report."

### newsapi 📰
- **Use for:** The latest financial news, press releases, and market sentiment analysis.
- **Best for:** "Latest news about the tech sector?", "Why did NVIDIA's stock drop yesterday?"

### coindesk 🪙
- **Use for:** Cryptocurrency prices (Bitcoin, Ethereum, etc.), crypto market news, and analysis of the digital asset space.
- **Best for:** "What is the price of Bitcoin?", "Latest news on Ethereum."

### tavily 🌐
- **Use for:** latest financial news, General web searches for broad, explanatory, or qualitative questions that don't fit other specialized APIs.
- **Best for:** "Explain what quantitative easing is.", "What are the main ESG investing strategies?", "List out the top companies in this field"

## Routing Heuristics (Guidelines):

- **Specific Stock Price/Fundamentals:** Prioritize `yahoo_finance`. For official, deep financials, use `sec_edgar` as a secondary source. For technicals, add `alpha_vantage`.
- **Complex Company Analysis:** For broad questions like "Is Apple a good investment?", you MUST combine sources. Start with `yahoo_finance` (fundamentals) and `sec_edgar` (official data), and add `newsapi` (sentiment) and `alpha_vantage` (technicals) as secondary.
- **Economic Questions:** Always use `fred` as the primary. Use `newsapi` for secondary context on *why* the economy is behaving a certain way.
- **News-Driven Queries:** Start with `newsapi`. If the news event concerns a specific company's stock, add `yahoo_finance` as a secondary source to correlate the news with price action.
- **Cryptocurrency Queries:** Prioritize `coindesk` for all crypto-related price, news, and analysis questions. Use `tavily` for broader concepts if needed.
- **Ambiguous or Broad Queries:** If the user's intent is unclear or very general, default to `tavily` as the primary source to get a general overview, and assign a lower confidence score (e.g., < 0.7).

## Examples:

"What's the current inflation rate?"
→ primary: fred, secondary: [newsapi], query_type: economic_data, confidence: 0.95

"What is the price of Bitcoin and what's the latest news about it?"
→ primary: coindesk, secondary: [newsapi], query_type: market_news, confidence: 0.98

"Give me a complete financial overview of Microsoft."
→ primary: yahoo_finance, secondary: [sec_edgar, newsapi, alpha_vantage], query_type: company_analysis, confidence: 0.9

"Tell me about the semiconductor industry."
→ primary: tavily, secondary: [newsapi], query_type: sector_analysis, confidence: 0.8

"Show me GOOG's latest annual report."
→ primary: sec_edgar, secondary: [], query_type: company_analysis, confidence: 1.0
"""

def route_financial_query(user_question: str) -> RouteQuery:
    """Route a user's financial question to appropriate data sources"""
    
    message = HumanMessage(content=f"Route this financial question: {user_question}")
    system_msg = SystemMessage(content=router_instructions)
    
    response = structured_llm_router.invoke([system_msg, message])
    return response

# Usage example

# def adaptive_rag():
#     # Test queries
#     test_queries = [
#         "What's Apple's current stock price?",
#         "Is Tesla a good investment right now?", 
#         "What's the current inflation rate in the US?",
#         "Show me technical analysis for Microsoft",
#         "Latest news about Amazon earnings",
#         "What is ESG investing and why is it important?"
#     ]
    
#     for query in test_queries:
#         result = route_financial_query(query)
#         print(f"Query: {query}")
#         print(f"Primary: {result.primary_datasource}")
#         print(f"Secondary: {result.secondary_sources}")
#         print(f"Type: {result.query_type}")
#         print(f"Confidence: {result.confidence}")
#         print("-" * 50)
