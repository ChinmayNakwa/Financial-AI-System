# backend/core/rag/self_rag.py

from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import settings
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re

# Use a more stable model for structured output
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    temperature=0,  # Lower temperature for more consistent JSON
    api_key=settings.GOOGLE_API_KEY
)

class QualityCheck(BaseModel):
    """Results of quality assessment for financial data"""
    is_recent: bool = Field(..., description="Is the information current (within last 3 months)?")
    is_reliable: bool = Field(..., description="Is the source trustworthy?")
    is_relevant: bool = Field(..., description="Does this directly answer the query?")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in quality")
    issues: List[str] = Field(default=[], description="Any quality issues found")

# DON'T use with_structured_output - it's unreliable with Gemini
# Instead, we'll manually parse JSON from the response

quality_check_instructions = """
You are a meticulous financial data quality analyst. Your task is to evaluate if retrieved information is useful for answering the user's query.
Your System's Data Sources (These are considered RELIABLE):

yahoo_finance: Stock prices, company info, historical data
polygon_io: Technical indicators, market data for stocks
fred: US economic data, macroeconomic indicators
newsapi: Financial news articles, market updates
tavily: General web search, broader financial context
sec_edgar: Official company SEC filings, regulatory documents
coindesk: Cryptocurrency price data, crypto market news

Evaluation Criteria:
1. RELEVANCE Assessment
Does the content help answer the user's query?
Factual Data Queries:

Rule: For specific data requests (e.g., "What is Apple's stock price?", "What is Tesla's P/E ratio?"), content MUST contain the exact requested data to be relevant.
Examples:

Query: "What is AAPL's current price?" + Content: "Apple (AAPL) is trading at $150.25" → ✅ RELEVANT
Query: "What is MSFT's P/E ratio?" + Content: "Apple's financial data..." → ❌ NOT RELEVANT (wrong company)



Comparison/Analysis Queries:

Rule: For comparative or advisory queries (e.g., "Apple vs Microsoft?", "Should I invest in Tesla?"), factual data about ANY mentioned entity IS relevant for synthesis.
Examples:

Query: "Should I buy Apple or Amazon?" + Content: "Apple's Q3 revenue increased 8%..." → ✅ RELEVANT
Query: "Compare tech stocks" + Content: "Microsoft reported strong cloud growth..." → ✅ RELEVANT



Critical Exclusions:

Error messages: "Could not find ticker", "API limit exceeded", "No data available" → ❌ NEVER RELEVANT
Redirect responses: "Please try again later", "Service unavailable" → ❌ NOT RELEVANT
Empty or truncated data: Incomplete tables, cut-off sentences → ❌ NOT RELEVANT
Completely unrelated content: Sports news for financial query → ❌ NOT RELEVANT
News with "No title - Unknown" → ❌ NOT RELEVANT (data retrieval failed)

2. RECENCY Assessment
Is the data current enough for financial decision-making?
Time Sensitivity Rules:

Real-time market data (stock prices, crypto prices): Must be from today or last trading day
Financial news: Should be within last 30 days for relevance
Company earnings/reports: Within last quarter (3 months) for current relevance
Economic indicators (GDP, inflation, employment): Within last year
SEC filings: Within last year unless historical analysis is specifically requested

3. RELIABILITY Assessment
Is the source trustworthy and properly functioning?
Approved Sources: All sources in the system list above are considered reliable ✅

Red Flags for Reliability:
- Content contains "Error 500", "Rate limit exceeded", "Invalid response"
- Garbled or nonsensical data (encoding issues)
- Missing critical fields (e.g., "No title - Unknown")

4. CONFIDENCE Scoring Guidelines
High Confidence (0.8-1.0):
✅ Directly relevant to query
✅ Recent data (within appropriate timeframe)
✅ Complete, well-formatted information
✅ From approved source with no error indicators

Medium Confidence (0.5-0.79):
⚠️ Somewhat relevant but indirect
⚠️ Slightly outdated but still useful
⚠️ Minor formatting issues but data is extractable
⚠️ Partial information that contributes to answer

Low Confidence (0.0-0.49):
❌ Poor relevance to query
❌ Significantly outdated
❌ Major data quality issues
❌ Contains error messages or failed retrievals

5. Issues Identification
Document specific problems found.

YOU MUST RESPOND WITH ONLY VALID JSON IN THIS EXACT FORMAT:
{
  "is_recent": true/false,
  "is_reliable": true/false,
  "is_relevant": true/false,
  "confidence": 0.0-1.0,
  "issues": ["issue1", "issue2"]
}

DO NOT include any other text, explanations, or markdown formatting. ONLY the JSON object.
"""

def check_quality(source: str, content: str, query: str) -> QualityCheck:
    """Evaluate the quality of financial information"""
    try:
        # Check for empty or very short content
        if not content or len(content.strip()) < 10:
            print(f"[Quality Check] Content too short for {source}")
            return QualityCheck(
                is_recent=False, 
                is_reliable=False, 
                is_relevant=False, 
                confidence=0.0, 
                issues=["Content too short or empty"]
            )
        
        # Check for common error patterns in content
        error_keywords = ["error", "failed", "could not", "unavailable", "exception"]
        content_lower = content.lower()
        if any(keyword in content_lower for keyword in error_keywords) and len(content) < 200:
            print(f"[Quality Check] Error detected in content for {source}")
            return QualityCheck(
                is_recent=False, 
                is_reliable=False, 
                is_relevant=False, 
                confidence=0.0, 
                issues=["Content contains error messages"]
            )
        
        # Check for "No title - Unknown" pattern (failed news retrieval)
        if "No title - Unknown" in content:
            print(f"[Quality Check] Failed data retrieval detected in {source}")
            return QualityCheck(
                is_recent=False, 
                is_reliable=False, 
                is_relevant=False, 
                confidence=0.0, 
                issues=["News data retrieval failed - contains 'No title - Unknown'"]
            )
        
        prompt = f"""Source: {source}

            Content:
            {content[:3000]}

            Original Query: {query}

            Analyze this content and return your assessment as JSON."""
        
        response = llm.invoke([
            SystemMessage(content=quality_check_instructions),
            HumanMessage(content=prompt)
        ])
        
        response_text = response.content
        print(f"[DEBUG Quality Check] Raw LLM response for {source}: {response_text[:200]}")
        
        # Extract JSON from the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # Validate and create QualityCheck object
            result = QualityCheck(
                is_recent=data.get("is_recent", False),
                is_reliable=data.get("is_reliable", False),
                is_relevant=data.get("is_relevant", False),
                confidence=float(data.get("confidence", 0.0)),
                issues=data.get("issues", [])
            )
            
            print(f"Quality Check for {source}: Relevant={result.is_relevant}, Confidence={result.confidence}")
            return result
        else:
            print(f"[Quality Check] No JSON found in LLM response for {source}")
            return QualityCheck(
                is_recent=False, 
                is_reliable=False, 
                is_relevant=False, 
                confidence=0.0, 
                issues=["Failed to parse LLM response as JSON"]
            )
        
    except json.JSONDecodeError as je:
        print(f"[Quality Check] JSON decode error for {source}: {je}")
        return QualityCheck(
            is_recent=False, 
            is_reliable=False, 
            is_relevant=False, 
            confidence=0.0, 
            issues=[f"JSON parsing error: {str(je)}"]
        )
    except Exception as e:
        print(f"Quality check internal error for {source}: {e}")
        import traceback
        print(traceback.format_exc())
        return QualityCheck(
            is_recent=False, 
            is_reliable=False, 
            is_relevant=False, 
            confidence=0.0, 
            issues=[f"Exception: {str(e)}"]
        )