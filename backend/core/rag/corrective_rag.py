# backend/core/rag/corrective_rag.py

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import settings
# from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

# llm = ChatMistralAI(model="mistral-large-latest", temperature=0, api_key=settings.MISTRAL_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, api_key=settings.GOOGLE_API_KEY)
    

class FactCheckResult(BaseModel):
    """Results of cross-verifying financial data"""
    consistent: bool = Field(..., description="Do sources agree on core facts?")
    consensus_score: float = Field(..., ge=0.0, le=1.0, description="Degree of agreement between sources")
    reliable_sources: List[str] = Field(..., description="Which sources were most trustworthy")
    final_value: Optional[str] = Field(..., description="Resolved value after verification")
    discrepancies: List[str] = Field(default=[], description="Any inconsistencies found")

structured_fact_checker = llm.with_structured_output(FactCheckResult)

fact_check_instructions = """
You are a financial data reconciliation expert. Compare information from multiple sources:

1. Identify core facts that should match (prices, dates, figures)
2. Note any significant discrepancies (>2% difference for numbers)
3. Determine most reliable sources (prioritize official filings)
4. Calculate consensus score:
   - 1.0 = perfect agreement
   - 0.8 = minor differences
   - 0.5 = conflicting info
5. Resolve final value based on most reliable sources
"""

def verify_facts(sources: List[Dict[str, str]], query: str) -> FactCheckResult:
    """Cross-check financial information from multiple sources"""
    sources_text = "\n".join(
        f"Source {i+1} ({s['source']}): {s['content'][:1000]}"
        for i, s in enumerate(sources)
    )
    
    message = HumanMessage(
        content=f"Query: {query}\n\nSources:\n{sources_text}"
    )
    system_msg = SystemMessage(content=fact_check_instructions)
    
    return structured_fact_checker.invoke([system_msg, message])