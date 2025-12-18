# backend/core/rag/financial_workflow.py

from typing import List, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
# from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import settings
from backend.core.rag.adaptive_rag import route_financial_query, RouteQuery
from backend.core.rag.self_rag import check_quality, QualityCheck
from backend.core.rag.corrective_rag import verify_facts

from core.data_sources import (
    yahoo_finance,
    # alpha_vantage,
    fred,
    newsapi,
    tavily,
    sec_edgar,
    coindesk,
    polygon,
)

# --- 1. Define Graph State ---

class Document(TypedDict):
    """Represents a retrieved document with its source and quality check."""
    source: str
    content: str
    quality_check: QualityCheck | None

class GraphState(TypedDict):
    """Represents the state of our graph."""
    user_question: str
    route: RouteQuery
    documents: List[Document]
    final_answer: str

# --- 2. Define Tool Mapping ---

# This maps the router's decision string to the actual function to call.
# The keys MUST match the Literals in adaptive_rag.py's RouteQuery.
tool_map = {
    "yahoo_finance": yahoo_finance.get_stock_data,
    # "alpha_vantage": alpha_vantage.get_technical_indicators,
    "fred": fred.get_economic_data,
    "newsapi": newsapi.get_financial_news,
    "tavily": tavily.search_web,
    "sec_edgar": sec_edgar.get_sec_filings,
    "coindesk": coindesk.get_latest_tick_data,
    "polygon_io": polygon.get_technical_indicators,
}

# --- 3. Define Graph Nodes ---

def route_query_node(state: GraphState) -> dict:
    """Node 1: Route the user's query (Adaptive RAG)."""
    print("--- NODE: ROUTING QUERY ---")
    question = state["user_question"]
    route = route_financial_query(question)
    print(f"Route: Primary -> {route.primary_datasource}, Secondary -> {route.secondary_sources}")
    return {"route": route}

def retrieve_documents_node(state: GraphState) -> dict:
    """Node 2: Retrieve documents from the chosen sources."""
    print("--- NODE: RETRIEVING DOCUMENTS ---")
    question = state["user_question"]
    route = state["route"]
    
    # Combine primary and secondary sources, removing duplicates
    all_source_names = list(set([route.primary_datasource] + route.secondary_sources))
    documents = []
    
    for source_name in all_source_names:
        if source_name in tool_map:
            tool = tool_map[source_name]
            print(f"Calling tool: {source_name}")
            content = tool(question)
            documents.append({"source": source_name, "content": content, "quality_check": None})
        else:
            print(f"Warning: Source '{source_name}' not found in tool map.")
    
    return {"documents": documents}

def quality_filter_node(state: GraphState) -> dict:
    """Node 3: Assess document quality and filter out bad ones (Self-RAG)."""
    print("--- NODE: ASSESSING DOCUMENT QUALITY ---")
    question = state["user_question"]
    documents_with_checks = []

    print("[DEBUG workflow] Content received by quality_filter_node:")
    for doc in state["documents"]:
        print(f"  - Source: {doc['source']}")
        print(f"    Content: {doc['content']}")
        print("    ---")

    for doc in state["documents"]:
        try:
            quality_check = check_quality(doc["source"], doc["content"], question)
            if quality_check and quality_check.is_relevant and quality_check.confidence >= 0.4:
                print(f"✅ Quality Check Passed for {doc['source']}: Confidence={quality_check.confidence}")
                doc_copy = doc.copy()
                doc_copy["quality_check"] = quality_check
                documents_with_checks.append(doc_copy)
            else:
                reason = "Irrelevant" if quality_check and not quality_check.is_relevant else "LLM Parsing Error/Low Confidence"
                print(f"--- FILTERED OUT: {doc['source']} ({reason}) ---")
        except Exception as e:
            print(f"--- ERROR checking quality for {doc['source']}: {e} ---")
            
    return {"documents": documents_with_checks}

def reconcile_facts_node(state: GraphState) -> dict:
    """Node 4: Reconcile facts from multiple sources (Corrective RAG)."""
    print("--- NODE: RECONCILING FACTS ---")
    question = state["user_question"]
    documents = state["documents"]
    
    try:
        sources_for_verification = [{"source": doc["source"], "content": doc["content"]} for doc in documents]
        fact_check_result = verify_facts(sources_for_verification, question)
        
        # Add None check
        if fact_check_result is None:
            print("⚠️ Fact verification returned None. Skipping reconciliation.")
            return {"documents": documents}
        
        print(f"Fact Check: Consistent={fact_check_result.consistent}, Consensus Value='{fact_check_result.final_value}'")
        
        # Create a single "reconciled" document from the fact-checker's output
        if fact_check_result.consistent and fact_check_result.final_value:
            reconciled_doc = {
                "source": f"reconciled_from_{fact_check_result.reliable_sources}",
                "content": fact_check_result.final_value,
                "quality_check": None
            }
            return {"documents": [reconciled_doc]}
        else:
            # If not consistent, pass all documents to the generator to summarize the conflict
            return {"documents": documents}
    
    except Exception as e:
        print(f"⚠️ Error in reconcile_facts_node: {e}")
        import traceback
        print(traceback.format_exc())
        # Return original documents on error
        return {"documents": documents}


def generate_answer_node(state: GraphState) -> dict:
    """Node 5: Generate a final, user-facing answer."""
    print("--- NODE: GENERATING FINAL ANSWER ---")
    question = state["user_question"]
    documents = state["documents"]
    
    try:
        # Check if we have any documents
        if not documents:
            return {"final_answer": "I apologize, but I couldn't retrieve any relevant information to answer your question. Please try rephrasing your query or checking if the data sources are available."}
        
        context = "\n\n---\n\n".join([f"Source: {doc['source']}\nContent: {doc['content']}" for doc in documents])
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, api_key=settings.GOOGLE_API_KEY)
        
        system_message = (
            "You are a financial data analyst AI. Your ONLY task is to answer a user's question based *strictly* on the data provided. You are forbidden from using any external knowledge. If the data is not present in the given document, you must state that the information is not available in the provided data. \n\n"
            "**Your Process:**\n"
            "1.  **Acknowledge the Data:** Begin your response by stating what data you have successfully retrieved from the context.\n"
            "2.  **Address Each Part of the Question:** Go through the user's question piece by piece.\n"
            "3.  **Cite Your Source:** For every piece of data you present, you must cite it directly from the provided JSON. For example: 'According to the yahoo_finance data, the trailing P/E for AAPL is X.'\n"
            "4.  **State Missing Information:** If a piece of requested information (like a specific metric or ticker) is not present in the JSON, you MUST explicitly state that it was not found in the provided data.\n"
            "5.  **Perform Analysis (If Requested):** If asked to perform analysis (like comparing growth prospects), base your analysis *only* on the numbers and facts present in the data.\n\n"
            "Failure to adhere to these rules and sourcing data from outside the provided data context will result in a failed task."
        )
        
        prompt = f"{system_message}\n\nCONTEXT:\n\n{context}\n\nBased on the context above, please answer the following question: {question}"
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        return {"final_answer": response.content}
    
    except Exception as e:
        print(f"⚠️ Error in generate_answer_node: {e}")
        import traceback
        print(traceback.format_exc())
        return {"final_answer": f"I encountered an error while generating the answer: {str(e)}. Please try again."}


# --- 4. Define Conditional Edges ---

def should_reconcile(state: GraphState) -> str:
    """Decide whether to reconcile facts or generate an answer directly."""
    print("--- DECISION: RECONCILE OR GENERATE? ---")
    filtered_documents = state.get("documents", [])
    
    if len(filtered_documents) == 0:
        print("Outcome: No high-quality documents found. Ending.")
        return "end"
    elif len(filtered_documents) > 1:
        print("Outcome: Multiple documents found. Reconciling facts.")
        return "reconcile"
    else:
        print("Outcome: Single document found. Generating answer directly.")
        return "generate"

# --- 5. Build the Graph ---

workflow = StateGraph(GraphState)
        
workflow.add_node("router", route_query_node)
workflow.add_node("retriever", retrieve_documents_node)
workflow.add_node("quality_filter", quality_filter_node)
workflow.add_node("reconciler", reconcile_facts_node)
workflow.add_node("generator", generate_answer_node)

workflow.set_entry_point("router")
workflow.add_edge("router", "retriever")
workflow.add_edge("retriever", "quality_filter")
workflow.add_conditional_edges(
    "quality_filter",
    should_reconcile,
    {
        "reconcile": "reconciler",
        "generate": "generator",
        "end": END
    }
)
workflow.add_edge("reconciler", "generator")
workflow.add_edge("generator", END)

app = workflow.compile()

print("--- GRAPH STRUCTURE ---")
# app.get_graph().print_ascii()
print("✅ Financial RAG Workflow Compiled!")