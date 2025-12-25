from tavily import TavilyClient
from backend.config import settings


def search_web(query: str, api_key: str) -> str:
    """
    Performs a general web search using Tavily for broad financial topics,
    definitions, or recent events not covered by other specific APIs.
    
    Uses the native Tavily Python client (recommended and future-proof).
    """
    try:
        # Initialize Tavily client with API key
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        
        # Perform search with financial context
        response = client.search(
            query=query,
            max_results=5,  # Get more results for better context
            search_depth="advanced",  # Use advanced for financial queries
            include_answer=True,   # Get AI-generated summary
            include_raw_content=False,  # Don't need full HTML
            topic="finance"  # Optional: can be "general", "news", or "finance"
        )
        
        # Check if we got results
        if not response or 'results' not in response:
            return "Tavily web search found no results."
        
        results = response.get('results', [])
        
        if not results:
            return "Tavily web search found no relevant results for the query."
        
        # Build formatted output
        output_parts = []
        
        # Add the AI-generated answer/summary at the top (very useful!)
        if response.get('answer'):
            output_parts.append(f"=== AI Summary ===\n{response['answer']}\n")
        
        # Add detailed search results
        output_parts.append("=== Web Search Results ===\n")
        
        for idx, result in enumerate(results, 1):
            result_parts = [f"[Result {idx}]"]
            
            # Title
            if result.get('title'):
                result_parts.append(f"Title: {result['title']}")
            
            # URL
            if result.get('url'):
                result_parts.append(f"Source: {result['url']}")
            
            # Published date (if available)
            if result.get('published_date'):
                result_parts.append(f"Published: {result['published_date']}")
            
            # Content snippet
            if result.get('content'):
                # Limit content to reasonable length
                content = result['content']
                if len(content) > 500:
                    content = content[:500] + "..."
                result_parts.append(f"Content: {content}")
            
            # Relevance score (if available)
            if result.get('score'):
                result_parts.append(f"Relevance: {result['score']:.2f}")
            
            output_parts.append("\n".join(result_parts))
        
        return "\n\n---\n\n".join(output_parts)
        
    except Exception as e:
        print(f"[Tavily] Error during web search: {e}")
        import traceback
        print(traceback.format_exc())
        return f"Error during Tavily web search: {str(e)}"