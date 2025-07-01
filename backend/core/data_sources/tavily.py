from langchain_community.tools.tavily_search import TavilySearchResults
from backend.config import settings

def search_web(query: str) -> str:
    """
    Performs a general web search using Tavily for broad financial topics,
    definitions, or recent events not covered by other specific APIs.
    """
    try:
        # include_answer=True gives a direct summary answer from Tavily
        search = TavilySearchResults(
            max_results=3,
            api_key=settings.TAVILY_API_KEY,
            include_answer=True 
        )
        results = search.invoke(query)
        
        if not results:
            return "Tavily web search found no results."

        # The first result is often the summarized answer
        answer = results[0].get('answer', '')
        
        # Then, format the source contexts
        contexts = [
            f"Title: {res['title']}\nURL: {res['url']}\nContent: {res['content']}"
            for res in results if 'content' in res
        ]
        
        response = f"Tavily Summary: {answer}\n\nSources:\n" + "\n\n---\n\n".join(contexts)
        return response
        
    except Exception as e:
        return f"Error during Tavily web search: {e}"