# backend/core/data_sources/newsapi.py

from newsapi import NewsApiClient
from backend.config import settings
from .yahoo_finance import extract_financial_entities

newsapi = NewsApiClient(api_key=settings.NEWS_API_KEY)

def get_financial_news(query: str) -> str:
    """
    Fetches news from NewsAPI. It uses the central AI extractor to find tickers,
    and if none are found, it falls back to the original query.
    """
    entities = extract_financial_entities(query)
    search_terms = entities.get("tickers", [])
    
    if not search_terms:
        search_terms = [query]
        print(f"[NewsAPI] No specific tickers found. Searching for general query: '{query}'")
    else:
        print(f"[NewsAPI] Identified tickers for news search: {search_terms}")
        
    all_news_summaries = []
    for term in search_terms:
        try:
            q_param = f'"{term}"' if len(term) <= 5 else term
            
            top_headlines = newsapi.get_everything(
                q=q_param,
                language='en',
                sort_by='relevancy',
                page_size=3
            )

            articles = top_headlines.get('articles', [])
            if not articles:
                all_news_summaries.append(f"No recent news found for '{term}' via NewsAPI.")
                continue

            term_header = f"Top news related to '{term}':\n"
            
            article_details = []
            for article in articles:
                # Add safe extraction with fallbacks
                title = article.get('title', 'No title available')
                source_name = article.get('source', {}).get('name', 'Unknown source')
                published_at = article.get('publishedAt', 'Unknown date')
                description = article.get('description', 'No description available')
                
                # Format date safely
                if published_at and published_at != 'Unknown date':
                    try:
                        published_at = published_at[:10]
                    except:
                        published_at = 'Unknown date'
                
                article_details.append(
                    f"Title: {title}\n"
                    f"Source: {source_name}\n"
                    f"Published: {published_at}\n"
                    f"Summary: {description[:200]}..."
                )

            all_news_summaries.append(term_header + "\n---\n".join(article_details))

        except Exception as e:
            print(f"[NewsAPI Error] Failed to fetch news for '{term}': {e}")
            all_news_summaries.append(f"An error occurred with NewsAPI for query '{term}': {e}")
            
    return "\n\n=====\n\n".join(all_news_summaries) if all_news_summaries else "No news data could be retrieved."