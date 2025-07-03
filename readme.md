# Financial AI Assistant: An Advanced RAG Agent

This project showcases a sophisticated Financial AI Assistant built with Python, LangChain, and LangGraph. It moves beyond basic RAG to implement a robust, agentic workflow capable of answering complex, multi-step financial questions with a high degree of accuracy and reliability.

The system leverages a suite of live financial data APIs and is architected around three advanced RAG patterns: **Adaptive RAG**, **Self-RAG**, and **Corrective RAG**.


## Key Features

- **Multi-Source Data Integration:** Connects to live APIs for stock prices (Yahoo Finance), economic data (FRED), financial news (NewsAPI), cryptocurrency prices (CoinDesk), SEC filings (EDGAR), and general web search (Tavily).
- **Intelligent Tool Use:** The agent can reason about which tool is best for a given query and can use tools in sequence to solve multi-step problems.
- **Self-Correction & Planning:** For complex queries, the agent can recognize when it lacks information, use a search tool to find the missing pieces, and then continue its task.
- **Dynamic & Flexible:** Can answer questions about financial instruments and economic indicators it wasn't explicitly pre-programmed to know about.
- **API-Ready:** Exposed via a FastAPI endpoint for easy integration into other applications.

---

## Core Concepts: The Advanced RAG Architecture

This project is built on three pillars of advanced Retrieval-Augmented Generation. Here's how they are implemented in the LangGraph workflow:

### 1. Adaptive RAG: The Intelligent Router

-   **Concept:** The system "adapts" its strategy by choosing the best data source for a given query, rather than relying on a single, one-size-fits-all approach.
-   **Implementation:** The core agent analyzes the user's prompt and its list of available tools. Based on the query's intent ("stock price" vs. "inflation rate"), it intelligently selects the appropriate function to call (e.g., `get_stock_data` or `get_economic_data`). This happens dynamically in the main `agent_node`.

### 2. Self-RAG: The Quality Analyst

-   **Concept:** The system critically evaluates the information it has just retrieved *before* using it, reflecting on its relevance and quality.
-   **Implementation:** After a tool returns information, the result is fed back into the agent's reasoning loop. The agent then assesses this new information against its original goal. If a tool returns irrelevant data (e.g., a niche "Flexible Price Index" when asked for headline inflation), the agent recognizes the mismatch and can decide to try another tool or refine its query. This prevents low-quality data from reaching the user.

### 3. Corrective RAG: The Multi-Step Researcher

-   **Concept:** The system can fix its own knowledge gaps or resolve conflicting information by taking additional, corrective steps.
-   **Implementation:** This is the primary strength of the LangGraph agent's cyclical workflow (`agent` -> `tools` -> `agent`).
    -   **Discovery & Enrichment:** For a query like *"Show me news for the top 3 tech stocks"*, the agent first recognizes it doesn't know the top 3 stocks.
        1.  **Correction (Step 1):** It calls the `search_the_web` tool to *discover* the list of stocks.
        2.  **Execution (Step 2):** With the new list, it *loops* through each stock ticker and calls the `get_financial_news` tool for each one.
    -   **Synthesis:** Finally, it synthesizes all the retrieved pieces of information into a single, comprehensive answer.

---

## Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

-   Python 3.10+
-   Git

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/Financial-AI-System.git
cd Financial-AI-System
```

### 3. Set Up the Environment

It's highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv myenv

# Activate it
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate
```

### 4. Install Dependencies

Install all the required Python packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 5. Configure API Keys

You will need API keys from several services.

1.  Create a file named `.env` in the root directory of the project.
2.  Copy the contents of `.env.example` (or the structure below) into your new `.env` file.
3.  Fill in your personal API keys.

```env
# .env file

# Financial Data APIs
ALPHA_VANTAGE_API_KEY="YOUR_KEY_HERE"
FRED_API_KEY="YOUR_KEY_HERE"
NEWS_API_KEY="YOUR_KEY_HERE"
SEC_API_KEY="YOUR_KEY_HERE"

# LLM & Search APIs
MISTRAL_API_KEY="YOUR_KEY_HERE"
GOOGLE_API_KEY="YOUR_KEY_HERE" # For Gemini models
TAVILY_API_KEY="YOUR_KEY_HERE"
```

### 6. Run the API Server

Once the setup is complete, you can start the FastAPI server using Uvicorn.

```bash
python main.py
```

The server will start, and you can access the interactive API documentation:

-   **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
-   **Redoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## How to Use

You can interact with the API using the `/docs` page or by sending a POST request to the `/query` endpoint.

**Example `curl` command:**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "What is the current US unemployment rate and what is the latest news about it?"
}'
```

### Example Queries to Try

-   **Simple:** `"What is the current stock price of Microsoft (MSFT)?"`
-   **Economic:** `"What is the 30-year fixed mortgage rate in the US?"`
-   **Crypto:** `"What is the price of ethereum and bitcoin?"`
-   **Complex / Multi-Step:** `"Who are the top 3 competitors of Tesla? Show me their latest stock price and market cap."`

---

## Project Structure

A brief overview of the key files in the `backend`:

-   `main.py`: The entry point to start the Uvicorn server.
-   `api.py`: Defines the FastAPI application and the `/query` endpoint.
-   `core/rag/financial_workflow.py`: The heart of the project. Defines the LangGraph agent, tools, and the graph itself.
-   `core/rag/adaptive_rag.py`, `self_rag.py`, `corrective_rag.py`: Contain the logic and prompts for the different RAG strategies.
-   `core/data_sources/`: A directory where each file is a wrapper around a specific financial data API, acting as a "tool" for the agent.
-   `config.py`: Handles loading environment variables and API keys.