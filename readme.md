
# Financial AI Assistant: An Advanced RAG Agent

This project showcases a sophisticated Financial AI Assistant built with Python, LangChain, and LangGraph. It moves beyond basic RAG to implement a robust, agentic workflow capable of answering complex, multi-step financial questions with a high degree of accuracy and reliability.

The system leverages a suite of live financial data APIs and is architected around three advanced RAG patterns: **Adaptive RAG**, **Self-RAG**, and **Corrective RAG**.

## Key Features

- **Multi-Source Data Integration:** Connects to live APIs for stock prices (Yahoo Finance), economic data (FRED), financial news (NewsAPI), cryptocurrency prices (CoinDesk), SEC filings (EDGAR), Technical Indicators (Polygon.io), and general web search (Tavily).
- **Intelligent Tool Use:** The agent can reason about which tool is best for a given query and can use tools in sequence to solve multi-step problems.
- **Self-Correction & Planning:** For complex queries, the agent can recognize when it lacks information, use a search tool to find the missing pieces, and then continue its task.
- **Dynamic & Flexible:** Can answer questions about financial instruments and economic indicators it wasn't explicitly pre-programmed to know about.
- **API-Ready:** Exposed via a FastAPI endpoint for easy integration into other applications.
- **Modern Frontend:** A clean, responsive Next.js chat interface with real-time "Agent Reasoning" status updates.

---

## Core Concepts: The Advanced RAG Architecture

This project is built on three pillars of advanced Retrieval-Augmented Generation. Here's how they are implemented in the LangGraph workflow:

### 1. Adaptive RAG: The Intelligent Router

-   **Concept:** The system "adapts" its strategy by choosing the best data source for a given query, rather than relying on a single, one-size-fits-all approach.
-   **Implementation:** The core agent analyzes the user's prompt and its list of available tools. Based on the query's intent ("stock price" vs. "inflation rate"), it intelligently selects the appropriate function to call (e.g., `get_stock_data` or `get_economic_data`). This happens dynamically in the main `router` node.

### 2. Self-RAG: The Quality Analyst

-   **Concept:** The system critically evaluates the information it has just retrieved *before* using it, reflecting on its relevance and quality.
-   **Implementation:** After a tool returns information, the result is fed into the `quality_filter` node. The agent assesses this new information against the original goal. If a tool returns irrelevant data or an error message, the agent filters it out, preventing low-quality data from reaching the user.

### 3. Corrective RAG: The Multi-Step Researcher

-   **Concept:** The system can fix its own knowledge gaps or resolve conflicting information by taking additional, corrective steps.
-   **Implementation:** This is the primary strength of the LangGraph agent's cyclical workflow (`router` -> `retriever` -> `reconciler` -> `generator`).
    -   **Discovery & Enrichment:** For complex queries, the agent can recognize it doesn't know a specific detail (e.g., a list of competitors).
    -   **Correction:** It uses the `tavily` search tool to discover missing entities, then loops back to fetch specific financial data for those new entities.
    -   **Synthesis:** The `reconciler` node identifies discrepancies between sources and resolves them before the final answer is generated.

---

## Getting Started

### 1. Prerequisites

-   Python 3.10+
-   Node.js 18+ (for the frontend)
-   Git

### 2. Clone the Repository

```bash
git clone https://github.com/ChinmayNakwa/Financial-AI-System.git
cd Financial-AI-System
```

### 3. Set Up the Backend

It's highly recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Set Up the Frontend

```bash
cd client
npm install
cd ..
```

### 5. Configure API Keys

Create a `.env` file in the root directory and fill in your keys:

```env
# LLM & Search APIs
GOOGLE_API_KEY="YOUR_KEY_HERE"
TAVILY_API_KEY="YOUR_KEY_HERE"

# Financial Data APIs
FRED_API_KEY="YOUR_KEY_HERE"
NEWS_API_KEY="YOUR_KEY_HERE"
SEC_API_KEY="YOUR_KEY_HERE"
COINDESK_API_KEY="YOUR_KEY_HERE"
POLYGON_API_KEY="YOUR_KEY_HERE"
```

---

## Running Locally

### Backend Server
From the root directory:
```bash
python main.py
```
The API will be available at `http://localhost:8000`. You can access the Swagger UI at `http://localhost:8000/docs`.

### Frontend Client
From the `client/` directory:
```bash
npm run dev
```
The interface will be available at `http://localhost:3000`.

---

## Deployment (Vercel)

This project is optimized for Vercel using a **Proxy Pattern** to bridge the Next.js frontend and the Python FastAPI backend.

### Configuration Highlights:
- **Framework Preset:** `Other`
- **Build Command:** `cd client && npm run build`
- **Output Directory:** `client/out`
- **Install Command:** `cd client && npm install`
- **Backend Entry:** `api/index.py` (Proxies requests to the `backend/` package)

The `vercel.json` handles the routing, ensuring that requests to `/backend/*` are routed to the Python serverless function, while all other requests serve the static Next.js frontend.

---

## Project Structure

-   `api/index.py`: Vercel-specific entry point that bridges the serverless environment to the backend package.
-   `backend/`:
    -   `api.py`: FastAPI routes and application logic.
    -   `core/rag/`: Contains the LangGraph workflow definitions (`financial_workflow.py`) and RAG logic (`adaptive_rag.py`, `self_rag.py`, `corrective_rag.py`).
    -   `core/data_sources/`: Individual tool implementations for each financial API.
-   `client/`:
    -   `app/`: Next.js 14+ App Router pages (Chat interface and Landing page).
    -   `components/`: Reusable UI components (Chat messages, Navbar, etc.).
    -   `next.config.ts`: Configured for `output: 'export'` to support static hosting.
-   `vercel.json`: Global routing and rewrite configuration.
-   `requirements.txt`: Python dependencies (located at root for Vercel detection).
