# backend/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware 

from backend.core.rag.financial_workflow import app as financial_rag_app

origins = [
    "*",  # Allows all origins. For production, you might want to restrict this
          # to your actual frontend domain, e.g., "https://yourapp.com"
]

# Initialize FastAPI Application
app = FastAPI(
    title="Financial RAG API",
    description="An API for a financial assistant using Adaptive, Self, and Corrective RAG.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define API Request and Response Models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str


@app.post("/backend/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        print(f"🚀 Received API query: '{request.query}'")
        inputs = {"user_question": request.query}
        final_state = await financial_rag_app.ainvoke(inputs, {"recursion_limit": 15})
        answer = final_state.get("final_answer", "Sorry, I could not find enough information to answer.")
        print("✅ Sending API response.")
        return QueryResponse(answer=answer)
    except Exception as e:
        print(f"🚨 An error occurred in the API: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.get("/backend/", include_in_schema=False)
def root():
    return {"message": "Financial RAG API is running. Go to /docs for the API documentation."}