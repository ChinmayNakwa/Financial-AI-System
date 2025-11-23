# 
# backend/api.py (Temporary Smoke Test)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Financial AI API",
    description="An API for the Financial AI Research",
    version="0.1.1"
)

origins = [
    "http://localhost:8501",  # Default for local Streamlit
    "http://localhost:8000", 
    "http://127.0.0.1:3000",
    "http://localhost:3000",  
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/backend/query")
def handle_query(request: QueryRequest):
    print(f"Smoke test received query: {request.query}")
    return QueryResponse(answer=f"Successfully received your message: {request.query}")

@app.get("/")
def root():
    return {"message": "Smoke Test API is running."}