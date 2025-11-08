# 
# backend/api.py (Temporary Smoke Test)

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

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