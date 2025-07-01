# main.py

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting Financial RAG API server...")
    print("➡️ Go to http://127.0.0.1:8000/docs for the interactive API documentation.")
    
    # This string tells uvicorn where to find the FastAPI app instance:
    # "backend.api": the Python module path (folder.file)
    # "app": the variable inside that module that holds the FastAPI instance
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)