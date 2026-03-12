import sys
import os
from fastapi import FastAPI, HTTPException, Depends, Body
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.langchain import langchain_mode
from src.agent import langgraph_route_and_respond
from src.vectorstores.creator import create_vector_store
from src.config import VALID_CATEGORIES
from src.langchain.utils import rewrite_query
from src.langchain.history import get_session_history

app = FastAPI(title="Benny AI Backend")

# User-provided Admin Credentials
ADMIN_DATA = {
    "name": "JACKIV GARG",
    "email": "jackiv@gmail.com",
    "password": "admin@123"
}

class ChatRequest(BaseModel):
    query: str
    session_id: str
    mode: str  # "LangChain" or "LangGraph"
    use_human_review: bool = False
    edited_query: Optional[str] = None

class RewriteRequest(BaseModel):
    query: str
    session_id: str

class AdminLoginRequest(BaseModel):
    email: str
    password: str

class AddContentRequest(BaseModel):
    category: str
    content: str
    email: str
    password: str

@app.get("/")
async def root():
    return {"message": "Benny AI Backend is running"}

from fastapi.responses import StreamingResponse
import json

@app.post("/chat")
async def chat(request: ChatRequest):
    def stream_response():
        try:
            if request.mode == "LangChain":
                for chunk in langchain_mode(request.query, request.session_id):
                    if isinstance(chunk, dict) and "__stats__" in chunk:
                        continue # Skip stats in stream for now
                    yield chunk
            
            elif request.mode == "LangGraph":
                for chunk in langgraph_route_and_respond(
                    request.query, 
                    request.session_id, 
                    use_human_review=request.use_human_review, 
                    edited_query=request.edited_query
                ):
                    if isinstance(chunk, dict) and "__stats__" in chunk:
                        continue # Skip stats in stream
                    yield chunk
            else:
                yield "Error: Invalid mode"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"Error: {str(e)}"

    return StreamingResponse(stream_response(), media_type="text/plain")

@app.post("/rewrite")
async def rewrite_endpoint(request: RewriteRequest):
    try:
        history = get_session_history(request.session_id)
        history_messages = history.messages
        
        history_str = "\n".join(
            f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
            for m in history_messages
        )
        
        rewritten = rewrite_query(request.query, history_str)
        return {"rewritten_query": rewritten}
    except Exception as e:
        return {"rewritten_query": request.query}

@app.post("/admin/login")
async def admin_login(request: AdminLoginRequest):
    if request.email == ADMIN_DATA["email"] and request.password == ADMIN_DATA["password"]:
        return {"status": "success", "name": ADMIN_DATA["name"]}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/admin/add_content")
async def add_content(request: AddContentRequest):
    # Verify admin
    if request.email != ADMIN_DATA["email"] or request.password != ADMIN_DATA["password"]:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if request.category not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    # Path to data file
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    file_path = os.path.join(data_dir, f"{request.category}.txt")
    
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n{request.content}")
        return {"status": "success", "message": f"Content added to {request.category}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/refresh")
async def refresh_vector_store(category: str, email: str = Body(...), password: str = Body(...)):
    # Verify admin
    if email != ADMIN_DATA["email"] or password != ADMIN_DATA["password"]:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if category not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    faiss_dir = os.path.join(os.path.dirname(__file__), "faiss_stores")
    
    txt_file = os.path.join(data_dir, f"{category}.txt")
    save_path = os.path.join(faiss_dir, f"faiss_store_{category}")
    
    try:
        create_vector_store(txt_file, save_path)
        return {"status": "success", "message": f"Vector store for {category} refreshed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
