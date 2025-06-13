# backend/app/main.py

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .auth import router as auth_router, get_current_user
from .chat import ask_question
from pydantic import BaseModel

# Define request schema
class ChatRequest(BaseModel):
    query: str

app = FastAPI()

# Serve frontend (adjust path as needed)
# if frontend is inside the root project dir
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")



# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication router
app.include_router(auth_router)


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

# Chat endpoint
@app.post("/chat")
def chat_endpoint(request: ChatRequest, user: dict = Depends(get_current_user)):
    answer = ask_question(request.query)
    return {"answer": answer}
