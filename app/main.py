from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.routes import file_router, index_router, genai_router, chat_router

class Question(BaseModel):
    prompt: str

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # List of allowed origins
    allow_credentials=True,           # Allow cookies to be sent with cross-domain requests
    allow_methods=["*"],              # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],              # Allow all headers
)

app.include_router(file_router.router)
app.include_router(index_router.router)
app.include_router(genai_router.router)
app.include_router(chat_router.router)
