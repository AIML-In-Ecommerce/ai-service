from fastapi import FastAPI
from pydantic import BaseModel


from app.routes import file_router, index_router, genai_router, chat_router

class Question(BaseModel):
    prompt: str

app = FastAPI()

app.include_router(file_router.router)
app.include_router(index_router.router)
app.include_router(genai_router.router)
app.include_router(chat_router.router)
