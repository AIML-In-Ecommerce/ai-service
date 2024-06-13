from fastapi import APIRouter
from app.services import genai_service

router = APIRouter()

@router.get("/index")
def index():
    return "Hello World"