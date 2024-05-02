from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from app.services import genai_service
import json


router = APIRouter(
    prefix="/genai",
    tags=["GenAI"],
    responses={404: {"description": "Not found"}},
)

@router.post("/review-synthesis")
async def upload_files(data: dict):
    if "reviews" not in data:
        raise HTTPException(status_code=400, detail="Reviews not found in request body")
    
    reviews = json.dumps(data["reviews"])
    response = genai_service.getReviewSynthesis(reviews)
    return JSONResponse(content={"message": "Reviews processed successfully", "data": response})