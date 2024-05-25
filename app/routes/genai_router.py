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
async def uploadFiles(data: dict):
    if "reviews" not in data:
        raise HTTPException(status_code=400, detail="Reviews not found in request body")
    
    reviews = json.dumps(data["reviews"])
    response = genai_service.getReviewSynthesis(reviews)
    return JSONResponse(content={"message": "Reviews processed successfully", "data": response})

@router.post("/generate-product-image")
async def generateProductImage(data:dict):
    prompt = json.dumps(data["prompt"])
    context =  json.dumps(data["context"])
    garmentImg = json.dumps(data["garmentImgUrl"])
    response = genai_service.generateProductImage(prompt=prompt, context=context, garmentImage=garmentImg)
    return JSONResponse(content={"message": "Generate product image successfully", "data": response})

@router.post("/virtual-try-on")
async def generateTryOnImage(data:dict):
    modelImg = json.dumps(data["modelImgUrl"])
    garmentImg =  json.dumps(data["garmentImgUrl"])
    response = genai_service.generateTryOnImage(modelImage=modelImg, garmentImage=garmentImg)
    return JSONResponse(content={"message": "Generate try on image successfully", "data": response})