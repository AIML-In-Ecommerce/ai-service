from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from typing import List
import base64
import os

from pathlib import Path


router = APIRouter(
    prefix="/file",
    tags=["File"],
    responses={404: {"description": "Not found"}},
)

UPLOAD_DIRECTORY = "uploaded_files"
Path(UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_location = f"{UPLOAD_DIRECTORY}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())
        return JSONResponse(status_code=200, content={"message": "File uploaded successfully.", "file_path": file_location})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "An error occurred", "details": str(e)})