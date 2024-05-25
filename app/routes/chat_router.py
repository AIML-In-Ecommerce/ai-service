from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.services import rag_service, agent_service

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    responses={404: {"description": "Not found"}},
)

@router.post("/agent")
async def upload_files(data: dict):
    prompt = data["prompt"]

    conversation = agent_service.agentResponse(prompt)
    return JSONResponse(content={"message": "Reviews processed successfully", "data": conversation})
    # return JSONResponse(content={"message": "Reviews processed successfully"})
