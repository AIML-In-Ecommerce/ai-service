from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.services import rag_service, agent_service, seller_agent_service
from app.models.ChatModel import ChatModel

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    responses={404: {"description": "Not found"}},
)

@router.post("/agent")
async def agent(data: ChatModel):
    history_conservation = data.history_conservation
    prompt = data.prompt
    print("Prompt: ", prompt)

    conversation = agent_service.agentResponse(history_conservation, prompt)
    return JSONResponse(content={"message": "Reviews processed successfully", "data": str(conversation)})

@router.post("/agent/seller-center")
async def sellerAgent(data: ChatModel):
    history_conservation = data.history_conservation
    prompt = data.prompt
    print("Prompt: ", prompt)

    conversation = seller_agent_service.agentResponse(history_conservation, prompt)
    return JSONResponse(content={"message": "Reviews processed successfully", "data": str(conversation)})