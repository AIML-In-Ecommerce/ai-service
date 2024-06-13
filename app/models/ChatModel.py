from pydantic import BaseModel
from typing import List
from app.models.MessageModel import MessageModel

class ChatModel(BaseModel):
    history_conservation: List[MessageModel]
    prompt: str
