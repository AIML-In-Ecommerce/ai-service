from pydantic import BaseModel

class TryOnModel(BaseModel):
    modelImgUrl: str
    garmentImgUrl: str