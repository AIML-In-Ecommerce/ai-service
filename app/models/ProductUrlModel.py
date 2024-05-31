from pydantic import BaseModel

class ProductUrlModel(BaseModel):
    prompt: str
    context: str
    garmentImgUrl: str