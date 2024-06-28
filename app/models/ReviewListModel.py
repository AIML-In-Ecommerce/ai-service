from pydantic import BaseModel
from typing import List

class ReviewListModel(BaseModel):
    reviews: List[str]