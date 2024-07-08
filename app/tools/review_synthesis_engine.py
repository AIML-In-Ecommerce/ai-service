from llama_index.core.tools import FunctionTool
from app.services.genai_service import getReviewSynthesis
import httpx
import requests
    
review_synthesis_engine = FunctionTool.from_defaults(
    fn = getReviewSynthesis,
    name  = "review_synthesis_engine",
    description="This tool is used to summarize a product's review list, parameter reviewList is list review of a product.",
)