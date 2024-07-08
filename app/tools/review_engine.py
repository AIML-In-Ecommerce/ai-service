from llama_index.core.tools import FunctionTool
import httpx
import requests


def getReviewListApi(productId: str):
    base_url = 'http://14.225.218.109:3007/ai/productReviews'
    url = f'{base_url}/{productId}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data  = response.json()
        return data['data']
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    
review_engine = FunctionTool.from_defaults(
    fn = getReviewListApi,
    name  = "review_getter",
    description="This tool can get list view of a product by productId, parameter productId is _id field of product. The response of this tool is a list review of product",
)