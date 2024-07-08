from llama_index.core.tools import FunctionTool
import requests


def callAddToCartApi(userId, productId):
    base_url = 'https://apis.fashionstyle.io.vn'
    url = f'{base_url}?userId={userId}&productId={productId}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data  = response.json()
        return data['data']
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    
cart_engine = FunctionTool.from_defaults(
    fn = callAddToCartApi,
    name  = "cart_adding",
    description="this tool can add a product to cart, the parameter include userId is id of user and productId is id of product what would be added",
)