from llama_index.core.tools import FunctionTool
import httpx
import requests


def callProductListApi(request_param_str: str):
    base_url = 'http://14.225.218.109:3006/products/search'
    url = f'{base_url}?{request_param_str}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data  = response.json()
        return data['data']
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    
product_engine = FunctionTool.from_defaults(
    fn = callProductListApi,
    name  = "product_getter",
    description="this tool can get list product by keyword, parameter request_param_str is a URL's query string which provide additional information include keyword (the keyword of product being found bu user), quantity (the number of product user want to receive), sortBy ({{soldQuantity}} in case related to to number of sold product and {{avgRating}} in case related to rating value). Example: {{request_param_str: '?keyword=ao-thun?quantity=2?sortBy=soldQuantity'}}  It's not necessary to always have all three request parameters.",
)