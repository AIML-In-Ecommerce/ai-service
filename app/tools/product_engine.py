from llama_index.core.tools import FunctionTool
import httpx
import requests


def callProductListApi(keyword: str):
    base_url = 'http://14.225.218.109:3006/products/search'
    url = f'{base_url}?keyword={keyword}'
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
    description="this tool can get list product by keyword, parameter keyword is the product which being found by client",
)