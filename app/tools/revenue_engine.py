from llama_index.core.tools import FunctionTool
import httpx
import requests


def getRevenue(request_param_str: str):
    # DO SOMETHING
    # revenues = [
    # {"year": "2024", "month": "01", "totalRevenue": 17500000},
    # {"year": "2024", "month": "02", "totalRevenue": 18500000},
    # {"year": "2024", "month": "03", "totalRevenue": 19000000},
    # {"year": "2024", "month": "04", "totalRevenue": 20000000},
    # {"year": "2024", "month": "05", "totalRevenue": 19500000},
    # {"year": "2024", "month": "06", "totalRevenue": 18830000},
    # {"year": "2024", "month": "07", "totalRevenue": 19200000},
    # {"year": "2024", "month": "08", "totalRevenue": 19800000},
    # {"year": "2024", "month": "09", "totalRevenue": 20500000},
    # {"year": "2024", "month": "10", "totalRevenue": 21000000},
    # {"year": "2024", "month": "11", "totalRevenue": 21500000},
    # {"year": "2024", "month": "12", "totalRevenue": 22000000}
    # ]

    base_url = 'https://apis.fashionstyle.io.vn/order/seller/revenue'
    url = f'{base_url}{request_param_str}'
    print("URL: ", url)
    try:
        response = requests.get(url)
        response.raise_for_status()
        data  = response.json()
        return data['data']
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    
    
revenue_engine = FunctionTool.from_defaults(
    fn = getRevenue,
    name  = "revenue_getter",
    description="This tool retrieves the revenue for a specific shop in a timestamp. The parameter request_param_str is a URL's query string which provide additional information include shop (a string representing the shop's identifier), year (an integer indicating the year for which the user wants to retrieve revenue data), month (an integer indicating the month for which the user wants to retrieve revenue data). The shop request parameter is always required. The year and month parameter are not necessary to required.",
)