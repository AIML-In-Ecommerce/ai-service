from llama_index.core.tools import FunctionTool
import httpx
import requests


def getRevenue(shopId: str, year: int):
    # DO SOMETHING
    revenues = [
    {"year": "2024", "month": "01", "totalRevenue": 17500000},
    {"year": "2024", "month": "02", "totalRevenue": 18500000},
    {"year": "2024", "month": "03", "totalRevenue": 19000000},
    {"year": "2024", "month": "04", "totalRevenue": 20000000},
    {"year": "2024", "month": "05", "totalRevenue": 19500000},
    {"year": "2024", "month": "06", "totalRevenue": 18830000},
    {"year": "2024", "month": "07", "totalRevenue": 19200000},
    {"year": "2024", "month": "08", "totalRevenue": 19800000},
    {"year": "2024", "month": "09", "totalRevenue": 20500000},
    {"year": "2024", "month": "10", "totalRevenue": 21000000},
    {"year": "2024", "month": "11", "totalRevenue": 21500000},
    {"year": "2024", "month": "12", "totalRevenue": 22000000}
]
    return revenues
    
revenue_engine = FunctionTool.from_defaults(
    fn = getRevenue,
    name  = "revenue_getter",
    description="This tool retrieves the revenue for a specific shop in a given year. The parameters include shopId that is a string representing the shop's identifier and year that is an integer indicating the year for which the user wants to retrieve revenue data.",
)