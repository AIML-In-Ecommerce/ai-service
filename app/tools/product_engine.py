from llama_index.core.tools import FunctionTool

def callProductListApi():
    print("Product Engine")
    return "Điện thoại Samsung, Điện thoại Iphone"
    
product_engine = FunctionTool.from_defaults(
    fn = callProductListApi,
    name  = "product_getter",
    description="this tool can get list product ",
)