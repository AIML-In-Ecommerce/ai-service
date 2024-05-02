from llama_index.core.tools import FunctionTool

def callAddToCartApi(productId):
    print("Call Add To Cart API")
    # CALL ADD TO CART FUNCTION
    return "Call Add To Cart API"
    
cart_engine = FunctionTool.from_defaults(
    fn = callAddToCartApi,
    name  = "cart_saver",
    description="this tool can add a product to cart, the parameter include userId is id of user and productId is id of product what would be added",
)