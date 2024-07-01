from llama_index.core.tools import FunctionTool

def callAddToCartApi(productId):
    print("Call Add To Cart API")
    return "Sản phẩm đã được thêm thành công vào giỏ hàng của bạn"
    
cart_engine = FunctionTool.from_defaults(
    fn = callAddToCartApi,
    name  = "cart_adding",
    description="this tool can add a product to cart, the parameter include userId is id of user and productId is id of product what would be added",
)