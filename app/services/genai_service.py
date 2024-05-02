from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_KEY = os.getenv('OPENAI_API_KEY')

REVIEW_SYSTHESIS_SYSTEM_MSG = """Bạn là một chuyên gia phân tích đánh giá, bạn sẽ được cung cấp một danh sách các đánh giá của một sản phẩm, hãy phân tích các đánh giá đó. Hãy đảm bảo kết quả trả về luôn luôn là một json với cấu trúc {"positiveCount" : đây là số lượng đánh giá tích cực , "negativeCount": đây là số lượng đánh giá tiêu cực, "positiveSumary" : "Đây là một đoạn tóm tắt mô tả ngắn về các đánh giá tích cực, độ dài đoạn tóm tắt khoảng 50 từ. Ví dụ: Hầu hết người mua đánh giá tích cực về chất lượng sản phẩm, bao gồm vải đẹp, chất jean dày dặn, co giãn tốt và form chuẩn. Một số khách hàng nhận xét sản phẩm đáng mua, đẹp, sang trọng và bền chắc. Đa số khách hàng hài lòng với dịch vụ giao hàng nhanh, đúng hẹn và đóng gói cẩn thận. Một số khách hàng đánh giá tích cực về sự nhiệt tình và trách nhiệm của shop.", "negativeSumary" : "Đây là một đoạn tóm tắt mô tả ngắn về các đánh giá tiêu cực,độ dài đoạn tóm tắt khoảng 30 từ. Ví dụ: Tuy nhiên, có một số nhận xét tiêu cực về khuy nút bị lỏng và màu không thích, có một nhận xét tiêu cực về việc nhầm hàng."}.
"""

GENERATE_DESCRIPTION_SYSTEM_MSG = """"""

def getReviewSynthesis(query):
    client = OpenAI(
        api_key = OPENAI_KEY,
    )
    print("Query", query)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": REVIEW_SYSTHESIS_SYSTEM_MSG},
            {"role": "user", "content": query}
        ],
    )
    message = response.choices[0].message.content
    return message

def generateDesciption(shortDescription):
    return