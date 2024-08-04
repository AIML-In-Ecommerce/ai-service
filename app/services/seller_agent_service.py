from app.tools import cart_engine, rag_engine, product_engine, revenue_engine, data_visualization_engine, review_engine, review_synthesis_engine
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from app.prompts.prompt import  context_str,qa_prompt_tmpl_str, seller_react_system_header_str
from llama_index.core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

promptTemplate = PromptTemplate(qa_prompt_tmpl_str)

tools =[
    product_engine.product_engine,
    review_engine.review_engine,
    review_synthesis_engine.review_synthesis_engine,
    revenue_engine.revenue_engine,
    data_visualization_engine.chart_engine,
    rag_engine.rag_engine
]

react_system_prompt = PromptTemplate(seller_react_system_header_str)
llm = OpenAI(model="gpt-4")

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context_str)
agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})
agent.reset()

prompt_dict = agent.get_prompts()
for k, v in prompt_dict.items():
    print(f"Prompt: {k}\n\nValue: {v.template}")

def agentResponse(history_conservation, query_str):
    prompt = promptTemplate.format(
        context_str = context_str,
        history_conservation = history_conservation,
        query_str=query_str
    )
    response = agent.query(prompt)
    return response
    
    
