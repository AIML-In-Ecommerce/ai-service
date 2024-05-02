from app.tools import cart_engine, rag_engine, product_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from app.prompts.prompt import  context_str,qa_prompt_tmpl_str
from llama_index.core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

promptTemplate = PromptTemplate(qa_prompt_tmpl_str)

tools =[
    cart_engine.cart_engine,
    product_engine.product_engine,
    QueryEngineTool(
        query_engine=rag_engine.rag_engine,
        metadata=ToolMetadata(
            name="platform_data",
            description="this gives detailed information about platform",
        ),
    ),
]

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context_str)

def agentResponse(query_str):
    prompt = promptTemplate.format(
        query_str=query_str,
    )
    response = agent.query(prompt)
    print("Agent Response: ", response)
    # DO SOMETHING

    
    return response
    
    
