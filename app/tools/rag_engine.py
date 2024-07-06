# # Using Llama-index
# from llama_index.core import VectorStoreIndex
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# import qdrant_client
# from dotenv import load_dotenv
# import os
# load_dotenv()

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# QDRANT_HOST = os.getenv('QDRANT_HOST')
# QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

# client = qdrant_client.QdrantClient(
#     QDRANT_HOST,
#     api_key=QDRANT_API_KEY, 
# )

# vector_store = QdrantVectorStore(client=client, collection_name="DOC")

# index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# rag_engine = index.as_query_engine()

# Using Langchain

from llama_index.core.tools import FunctionTool
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
import qdrant_client
import os
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
QDRANT_HOST = os.getenv('QDRANT_HOST')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')



print("OPENAI_KEY: ", OPENAI_API_KEY)
print("QDRANT_HOST: ", QDRANT_HOST)
print("QDRANT_API_KEY: ", QDRANT_API_KEY)


embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

client = qdrant_client.QdrantClient(
        QDRANT_HOST,
        api_key=QDRANT_API_KEY
    )

vectorstore = Qdrant(
        client=client,
        collection_name="DOC",
        embeddings=embeddings
    )

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
    )

def ragResponse(query):
    return qa.run(query)

rag_engine = FunctionTool.from_defaults(
    fn = ragResponse,
    name  = "rag_tool",
    description="this gives detailed information of development team, parameter query is full question of user",
    return_direct=True
)