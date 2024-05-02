from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
QDRANT_HOST = os.getenv('QDRANT_HOST')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

client = qdrant_client.QdrantClient(
    QDRANT_HOST,
    api_key=QDRANT_API_KEY, 
)

vector_store = QdrantVectorStore(client=client, collection_name="DOC")

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

rag_engine = index.as_query_engine()