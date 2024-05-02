from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os

class VectorStore:
    def __init__(self, collectionId) -> None:
        OPENAI_KEY = os.environ.get('OPENAI_API_KEY')
        QDRANT_HOST = os.environ.get('QDRANT_HOST')
        QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')

        self.client = qdrant_client.QdrantClient(
            QDRANT_HOST,
            api_key=QDRANT_API_KEY
        )

        self.collection_config = qdrant_client.http.models.VectorParams(
            size=1536,
            distance=qdrant_client.http.models.Distance.COSINE
        )

        self.client.recreate_collection(
            collection_name="DOC",
            vectors_config=self.collection_config
        )

        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)

        self.vectorstore = Qdrant(
            client=self.client,
            collection_name=collectionId,
            embeddings=self.embeddings
        )

