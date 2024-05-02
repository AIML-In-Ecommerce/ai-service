from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
import qdrant_client
from dotenv import load_dotenv
import os
load_dotenv()

# OPENAI_KEY = os.environ.get('OPENAI_API_KEY')
# QDRANT_HOST = os.environ.get('QDRANT_HOST')
# QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
print("OPENAI_API_KEY", os.getenv('A'))

client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )


collection_config = qdrant_client.http.models.VectorParams(
        size=1536,
        distance=qdrant_client.http.models.Distance.COSINE
    )

client.recreate_collection(
    collection_name="DOC",
    vectors_config=collection_config
)

embeddings = OpenAIEmbeddings()

vectorstore = Qdrant(
        client=client,
        collection_name="DOC",
        embeddings=embeddings
    )

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

with open("story.txt") as f:
    raw_text = f.read()

texts = get_chunks(raw_text)

vectorstore.add_texts(texts)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
    )

query = "Dữ liệu viết về ai?"
response = qa.run(query)

print(response)