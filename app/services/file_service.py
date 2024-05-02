from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import qdrant_client
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_KEY = os.getenv('OPENAI_API_KEY')
QDRANT_HOST = os.getenv('QDRANT_HOST')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')


embeddings = OpenAIEmbeddings()

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


def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def saveDataToVectorStore():
    file_path = "Đề cương TTDATN.pdf"

    with open(file_path, "rb") as f:
        raw_text = ""
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    texts = get_chunks(raw_text)

    vectorstore.add_texts(texts)