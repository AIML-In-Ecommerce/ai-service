# from langchain.vectorstores import Qdrant
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
# from dotenv import load_dotenv
# import qdrant_client
# import os
# load_dotenv()

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# QDRANT_HOST = os.getenv('QDRANT_HOST')
# QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')



# print("OPENAI_KEY: ", OPENAI_API_KEY)
# print("QDRANT_HOST: ", QDRANT_HOST)
# print("QDRANT_API_KEY: ", QDRANT_API_KEY)


# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# client = qdrant_client.QdrantClient(
#         QDRANT_HOST,
#         api_key=QDRANT_API_KEY
#     )

# vectorstore = Qdrant(
#         client=client,
#         collection_name="DOC",
#         embeddings=embeddings
#     )

# qa = RetrievalQA.from_chain_type(
#     llm=OpenAI(),
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever()
#     )

# def ragResponse(query):
#     return qa.run(query)