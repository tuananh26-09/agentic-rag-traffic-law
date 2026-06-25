import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# def get_embeddings():
#     return HuggingFaceEmbeddings(model_name=os.getenv("embedding_model"))

def get_embeddings():
    return OpenAIEmbeddings(
        model="BAAI/bge-m3",
        openai_api_base="http://embedder_api:7997", 
        openai_api_key="sk-local", # API nội bộ
        check_embedding_ctx_length=False,
    )
