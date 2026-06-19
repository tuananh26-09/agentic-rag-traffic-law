import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=os.getenv("embedding_model"))