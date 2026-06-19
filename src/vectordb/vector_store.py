from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from src.embeddings.embedder import get_embeddings
from dotenv import load_dotenv
import os

load_dotenv() 
def get_retriever():
    # Kết nối tới Qdrant đang chạy
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    
    vector_db = QdrantVectorStore(
        client=client, 
        collection_name="luat_giao_thong_viet_nam", 
        embedding=get_embeddings()
    )
    
    # Trả về retriever lấy top * tài liệu
    return vector_db.as_retriever(search_kwargs={"k": 10})