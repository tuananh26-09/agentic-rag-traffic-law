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

def upload_to_qdrant(all_docs):
        print(" Đang nạp Vector vào QDRANT trên Docker...")
        try:
            embeddings = get_embeddings()
            
            QdrantVectorStore.from_documents(
                documents=all_docs,
                embedding=embeddings,
                url=os.getenv("QDRANT_URL"),
                collection_name="luat_giao_thong_viet_nam",
                force_recreate=False
            )
            print(f" HOÀN TẤT! Hãy mở {os.getenv('QDRANT_URL')}/dashboard để xem dữ liệu.")
        except Exception as e:
            print(f" Lỗi khi nạp vào Qdrant: {e}")
