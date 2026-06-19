import os
import pickle
import io
from minio import Minio
from src.ingestion.loader import LawDataLoader
from langchain_qdrant import QdrantVectorStore 
from src.embeddings.embedder import get_embeddings
from src.chunking.chunker import LawTextChunker
from dotenv import load_dotenv

load_dotenv() 

def main():
    print("1. Đang đọc thư mục Luật và làm sạch văn bản...")
    
    loader = LawDataLoader()
    # Các file .md đang nằm trong thư mục 'data'
    documents = loader.load_dir("data")
    print("2. Đang băm nhỏ văn bản bằng LawTextChunker (Chunking)...")
    
    # Khởi tạo cắt luật
    law_chunker = LawTextChunker(chunk_size=2000, chunk_overlap=200)
    
    # Thực hiện băm văn bản
    chunks = law_chunker.split_documents(documents)

    print("-> Đang đẩy file chunks.pkl lên MinIO...")
    # Khởi tạo kết nối tới MinIO
    minio_client = Minio(
        os.getenv("MINIO_URL"),
        access_key=os.getenv("access_key"),
        secret_key=os.getenv("secret_key"),
        secure=False
    )
    
    found = minio_client.bucket_exists("rag-data")
    if not found:
        minio_client.make_bucket("rag-data")
        print("Đã tạo bucket 'rag-data' trên MinIO.")
    
    # Biến dữ liệu chunks thành dạng bytes
    pickled_data = pickle.dumps(chunks)
    data_stream = io.BytesIO(pickled_data)
    
    # Đẩy thẳng lên bucket 'rag-data'
    minio_client.put_object(
        bucket_name="rag-data",
        object_name="chunks.pkl",
        data=data_stream,
        length=len(pickled_data)
    )
    print("Đã lưu chunks.pkl lên MinIO thành công!")

    print("3. Đang nạp Vector vào QDRANT trên Docker...")
    embeddings = get_embeddings()
    
    # KẾT NỐI VÀ NẠP VÀO QDRANT
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        collection_name="luat_giao_thong_viet_nam",
        force_recreate=True
    )
    print(" HOÀN TẤT! Hãy mở " + os.getenv("QDRANT_URL") + "/dashboard để xem dữ liệu.")

if __name__ == "__main__":
    main()