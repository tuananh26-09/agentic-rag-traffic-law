import os
from minio import Minio
from dotenv import load_dotenv

load_dotenv()

def upload_to_minio(file_path, object_name="chunk.pkl"):
    print(f" Đang kết nối tới MinIO để tải lên file {object_name}...")
    try:
        minio_client = Minio(
            os.getenv("MINIO_URL"),
            access_key=os.getenv("access_key"),
            secret_key=os.getenv("secret_key"),
            secure=False
        )
        
        bucket_name = "rag-data"
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f" Đã tạo bucket '{bucket_name}' trên MinIO.")
        
        minio_client.fput_object(bucket_name, object_name, file_path)
        print(f" Đã lưu {object_name} lên MinIO thành công!")
        
    except Exception as e:
        print(f" Lỗi khi tải lên MinIO: {e}")