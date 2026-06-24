import os
import fitz 
import easyocr
import pickle
from langchain_core.documents import Document as LangchainDocument
from src.chunking.chunker import LawTextChunker
from src.embeddings.embedder import get_embeddings
from langchain_qdrant import QdrantVectorStore
from minio import Minio
from dotenv import load_dotenv

load_dotenv()

class DocumentProcessor:
    def __init__(self):
        # Khởi tạo bộ OCR tiếng Việt
        print(" Đang khởi động bộ mã hóa hình ảnh EasyOCR (Tiếng Việt)...")
        self.reader = easyocr.Reader(['vi'])
        
        # Cấu hình bộ băm nhỏ văn bản
        self.chunker = LawTextChunker(chunk_size=1024, chunk_overlap=128)

    def extract_text_from_pdf(self, pdf_path):  
        print(f" Đang xử lý Digital PDF: {os.path.basename(pdf_path)}")
        doc = fitz.open(pdf_path)
        
        full_text = "" # Gom toàn bộ PDF vào 1 trang liền mạch
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = ""
            
            # Kỹ thuật bóc tách theo "Khối" (Block) để giữ đúng định dạng đoạn văn
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: b[1]) # Sắp xếp dọc từ trên xuống dưới
            
            for b in blocks:
                if b[6] == 0: # 0 nghĩa là Khối chứa chữ
                    block_text = b[4].strip()
                    
                    # BỘ LỌC RÁC: Bỏ qua các khối chỉ chứa số (chính là số trang)
                    if block_text.isdigit() and len(block_text) < 5:
                        continue
                        
                    page_text += block_text + "\n\n"
            
            # Hỗ trợ OCR nếu trang PDF là ảnh scan
            if not page_text.strip():
                print(f" Trang {page_num + 1} không có chữ kỹ thuật số, kích hoạt OCR...")
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                page_text = self.extract_text_from_image_bytes(img_bytes) + "\n\n"
                
            # THUẬT TOÁN NỐI TRANG: 
            # Nếu cuối trang trước KHÔNG có dấu chấm câu, nó sẽ tự động nối liền mạch với trang sau
            if full_text and full_text.strip() and full_text.strip()[-1] not in ['.', ';', ':']:
                full_text = full_text.rstrip() + " " + page_text.lstrip()
            else:
                full_text += page_text

        # Trả về duy nhất 1 tài liệu nguyên vẹn cho cả file PDF
        return [{
            "text": full_text,
            "metadata": {
                "source": os.path.basename(pdf_path),
                "type": "pdf"
            }
        }]

    def extract_text_from_image(self, img_path):
        print(f" Đang xử lý Hình ảnh (OCR): {os.path.basename(img_path)}")
        result = self.reader.readtext(img_path, detail=0)
        text = " ".join(result)
        
        return [{
            "text": text,
            "metadata": {
                "source": os.path.basename(img_path),
                "page": 1,
                "type": "image"
            }
        }]

    def extract_text_from_image_bytes(self, img_bytes):
        result = self.reader.readtext(img_bytes, detail=0)
        return " ".join(result)
    
    def upload_to_minio(self, file_path, object_name="chunk.pkl"):
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
            
    def upload_to_qdrant(self, all_docs):
        print(" Đang nạp Vector vào QDRANT trên Docker...")
        try:
            embeddings = get_embeddings()
            
            QdrantVectorStore.from_documents(
                documents=all_docs,
                embedding=embeddings,
                url=os.getenv("QDRANT_URL"),
                collection_name="luat_giao_thong_viet_nam",
                force_recreate=True
            )
            print(f" HOÀN TẤT! Hãy mở {os.getenv('QDRANT_URL')}/dashboard để xem dữ liệu.")
        except Exception as e:
            print(f" Lỗi khi nạp vào Qdrant: {e}")

    def process_folder(self, input_folder, output_pickle_path):
        all_chunks = []
        all_qdrant_docs = []
        
        if not os.path.exists(input_folder):
            print(f" Thư mục đầu vào {input_folder} không tồn tại!")
            return

        for file_name in os.listdir(input_folder):
            file_path = os.path.join(input_folder, file_name)
            ext = file_name.split('.')[-1].lower()
            
            extracted_docs = []
            if ext == 'pdf':
                extracted_docs = self.extract_text_from_pdf(file_path)
            elif ext in ['png', 'jpg', 'jpeg']:
                extracted_docs = self.extract_text_from_image(file_path)
            else:
                continue
            
            raw_documents = [
                LangchainDocument(page_content=doc["text"], metadata=doc["metadata"]) for doc in extracted_docs
            ]
            
            split_docs = self.chunker.split_documents(raw_documents)
            
            # Gom toàn bộ mảnh của file hiện tại vào giỏ lớn
            all_qdrant_docs.extend(split_docs)
            
            for i, chunk in enumerate(split_docs):
                source_name = chunk.metadata.get('source', 'unknown')
                all_chunks.append({
                    "id": f"{source_name}_c{i}",
                    "text": chunk.page_content,
                    "metadata": chunk.metadata
                })

        os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(all_chunks, f)
            
        print(f"Đã xử lý tự động thành công! Tổng số chunks lưu xuống MinIO: {len(all_chunks)}")
        print(f"File đã được đóng gói và lưu tại: {output_pickle_path}")
        
        self.upload_to_minio(
            file_path=output_pickle_path,
            object_name=os.path.basename(output_pickle_path)
        )
        
        if all_qdrant_docs:
            print(f" Đang chuẩn bị nạp tổng cộng {len(all_qdrant_docs)} đoạn luật vào Qdrant...")
            self.upload_to_qdrant(all_qdrant_docs)

if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_folder(
        input_folder="data/raw", 
        output_pickle_path="data/chunk.pkl"
    )
