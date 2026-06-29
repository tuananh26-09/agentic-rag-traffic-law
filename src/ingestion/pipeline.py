import os
import re
import pickle
from langchain_core.documents import Document as LangchainDocument
from src.chunking.chunker import LawTextChunker
from src.ingestion.extractor import OCRExtractor
from src.utils.storage import upload_to_minio
from src.vectordb.vector_store import upload_to_qdrant
from dotenv import load_dotenv

load_dotenv()

class DocumentProcessor:
    def __init__(self):
        self.extractor = OCRExtractor()
        self.chunker = LawTextChunker(chunk_size=1024, chunk_overlap=128)

    def process_folder(self, input_folder, output_pickle_path):
        all_chunks = []
        all_qdrant_docs = []
        
        if not os.path.exists(input_folder):
            print(f" Thư mục đầu vào {input_folder} không tồn tại!")
            return
        
        file_names = sorted(os.listdir(input_folder))
        combined_image_text = ""
        image_sources = []

        for file_name in file_names:
            file_path = os.path.join(input_folder, file_name)
            ext = file_name.split('.')[-1].lower()
            
            if ext == 'pdf':
                extracted_docs = self.extractor.extract_text_from_pdf(file_path)
                raw_documents = [LangchainDocument(page_content=doc["text"], metadata=doc["metadata"]) for doc in extracted_docs]
                all_qdrant_docs.extend(self.chunker.split_documents(raw_documents))
                
            elif ext in ['png', 'jpg', 'jpeg']:
                extracted_docs = self.extractor.extract_text_from_image(file_path)
                if extracted_docs:
                    combined_image_text += extracted_docs[0]["text"] + "\n\n"
                    image_sources.append(file_name)
                    
        # Xử lý Mega-Document cho Ảnh
        if combined_image_text:
            print(f" Đang gộp {len(image_sources)} ảnh và gắn thẻ Metadata thông minh...")
            docs_to_chunk = []
            current_chuong, current_dieu = "Chưa xác định", "Chưa xác định"
            current_content = []
            
            for line in combined_image_text.split('\n'):
                line = line.strip()
                if not line: continue
                    
                if re.match(r'^Chương\s+[IVXLCDM]+', line, re.IGNORECASE):
                    current_chuong = line
                    current_content.append(line)
                elif re.match(r'^Điều\s+\d+', line, re.IGNORECASE):
                    if current_content:
                        docs_to_chunk.append(LangchainDocument(
                            page_content="\n\n".join(current_content),
                            metadata={"source": f"Gộp ({image_sources[0]} -> {image_sources[-1]})", "type": "merged_images", "Tên Chương": current_chuong, "Tên Điều Luật": current_dieu}
                        ))
                        current_content = [] 
                    current_dieu = line
                    current_content.append(line)
                else:
                    current_content.append(line)
                    
            if current_content:
                docs_to_chunk.append(LangchainDocument(
                    page_content="\n\n".join(current_content),
                    metadata={"source": f"Gộp ({image_sources[0]} -> {image_sources[-1]})", "type": "merged_images", "Tên Chương": current_chuong, "Tên Điều Luật": current_dieu}
                ))
                
            split_docs = self.chunker.split_documents(docs_to_chunk)
            all_qdrant_docs.extend(split_docs)
            
        for i, chunk in enumerate(all_qdrant_docs):
            all_chunks.append({"id": f"chunk_{i}", "text": chunk.page_content, "metadata": chunk.metadata})

        # Lưu dữ liệu
        os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(all_chunks, f)
            
        print(f" Đã xử lý xong! Chuẩn bị tải dữ liệu lên Server...")
        upload_to_minio(output_pickle_path, os.path.basename(output_pickle_path))
        
        if all_qdrant_docs:
            upload_to_qdrant(all_qdrant_docs)

if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_folder("data/raw", "data/chunk_test.pkl")
