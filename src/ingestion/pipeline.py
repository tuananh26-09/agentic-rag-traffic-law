import os
import re
import fitz
import pickle
import numpy as np
import gc
import cv2
from PIL import Image
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
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
        print("Đang khởi động PaddleOCR (Chuyên gia Bố cục)...")
        self.detector = PaddleOCR(use_angle_cls=True, lang='vi',rec = False, show_log=False)
        print("Đang khởi động VietOCR (Chuyên gia TIếng Việt)...")
        config = Cfg.load_config_from_name('vgg_transformer')
        
        config['device'] = 'cpu'
        # Cấu hình bộ băm nhỏ văn bản
        self.recognizer = Predictor(config)
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
        print(f" Đang xử lý Hình ảnh Hybrid (Paddle+VietOCR): {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        
        if img is None:
            print(f" Không thể đọc file hình ảnh: {img_path}")
            return []
        
        # 1. PADDLE-OCR LÀM HOA TIÊU TÌM KHUNG CHỮ
        # Kết quả trả về là tọa độ 4 góc của từng dòng chữ
        boxes = self.detector.ocr(img_path, det=True, rec=False, cls=True)
        text_lines = []
        
        if boxes and boxes[0]:
            # Sắp xếp các dòng chữ từ trên xuống dưới theo tọa độ trục Y
            sorted_boxes = sorted(boxes[0], key=lambda x: x[0][1])
            
            for box in sorted_boxes:
                # Trích xuất tọa độ điểm trên cùng bên trái và dưới cùng bên phải
                box = np.array(box).astype(np.int32)
                
                x_min = np.min(box[:, 0])
                x_max = np.max(box[:, 0])
                y_min = np.min(box[:, 1])
                y_max = np.max(box[:, 1])
                
                pad = 4
                y_min = max(0, y_min - pad)
                y_max = min(img.shape[0], y_max + pad)
                x_min = max(0, x_min - pad)
                x_max = min(img.shape[1], x_max + pad)
                
                # Cắt gọn cái khung chứa dòng chữ đó ra
                cropped_img = img[y_min:y_max, x_min:x_max]
                
                if cropped_img.size == 0 or cropped_img.shape[0] < 5 or cropped_img.shape[1] < 5:
                    continue
                    
                # Chuyển OpenCV Image (BGR) sang PIL Image (RGB) cho VietOCR hiểu
                cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cropped_img_rgb)
                
                # 2. VIET-OCR LÀM XẠ THỦ ĐỌC CHỮ
                text = self.recognizer.predict(pil_img)
                clean_text = text.strip()
                
                if clean_text:
                    # 🟢 BỘ LỌC 1: Bỏ qua các số trang rác (VD: số "8" đứng trơ trọi)
                    if clean_text.isdigit() and len(clean_text) < 4:
                        continue
                        
                    # 🟢 BỘ LỌC 2: Khử ảo giác dấu câu của VietOCR
                    if clean_text.endswith("Thuận"):
                        clean_text = clean_text[:-5] + ":"
                    elif clean_text.endswith("thuận"):
                        clean_text = clean_text[:-5] + ","
                        
                    text_lines.append(text.strip())
                
        # Dọn RAM
        gc.collect()
        
        # 🟢 BÍ KÍP TỐI THƯỢNG: GHÉP DÒNG THÔNG MINH (Heuristic Line Merging)
        healed_lines = []
        current_sentence = ""
        
        for line in text_lines:
            if not current_sentence:
                current_sentence = line
            else:
                # Nếu dòng trước KHÔNG kết thúc bằng dấu chấm (.), hai chấm (:), chấm phẩy (;)
                # thì chắc chắn nó là một câu đang bị rớt dòng -> Nối lại bằng 1 khoảng trắng!
                if current_sentence[-1] not in ['.', ':', ';']:
                    current_sentence += " " + line
                else:
                # KIỂM TRA CẤU TRÚC: Dòng này có phải là bắt đầu của một Điều/Khoản/Điểm mới không?
                # Bắt các pattern như: "Điều 8.", "7.", "a)", "i)"
                    is_new_paragraph = re.match(r'^(Điều\s+\d+|[a-zđ]\)|[0-9]+\.)', line, re.IGNORECASE)
                    
                    if is_new_paragraph:
                        # Chắc chắn là đoạn mới, cất câu cũ đi và xuống dòng!
                        healed_lines.append(current_sentence)
                        current_sentence = line
                    elif current_sentence[-1] not in ['.', ':', ';']:
                        # Không phải đoạn mới, và câu trước cũng chưa hết -> Nối liền (cách 1 khoảng trắng)
                        current_sentence += " " + line
                    else:
                        # Không phải đoạn mới, nhưng câu trước đã có dấu chấm -> Xuống dòng
                        healed_lines.append(current_sentence)
                        current_sentence = line
        # Đẩy câu cuối cùng vào danh sách
        if current_sentence:
            healed_lines.append(current_sentence)
        
        final_text = "\n\n".join(healed_lines)
        print(f"   -> Đã bóc được {len(final_text)} ký tự (Hybrid). Trích đoạn: {final_text[:80]}...")
        
        return [{
            "text": final_text,
            "metadata": {
                "source": os.path.basename(img_path),
                "page": 1,
                "type": "image"
            }
        }]

    def extract_text_from_image_bytes(self, img_bytes):
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Gọi Hoa tiêu PaddleOCR tìm khung
        boxes = self.detector.ocr(img, det=True, rec=False, cls=True)
        text_lines = []
        
        if boxes and boxes[0]:
            sorted_boxes = sorted(boxes[0], key=lambda x: x[0][1])
            for box in sorted_boxes:
                box = np.array(box).astype(np.int32)
                x_min, x_max = np.min(box[:, 0]), np.max(box[:, 0])
                y_min, y_max = np.min(box[:, 1]), np.max(box[:, 1])
                
                cropped_img = img[y_min:y_max, x_min:x_max]
                if cropped_img.size == 0:
                    continue
                    
                cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cropped_img_rgb)
                
                # Gọi Xạ thủ VietOCR đọc chữ
                text = self.recognizer.predict(pil_img)
                text_lines.append(text)
                
        return " ".join(text_lines)
    
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
                collection_name="test_ocr",
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
        
        file_names = sorted(os.listdir(input_folder))
        
        combined_image_text = ""
        image_sources = []

        for file_name in file_names:
            file_path = os.path.join(input_folder, file_name)
            ext = file_name.split('.')[-1].lower()
            
            if ext == 'pdf':
                extracted_docs = self.extract_text_from_pdf(file_path)
                raw_documents = [LangchainDocument(page_content=doc["text"], metadata=doc["metadata"]) for doc in extracted_docs]

                split_docs = self.chunker.split_documents(raw_documents)
                all_qdrant_docs.extend(split_docs)
                
            elif ext in ['png', 'jpg', 'jpeg']:
                extracted_docs = self.extract_text_from_image(file_path)
                if extracted_docs:
                    combined_image_text += extracted_docs[0]["text"] + "\n\n"
                    image_sources.append(file_name)
            else:
                continue
            
        # 🟢 BÍ KÍP MỚI: GẮN CHIP ĐỊNH VỊ METADATA TRƯỚC KHI CHẶT
        if combined_image_text:
            print(f" Đang gộp {len(image_sources)} ảnh và gắn thẻ Metadata thông minh...")
            
            docs_to_chunk = []
            current_chuong = "Chưa xác định"
            current_dieu = "Chưa xác định"
            current_content = []
            
            # Quét từng dòng của Mega-Document
            for line in combined_image_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # 1. Bắt tín hiệu Chương mới
                if re.match(r'^Chương\s+[IVXLCDM]+', line, re.IGNORECASE):
                    current_chuong = line
                    current_content.append(line)
                    
                # 2. Bắt tín hiệu Điều mới
                elif re.match(r'^Điều\s+\d+', line, re.IGNORECASE):
                    # Đóng gói Điều cũ (nếu có) thành 1 Document hoàn chỉnh mang đầy đủ thẻ tên
                    if current_content:
                        docs_to_chunk.append(
                            LangchainDocument(
                                page_content="\n\n".join(current_content),
                                metadata={
                                    "source": f"Gộp ảnh ({image_sources[0]} -> {image_sources[-1]})",
                                    "type": "merged_images",
                                    "Tên Chương": current_chuong,
                                    "Tên Điều Luật": current_dieu
                                }
                            )
                        )
                        current_content = [] # Dọn giỏ để chứa nội dung của Điều mới
                        
                    current_dieu = line
                    current_content.append(line)
                    
                # 3. Các dòng chữ bình thường (Khoản, Điểm)
                else:
                    current_content.append(line)
                    
            # Đóng gói phần nội dung cuối cùng còn sót lại
            if current_content:
                docs_to_chunk.append(
                    LangchainDocument(
                        page_content="\n\n".join(current_content),
                        metadata={
                            "source": f"Gộp ảnh ({image_sources[0]} -> {image_sources[-1]})",
                            "type": "merged_images",
                            "Tên Chương": current_chuong,
                            "Tên Điều Luật": current_dieu
                        }
                    )
                )
                
            # Đưa toàn bộ các Điều luật (Đã được gắn mác chuẩn chỉ 100%) cho Chunker
            split_docs = self.chunker.split_documents(docs_to_chunk)
            all_qdrant_docs.extend(split_docs)
            
            # Lưu các mảnh vỡ (Chunks) vào danh sách để tải lên MinIO
            for i, chunk in enumerate(split_docs):
                all_chunks.append({
                    "id": f"chunk_{i}",
                    "text": chunk.page_content,
                    "metadata": chunk.metadata
                })

        os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(all_chunks, f)
            
        print(f" Đã xử lý tự động thành công! Tổng số chunks lưu xuống MinIO: {len(all_chunks)}")
        
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
        output_pickle_path="data/chunk_test.pkl"
    )
