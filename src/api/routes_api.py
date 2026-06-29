# api/routes_api.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import os
import shutil
import uuid
from src.graph.nodes import build_graph
from src.ingestion.pipeline import DocumentProcessor

router = APIRouter()
app = build_graph() # Load đồ thị một lần duy nhất
processor = DocumentProcessor()

@router.post("/chat")
async def chat(question: str, session_id: str = "default"):
    config = {"configurable": {"thread_id": session_id}}
    result = app.invoke({"question": question}, config=config)
    return {"answer": result["answer"]}


@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    API Upload tài liệu (Hỗ trợ PDF và nhiều ảnh cùng lúc).
    Tự động OCR, Chunking, Embedding và lưu vào Qdrant + MinIO.
    """
    # 1. Tạo một thư mục tạm duy nhất cho lượt upload này
    session_id = uuid.uuid4().hex
    session_folder = f"data/temp_uploads_{session_id}"
    os.makedirs(session_folder, exist_ok=True)
    
    try:
        saved_files = []
        # 2. Lưu các file gửi qua API xuống ổ cứng tạm
        for file in files:
            file_path = os.path.join(session_folder, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
            
        print(f"\n📥 [API UPLOAD] Đã nhận {len(files)} file: {saved_files}")
        
        # 3. Kích hoạt Cỗ máy Pipeline (Tận dụng logic gộp ảnh siêu việt của bạn)
        output_pickle = os.path.join(session_folder, f"backup_chunks_{session_id[:6]}.pkl")
        processor.process_folder(input_folder=session_folder, output_pickle_path=output_pickle)
        
        # 4. Lưu Bản Gốc lên MinIO để lưu trữ (Backup File Raw)
        for filename in saved_files:
            local_path = os.path.join(session_folder, filename)
            # Up thẳng vào bucket rag-data với tiền tố raw_docs/
            processor.upload_to_minio(file_path=local_path, object_name=f"raw_docs/{filename}")
            
        return {
            "status": "success",
            "message": f"Đã bóc tách, băm nhỏ, nhúng Vector và nạp thành công {len(files)} tài liệu vào Qdrant!",
            "files_processed": saved_files
        }
        
    except Exception as e:
        print(f"❌ [API UPLOAD] Lỗi: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 5. Rút êm: Dọn sạch sẽ thư mục tạm sau khi làm xong để nhẹ máy
        if os.path.exists(session_folder):
            shutil.rmtree(session_folder)
            print(f"🧹 [API UPLOAD] Đã dọn dẹp thư mục rác: {session_folder}")
