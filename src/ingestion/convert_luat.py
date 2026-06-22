import os
import re
from docx import Document 

def docx_to_md(docx_path, md_path):
    """Đọc file Word (.docx) và tự động bóc tách Heading (Chương, Điều) chuẩn xác"""
    doc = Document(docx_path)
    
    # Biểu thức chính quy (Regex) bắt Chương và Điều
    chuong_pattern = re.compile(r"^Chương\s+[IVXLCDM]+", re.IGNORECASE)
    dieu_pattern = re.compile(r"^Điều\s+\d+", re.IGNORECASE)
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Tài liệu: {os.path.basename(docx_path)}\n\n")
        
        for para in doc.paragraphs:
            text = para.text.strip()
            
            # Bỏ qua các dòng trống để file md gọn gàng
            if not text:
                continue
            
            # Kiểm tra xem dòng này có được in đậm không
            is_bold = False
            for run in para.runs:
                if run.text.strip(): # Chỉ lấy text có ý nghĩa để xét in đậm
                    if run.bold:
                        is_bold = True
                    break
            
            # Bộ lọc kép: Vừa in đậm VỪA đúng cấu trúc từ khóa
            if is_bold and chuong_pattern.match(text):
                f.write(f"# {text}\n\n")
            elif is_bold and dieu_pattern.match(text):
                f.write(f"## {text}\n\n")
            else:
                # Tất cả nội dung còn lại quy về chữ thường
                f.write(f"{text}\n\n")

def process_directory(input_dir="data/raw", output_dir="data/clean"):
    """Quét thư mục data_raw và tự động chuyển MỌI file .docx sang .md"""
    # Tạo thư mục đầu ra nếu chưa có
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        name, ext = os.path.splitext(filename)
        md_path = os.path.join(output_dir, f"{name}.md")
        
        # XỬ LÝ ĐỊNH DẠNG .DOCX
        if ext.lower() == '.docx':
            print(f"Đang xử lý: {filename}...")
            try:
                docx_to_md(file_path, md_path)
                print(f" Đã chuyển thành công: {name}.md")
            except Exception as e:
                print(f" Lỗi khi xử lý {filename}: {e}")
        else:
            print(f" Bỏ qua {filename} (Chỉ hỗ trợ file .docx)")

# Chạy thử
process_directory()