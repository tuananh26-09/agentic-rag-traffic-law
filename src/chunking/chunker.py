import re
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

class LawTextChunker:
    def __init__(self, chunk_size=1024, chunk_overlap=128):
        headers_to_split_on = [
            ("#", "Tên Chương"), 
            ("##", "Tên Điều Luật"), 
        ]
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False 
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=True,
            keep_separator=True,
            separators=[
            
                r"\n# ",             # Tiêu đề H1
                r"\n## ",            # Tiêu đề H2
                r"\n### ",           # Tiêu đề H3
                
                r"\n## Điều \d+",     # Ưu tiên 1: Cắt theo ## Điều (đã được tự động tiêm vào)
                r"\n\d+\.",          # Ưu tiên 2: Cắt theo Khoản (1., 2., 3.)
                r"\n[a-zđ]\)",       # Ưu tiên 3: Cắt theo Điểm (a), b), đ))
                r"\n-\s",            # Gạch đầu dòng
                r"\n\*\s",           # Dấu sao đầu dòng
                r"\n\n",             # Ưu tiên 4: Đoạn văn
                r"\n",               # Ưu tiên 5: Dòng
                r"(?<=[.?!])\s+",    # Ưu tiên 6: Câu
                r"\s+",              # Ưu tiên cuối: Khoảng trắng
                ""                   # Cắt theo từng ký tự (Character) - Khóa an toàn cuối cùng
                
            ]
        )

    def inject_markdown_headers(self, text):
        """
        Hàm tự động tiêm thẻ Markdown dựa trên quy tắc Regex.
        Kế thừa logic chuẩn xác từ convert_luat.py của bạn.
        """
        # Thêm "# " vào các dòng bắt đầu bằng chữ "Chương" và số La Mã
        text = re.sub(r"(?im)^(Chương\s+[IVXLCDM]+)", r"# \1", text)
        
        # Thêm "## " vào các dòng bắt đầu bằng chữ "Điều" và số
        text = re.sub(r"(?im)^(Điều\s+\d+)", r"## \1", text)
        
        return text

    def split_documents(self, documents):
        final_chunks = []
        for doc in documents:
            # 1. LƯU LẠI METADATA GỐC (từ PDF/Ảnh gồm source, page...)
            original_metadata = doc.metadata.copy()
            
            # 2. PHỤC CHẾ THẺ MARKDOWN (Tiêm # và ## vào văn bản thô)
            processed_text = self.inject_markdown_headers(doc.page_content)
            
            # Bước 1: Chia theo Điều luật 
            md_splits = self.md_splitter.split_text(processed_text)
            
            # Bước 2: Băm nhỏ các Điều luật dài
            for split in md_splits:
                # 3. GỘP METADATA: Lấy gốc ráp chung với Tên Chương/Điều
                merged_metadata = original_metadata.copy()
                merged_metadata.update(split.metadata)
                
                # Ép định dạng Document chuẩn
                base_doc = Document(page_content=split.page_content, metadata=merged_metadata)
                
                # Băm nhỏ
                sub_chunks = self.text_splitter.split_documents([base_doc])
                
                # Cấy tiền tố bối cảnh cho AI hiểu
                for sub_chunk in sub_chunks:
                    chuong = sub_chunk.metadata.get("Tên Chương", "")
                    dieu_luat = sub_chunk.metadata.get("Tên Điều Luật", "")
                    context_prefix = ""
                    
                    if chuong:
                        context_prefix += f"[{chuong}] "
                    if dieu_luat:
                        context_prefix += f"[{dieu_luat}]\n"
                        
                    if dieu_luat and dieu_luat not in sub_chunk.page_content:
                        sub_chunk.page_content = context_prefix + sub_chunk.page_content
                        
                final_chunks.extend(sub_chunks)
                
        return final_chunks
