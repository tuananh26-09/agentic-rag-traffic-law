from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

class LawTextChunker:
    def __init__(self, chunk_size=1024, chunk_overlap=128):
        # 1. BỘ CẮT THEO CẤU TRÚC MARKDOWN (Giữ lại tiêu đề Điều luật)
        # Nó sẽ tìm các thẻ ## và coi đó là "Tên Điều Luật"
        headers_to_split_on = [
            ("##", "Tên Điều Luật"), 
            ("#", "Tên Chương"), 
        ]
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False # Giữ lại chữ "## Điều..." trong văn bản
        )
        
        # 2. BỘ BĂM NHỎ (Dành cho các Điều/Khoản quá dài)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Ưu tiên tìm chữ "\nĐiều " để cắt trước. 
            # Nếu Điều đó quá dài, nó mới cắt tiếp ở "\n1." (Khoản) và "\na)" (Điểm)
            separators=[
                "\n## Điều ", 
                "\nĐiều ",
                "\n1.", "\n2.", "\n3.", "\n4.", "\n5.","\n6.", "\n7.", "\n8.", "\n9.", "\n10.", "\n11.", "\n12.", "\n13.", "\n14.", "\n15.", "\n16.", "\n17.", "\n18.", "\n19.", "\n20.","\n21.", "\n22.", "\n23.", "\n24.", "\n25.", "\n26.", "\n27.", "\n28.", "\n29.", "\n30.",
                "\na)", "\nb)", "\nc)", "\nd)", "\nđ)", "\ne)", "\ng)", "\nh)", "\ni)", "\nj)", "\nk)", "\nl)", "\nm)", "\nn)", "\no)", "\np)", "\nq)", "\nr)", "\ns)", "\nt)", "\nu)", "\nv)", "\nw)", "\nx)", "\ny)", "\nz)",
                "\n\n", "\n", ".", " "
            ]
        )

    def split_documents(self, documents):
        final_chunks = []
        for doc in documents:
            # Bước 1: Chia theo Điều luật và dán tên Điều vào Metadata
            md_splits = self.md_splitter.split_text(doc.page_content)
            
            # Bước 2: Băm nhỏ các Điều luật dài quá 2000 ký tự
            for split in md_splits:
                # Ép lại thành định dạng Document chuẩn của Langchain
                base_doc = Document(page_content=split.page_content, metadata=split.metadata)
                
                # Băm nhỏ
                sub_chunks = self.text_splitter.split_documents([base_doc])
                
                # Để LLM đọc hiểu tốt hơn, ta tự động ghép tên Điều vào đầu mỗi đoạn chunk con
                for sub_chunk in sub_chunks:
                    # Rút tên Điều luật từ metadata (do md_splitter tự tạo ra)
                    dieu_luat = sub_chunk.metadata.get("Tên Điều Luật", "")
                    if dieu_luat and dieu_luat not in sub_chunk.page_content:
                        sub_chunk.page_content = f"[{dieu_luat}]\n" + sub_chunk.page_content
                        
                final_chunks.extend(sub_chunks)
                
        return final_chunks
    
    
