# 🚦 Hệ Thống Trợ Lý AI Pháp Luật Giao Thông Việt Nam (Agentic RAG)

Dự án này xây dựng một hệ thống hỏi đáp (QA) thông minh chuyên sâu về Luật Giao thông đường bộ Việt Nam. Khác với các mô hình Chatbot thông thường, hệ thống ứng dụng kiến trúc **Agentic RAG (Retrieval-Augmented Generation)**, cho phép AI tự động phân tích tình huống, truy xuất chính xác các điều luật từ cơ sở dữ liệu Vector và đưa ra tư vấn pháp lý có tính chính xác cao, không bịa đặt (Anti-Hallucination).

Đây là sản phẩm phục vụ nghiên cứu và phát triển trong lĩnh vực Trí tuệ nhân tạo (Khóa luận tốt nghiệp).

---

## 🏗️ Kiến trúc Hệ thống (Tech Stack)

Hệ thống được thiết kế theo chuẩn Microservices, đóng gói hoàn toàn bằng Docker, bao gồm các thành phần cốt lõi:

* **Mô hình Ngôn ngữ Lớn (LLM):** `Google Gemini 2.5 Flash` (Tốc độ cao, tối ưu context dài).
* **Vector Database:** `Qdrant` (Lưu trữ và tìm kiếm ngữ nghĩa các văn bản luật).
* **Object Storage (Data Lake):** `MinIO` (Lưu trữ lịch sử hội thoại JSON và tệp tin).
* **Orchestration / Agent:** `LangGraph` & `LangChain` (Quản lý đồ thị luồng suy luận và bộ nhớ).
* **Embeddings:** `HuggingFace (MiniLM-L12-v2)` (Chuyển đổi văn bản tiếng Việt sang Vector).
* **Frontend UI:** `Streamlit` (Giao diện người dùng tương tác trực quan).
* **Backend API:** `FastAPI` (Cổng giao tiếp chuẩn RESTful).
* **Observability:** `LangSmith` (Giám sát, theo dõi luồng chạy và đánh giá token).

---

## 🚀 Hướng dẫn Cài đặt & Triển khai

### 1. Yêu cầu hệ thống (Prerequisites)
* [Docker](https://docs.docker.com/get-docker/) và [Docker Compose](https://docs.docker.com/compose/install/) đã được cài đặt.
* Python 3.10+ (Nếu muốn chạy thử nghiệm môi trường ảo hóa ngoài Docker).
* API Keys từ [Google AI Studio](https://aistudio.google.com/) và [LangSmith](https://smith.langchain.com/).

### 2. Thiết lập môi trường
Clone repository về máy:
```bash
git clone [[https://github.com/tuananh26-09/agentic-rag-traffic-law.git](https://github.com/tuananh26-09/agentic-rag-traffic-law.git)
cd agentic-rag-traffic-law](https://github.com/tuananh26-09/agentic-rag-traffic-law.git)
