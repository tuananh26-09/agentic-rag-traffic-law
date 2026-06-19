#  Hệ Thống Trợ Lý AI Pháp Luật Giao Thông Việt Nam (Agentic RAG)

Dự án này xây dựng một hệ thống hỏi đáp (QA) thông minh chuyên sâu về Luật Giao thông đường bộ Việt Nam. Khác với các mô hình Chatbot thông thường, hệ thống ứng dụng kiến trúc **Agentic RAG (Retrieval-Augmented Generation)**, cho phép AI tự động phân tích tình huống, truy xuất chính xác các điều luật từ cơ sở dữ liệu Vector và đưa ra tư vấn pháp lý có tính chính xác cao, không bịa đặt (Anti-Hallucination).

Đây là sản phẩm phục vụ nghiên cứu và phát triển trong lĩnh vực Trí tuệ nhân tạo (Khóa luận tốt nghiệp).

---

##  Kiến trúc Hệ thống (Tech Stack)

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

##  Hướng dẫn Cài đặt & Triển khai

### 1. Yêu cầu hệ thống (Prerequisites)
* [Docker](https://docs.docker.com/get-docker/) và [Docker Compose](https://docs.docker.com/compose/install/) đã được cài đặt.
* Python 3.10+ (Nếu muốn chạy thử nghiệm môi trường ảo hóa ngoài Docker).
* API Keys từ [Google AI Studio](https://aistudio.google.com/) và [LangSmith](https://smith.langchain.com/).

### 2. Thiết lập môi trường
Clone repository về máy:
```bash
git clone [[https://github.com/tuananh26-09/agentic-rag-traffic-law.git](https://github.com/tuananh26-09/agentic-rag-traffic-law.git)
cd agentic-rag-traffic-law](https://github.com/tuananh26-09/agentic-rag-traffic-law.git)
```



# 🚦 Hệ Thống Trợ Lý AI Pháp Luật Giao Thông Việt Nam (Agentic RAG)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)](https://www.docker.com/)
[![LLM](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-orange.svg)](https://deepmind.google/technologies/gemini/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-green.svg)](https://python.langchain.com/v0.1/docs/langgraph/)

Dự án này xây dựng một hệ thống hỏi đáp (QA) thông minh chuyên sâu về **Luật Giao thông đường bộ Việt Nam**. Khác với các mô hình Chatbot thông thường, hệ thống ứng dụng kiến trúc **Agentic RAG (Retrieval-Augmented Generation)** kết hợp cùng đồ thị tư duy, cho phép AI tự động phân tích tình huống, truy xuất chính xác điều luật từ cơ sở dữ liệu Vector và đưa ra tư vấn pháp lý có tính chính xác cao.

Sản phẩm được phát triển nhằm phục vụ nghiên cứu và trình bày báo cáo khóa luận tốt nghiệp chuyên ngành Trí tuệ nhân tạo.

---

## ✨ Điểm Nổi Bật & Tính Năng Lõi (Key Features)

Dự án tập trung giải quyết các bài toán khó nhất của RAG Tiếng Việt trong lĩnh vực Pháp lý:

* **🧠 Luồng Suy Luận Pháp Lý 4 Bước (Agentic Workflow):** Tránh việc AI vội vàng kết luận, hệ thống ép LLM phải suy luận theo quy trình chuẩn:
  1. Phân tích và liệt kê hành vi thực tế của người hỏi.
  2. Truy xuất và đối chiếu với các `chunks` vi phạm trong CSDL Luật.
  3. Kết luận lỗi (lỗi đơn hoặc lỗi hỗn hợp) dựa trên dữ liệu thật.
  4. Đưa ra lưu ý thẩm quyền (VD: *"Tỷ lệ % lỗi cuối cùng, mức phạt chính xác phải do CSGT xác định tại hiện trường"*).
* **🔍 Hybrid Retrieval (Tìm kiếm Lai):** Kết hợp sức mạnh của tìm kiếm theo từ khóa (BM25) và tìm kiếm theo ngữ nghĩa Vector (Qdrant), giúp không bỏ sót các thuật ngữ luật đặc thù.
* **🛡️ Cơ chế Anti-Hallucination:** Ngăn chặn triệt để tình trạng AI "bịa" luật. Nếu không tìm thấy văn bản quy phạm pháp luật liên quan, hệ thống sẽ tự động từ chối trả lời (Refusal Rate).
* **📊 Observability:** Tích hợp toàn diện `LangSmith` để giám sát chi tiết từng bước chạy (Trace), số lượng Token tiêu thụ và độ trễ (Latency) của mô hình.

---

## 🏗️ Kiến trúc Công nghệ (Tech Stack)

Hệ thống được thiết kế theo chuẩn Microservices, cô lập môi trường hoàn toàn bằng Docker:

| Thành phần | Công nghệ sử dụng | Chức năng chính |
| :--- | :--- | :--- |
| **LLM Engine** | `Google Gemini 2.5 Flash` | Tốc độ cao, ngữ cảnh dài, xử lý tư duy Agent. |
| **Vector DB** | `Qdrant` | Lưu trữ và tìm kiếm ngữ nghĩa các văn bản luật. |
| **Data Lake** | `MinIO` | Lưu trữ file luật (PDF, Markdown) và lịch sử hội thoại (JSON). |
| **Orchestration**| `LangGraph` & `LangChain` | Quản lý đồ thị luồng suy luận và bộ nhớ người dùng. |
| **Embeddings** | `MiniLM-L12-v2` | Chuyển đổi văn bản tiếng Việt sang Vector siêu nhẹ. |
| **Frontend** | `Streamlit` | Giao diện tương tác trực quan (UI). |
| **Backend API** | `FastAPI` | Cổng giao tiếp chuẩn RESTful phục vụ mở rộng Microservices. |

---

## 📂 Cấu Trúc Thư Mục (Project Structure)

```text
📦 agentic-rag-traffic-law
├── 📁 data                   # [Cần thêm tay] Chứa file tài liệu gốc (.md, .pdf)
├── 📁 src                    # Source code lõi của hệ thống
│   ├── 📁 api                # FastAPI endpoints (Microservices)
│   ├── 📁 chunking           # Xử lý chia nhỏ văn bản, bảo toàn cấu trúc luật
│   ├── 📁 embeddings         # Khởi tạo mô hình Embedding HuggingFace
│   ├── 📁 graph              # Xây dựng đồ thị Agentic (nodes, edges, state)
│   ├── 📁 ingestion          # Đọc dữ liệu từ file và đẩy vào CSDL
│   ├── 📁 llm                # Kết nối Gemini 2.5 Flash API
│   ├── 📁 prompts            # Quản lý template câu lệnh (System Prompts)
│   └── 📁 retrieval          # Xử lý truy vấn kết hợp (Qdrant + BM25)
├── 📄 .env.example           # File mẫu chứa các biến môi trường
├── 📄 build_index.py         # Script chạy đường ống Ingestion (Nạp dữ liệu)
├── 📄 docker-compose.yml     # Khởi tạo MinIO, Qdrant và AI Web App
├── 📄 Dockerfile             # Kịch bản đóng gói ứng dụng AI
├── 📄 requirements.txt       # Danh sách thư viện Python
└── 📄 web_app.py             # Script chạy giao diện Streamlit
