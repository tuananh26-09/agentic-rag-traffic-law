#  Hệ Thống Trợ Lý AI Pháp Luật Giao Thông Việt Nam (Agentic RAG)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)](https://www.docker.com/)
[![LLM](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-orange.svg)](https://deepmind.google/technologies/gemini/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-green.svg)](https://python.langchain.com/v0.1/docs/langgraph/)

Dự án này xây dựng một hệ thống hỏi đáp (QA) thông minh chuyên sâu về **Luật Giao thông đường bộ Việt Nam**. Khác với các mô hình Chatbot thông thường, hệ thống ứng dụng kiến trúc **Agentic RAG (Retrieval-Augmented Generation)** kết hợp cùng đồ thị tư duy, cho phép AI tự động phân tích tình huống, truy xuất chính xác điều luật từ cơ sở dữ liệu Vector và đưa ra tư vấn pháp lý có tính chính xác cao.

Sản phẩm được phát triển nhằm phục vụ nghiên cứu và trình bày báo cáo khóa luận tốt nghiệp chuyên ngành Trí tuệ nhân tạo.

---

##  Điểm Nổi Bật & Tính Năng Lõi (Key Features)

Dự án tập trung giải quyết các bài toán khó nhất của RAG Tiếng Việt trong lĩnh vực Pháp lý:

* ** Luồng Suy Luận Pháp Lý 4 Bước (Agentic Workflow):** Tránh việc AI vội vàng kết luận, hệ thống ép LLM phải suy luận theo quy trình chuẩn:
  1. Phân tích và liệt kê hành vi thực tế của người hỏi.
  2. Truy xuất và đối chiếu với các `chunks` vi phạm trong CSDL Luật.
  3. Kết luận lỗi (lỗi đơn hoặc lỗi hỗn hợp) dựa trên dữ liệu thật.
  4. Đưa ra lưu ý thẩm quyền (VD: *"Tỷ lệ % lỗi cuối cùng, mức phạt chính xác phải do CSGT xác định tại hiện trường"*).
* ** Hybrid Retrieval (Tìm kiếm Lai):** Kết hợp sức mạnh của tìm kiếm theo từ khóa (BM25) và tìm kiếm theo ngữ nghĩa Vector (Qdrant), giúp không bỏ sót các thuật ngữ luật đặc thù.
* ** Cơ chế Anti-Hallucination:** Ngăn chặn triệt để tình trạng AI "bịa" luật. Nếu không tìm thấy văn bản quy phạm pháp luật liên quan, hệ thống sẽ tự động từ chối trả lời (Refusal Rate).
* ** Observability:** Tích hợp toàn diện `LangSmith` để giám sát chi tiết từng bước chạy (Trace), số lượng Token tiêu thụ và độ trễ (Latency) của mô hình.

---

##  Kiến trúc Công nghệ (Tech Stack)

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

##  Cấu Trúc Thư Mục (Project Structure)

```text
📦 agentic-rag-traffic-law
├── 📁 data                   # Chứa file tài liệu gốc (.md, .pdf)
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
```

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
### 3) Chuẩn bị API keys và file `.env`

**A) Lấy Gemini key từ Google AI Studio**

1. Mở trang `https://aistudio.google.com/` và đăng nhập bằng tài khoản Google.
2. Nhìn sang menu bên trái, chọn **Get API key**.
3. Bấm vào nút **Create API key** (Tạo khóa API).
4. Nếu đây là lần đầu tiên, hệ thống sẽ yêu cầu tạo một project Google Cloud. Bạn cứ bấm đồng ý để tạo mới.
5. Sau khi tạo xong, sao chép chuỗi mã Key đó.

**B) Lấy LangSmith key (Dùng để giám sát hệ thống)**

1. Mở trang `https://smith.langchain.com/` và đăng nhập hoặc tạo tài khoản.
2. Nhìn xuống góc dưới cùng bên trái, bấm vào biểu tượng bánh răng **Settings** -> Chọn **API Keys**.
3. Bấm **Create API Key**. 
4. *Lưu ý quan trọng:* Ở bảng hiện ra, đảm bảo bạn chọn loại **Personal** (Cá nhân).
5. Sao chép chuỗi mã Key (thường bắt đầu bằng `lsv2_...`).

**C) Tạo file `.env` tại thư mục gốc của dự án**

Tạo một file có tên chính xác là `.env` tại thư mục gốc và điền các API Key của bạn vào:
```bash
# Lõi AI
GOOGLE_API_KEY=dien_api_key_cua_ban_vao_day
embedding_model=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
llm_model=gemini-2.5-flash

# Cơ sở dữ liệu
QDRANT_URL=http://qdrant:6333
MINIO_URL=minio:9000
access_key=admin
secret_key=thay_bang_mat_khau_cua_ban

# Giám sát (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=dien_langsmith_key_vao_day
LANGCHAIN_PROJECT=LuatGiaoThong
```

### 4. Chuẩn bị Dữ liệu Huấn luyện (Data Ingestion)
* Ta cần phải tải dữ liệu trên ```https://vbpl.vn/van-ban/trung-uong``` hoặc ```https://thuvienphapluat.vn/phap-luat/ho-tro-phap-luat/luat-giao-thong-2026-va-cac-nghi-dinh-thong-tu-huong-dan-moi-nhat-luat-giao-thong-2026-gom-cac-luat-325614-246822.html``` ở định dạng `.docx`.
* Rồi sau đó chạy file `convert_luat.py` để chuyển dữ liệu từ `.docx` sang `.md`

### 5. Khởi chạy toàn hệ thống bằng Docker
Chỉ với 1 câu lệnh, Docker sẽ tự động pull các image cần thiết (Qdrant, MinIO), cài đặt thư viện và khởi động toàn bộ Microservices:

```Bash
docker-compose up --build -d
```
Sau khi hệ thống báo Started, hãy nạp dữ liệu từ thư mục data/ vào cơ sở dữ liệu Vector bằng lệnh:

```Bash
docker exec -it ai_web_app python build_index.py
```
## Sử Dụng Hệ Thống
Khi toàn bộ Container đã chuyển trạng thái sang Running, bạn có thể truy cập các thành phần qua trình duyệt:

* Giao Diện AI (Streamlit): http://localhost:8501

* Bảng Điều Khiển MinIO: http://localhost:9001 (Tài khoản: admin / password123)

* Bảng Điều Khiển Qdrant: http://localhost:6333/dashboard

Kịch bản hỏi đáp mẫu:

"Tôi đi xe máy Honda Vision bị CSGT tuýt còi vì lỗi chuyển hướng không xi nhan, đồng thời tôi quên mang bằng lái xe. Xin hỏi tổng mức phạt của tôi là bao nhiêu và tôi có bị giữ xe không?"

## Đánh giá & Kiểm thử (Evaluation)
Dự án áp dụng bộ tiêu chuẩn Evaluation dành cho LLM Operations (LLMOps):

Tracing: Sử dụng bảng điều khiển LangSmith để gỡ lỗi và trực quan hóa từng bước trong đồ thị LangGraph. Đảm bảo luồng Retriever kéo đúng văn bản và Generator sử dụng đúng ngữ cảnh.

Accuracy: Hệ thống đảm bảo tính Faithfulness (Trung thành với văn bản gốc) và Answer Relevance (Sát với câu hỏi người dùng).
