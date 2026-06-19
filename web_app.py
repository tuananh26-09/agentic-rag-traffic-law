import streamlit as st
import uuid
import json
import io
import os
from minio import Minio
from src.graph.nodes import build_graph
from dotenv import load_dotenv

load_dotenv()


# 1. CẤU HÌNH TRANG WEB
st.set_page_config(page_title="AI Luật Giao Thông", page_icon="🚦", layout="wide")

# 2. KẾT NỐI MINIO ĐỂ LƯU LỊCH SỬ
# (Tự động dùng localhost nếu chạy ở ngoài, dùng minio nếu chạy trong Docker)
minio_url = os.getenv("MINIO_URL")

minio_client = Minio(
    minio_url,
    access_key=os.getenv("access_key"),
    secret_key=os.getenv("secret_key"),
    secure=False
)

# Tạo Bucket lưu lịch sử
BUCKET_NAME = "chat-history"
if not minio_client.bucket_exists(BUCKET_NAME):
    minio_client.make_bucket(BUCKET_NAME)

# Hàm hỗ trợ: Lưu và Tải file JSON từ MinIO
def save_chat_to_minio(thread_id, messages_list):
    data = json.dumps(messages_list, ensure_ascii=False).encode('utf-8')
    minio_client.put_object(
        BUCKET_NAME,
        f"{thread_id}.json",
        io.BytesIO(data),
        length=len(data),
        content_type="application/json"
    )

def load_chat_from_minio(thread_id):
    try:
        response = minio_client.get_object(BUCKET_NAME, f"{thread_id}.json")
        return json.loads(response.read().decode('utf-8'))
    except Exception:
        return []

# 3. KHỞI TẠO BỘ NHỚ
if "app" not in st.session_state:
    st.session_state.app = build_graph()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Lấy tin nhắn của phiên hiện tại từ MinIO
current_messages = load_chat_from_minio(st.session_state.thread_id)

# 4. THANH BÊN (SIDEBAR) - LỊCH SỬ MINIO
with st.sidebar:
    st.title("📚 Lịch sử Chat")
    
    # Nút tạo Chat mới
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
        
    st.divider()
    
    # Quét Bucket 'chat-history' để lấy danh sách các phiên chat
    st.markdown("**Các phiên chat trước:**")
    objects = minio_client.list_objects(BUCKET_NAME)
    
    for obj in objects:
        sess_id = obj.object_name.replace(".json", "")
        # Đọc 1 đoạn JSON ngắn để làm tiêu đề
        try:
            old_chat = json.loads(minio_client.get_object(BUCKET_NAME, obj.object_name).read().decode('utf-8'))
            if old_chat:
                first_msg = old_chat[0]['content']
                title = first_msg[:25] + "..." if len(first_msg) > 25 else first_msg
                if st.button(f"💬 {title}", key=sess_id, use_container_width=True):
                    st.session_state.thread_id = sess_id
                    st.rerun()
        except:
            continue

# 5. GIAO DIỆN CHAT
st.title("🚦 Trợ Lý AI - Luật Giao Thông Việt Nam")

# Hiển thị tin nhắn cũ
for msg in current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Xử lý khi người dùng nhắn tin
if prompt := st.chat_input("Hỏi AI về luật giao thông..."):
    # 1. In ra màn hình và lưu vào mảng hiện tại
    with st.chat_message("user"):
        st.markdown(prompt)
    current_messages.append({"role": "user", "content": prompt})
    save_chat_to_minio(st.session_state.thread_id, current_messages)

    # 2. Gọi AI suy luận
    with st.chat_message("assistant"):
        with st.spinner("AI đang phân tích luật..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = st.session_state.app.invoke({"question": prompt}, config=config)
            answer = result["answer"]
            st.markdown(answer)
            
    # 3. Lưu câu trả lời của AI và up lại lên MinIO
    current_messages.append({"role": "assistant", "content": answer})
    save_chat_to_minio(st.session_state.thread_id, current_messages)