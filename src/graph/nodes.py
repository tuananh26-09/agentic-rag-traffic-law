import pickle
import os
import requests
from minio import Minio
from langgraph.graph import StateGraph, START, END
from src.graph.state import RAGState
from src.vectordb.vector_store import get_retriever as get_vector_retriever
from src.retrieval.retriever import BM25Retriever, HybridRetriever
from src.llm.llm_client import get_llm
from src.prompts.prompt_templates import BASELINE_PROMPT, REWRITE_PROMPT
from src.utils.helpers import parse_focused_answer 
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from langchain_core.documents import Document as LangchainDocument
from langchain_google_genai import ChatGoogleGenerativeAI 

load_dotenv()

# Khai báo biến toàn cục
hybrid_retriever_instance = None

def get_hybrid_retriever():
    global hybrid_retriever_instance
    if hybrid_retriever_instance is not None:
        return hybrid_retriever_instance
        
    print("\n[System] Đang kết nối MinIO để tải dữ liệu BM25...")
    try:
        minio_client = Minio(
            os.getenv("MINIO_URL"),
            access_key=os.getenv("access_key"),
            secret_key=os.getenv("secret_key"),
            secure=False
        )
        
        # Kéo file chunks.pkl từ bucket 'rag-data'
        response = minio_client.get_object("rag-data", "chunk.pkl")
        saved_chunks = pickle.loads(response.read())
        response.close()
        response.release_conn()
        print(" Đã tải xong dữ liệu BM25 từ MinIO!")
        
        langchain_docs = [
            LangchainDocument(page_content=item["text"], metadata=item["metadata"])
            for item in saved_chunks
        ]
        
        # Khởi tạo Retriever
        vector_retriever = get_vector_retriever()
        bm25_retriever = BM25Retriever.from_documents(documents=langchain_docs, k=15)
        hybrid_retriever_instance = HybridRetriever(
            vector_retriever=vector_retriever, 
            bm25_retriever=bm25_retriever, 
            k=15
        )
        return hybrid_retriever_instance
        
    except Exception as e:
        raise e

def retrieve_node(state: RAGState):
    print("\n[Node] Đang tìm kiếm bằng HYBRID SEARCH...")
    
    retriever = get_hybrid_retriever()
    
    #1: LẤY CÂU HỎI VÀ XỬ LÝ NGỮ CẢNH
    search_query = state["question"]
    history = state.get("chat_history", [])
    
    # Nếu đang chat nối tiếp, lấy câu hỏi của người dùng ở lượt ngay trước đó ghép vào
    if len(history) >= 2:
        last_user_question = history[-2].replace("Người dùng: ", "")
        search_query = f"{last_user_question} - {search_query}"
        
    # 2. MỞ RỘNG TRUY VẤN (QUERY EXPANSION)
    try:
        lite_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            temperature=0,
            max_tokens=50,
            api_key=os.getenv("GOOGLE_API_KEY2")
        )
        
        formatted_rewrite_prompt = REWRITE_PROMPT.format(question=search_query)
        
        final_search_query = lite_llm.invoke(formatted_rewrite_prompt).content.strip()
        print(f"\n [Trinh sát AI] Đã dịch: '{search_query}' \n   -> '{final_search_query}'\n")
    except Exception as e:
        print(f"\n [Trinh sát AI] Gặp sự cố: {e}. Sẽ dùng câu hỏi gốc.\n")
        final_search_query = search_query
        
    # 3. TIẾN HÀNH TÌM KIẾM BẰNG TỪ KHÓA ĐÃ DỊCH
    docs = retriever.invoke(final_search_query)
    
    unique_docs = []
    seen = set()
    for doc in docs:
        content = (doc.page_content or "").strip()
        if content and len(content) > 40 and content not in seen:
            unique_docs.append(content)
            seen.add(content)
            
    top_k_final = 3
    final_docs = []
    
    if unique_docs:
        print(f" [Trạm Kiểm Duyệt] Gửi {len(unique_docs)} tài liệu cho Reranker chấm điểm...")
        rerank_url = os.getenv("RERANKER_URL", "http://reranker_api:80/rerank")
        payload = {
            "query": final_search_query,
            "texts": unique_docs
        }
        
        try:
            # Bắn API sang container Rerank
            response = requests.post(rerank_url, json=payload, timeout=10)
            response.raise_for_status()
            rerank_results = response.json()
            
            # rerank_results trả về dạng: [{"index": 2, "score": 0.95}, ...]
            for item in rerank_results[:top_k_final]:
                idx = item["index"]
                final_docs.append(unique_docs[idx])
                
            print(f" [Trạm Kiểm Duyệt] Đã chốt {len(final_docs)} tài liệu chuẩn xác nhất!")
            
        except Exception as e:
            print(f" ⚠️ Reranker API lỗi hoặc chưa bật ({e}). Dùng kết quả gốc của Hybrid.")
            final_docs = unique_docs[:top_k_final]
    else:
        final_docs = []

    return {"documents": final_docs}

def generate_node(state: RAGState):
    print("\n[Node] AI đang phân tích luật...")
    llm = get_llm()
    
    # Nối các tài liệu đã lọc thành 1 chuỗi dài
    context = "\n\n".join(state["documents"])
    
    history_text = "\n".join(state.get("chat_history", []))
    
    combined_question = f"Lịch sử trò chuyện:\n{history_text}\n\nCâu hỏi hiện tại: {state['question']}"
    
    # Gắn vào Prompt chuẩn
    prompt = BASELINE_PROMPT.format(context=context, question=combined_question)
    
    # LLM sinh câu trả lời
    raw_response = llm.invoke(prompt).content
    
    # Ép qua bộ lọc output để làm sạch văn bản
    final_answer = parse_focused_answer(raw_response)
    
    #3: GHI SỔ 
    return {
        "answer": final_answer,
        "chat_history": [f"Người dùng: {state['question']}", f"AI: {final_answer}"]
    }

def build_graph():
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app
