from src.graph.nodes import build_graph
import os


def main():
    print(" HỆ THỐNG AI LUẬT GIAO THÔNG (LANGGRAPH)")
    
    # Tải đồ thị
    app = build_graph()
    
    config = {"configurable": {"thread_id": "phien_chat_so_1"}}
    
    while True:
        user_query = input("\nBạn muốn hỏi gì (Gõ 'q' để thoát): ")
        if user_query.lower() == 'q':
            break
            
        # Chạy LangGraph
        result = app.invoke({"question": user_query}, config=config)
        
        print("\n TRẢ LỜI:")
        print("=" * 50)
        print(result["answer"])
        print("=" * 50)

if __name__ == "__main__":
    main()