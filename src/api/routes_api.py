# api/routes_api.py
from fastapi import APIRouter
from src.graph.nodes import build_graph

router = APIRouter()
app = build_graph() # Load đồ thị một lần duy nhất

@router.post("/chat")
async def chat(question: str, session_id: str = "default"):
    config = {"configurable": {"thread_id": session_id}}
    result = app.invoke({"question": question}, config=config)
    return {"answer": result["answer"]}