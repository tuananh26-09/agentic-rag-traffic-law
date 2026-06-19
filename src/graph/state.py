from typing import List, Annotated
from typing_extensions import TypedDict
import operator

class RAGState(TypedDict):
    question: str
    documents: List[str]
    answer: str
    chat_history: Annotated[List[str], operator.add]