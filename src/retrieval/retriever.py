import numpy as np
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from typing import List, Any
from collections import defaultdict

# 1. Class BM25 Retriever (Tìm kiếm từ khóa)
class BM25Retriever(BaseRetriever):
    documents: List[Any] = Field(description="Danh sách tài liệu")
    bm25: Any = Field(description="Index của BM25")
    k: int = Field(default=5, description="Số lượng tài liệu trả về")

    @classmethod
    def from_documents(cls, documents: List[Document], k: int = 5):
        # Tách từ tiếng Việt bằng underthesea
        tokenized_docs = [word_tokenize(doc.page_content.lower()) for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        return cls(documents=documents, bm25=bm25, k=k)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(bm25_scores)[::-1][:self.k]
        return [self.documents[i] for i in top_k_indices]

# 2. Class Hybrid Retriever (Gộp Vector Search và BM25)
class HybridRetriever(BaseRetriever):
    vector_retriever: Any
    bm25_retriever: Any
    k: int = 5
    rrf_k: int = 60 

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        results_list = [vector_docs, bm25_docs]
        rrf_scores = defaultdict(float)
        doc_content_map = {}
        
        for results in results_list:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc.page_content
                rrf_scores[doc_id] += 1 / (self.rrf_k + rank)
                doc_content_map[doc_id] = doc
                
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_content_map[doc_id] for doc_id, score in sorted_docs][:self.k]