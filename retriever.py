from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


class FlatRetriever(BaseRetriever):
    k: int = 4
    """Number of top results to return"""
    db: VectorStore

    def _get_relevant_documents(self, query_dict: dict, *, run_manager: CallbackManagerForRetrieverRun) -> (
            List)[Document]:
        """Sync implementations for retriever."""
        query = query_dict.get('question', '')
        documents = self.db.as_retriever(search_kwargs={"k": 20}).get_relevant_documents(query)
        return documents[:self.k]
