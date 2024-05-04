from typing import List
import pandas as pd

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import MultiQueryRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel

query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is 
    to generate 3 different versions of the given user 
    question to retrieve relevant documents from a vector database. 
    By generating multiple perspectives on the user question, 
    your goal is to help the user overcome some of the limitations 
    of distance-based similarity search. Provide these alternative 
    questions separated by newlines without ordinal numbers. 
    Original question: {question}""",
)


class FlatRetriever(BaseRetriever):
    k: int = 8
    """Number of top results to return"""
    db: VectorStore
    chat_model: BaseChatModel

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""
        multi_retriever = MultiQueryRetriever.from_llm(retriever=self.db.as_retriever(),
                                                       llm=self.chat_model,
                                                       include_original=True, prompt=query_prompt)
        documents = multi_retriever.get_relevant_documents(query=query)
        matching_documents = []
        dataframes = []
        df = pd.read_csv(documents[0].metadata['source'])
        for document in documents:
            dataframes.append(df.loc[document.metadata['row'], :])
        df = pd.DataFrame(dataframes)
        if 'arzon' in query.lower():
            matching_documents.append(documents[df['total_price'].argmin()])
        elif 'qimmat' in query.lower():
            matching_documents.append(documents[df['total_price'].argmax()])
        else:
            matching_documents = documents[:self.k]
        return matching_documents
