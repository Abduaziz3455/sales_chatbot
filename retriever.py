import re
from typing import List

import pandas as pd
from langchain.chains.llm import LLMChain
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

room_prompt = PromptTemplate(input_variables=["question"], template="""As an expert in text extraction, your task is to extract room numbers from user questions. 
    
    ### Instructions: Your goal is to identify and extract any room numbers mentioned in the user's question. 
    If the user's question contains a room number, return the number. For example, if the user's question is '2xonali uylar narxi qancha?' 
    you should only return '2' without text. If no room number is detected, return 'None'. 
    
    ### Context: You are developing a text extraction algorithm that specifically focuses on identifying room numbers within user queries. 
    Your algorithm needs to accurately recognize and extract room numbers to provide relevant responses. Original question: {question}""")


class FlatRetriever(BaseRetriever):
    k: int = 3
    """Number of top results to return"""
    db: VectorStore
    chat_model: BaseChatModel

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Sync implementations for retriever."""
        documents = self.db.as_retriever(search_kwargs={"k": 50}).get_relevant_documents(query)

        dataframes = []
        df = pd.read_csv(documents[0].metadata['source'])
        for document in documents:
            dataframes.append(df.loc[document.metadata['row'], :])
        df = pd.DataFrame(dataframes)
        # room extractor
        chain = LLMChain(llm=self.chat_model, prompt=room_prompt, verbose=False, output_key='rooms')
        text = chain({'question': query})['rooms']
        if text != 'None':
            try:
                room = int(re.findall(r'\d+', text)[0])
                df = df[df['rooms'] == room]
            except:
                pass
        # price extractor
        if 'arzon' in query.lower():
            return [df.loc[df['total_price_sum'].idxmin()]]
        elif 'qimmat' in query.lower():
            return [df.loc[df['total_price_sum'].idxmax()]]
        else:
            return documents[:self.k]
