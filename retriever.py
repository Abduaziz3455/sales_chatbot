from typing import List

import pandas as pd
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from environs import Env

env = Env()


def extract_chain(text: str):
    schemas = [
        ResponseSchema(name="Rooms", description="The number of rooms the client wants. Options: from 1 to 6 or null", type='number'),
        ResponseSchema(name="Floor",
                       description="The floor number the client wants an apartment on. Options: from 1 to 20 or null", type='number'),
        ResponseSchema(name="FlatStatus",
                       description="The status of the flat, indicating whether the apartment is repaired or not. Options: 1 (repaired), 0 (not repaired), or null", type='number'),
        ResponseSchema(name="Area",
                       description="The desired area of the apartment in square meters. Options: from 1 to 200 or null", type='number')]

    output_parser = StructuredOutputParser.from_response_schemas(schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(template="As an expert in text extraction, please carefully analyze the given client question about apartment features. "
                                     "If there is nothing to extract, respond with `null`.{format_instructions}\n{text}.",
                            input_variables=["text"],
                            partial_variables={"format_instructions": format_instructions})
    chat_model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    chain = prompt | chat_model | output_parser
    dict_response = chain.invoke({"text": text})
    return dict_response


class FlatRetriever(BaseRetriever):
    k: int = 3
    """Number of top results to return"""
    db: VectorStore

    def _get_relevant_documents(self, dict_query: dict, *, run_manager: CallbackManagerForRetrieverRun) -> List[
        Document]:
        """Sync implementations for retriever."""
        query = dict_query["question"]
        documents = self.db.as_retriever(search_kwargs={"k": 50}).get_relevant_documents(query)

        dataframes = []
        df = pd.read_csv(documents[0].metadata['source'])
        for document in documents:
            dataframes.append(df.loc[document.metadata['row'], :])
        df = pd.DataFrame(dataframes)
        # room extractor
        response = extract_chain(query)
        room = response['Rooms']
        if room != 'None':
            try:
                df = df[df['rooms'] == room].sort_values(by=['total_price_sum'], ascending=True)
            except:
                pass
        # price extractor
        if 'arzon' in query.lower():
            return [df.loc[df['total_price_sum'].idxmin()]]
        elif 'qimmat' in query.lower():
            return [df.loc[df['total_price_sum'].idxmax()]]
        elif 'kichik' in query.lower():
            return [df.loc[df['area'].idxmin()]]
        elif 'katta' in query.lower():
            return [df.loc[df['area'].idxmax()]]
        else:
            return df[:self.k].to_dict(orient="records")
