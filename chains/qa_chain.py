import os

import dotenv
from environs import Env
from langchain.prompts import (PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate,
                               ChatPromptTemplate)
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from retriever import FlatRetriever

env = Env()
dotenv.load_dotenv()

persist_directory = "chroma_data/"

review_template_str = """
Use the following context to answer the questions.
Be as detailed as possible, but don't make up any information that's not from the context.
If you don't know an answer just say don't know. All answers should be in Uzbek.
Context:
{context}
"""

# Restrictions:
# Avoid offering expensive apartments
# Not providing information about prices
# Submit information for one household only.

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=review_template_str, ))

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="Question: {question}", ))

messages = [review_system_prompt, review_human_prompt]

prompt_template = ChatPromptTemplate(input_variables=["context", "question"], messages=messages)

chat_model = ChatOpenAI(model=env.str('MODEL'), temperature=0)

if not os.path.exists(persist_directory):
    loader = CSVLoader("data/ready.csv")
    pages = loader.load()
    print('Database load started')
    vector_db = Chroma.from_documents(pages, OpenAIEmbeddings(), persist_directory=persist_directory)
    print('Database copied to vector database')
else:
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())

retriever_from_llm = FlatRetriever(db=vector_db)
# retriever_from_llm = vector_db.as_retriever(search_kwargs={"k": 5})

chain = ({"context": retriever_from_llm,
         "question": RunnablePassthrough()} | prompt_template | chat_model | StrOutputParser())

qa_chain = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///sql_app.db"
    ),
    input_messages_key="question",
    history_messages_key="history",
)
