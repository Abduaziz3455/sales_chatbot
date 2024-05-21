import os

import dotenv
import pandas as pd
from environs import Env
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from retriever import FlatRetriever

env = Env()

df = pd.read_csv('data/ready.csv')
phone = df['builder_phone'][0]
with open('data/questions.txt') as file:
    questions = file.read()
    file.close()

output_parser = StrOutputParser()

dotenv.load_dotenv()

persist_directory = "chroma_data/"

review_template_str = f"""You are Sales Assistant who answer the questions about flat features.
Use the following context to answer the questions.
Be concise, as detailed as possible, but don't make up any information that's not from the context. 
send information in an ordered format. If you don't know an answer just say don't know 
and send admins phone {phone} to contact. All answers should be in Uzbek.

Here is additional questions with answers:
{questions}

Context:
{{context}}
"""

prompt_template = ChatPromptTemplate.from_messages(
    [("system", review_template_str), MessagesPlaceholder("chat_history", optional=True),
     ("human", "Question: {question}"), ])

chat_model = ChatAnthropic(model=env.str('MODEL'), temperature=0)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

if not os.path.exists(persist_directory):
    loader = CSVLoader("data/ready.csv")
    pages = loader.load()
    print('Database load started')
    vector_db = Chroma.from_documents(pages, embeddings, persist_directory=persist_directory)
    print('Database copied to vector database')
else:
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

retriever_from_llm = FlatRetriever(db=vector_db, chat_model=chat_model)

qa_chain = ({"context": retriever_from_llm,
             "question": RunnablePassthrough()} | prompt_template | chat_model | StrOutputParser())

agent = RunnableWithMessageHistory(qa_chain,
                                   lambda session_id: SQLChatMessageHistory(session_id=session_id,
                                                                            connection_string="sqlite:///sql_app.db"),
                                   input_messages_key="question", history_messages_key="chat_history", )
