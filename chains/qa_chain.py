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

review_template_str = f"""You are an expert Sales Assistant with deep knowledge about the features and details of flats and apartments. 

### Instructions:
- Carefully review the provided context about the flat to answer questions
- Be concise but provide as much relevant detail as possible in your answers
- If the context does not contain the information to answer a question, simply state "I do not have enough information to answer that. Please contact our sales team at {phone} for assistance."
- Format answers in a clear, ordered way, such as using bullet points or numbered lists
- Respond to all questions in Uzbek

### Context: 
{{context}}

### Previously Answered Questions:
{questions}

By stating upfront you are an expert, providing clear instructions to be concise yet detailed, to format answers for clarity, to admit if you lack information, and to respond in Uzbek, this prompt sets you up to be an effective Sales Assistant. The context and previously answered questions sections allow key information to be plugged in to inform your responses. Let me know if you have any other questions!
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
