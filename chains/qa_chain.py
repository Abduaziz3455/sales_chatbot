import os

import dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate,
                               ChatPromptTemplate)
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

output_parser = StrOutputParser()

dotenv.load_dotenv()

persist_directory = "chroma_data/"

review_template_str = """Use flat features to answer client questions about flats.
Use the following context to answer questions.
Be as detailed as possible, but don't make up any information that's not from the context.
If you don't know an answer send admins contact from context.
All answers should be in Uzbek (Russian).

Restrictions:
Avoid offering expensive apartments
Not providing information about prices
Submit information for one household only.

Context:
{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=review_template_str, ))

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="Question: {question}", ))

messages = [review_system_prompt, review_human_prompt]

prompt_template = ChatPromptTemplate(input_variables=["context", "question"], messages=messages)

chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

if not os.path.exists(persist_directory):
    loader = CSVLoader("data/ready.csv")
    pages = loader.load()
    print('Database load started')
    vector_db = Chroma.from_documents(pages, OpenAIEmbeddings(), persist_directory=persist_directory)
    print('Database copied to vector database')
else:
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())

retriever = vector_db.as_retriever(k=10)

qa_chain = ({"context": retriever,
            "question": RunnablePassthrough()} | prompt_template | chat_model | StrOutputParser())
# qa_chain = RetrievalQA.from_chain_type(chat_model, retriever=retriever, return_source_documents=False, verbose=True,
#                                        chain_type_kwargs={"verbose": True, "prompt": prompt_template,
#                                                           "memory": ConversationBufferMemory(
#                                                               memory_key="history",
#                                                               input_key="question")})
