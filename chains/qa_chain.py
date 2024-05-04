import os

import dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate,
                               ChatPromptTemplate)
from langchain.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from environs import Env

env = Env()

output_parser = StrOutputParser()

dotenv.load_dotenv()

persist_directory = "chroma_data/"

review_template_str = """Use flat features to answer client questions about flats.
Use the following context to answer questions.
Be as detailed as possible, but don't make up any information that's not from the context.
If you don't know an answer just say don't know and send admins contact from context.
All answers should be in Uzbek.

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

# retriever = vector_db.as_retriever(search_kwargs={"k": 10})
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vector_db.as_retriever(), llm=chat_model,
                                                  include_original=True, prompt=query_prompt)

# qa_chain = ({"context": retriever_from_llm,
#             "question": RunnablePassthrough()} | prompt_template | chat_model | StrOutputParser())
qa_chain = RetrievalQA.from_chain_type(chat_model, retriever=retriever_from_llm, return_source_documents=False, verbose=True,
                                       chain_type_kwargs={"verbose": True, "prompt": prompt_template})
