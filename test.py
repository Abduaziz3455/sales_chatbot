from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOpenAI
load_dotenv()

chat_model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
room_prompt = PromptTemplate(
    input_variables=["question"],
    template="""As an expert in text extraction, your task is to extract room numbers from user questions. 

### Instructions:
Your goal is to identify and extract any room numbers mentioned in the user's question. If the user's question contains a room number, return the number. 
For example, if the user's question is '2xonali uylar narxi qancha?' you should only return '2' without text. If no room number is detected, return 'None'.

### Context:
You are developing a text extraction algorithm that specifically focuses on identifying room numbers within user queries. 
Your algorithm needs to accurately recognize and extract room numbers to provide relevant responses.

Original question: {question}""",
)

chain = LLMChain(llm=chat_model, prompt=room_prompt, verbose=False, output_key='rooms')

while True:
    question = input("Query: ")
    print(chain({'question': question})['rooms'])
