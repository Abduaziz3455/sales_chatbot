from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from environs import Env
from langsmith import Client

env = Env()
load_dotenv()
client = Client()
chat_model = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)
prompt_template = open('dataset_prompt.txt', 'r', encoding='utf-8').read()
prompt = PromptTemplate(input_variables=["input"], template=prompt_template)
chain = prompt | chat_model
