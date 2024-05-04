from dotenv import load_dotenv
from langchain import hub
from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOpenAI

load_dotenv()

chat_model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
prompt = hub.pull("hardkothari/prompt-maker")
chain = LLMChain(llm=chat_model, prompt=prompt, verbose=True, output_key='output')
task = input("Taskni kiriting: ")
prompt = input("Prompt: ")
response = chain({'task': task, 'lazy_prompt': prompt})
with open('generated_prompt.txt', 'w') as outfile:
    outfile.write(response['output'])
    outfile.close()
