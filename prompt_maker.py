from dotenv import load_dotenv
from langchain import hub
from langchain.chains.llm import LLMChain
from langchain_anthropic import ChatAnthropic
from environs import Env
env = Env()
load_dotenv()

chat_model = ChatAnthropic(model=env.str('MODEL'), temperature=0)
# chat_model = ChatOpenAI(model='gpt-4o', temperature=0)
prompt = hub.pull("hardkothari/prompt-maker")
chain = LLMChain(llm=chat_model, prompt=prompt, verbose=True, output_key='output')
task = input("Taskni kiriting: ")
prompt = input("Prompt: ")
response = chain({'task': task, 'lazy_prompt': prompt})
with open('prompt-claude-3-opus.txt', 'w') as outfile:
    outfile.write(response['output'])
    outfile.close()
