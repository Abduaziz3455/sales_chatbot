from dotenv import load_dotenv
from langchain import hub
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic

load_dotenv()
# chat_model = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0.5)
chat_model = ChatOpenAI(model='gpt-4o', temperature=0.7)
prompt = hub.pull("hardkothari/prompt-maker")
chain = LLMChain(llm=chat_model, prompt=prompt, verbose=True, output_key='output')
task = 'Text extraction'
prompt = input("Prompt: ")
response = chain({'task': task, 'lazy_prompt': prompt})
with open('generated_prompt.txt', 'w', encoding='utf-8') as outfile:
    outfile.write(response['output'])
    outfile.close()
