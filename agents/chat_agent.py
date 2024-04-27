from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain.chat_models.openai import ChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from chains.extract_chain import extract_chain
from chains.qa_chain import qa_chain
import pandas as pd

df = pd.read_csv('data/ready.csv')
phone = df['builder_phone'][0]
with open('questions.txt') as file:
    questions = file.read()
    file.close()

system_message = f"""You are Sales Assistant who answer the questions 
about buying a flat that customers are interested in.
For complete information take person name and phone number to contact with operators.
Be as detailed as possible, but don't make up any information.
If you don't know an answer send admins phone {phone}.
All answers should be in Uzbek (Russian).

Here is some default questions with answers:

{questions}
"""

agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

tools = [
    Tool(
        name="Features",
        func=qa_chain.invoke,
        description="""Always use this tool as main tool. you need to answer objective questions
        about flat features like area, rooms, district, plan image and other flat parameters
        that could be answered about a flat using semantic search. And you can give seller contact information if needed. 
        Use the entire prompt as input to the tool. For instance, if the prompt is
        "How much is the price per square meter of 2-room houses?", the input should be
        "How much is the price per square meter of 2-room houses?".
        """,
    ),
    Tool(
        name="Extract",
        func=extract_chain,
        description="""Use this only when client sent his name and phone number.
        Use the entire prompt as input to the tool. 
        Examples: "My name is Shokir 901234567", "917399962 ismim Aziz", "Shahboz".
        """,
    ),
]

chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
)

openai_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=agent_prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=openai_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)

agent = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///sql_app.db"
    ),
    input_messages_key="input",
    history_messages_key="chat_history",
)
