from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from chains.extract_chain import extract_chain
from chains.qa_chain import qa_chain
import pandas as pd
from environs import Env

env = Env()

df = pd.read_csv('data/ready.csv')
phone = df['builder_phone'][0]
with open('data/questions.txt') as file:
    questions = file.read()
    file.close()

system_message = f"""You are Sales Assistant who answer the questions 
about flat features. For complete information take users phone number to contact with operators.
Be concise, as detailed as possible, but don't make up any information. send information in an ordered format.
If you don't know an answer just say don't know and send admins phone {phone} to contact.
All answers should be in Uzbek.

Here is additional questions with answers:
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
        description="""Use this only when user sends his own phone number.
        Use whole prompt as input to the tool.
        """,
    ),
]

chat_model = ChatOpenAI(
    model=env.str('MODEL'),
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
