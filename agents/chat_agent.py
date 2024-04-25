from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain.chat_models.openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from chains.extract_chain import extract_chain
from chains.qa_chain import qa_chain

system_message = """You are Sales Assistant who answer the questions 
about buying a flat that customers are interested in
All answers should be in Uzbek (Russian).
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
        about flat features like area, rooms, district and other flat parameters
        that could be answered about a flat using semantic search. 
        Use the entire prompt as input to the tool. For instance, if the prompt is
        "How much is the price per square meter of 2-room houses?", the input should be
        "How much is the price per square meter of 2-room houses?".
        """,
    ),
    Tool(
        name="Extract",
        func=extract_chain,
        description="""Use this when you need to extract phone number or person name.
        Use the entire prompt as input to the tool.
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

memory = ChatMessageHistory(session_id="test-session")

agent = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)
