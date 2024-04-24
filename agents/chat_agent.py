from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain.chat_models.openai import ChatOpenAI

from chains.qa_chain import qa_chain
from chains.extract_chain import extract_chain

agent_prompt = hub.pull("hwchase17/openai-functions-agent")

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
        func=extract_chain.invoke,
        description="""Use this when you need to extract phone number or person name.
        Use the entire prompt as input to the tool. Return nothing.
        """,
    ),
]

chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
)

agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=agent_prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)
