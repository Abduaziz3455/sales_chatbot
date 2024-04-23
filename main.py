from agents.rag_agent import agent_executor
from fastapi import FastAPI
from models.rag_query import QueryInput, QueryOutput
from utils.async_utils import async_retry

app = FastAPI(
    title="Sales Chatbot",
    description="Endpoints for a sales RAG chatbot",
    docs_url='/api'
)


@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """
    Retry the agent if a tool fails to run. This can help when there
    are intermittent connection issues to external APIs.
    """

    return await agent_executor.ainvoke({"input": query})


@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/agent")
async def query_agent(
    query: QueryInput,
) -> QueryOutput:
    query_response = await invoke_agent_with_retry(query.text)
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]

    return query_response
