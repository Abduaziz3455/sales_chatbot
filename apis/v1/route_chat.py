from fastapi import APIRouter, status
from fastapi import Depends
from sqlalchemy.orm import Session

from agents.chat_agent import agent_executor
from db.repository.chat_crud import create_new_message
from db.session import get_db
from schemas.chat_query import QueryInput, QueryOutput
from utils.async_utils import async_retry

router = APIRouter()


@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """
    Retry the agent if a tool fails to run. This can help when there
    are intermittent connection issues to external APIs.
    """
    return await agent_executor.ainvoke({"input": query})


@router.get("/")
async def get_status():
    return {"status": "running"}


@router.post("/agent", response_model=QueryOutput, status_code=status.HTTP_201_CREATED)
async def send_message(query: QueryInput, db: Session = Depends(get_db)):
    agent_response = await invoke_agent_with_retry(query.message)
    response = {'input': query.message, 'output': agent_response['output'], 'user_id': query.user_id,
                'company_id': query.company_id, 'intermediate_steps': agent_response['intermediate_steps']}
    response["intermediate_steps"] = [str(s) for s in response["intermediate_steps"]]
    create_new_message(response.copy(), db)
    return response
