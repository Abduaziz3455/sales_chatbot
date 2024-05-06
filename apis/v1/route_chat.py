import requests
from fastapi import APIRouter, status, UploadFile, File
from fastapi import Depends
from sqlalchemy import delete, table
from sqlalchemy.orm import Session

from agents.chat_agent import agent
from db.repository.chat_crud import create_new_message
from db.session import get_db
from schemas.chat_query import QueryInput, QueryOutput
from utils.async_utils import async_retry

router = APIRouter()


@async_retry(max_retries=3, delay=1)
async def invoke_agent_with_retry(query: str, user_id):
    """
    Retry the agent if a tool fails to run. This can help when there
    are intermittent connection issues to external APIs.
    """
    return await agent.ainvoke({"input": query}, config={"configurable": {"session_id": user_id}})


async def stt(file):
    url = 'https://mohir.ai/api/v1/stt'
    headers = {
        "Authorization": '8d700409-246f-4799-bf6f-3d1cdfea7561:c204f060-db45-46f0-b5ed-d6a5db47c13a'
    }

    files = {
        "file": ("audio.mp3", file),
    }
    data = {
        "return_offsets": "true",
        "run_diarization": "false",
        "blocking": "true",
    }

    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            return f"Request failed with status code {response.status_code}: {response.text}"
    except requests.exceptions.Timeout:
        return "Request timed out. The API response took too long to arrive."


@router.get("/")
async def get_status():
    return {"status": "running"}


@router.post("/agent", response_model=QueryOutput, status_code=status.HTTP_201_CREATED)
async def send_message(query: QueryInput, db: Session = Depends(get_db)):
    agent_response = await invoke_agent_with_retry(query.message, query.user_id)
    response = {'input': query.message, 'output': agent_response['output'], 'user_id': query.user_id,
                'company_id': query.company_id, 'intermediate_steps': agent_response['intermediate_steps']}
    response["intermediate_steps"] = [str(s) for s in response["intermediate_steps"]]
    create_new_message(response.copy(), db)
    return response


@router.post("/stt_agent")
async def upload_voice(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        response = await stt(contents)
        # with open(file.filename, 'wb') as f:
        #     f.write(contents)
        return response
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()


@router.get("/delete_history", status_code=status.HTTP_200_OK)
async def delete_chat_histories(db: Session = Depends(get_db)):
    db.execute(
        delete(table('message_store'))
    )
    db.commit()
    return {'status': 'Chat history was cleaned successfully!'}
