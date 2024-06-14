import requests
from fastapi import APIRouter, status
from fastapi import Depends
from sqlalchemy import delete, table
from sqlalchemy.orm import Session

from chains.qa_chain import qa_chain
from db.repository.chat_crud import create_new_message
from db.session import get_db
from schemas.chat_query import QueryInput, QueryOutput, VoiceInput
from utils.async_utils import async_retry
from environs import Env

env = Env()

router = APIRouter()


@async_retry(max_retries=3, delay=1)
async def invoke_agent_with_retry(query: str, user_id: str):
    """
    Retry the agent if a tool fails to run. This can help when there
    are intermittent connection issues to external APIs.
    """
    return await qa_chain.ainvoke(input={'question': query}, config={"configurable": {"session_id": user_id}})


def stt(file):
    url = 'https://mohir.ai/api/v1/stt'
    headers = {
        "Authorization": env.str("MOHIRAI_TOKEN")
    }
    files = {
        "file": ("audio.mp3", file),
    }
    data = {
        "return_offsets": "false",
        "run_diarization": "false",
        "blocking": "true",
    }

    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code == 200:
            return response.json()['result']['text']
        else:
            return None
    except requests.exceptions.Timeout:
        return None


@router.get("/")
async def get_status():
    return {"status": "running"}


@router.post("/agent", response_model=QueryOutput, status_code=status.HTTP_201_CREATED)
async def send_message(query: QueryInput, db: Session = Depends(get_db)):
    agent_response = await invoke_agent_with_retry(query.message, query.user_id)
    response = {'input': query.message, 'output': agent_response, 'user_id': query.user_id,
                'company_id': query.company_id}
    # response["intermediate_steps"] = [str(s) for s in response["intermediate_steps"]]
    create_new_message(response.copy(), db)
    return response


@router.post("/stt_agent")
async def upload_voice(query: VoiceInput, db: Session = Depends(get_db)):
    try:
        voice = requests.get(query.voice_url).content
        text = stt(voice)
        if text:
            agent_response = await invoke_agent_with_retry(text, query.user_id)
            response = {'input': text, 'output': agent_response['output'], 'user_id': query.user_id,
                        'company_id': query.company_id, 'intermediate_steps': agent_response['intermediate_steps']}
            response["intermediate_steps"] = [str(s) for s in response["intermediate_steps"]]
            create_new_message(response.copy(), db)
            return response
        else:
            print('Xatolik MohirAI')
            return {"message": "Error with MohirAI", "status": 400}
    except Exception as e:
        return {"message": f"{e}", "status": 400}


@router.get("/delete_history", status_code=status.HTTP_200_OK)
async def delete_chat_histories(db: Session = Depends(get_db)):
    db.execute(
        delete(table('message_store'))
    )
    db.commit()
    return {'status': 'Chat history was cleaned successfully!'}
