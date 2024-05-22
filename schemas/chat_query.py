from typing import Optional

from pydantic import BaseModel


class QueryInput(BaseModel):
    message: str
    user_id: Optional[str]
    company_id: Optional[int]


class QueryOutput(BaseModel):
    input: str
    output: str
    user_id: str
    company_id: int
    intermediate_steps: list[str]

    class Config:  # tells pydantic to convert even non dict obj to json
        from_attributes = True


class VoiceInput(BaseModel):
    user_id: Optional[str]
    company_id: Optional[int]
    voice_url: str

    class Config:  # tells pydantic to convert even non dict obj to json
        from_attributes = True
