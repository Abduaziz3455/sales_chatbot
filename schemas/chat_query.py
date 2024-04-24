from typing import Optional

from pydantic import BaseModel


class QueryInput(BaseModel):
    message: str
    user_id: Optional[int]
    company_id: Optional[int]


class QueryOutput(BaseModel):
    input: str
    output: str
    user_id: int
    company_id: int
    intermediate_steps: list[str]

    class Config:  # tells pydantic to convert even non dict obj to json
        from_attributes = True
