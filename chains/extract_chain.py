from typing import Optional

import dotenv
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

dotenv.load_dotenv()


class Person(BaseModel):
    phone_number: Optional[str] = Field(None, description="The person's phone number")


prompt = ChatPromptTemplate.from_messages([("system", "You are an expert at identifying person phone number. "
                                                      "Extract nothing if no phone number can be found in the text."
                                                      "All answers should be in Uzbek.",),
                                           ("human", "{text}"), ])

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def extract_chain(input: str):
    response = create_structured_output_runnable(Person, llm, prompt).invoke(input)
    return response
