from typing import Optional

import dotenv
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

dotenv.load_dotenv()


class Person(BaseModel):
    name: Optional[str] = Field(None, description="The person's name")
    phone_number: Optional[str] = Field(None, description="The person's phone number")


prompt = ChatPromptTemplate.from_messages([("system", "You are an expert at identifying person's identity.  Only "
                                                      "extract person name and phone number. Extract nothing if no "
                                                      "name or phone number can be found in the text.",),
                                           ("human", "{text}"), ])
# create_openai_fn_runnable
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

extract_chain = create_structured_output_runnable(Person, llm, prompt)  #