from sqlalchemy import Column, Integer, String

from db.base_class import Base


class Chat(Base):
    id = Column(Integer, primary_key=True, index=True)
    input = Column(String, nullable=False)
    output = Column(String, nullable=False)
    user_id = Column(Integer, nullable=False)
    company_id = Column(Integer, nullable=False)
