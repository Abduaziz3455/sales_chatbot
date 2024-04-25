from sqlalchemy.orm import Session

from db.models.chat import Chat


def create_new_message(query, db: Session):
    del query['intermediate_steps']
    message = Chat(**query)
    db.add(message)
    db.commit()
    db.refresh(message)
