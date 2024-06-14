from sqlalchemy.orm import Session

from db.models.chat import Chat


def create_new_message(query, db: Session):
    message = Chat(**query)
    db.add(message)
    db.commit()
    db.refresh(message)
