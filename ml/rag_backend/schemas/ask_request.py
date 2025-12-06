from pydantic import BaseModel


class AskRequest(BaseModel):
    uuid: str
    assignment_id: str
    content: str
    benchmark: bool = False  # When True, skip chat history in prompt