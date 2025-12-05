from pydantic import BaseModel
import typing as tp


class ChatHistoryRequest(BaseModel):
    uuid: str
    assignment_id: tp.Optional[str] = None  # Optional, kept for backward compatibility