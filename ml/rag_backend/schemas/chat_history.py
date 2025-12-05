from pydantic import BaseModel, Field
import typing as tp
from langchain_core.messages import BaseMessage


class ChatHistory(BaseModel):
    uuid: str
    assignment_id: tp.Optional[str] = None  # Optional, kept for backward compatibility
    history: tp.List[BaseMessage] = Field(default_factory=list)