from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from agents.schemas import PromptMessage
import typing as tp
import re

def format_prompt(
        user_message: str, 
        chat_history: tp.List[BaseMessage],
        context: tp.Optional[str] = None,
    ) -> tp.List[BaseMessage]:
    '''
    Formats prompt for llm as LangChain messages
    '''

    history: tp.List[BaseMessage] = []
    for message in chat_history[:-1]:
        if message.type == "human":
            history.append(HumanMessage(content=message.content))
        elif message.type == "ai":
            history.append(AIMessage(content=message.content))
        elif message.type == "system":
            history.append(SystemMessage(content=message.content))
        else:
            continue

    if context:
        history.append(SystemMessage(content=context))

    history.append(HumanMessage(content=user_message))

    return history