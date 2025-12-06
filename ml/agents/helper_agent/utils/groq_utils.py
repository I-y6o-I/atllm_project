from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from groq import APIStatusError
from agents.groq_key_manager import GroqKeyManager
import logging
import typing as tp

logger = logging.getLogger(__name__)


def generate_with_retry(
    prompt: list[BaseMessage],
    model_name: str,
    key_manager: GroqKeyManager,
) -> tp.Optional[BaseMessage]:
    """
    Generate response with automatic key switching on rate limit errors.
    
    Handles:
    - TPD (Tokens Per Day) errors: switches to next key
    - TPM (Tokens Per Minute) errors: raises RuntimeError (request too large)
    - 429 Rate limit errors: switches to next key
    
    Args:
        prompt: List of messages to send to the LLM
        model_name: Name of the Groq model to use
        key_manager: GroqKeyManager instance for API key management
        
    Returns:
        The LLM response message, or None if failed
        
    Raises:
        RuntimeError: If TPM error or unrecoverable API error occurs
    """
    llm = ChatGroq(
        model=model_name,
        api_key=key_manager.get_key(),
    )
    try:
        response = llm.invoke(prompt)
        return response
    except APIStatusError as e:
        if e.status_code == 413 and "TPD" in str(e):
            key_manager.switch_key()
            logger.info(
                f"TPD error encountered, switching Groq API key to index {key_manager.idx}..."
            )
            return generate_with_retry(prompt, model_name, key_manager)
        elif e.status_code == 413 and "TPM" in str(e):
            logger.error("TPM error encountered, request too large")
            raise RuntimeError("TPM error encountered, request too large")
        elif e.status_code == 429:
            key_manager.switch_key()
            logger.info(
                f"Rate limit (429) encountered, switching Groq API key to index {key_manager.idx}..."
            )
            return generate_with_retry(prompt, model_name, key_manager)
        else:
            raise RuntimeError(f"Groq API error: {e.status_code} - {e.message}") from e
