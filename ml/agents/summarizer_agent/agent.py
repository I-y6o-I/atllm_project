"""
Summarizer Agent for RAPTOR tree building.
Produces concise abstractive summaries of text chunks for hierarchical retrieval.
"""

import logging
from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from agents.groq_key_manager import GroqKeyManager
from .prompts import SUMMARIZE_SYSTEM_PROMPT, SUMMARIZE_USER_PROMPT

logger = logging.getLogger(__name__)


class SummarizerAgent:
    """
    Agent for generating abstractive summaries of text chunks.
    Used in RAPTOR tree building to create higher-level summary nodes.
    """
    
    def __init__(self, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        self._key_manager = GroqKeyManager()
        self._model_name = model_name
        self._load_llm()
        logger.info(f"SummarizerAgent initialized with model {model_name}")
    
    def _load_llm(self) -> None:
        self._llm = ChatGroq(
                    model=self._model_name,
                    api_key=self._key_manager.get_key(),
                    temperature=0.2,
                )

        try:
            self._llm.invoke("ping")
            logger.info("Gorq LLM is available")
            return None
        except Exception as e:
                self._key_manager.switch_key()
                logger.info(f"Key invalid, switched to next key {self._key_manager.idx}...")
                return self._load_llm()


    
    def _rotate_key_and_reload(self) -> None:
        self._key_manager.switch_key()
        self._llm = ChatGroq(
            model=self._model_name,
            api_key=self._key_manager.get_key(),
            temperature=0.2,
            # max_tokens=512
        )
        logger.info(f"Rotated to key index {self._key_manager.idx}")
    
    def summarize(self, texts: List[str], max_input_chars: int = 16000) -> str:
        """
        Generate a summary from multiple text chunks.
        
        Args:
            texts: List of text chunks to summarize
            max_input_chars: Maximum input characters (truncate if exceeded)
            
        Returns:
            Abstractive summary string
        """
        if not texts:
            raise ValueError("No texts provided for summarization")
        
        combined = "\n\n---\n\n".join(texts)
        
        if len(combined) > max_input_chars:
            combined = combined[:max_input_chars] + "\n\n[truncated...]"
            logger.warning(f"Input truncated to {max_input_chars} chars")
        
        messages = [
            SystemMessage(content=SUMMARIZE_SYSTEM_PROMPT),
            HumanMessage(content=SUMMARIZE_USER_PROMPT.format(chunks=combined))
        ]
        
        try:
            response = self._llm.invoke(messages)
            summary = response.content.strip()
            logger.info("=================================================")
            logger.info(summary)

            logger.debug(f"Generated summary: {len(summary)} chars from {len(texts)} chunks")
            return summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}, rotating key...")
            self._rotate_key_and_reload()
            try:
                response = self._llm.invoke(messages)
                return response.content.strip()
            except Exception as e2:
                logger.error(f"Summarization retry failed: {e2}")
                raise

    def summarize_batch(self, text_groups: List[List[str]]) -> List[str]:
        """
        Summarize multiple groups of texts.
        
        Args:
            text_groups: List of text chunk groups (one summary per group)
            
        Returns:
            List of summaries
        """
        summaries = []
        for i, group in enumerate(text_groups):
            try:
                summary = self.summarize(group)
                summaries.append(summary)
                logger.debug(f"Summarized group {i+1}/{len(text_groups)}")
            except Exception as e:
                logger.error(f"Failed to summarize group {i}: {e}")
                fallback = " ".join(group[:2])[:500]
                summaries.append(fallback)
        return summaries
