from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from agents.base import BaseAgent
from langchain_core.runnables import RunnableConfig
from langgraph.types import StateSnapshot
from langgraph.graph import START, END, StateGraph
from agents.helper_agent.schemas.rag_state import RAGState
from langchain_core.messages import AIMessage
from functools import partial
from agents.helper_agent.nodes import retrieve, query_rag_llm, query_llm, route_rag_usage
from agents.groq_key_manager import GroqKeyManager
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from agents.helper_agent.config import \
    RAG_DB_PATH, \
    EMBEDDING_MODEL_NAME, \
    LLM_MODEL_NAME, \
    SCORE_THRESHOLD
from rag_backend.repositories import QdrantRepository
from rag_backend.config import POSTGRES_URL
import logging
import os
import os
from agents.schemas import AgentState


logger = logging.getLogger(__name__)

class HelperAgent(BaseAgent):
    def __init__(self, qdrant_repo: QdrantRepository):
        self.model_name = "deepseek-r1-distill-llama-70b"
        self._key_manager = GroqKeyManager()

        self._qdrant_repo = qdrant_repo

        self._load_vector_storage()
        self._load_llm()
        self._load_graph_builder()

    def _load_vector_storage(self) -> None:
        try:
            self._qdrant_client = self._qdrant_repo.qdrant_client
            self._db = QdrantVectorStore(
                client=self._qdrant_client,
                collection_name="labs_collection",
                embedding=self._qdrant_repo.embedding_model,
                # metadata_payload_key="payload",
                content_payload_key="text"
            )
            self._retriever = self._db.as_retriever(
                search_type="similarity",
                k=3,
                search_kwargs={
                    "score_threshold": SCORE_THRESHOLD,
                }
            )

            logger.info(f"SCORE THRESHOLD {str(SCORE_THRESHOLD)}")

            logger.info("Vector storage loaded successfully")
        except Exception as e:
            logger.error(f"Vector storage load failed: {e}")
            raise RuntimeError("Vector storage load failed") from e
        

    def _load_llm(self) -> None:
        self._llm = ChatGroq(
            model=self.model_name,
            api_key=self._key_manager.get_key(),
        )

        try:
            self._llm.invoke("ping")
            logger.info("Gorq LLM is available")
            return None
        except Exception as e:
                self._key_manager.switch_key()
                logger.info(f"Key invalid, switched to next key {self._key_manager.idx}...")
                return self._load_llm()

    def _load_graph_builder(self) -> None:
        try:
            self._graph_builder = StateGraph(RAGState)

            self._graph_builder.add_node(
                "retrieve",
                partial(
                    retrieve,
                    retriever=self._retriever
                )
            )
            self._graph_builder.add_node(
                "query_rag_llm",
                partial(
                    query_rag_llm,
                    llm=self._llm,
                )
            )
            self._graph_builder.add_node(
                "query_llm",
                partial(
                    query_llm,
                    llm=self._llm,
                )
            )

            self._graph_builder.add_conditional_edges("retrieve", route_rag_usage)
            self._graph_builder.add_edge("query_rag_llm", END)
            self._graph_builder.add_edge("query_llm", END)

            self._graph_builder.set_entry_point("retrieve")
            logger.info("Graph loaded successfully")
        except Exception as e:
            logger.error(f"Graph load failed: {e}")

    async def get_last_state(self, config: RunnableConfig) -> StateSnapshot:
        async with AsyncPostgresSaver.from_conn_string(POSTGRES_URL) as saver:
            graph = self._graph_builder.compile(checkpointer=saver)
            return await graph.aget_state(config=config)

    async def prompt(self, input_state: AgentState, config: RunnableConfig) -> AIMessage:
        async with AsyncPostgresSaver.from_conn_string(POSTGRES_URL) as saver:
            graph = self._graph_builder.compile(checkpointer=saver)
            # Cast to RAGState for processing
            response_state = await graph.ainvoke(input_state, config=config)
            return response_state['msg_state']["messages"][-1]