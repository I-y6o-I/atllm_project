from agents.helper_agent.schemas.rag_state import RAGState
from rag_backend.repositories import RaptorRepository
import logging

logger = logging.getLogger(__name__)


def retrieve(state: RAGState, raptor_repo: RaptorRepository) -> dict[str, str]:
    """Retrieve relevant information using RAPTOR top-down search."""
    logger.info(f"RAPTOR retrieve - query: {state.query}, paper_id: {state.assignment_id}")

    try:
        results = raptor_repo.search_top_down(
            query=state.query,
            top_k_level2=1,
            top_k_level1=5,
            top_k_level0=5,
            score_threshold=0.6
        )
        
        
        if not results:
            logger.info(f"No RAPTOR results for paper {state.assignment_id}, using all results")

        
        serialized = "\n\n".join(
            f"[Chunk {i+1}]: {r['text']}"
            for i, r in enumerate(results)
        )
        
        logger.info(f"Retrieved {len(results)} chunks via RAPTOR")
        
        return {"docs": serialized}
        
    except Exception as e:
        logger.error(f"RAPTOR retrieval failed: {e}")
        return {"docs": ""}