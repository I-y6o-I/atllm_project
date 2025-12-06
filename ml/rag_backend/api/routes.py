from fastapi import APIRouter, Form, Depends, HTTPException, Request, Response
from rag_backend.schemas import \
    AgentResponse,\
    AskRequest,\
    ChatHistoryRequest,\
    ChatHistory
from rag_backend.services import AskService, ChatHistoryService
from rag_backend.dependencies import(
    get_ask_service,
    get_chat_history_service,
    get_raptor_repository
)
from rag_backend.repositories import RaptorRepository
import logging
import traceback

router = APIRouter(tags=["Model"])
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/ask", response_model=AgentResponse)
async def ask(
    request: AskRequest,
    ask_service: AskService = Depends(get_ask_service)
):
    try:
        return await ask_service.ask(request)
    except Exception as e:
        logger.error(f"Ask error: {e}")
        print(f"[TRACEBACK]\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/get_chat_history", response_model=ChatHistory)
async def get_chat_history(
    request: ChatHistoryRequest = Depends(),
    chat_history_service: ChatHistoryService = Depends(get_chat_history_service)
):
    try:
        return await chat_history_service.get_chat_history(request)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/")
async def webhook_listener(request: Request):
    data = await request.json()
    print("Webhook received:", data)
    return {"status": "received"}

    
@router.post("/index_assignment")
async def index_assignment(
    assignment_id: str = Form(...),
    raptor_repo: RaptorRepository = Depends(get_raptor_repository)
):
    try:
        raptor_repo.index_paper_level0_from_minio(assignment_id)
        return Response(status_code=204)
    except Exception as e:
        logger.error(f"Index assignment error: {e}")
        print(f"[TRACEBACK]\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


from typing import Optional, List, Dict, Any

@router.post("/build_tree")
async def build_tree(
    max_levels: int = 3,
    min_nodes_per_level: int = 3,
    raptor_repo: RaptorRepository = Depends(get_raptor_repository),
) -> Dict[str, Any]:
    try:
        result = raptor_repo.build_full_tree(
            max_levels=max_levels,
            min_nodes_per_level=min_nodes_per_level
        )

        print("[BUILD TREE RESULT]", raptor_repo.get_tree_stats(), flush=True)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Build tree error: {e}")
        print(f"[TRACEBACK]\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/raptor_search")
async def raptor_search(
    query: str,
    mode: Optional[str] = "raptor", # "raptor" or "all_chunks"
    raptor_repo: RaptorRepository = Depends(get_raptor_repository)
) -> List[Dict[str, Any]]:
    try:
        if mode == "all_chunks":
            results = raptor_repo.search_by_level(
                query=query,
                level=0,
                limit=20
            )
        else:
            results = raptor_repo.search_top_down(
                query=query,
                top_k_level2=1,   # default: 1
                top_k_level1=5,   # default: 5
                top_k_level0=20   # default: 20
            )
        return results
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAPTOR search error: {e}")
        print(f"[TRACEBACK]\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
