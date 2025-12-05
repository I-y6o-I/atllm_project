from fastapi import Request
from rag_backend.repositories import RaptorRepository

def get_raptor_repository(request: Request) -> RaptorRepository:
    return request.app.state.raptor_repository