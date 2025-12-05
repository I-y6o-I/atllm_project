from qdrant_client import QdrantClient, models
from .minio_repository import MinioRepository
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
import os
import logging
import uuid
from typing import Optional
import io
import re


logger = logging.getLogger(__name__)


class QdrantRepository:
    def __init__(self, minio_repo: MinioRepository):
        self._qdrant = QdrantClient(
            host=os.getenv("QDRANT_HOST"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
        )
        self.collection_name = "papers_collection"
        self._minio_repo = minio_repo
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=250,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
            model_kwargs={"device": os.getenv("DEVICE", "cpu")},
            encode_kwargs={"normalize_embeddings": True}
        )

        if self.collection_name not in [c.name for c in self._qdrant.get_collections().collections]:
            self._qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )

        logger.info("Qdrant repository initialized successfully")

    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text content from a PDF file."""
        text_parts = []
        try:
            stream = io.BytesIO(pdf_bytes)
            reader = PdfReader(stream)
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                if text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise ValueError(f"Failed to parse PDF: {e}")
        
        full_text = "\n\n".join(text_parts)
        full_text = self._remove_references_and_appendix(full_text)
        
        return full_text
    
    def _remove_references_and_appendix(self, text: str) -> str:
        references_patterns = [
            r'\n\s*References\s*\n',
            r'\n\s*REFERENCES\s*\n',
            r'\n\s*Bibliography\s*\n',
            r'\n\s*BIBLIOGRAPHY\s*\n',
            r'\n\s*Works\s+Cited\s*\n',
            r'\n\s*WORKS\s+CITED\s*\n',
            r'\n\s*Literature\s+Cited\s*\n',
            r'\n\s*LITERATURE\s+CITED\s*\n',
            r'\n\s*\d+\.?\s*References\s*\n',  # "7. References" or "7 References"
            r'\n\s*\d+\.?\s*REFERENCES\s*\n',
            r'\n\s*\[\s*References\s*\]\s*\n',  # "[References]"
        ]
        
        appendix_patterns = [
            r'\n\s*Appendix\s*[A-Z]?\s*[\.:]*\s*\n',
            r'\n\s*APPENDIX\s*[A-Z]?\s*[\.:]*\s*\n',
            r'\n\s*Appendices\s*\n',
            r'\n\s*APPENDICES\s*\n',
            r'\n\s*Supplementary\s+Material[s]?\s*\n',
            r'\n\s*SUPPLEMENTARY\s+MATERIAL[S]?\s*\n',
            r'\n\s*Supplementary\s+Information\s*\n',
            r'\n\s*SUPPLEMENTARY\s+INFORMATION\s*\n',
            r'\n\s*Supporting\s+Information\s*\n',
            r'\n\s*SUPPORTING\s+INFORMATION\s*\n',
            r'\n\s*\d+\.?\s*Appendix\s*\n',  # "A. Appendix" or "8. Appendix"
            r'\n\s*[A-Z]\.?\s*Appendix\s*\n',
        ]
        
        ref_start = len(text)
        for pattern in references_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.start() < ref_start:
                ref_start = match.start()
        
        appendix_start = len(text)
        for pattern in appendix_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.start() < appendix_start:
                appendix_start = match.start()
        
        cutoff = min(ref_start, appendix_start)
        
        min_content_ratio = 0.2  # Keep at least 20% of the document
        if cutoff < len(text) and cutoff > len(text) * min_content_ratio:
            cleaned_text = text[:cutoff].strip()
            removed_chars = len(text) - len(cleaned_text)
            logger.debug(
                f"Removed {removed_chars} chars (references/appendix). "
                f"Original: {len(text)}, Cleaned: {len(cleaned_text)}"
            )
            return cleaned_text
        
        return text

    def index_paper(self, paper_id: str, pdf_bytes: bytes, metadata: Optional[dict] = None):
        """Index a scientific paper (PDF) into the vector store."""
        if pdf_bytes is None or len(pdf_bytes) == 0:
            raise ValueError(f"Paper with id {paper_id} has no content")
        
        # Extract text from PDF
        paper_text = self._extract_text_from_pdf(pdf_bytes)
        
        if not paper_text.strip():
            raise ValueError(f"No text content extracted from paper {paper_id}")

        # Split into chunks
        chunks = self._splitter.split_text(paper_text)
        embeddings = self._embedding_model.embed_documents(chunks)

        # Prepare metadata
        base_metadata = metadata or {}
        base_metadata["paper_id"] = paper_id

        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            payload = {
                "metadata": {
                    **base_metadata,
                    "chunk_id": idx,
                    "total_chunks": len(chunks),
                },
                "text": chunk  
            }
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
                )
            )

        self._qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"Indexed paper {paper_id} with {len(chunks)} chunks")
        return len(chunks)

    def index_paper_from_minio(self, paper_id: str):
        """Index a paper stored in MinIO by its ID."""
        pdf_files = self._minio_repo.get_papers(paper_id)
        if pdf_files is None or len(pdf_files) == 0:
            raise ValueError(f"Paper with id {paper_id} not found in storage")
        
        total_chunks = 0
        for filename, pdf_bytes in pdf_files:
            metadata = {"filename": filename}
            chunks = self.index_paper(paper_id, pdf_bytes, metadata)
            total_chunks += chunks
        
        return total_chunks

    def search_papers(self, query: str, limit: int = 5, score_threshold: float = 0.5):
        """Search for relevant paper chunks based on a query."""
        query_embedding = self._embedding_model.embed_query(query)
        
        results = self._qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return [
            {
                "text": hit.payload.get("text", ""),
                "metadata": hit.payload.get("metadata", {}),
                "score": hit.score
            }
            for hit in results
        ]

    def delete_paper(self, paper_id: str):
        """Delete all chunks associated with a paper."""
        self._qdrant.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.paper_id",
                            match=models.MatchValue(value=paper_id)
                        )
                    ]
                )
            )
        )
        logger.info(f"Deleted all chunks for paper {paper_id}")

    def list_indexed_papers(self):
        """List all unique paper IDs in the collection."""
        # Scroll through all points and collect unique paper IDs
        paper_ids = set()
        offset = None
        
        while True:
            results, offset = self._qdrant.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=["metadata"]
            )
            
            for point in results:
                print(point.payload["metadata"])
                if point.payload and "metadata" in point.payload:
                    paper_id = point.payload["metadata"].get("paper_id")
                    if paper_id:
                        paper_ids.add(paper_id)
            
            if offset is None:
                break
        
        return list(paper_ids)


    @property
    def qdrant_client(self) -> QdrantClient:
        return self._qdrant
    
    @property
    def embedding_model(self) -> HuggingFaceEmbeddings:
        return self._embedding_model
