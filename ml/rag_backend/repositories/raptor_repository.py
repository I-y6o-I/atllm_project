from qdrant_client import models
from .qdrant_repository import QdrantRepository
from .minio_repository import MinioRepository
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import uuid
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class RaptorRepository(QdrantRepository):
    DEFAULT_CHUNK_SIZE = 500  # ~100-150 tokens
    DEFAULT_CHUNK_OVERLAP = 50
    
    def __init__(
        self, 
        minio_repo: MinioRepository,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        super().__init__(minio_repo)
        self.collection_name = "raptor_collection"
        if self.collection_name not in [c.name for c in self._qdrant.get_collections().collections]:
            self._qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )
        
        self._raptor_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )

        self._create_raptor_indexes()
        
        logger.info(f"RAPTOR repository initialized (chunk_size={chunk_size})")
    
    def _create_raptor_indexes(self):
        try:
            self._qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.level",
                field_schema=models.PayloadSchemaType.INTEGER
            )
            logger.info("Created payload index for metadata.level")
        except Exception as e:
            logger.debug(f"Level index creation skipped: {e}")
        
        try:
            self._qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.node_type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info("Created payload index for metadata.node_type")
        except Exception as e:
            logger.debug(f"Node type index creation skipped: {e}")
        
        try:
            self._qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.paper_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info("Created payload index for metadata.paper_id")
        except Exception as e:
            logger.debug(f"Paper ID index creation skipped: {e}")
    
    def index_paper_level0(
        self, 
        paper_id: str, 
        pdf_bytes: bytes, 
        metadata: Optional[dict] = None
    ) -> Dict[str, Any]:
        if pdf_bytes is None or len(pdf_bytes) == 0:
            raise ValueError(f"Paper with id {paper_id} has no content")
        
        paper_text = self._extract_text_from_pdf(pdf_bytes)
        
        if not paper_text.strip():
            raise ValueError(f"No text content extracted from paper {paper_id}")
        
        chunks = self._raptor_splitter.split_text(paper_text)
        
        if not chunks:
            raise ValueError(f"No chunks created from paper {paper_id}")
        
        embeddings = self._embedding_model.embed_documents(chunks)
        
        base_metadata = metadata or {}
        base_metadata["paper_id"] = paper_id

        points = []
        node_ids = []
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            node_id = str(uuid.uuid4())
            node_ids.append(node_id)
            
            token_count = len(chunk) // 4
            
            payload = {
                "text": chunk,
                "metadata": {
                    **base_metadata,
                    "level": 0, # Level 0 = leaf chunks
                    "node_type": "chunk",  # "summary" for higher levels
                    "node_id": node_id,
                    "children_ids": [],
                    "parent_ids": [],  # Will be populated when tree is built
                    "cluster_id": None,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "token_count": token_count,
                }
            }
            
            points.append(
                models.PointStruct(
                    id=node_id,
                    vector=embedding,
                    payload=payload
                )
            )
        
        self._qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(
            f"Indexed paper {paper_id} at level 0 with {len(chunks)} leaf chunks"
        )
        
        return {
            "paper_id": paper_id,
            "num_chunks": len(chunks),
            "node_ids": node_ids,
            "level": 0
        }
    
    def index_paper_level0_from_minio(self, paper_id: str) -> Dict[str, Any]:
        pdf_files = self._minio_repo.get_papers(paper_id)
        if pdf_files is None or len(pdf_files) == 0:
            raise ValueError(f"Paper with id {paper_id} not found in storage")
        
        filename, pdf_bytes = pdf_files[0]
        metadata = {"filename": filename}
        
        return self.index_paper_level0(paper_id, pdf_bytes, metadata)
    
    def get_level0_nodes(
        self, 
        paper_id: str, 
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all level 0 (leaf) nodes for a paper.
        
        This is useful for:
        - Building higher levels of the tree
        - Debugging/inspection
        - Re-clustering
        
        Args:
            paper_id: Paper ID to filter by
            limit: Maximum number of nodes to return
            
        Returns:
            List of node dicts with text, embedding, and metadata
        """
        results, _ = self._qdrant.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.paper_id",
                        match=models.MatchValue(value=paper_id)
                    ),
                    models.FieldCondition(
                        key="metadata.level",
                        match=models.MatchValue(value=0)
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=True
        )
        
        nodes = []
        for point in results:
            nodes.append({
                "node_id": point.id,
                "text": point.payload.get("text", ""),
                "metadata": point.payload.get("metadata", {}),
                "vector": point.vector
            })
        
        # Sort by chunk_index
        nodes.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
        
        return nodes
    
    def get_nodes_by_level(
        self, 
        paper_id: str, 
        level: int, 
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all nodes at a specific level for a paper.
        
        Args:
            paper_id: Paper ID to filter by
            level: Tree level (0 = leaves, 1+ = summaries)
            limit: Maximum number of nodes to return
            
        Returns:
            List of node dicts
        """
        results, _ = self._qdrant.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.paper_id",
                        match=models.MatchValue(value=paper_id)
                    ),
                    models.FieldCondition(
                        key="metadata.level",
                        match=models.MatchValue(value=level)
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=True
        )
        
        return [
            {
                "node_id": point.id,
                "text": point.payload.get("text", ""),
                "metadata": point.payload.get("metadata", {}),
                "vector": point.vector
            }
            for point in results
        ]
    
    def get_max_level(self, paper_id: str) -> int:
        """
        Get the maximum tree level for a paper.
        
        Returns:
            Maximum level number (0 if only leaves exist)
        """
        max_level = 0
        
        # Check levels from 0 upward until we find none
        for level in range(10):  # Assume max 10 levels
            results, _ = self._qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.paper_id",
                            match=models.MatchValue(value=paper_id)
                        ),
                        models.FieldCondition(
                            key="metadata.level",
                            match=models.MatchValue(value=level)
                        )
                    ]
                ),
                limit=1,
                with_payload=False
            )
            
            if results:
                max_level = level
            else:
                break
        
        return max_level
    
    def search_collapsed_tree(
        self, 
        query: str, 
        paper_id: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search using collapsed tree strategy (all levels flattened).
        
        This is the recommended RAPTOR retrieval approach - it searches
        across all levels (leaves and summaries) and returns top-k by similarity.
        
        Args:
            query: Search query
            paper_id: Optional paper ID to filter by
            limit: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of matching nodes with scores
        """
        query_embedding = self._embedding_model.embed_query(query)
        
        # Build filter
        search_filter = None
        if paper_id:
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.paper_id",
                        match=models.MatchValue(value=paper_id)
                    )
                ]
            )
        
        results = self._qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return [
            {
                "node_id": hit.id,
                "text": hit.payload.get("text", ""),
                "metadata": hit.payload.get("metadata", {}),
                "score": hit.score
            }
            for hit in results
        ]
    
    def search_by_level(
        self, 
        query: str, 
        level: int,
        paper_id: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search only within a specific tree level.
        
        Useful for tree-traversal retrieval strategy.
        
        Args:
            query: Search query
            level: Tree level to search
            paper_id: Optional paper ID to filter by
            limit: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of matching nodes with scores
        """
        query_embedding = self._embedding_model.embed_query(query)
        
        # Build filter
        must_conditions = [
            models.FieldCondition(
                key="metadata.level",
                match=models.MatchValue(value=level)
            )
        ]
        
        if paper_id:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.paper_id",
                    match=models.MatchValue(value=paper_id)
                )
            )
        
        results = self._qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=models.Filter(must=must_conditions),
            limit=limit,
            score_threshold=score_threshold
        )
        
        return [
            {
                "node_id": hit.id,
                "text": hit.payload.get("text", ""),
                "metadata": hit.payload.get("metadata", {}),
                "score": hit.score
            }
            for hit in results
        ]
    
    def delete_paper_all_levels(self, paper_id: str):
        """
        Delete all nodes (all levels) for a paper.
        
        Args:
            paper_id: Paper ID to delete
        """
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
        logger.info(f"Deleted all RAPTOR nodes for paper {paper_id}")
    
    def get_tree_stats(self, paper_id: str) -> Dict[str, Any]:
        """
        Get statistics about the RAPTOR tree for a paper.
        
        Returns:
            Dict with tree statistics
        """
        stats = {
            "paper_id": paper_id,
            "levels": {},
            "total_nodes": 0
        }
        
        max_level = self.get_max_level(paper_id)
        
        for level in range(max_level + 1):
            nodes = self.get_nodes_by_level(paper_id, level)
            stats["levels"][level] = {
                "count": len(nodes),
                "node_type": "chunk" if level == 0 else "summary"
            }
            stats["total_nodes"] += len(nodes)
        
        stats["max_level"] = max_level
        
        return stats
