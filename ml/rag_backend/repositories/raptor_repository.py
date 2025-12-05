from qdrant_client import models
from .qdrant_repository import QdrantRepository
from .minio_repository import MinioRepository
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import uuid
import numpy as np
from typing import Optional, List, Dict, Any
from sklearn.decomposition import PCA
import hdbscan
from agents.summarizer_agent import SummarizerAgent
import time

logger = logging.getLogger(__name__)


class RaptorRepository(QdrantRepository):
    DEFAULT_CHUNK_SIZE = 500  # 100-150 tokens
    DEFAULT_CHUNK_OVERLAP = 50
    
    def __init__(
        self, 
        minio_repo: MinioRepository,
        summarizer: SummarizerAgent,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        super().__init__(minio_repo)

        self._summarizer = summarizer
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
    
    
    def get_nodes_by_level(
        self, 
        level: int, 
        limit: int = 10000
    ) -> List[Dict[str, Any]]:
        results, _ = self._qdrant.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
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
    
    def get_max_level(self) -> int:
        max_level = 0
        
        for level in range(10):
            results, _ = self._qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
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

    def search_top_down(
        self,
        query: str,
        top_k_level2: int = 1,
        top_k_level1: int = 5,
        top_k_level0: int = 20,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Top-down tree traversal search.
        
        1. Retrieve top_k_level2 nodes from level 2
        2. Retrieve top_k_level1 nodes from level 1 (only children of step 1)
        3. Retrieve top_k_level0 nodes from level 0 (only children of step 2)
        
        Returns only level 0 chunks as final result.
        
        Args:
            query: Search query
            top_k_level2: Number of nodes to retrieve from level 2
            top_k_level1: Number of nodes to retrieve from level 1
            top_k_level0: Number of nodes to retrieve from level 0
            score_threshold: Minimum similarity score
            
        Returns:
            List of level 0 chunks sorted by score
        """
        query_embedding = self._embedding_model.embed_query(query)
        
        max_level = self.get_max_level()
        
        if max_level < 2:
            return self._search_nodes_by_ids(
                query_embedding=query_embedding,
                node_ids=None,
                level=0,
                limit=top_k_level0,
                score_threshold=score_threshold
            )
        
        level2_results = self._search_nodes_by_ids(
            query_embedding=query_embedding,
            node_ids=None,
            level=2,
            limit=top_k_level2,
            score_threshold=score_threshold
        )
        
        level1_candidates = []
        for node in level2_results:
            children_ids = node["metadata"].get("children_ids", [])
            level1_candidates.extend(children_ids)
        
        if not level1_candidates:
            return []
        
        level1_results = self._search_nodes_by_ids(
            query_embedding=query_embedding,
            node_ids=level1_candidates,
            level=1,
            limit=top_k_level1,
            score_threshold=score_threshold
        )
        
        level0_candidates = []
        for node in level1_results:
            children_ids = node["metadata"].get("children_ids", [])
            level0_candidates.extend(children_ids)
        
        if not level0_candidates:
            return []
        
        level0_results = self._search_nodes_by_ids(
            query_embedding=query_embedding,
            node_ids=level0_candidates,
            level=0,
            limit=top_k_level0,
            score_threshold=score_threshold
        )
        
        return level0_results

    def _search_nodes_by_ids(
        self,
        query_embedding: List[float],
        node_ids: Optional[List[str]],
        level: int,
        limit: int,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search nodes filtered by specific IDs and level.
        
        Args:
            query_embedding: Query vector
            node_ids: List of node IDs to search within (None = all nodes at level)
            level: Tree level to search
            limit: Number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of matching nodes with scores
        """
        must_conditions = [
            models.FieldCondition(
                key="metadata.level",
                match=models.MatchValue(value=level)
            )
        ]
        
        if node_ids is not None and len(node_ids) > 0:
            points = self._qdrant.retrieve(
                collection_name=self.collection_name,
                ids=node_ids,
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                return []
            
            scored_results = []
            for point in points:
                if point.vector is None:
                    continue
                    
                score = float(np.dot(query_embedding, point.vector))
                
                if score >= score_threshold:
                    scored_results.append({
                        "node_id": point.id,
                        "text": point.payload.get("text", ""),
                        "metadata": point.payload.get("metadata", {}),
                        "score": score
                    })
            
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            return scored_results[:limit]
        else:
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
    
    def get_tree_stats(self) -> Dict[str, Any]:
        stats = {
            "levels": {},
            "total_nodes": 0
        }
        
        max_level = self.get_max_level()
        
        for level in range(max_level + 1):
            nodes = self.get_nodes_by_level(level)
            stats["levels"][level] = {
                "count": len(nodes),
                "node_type": "chunk" if level == 0 else "summary"
            }
            stats["total_nodes"] += len(nodes)
        
        stats["max_level"] = max_level
        
        return stats

    def _reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_components: int = 10
    ) -> np.ndarray:
        n_samples = embeddings.shape[0]
        n_features = embeddings.shape[1]
        
        target_components = min(n_components, n_samples, n_features)
        
        if target_components < 2:
            return embeddings
        
        pca = PCA(n_components=target_components)
        return pca.fit_transform(embeddings)
    
    def _assign_noise_to_nearest_cluster(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        labels = labels.copy()
        
        unique_clusters = [l for l in set(labels) if l != -1]
        
        if not unique_clusters:
            return np.zeros(len(labels), dtype=int)
        
        centroids = {}
        for cluster_id in unique_clusters:
            cluster_mask = labels == cluster_id
            centroids[cluster_id] = embeddings[cluster_mask].mean(axis=0)
        
        noise_indices = np.where(labels == -1)[0]
        
        for idx in noise_indices:
            point = embeddings[idx]
            min_dist = float('inf')
            nearest_cluster = unique_clusters[0]
            
            for cluster_id, centroid in centroids.items():
                dist = np.linalg.norm(point - centroid)
                if dist < min_dist:
                    min_dist = dist
                    nearest_cluster = cluster_id
            
            labels[idx] = nearest_cluster
        
        if len(noise_indices) > 0:
            logger.debug(f"Reassigned {len(noise_indices)} noise points to nearest clusters")
        
        return labels

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        min_samples: int = 2,
        n_components: int = 50
    ) -> np.ndarray:
        n_samples = embeddings.shape[0]
        
        if n_samples < min_cluster_size:
            return np.zeros(n_samples, dtype=int)
        
        if embeddings.shape[1] > n_components:
            reduced = self._reduce_dimensions(embeddings, n_components=min(n_components, n_samples - 1))
        else:
            reduced = embeddings
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        labels = clusterer.fit_predict(reduced)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        logger.debug(f"HDBSCAN found {n_clusters} clusters, {n_noise} noise points")
        
        if n_noise > 0 and n_clusters > 0:
            labels = self._assign_noise_to_nearest_cluster(reduced, labels)
        elif n_clusters == 0:
            logger.debug("No clusters found, treating all as one cluster")
            return np.zeros(n_samples, dtype=int)
        
        return labels
    
    def _group_nodes_by_cluster(
        self,
        nodes: List[Dict[str, Any]],
        labels: np.ndarray
    ) -> Dict[int, List[Dict[str, Any]]]:
        groups = {}
        for node, label in zip(nodes, labels):
            label_int = int(label)
            if label_int not in groups:
                groups[label_int] = []
            groups[label_int].append(node)
        return groups
    
    def _update_parent_ids(
        self,
        children_ids: List[str],
        parent_id: str
    ):
        for child_id in children_ids:
            try:
                points = self._qdrant.retrieve(
                    collection_name=self.collection_name,
                    ids=[child_id],
                    with_payload=True
                )
                
                if points:
                    point = points[0]
                    metadata = point.payload.get("metadata", {})
                    parent_ids = metadata.get("parent_ids", [])
                    
                    if parent_id not in parent_ids:
                        parent_ids.append(parent_id)
                        
                    self._qdrant.set_payload(
                        collection_name=self.collection_name,
                        payload={"metadata.parent_ids": parent_ids},
                        points=[child_id]
                    )
            except Exception as e:
                logger.warning(f"Failed to update parent_ids for {child_id}: {e}")
    
    def build_next_level(
        self,
        current_level: int = 0,
        min_nodes_to_cluster: int = 5
    ) -> Dict[str, Any]:
        nodes = self.get_nodes_by_level(current_level)
        
        if len(nodes) < min_nodes_to_cluster:
            logger.info(
                f"Not enough nodes ({len(nodes)}) at level {current_level} to build next level"
            )
            return {
                "level": current_level + 1,
                "num_nodes": 0,
                "node_ids": [],
                "stopped": True,
                "reason": "insufficient_nodes"
            }
        
        embeddings = np.array([node["vector"] for node in nodes])
        
        labels = self._cluster_embeddings(embeddings)
        
        groups = self._group_nodes_by_cluster(nodes, labels)
        
        next_level = current_level + 1
        points = []
        node_ids = []
        
        for i, (cluster_id, cluster_nodes) in enumerate(groups.items()):
            texts = [node["text"] for node in cluster_nodes]
            children_ids = [node["node_id"] for node in cluster_nodes]
            
            try:
                summary = self._summarizer.summarize(texts)
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Summarization failed for cluster {cluster_id}: {e}")
                summary = " ".join(texts[:3])[:1000]
            
            summary_embedding = self._embedding_model.embed_query(summary)
            
            node_id = str(uuid.uuid4())
            node_ids.append(node_id)
            
            payload = {
                "text": summary,
                "metadata": {
                    "level": next_level,
                    "node_type": "summary",
                    "node_id": node_id,
                    "children_ids": children_ids,
                    "parent_ids": [],
                    "cluster_id": int(cluster_id),
                    "children_count": len(children_ids),
                    "token_count": len(summary) // 4,
                }
            }
            
            points.append(
                models.PointStruct(
                    id=node_id,
                    vector=summary_embedding,
                    payload=payload
                )
            )
            
            self._update_parent_ids(children_ids, node_id)

            logger.info(f"Built {i}/{len(groups.items())} nodes")
        
        if points:
            self._qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        logger.info(
            f"Built level {next_level}: "
            f"{len(groups)} clusters -> {len(points)} summary nodes"
        )
        
        return {
            "level": next_level,
            "num_nodes": len(points),
            "node_ids": node_ids,
            "num_clusters": len(groups),
            "stopped": False
        }
    
    def build_full_tree(
        self,
        max_levels: int = 3,
        min_nodes_per_level: int = 5
    ) -> Dict[str, Any]:
        stats = self.get_tree_stats()
        
        if stats["total_nodes"] == 0:
            raise ValueError("No level 0 nodes found. Index papers first.")
        
        current_level = stats["max_level"]
        levels_built = []
        
        while current_level < max_levels:
            result = self.build_next_level(
                current_level=current_level,
                min_nodes_to_cluster=min_nodes_per_level
            )
            
            levels_built.append(result)
            
            if result["stopped"]:
                break
            
            if result["num_nodes"] < min_nodes_per_level:
                logger.info(
                    f"Stopping tree build: level {result['level']} has only "
                    f"{result['num_nodes']} nodes"
                )
                break
            
            current_level = result["level"]
        
        final_stats = self.get_tree_stats()
        
        return {
            "levels_built": levels_built,
            "final_stats": final_stats
        }
