from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import ConfigDict
from sentence_transformers import CrossEncoder

from scripts.config import (
    BM25_FILE,
    BM25_WEIGHT,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    ENABLE_QUERY_REWRITE,
    ENABLE_RERANK,
    ENABLE_ROUTING,
    MIN_WEIGHTED_RRF_SCORE,
    NO_ANSWER_MIN_FUSED_SCORE,
    NO_ANSWER_MIN_RERANK_SCORE,
    RERANKER_MODEL,
    RETRIEVAL_PROFILES,
    REWRITE_HINTS,
    ROUTING_GROUP_TO_TITLES,
    ROUTING_RULES,
    RRF_K,
    TOP_K_BM25,
    TOP_K_DENSE,
    TOP_K_FINAL,
    TOP_K_FUSED,
    TOP_K_RERANK,
    bm25_tokenize,
)


class HybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bm25_file: Path = BM25_FILE
    chroma_dir: Path = CHROMA_DIR
    embedding_model: str = EMBEDDING_MODEL
    chroma_collection: str = CHROMA_COLLECTION
    profile_name: str = "hybrid_rerank"
    device: str = "cpu"

    bm25_payload: dict[str, Any] | None = None
    vectorstore: Chroma | None = None
    reranker: CrossEncoder | None = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.bm25_payload = self._load_bm25(self.bm25_file)
        self.vectorstore = self._load_chroma(
            self.chroma_dir,
            self.embedding_model,
            self.chroma_collection,
            self.device,
        )
    
        profile_cfg = RETRIEVAL_PROFILES.get(self.profile_name, {})
        needs_reranker = bool(profile_cfg.get("enable_reranker", False)) and ENABLE_RERANK
    
        if needs_reranker:
            self.reranker = CrossEncoder(RERANKER_MODEL)
        else:
            self.reranker = None

    @staticmethod
    def _load_bm25(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {path}. Run ingest/build_index first."
            )
        with path.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_chroma(
        chroma_dir: Path,
        embedding_model: str,
        collection: str,
        device: str,
    ) -> Chroma:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        return Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=str(chroma_dir),
        )

    def _rewrite_query(self, query: str) -> str:
        if not ENABLE_QUERY_REWRITE:
            return query
        q = query.strip()
        q_lower = q.lower()
        for needle, replacement in REWRITE_HINTS.items():
            if needle in q_lower:
                q_lower = q_lower.replace(needle, replacement)
        return q_lower

    def _route_group(self, query: str) -> str | None:
        if not ENABLE_ROUTING:
            return None
        q = query.lower()
        best_group = None
        best_score = 0
        for group, keywords in ROUTING_RULES.items():
            score = sum(1 for kw in keywords if kw in q)
            if score > best_score:
                best_score = score
                best_group = group
        return best_group if best_score > 0 else None

    def _title_boost(self, item: dict[str, Any], routed_group: str | None) -> float:
        if not ENABLE_ROUTING or not routed_group:
            return 1.0
        preferred_titles = ROUTING_GROUP_TO_TITLES.get(routed_group, [])
        title = str(item.get("title", ""))
        return 1.15 if title in preferred_titles else 1.0

    def _bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        bm25 = self.bm25_payload["bm25"]
        chunks = self.bm25_payload["chunks"]
        scores = bm25.get_scores(bm25_tokenize(query))
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "title": chunk["title"],
                "score": float(score),
                "rank": rank,
                "retriever": "bm25",
            }
            for rank, (chunk, score) in enumerate(ranked, start=1)
        ]

    def _dense_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        prefixed_query = f"query: {query}"
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            prefixed_query, k=top_k
        )

        return [
            {
                "chunk_id": doc.metadata["chunk_id"],
                "text": doc.page_content.removeprefix("passage: "),
                "source": doc.metadata["source"],
                "page": doc.metadata["page"],
                "title": doc.metadata["title"],
                "score": float(score),
                "rank": rank,
                "retriever": "dense",
            }
            for rank, (doc, score) in enumerate(docs_and_scores, start=1)
        ]

    def _weighted_rrf_fusion(
        self,
        bm25_results: list[dict[str, Any]],
        dense_results: list[dict[str, Any]],
        routed_group: str | None,
        top_k: int,
        rrf_k: int,
        bm25_weight: float,
        dense_weight: float,
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}

        for item in bm25_results:
            cid = item["chunk_id"]
            score = bm25_weight * (1.0 / (rrf_k + item["rank"]))
            score *= self._title_boost(item, routed_group)
            if cid not in fused:
                fused[cid] = {
                    **item,
                    "weighted_rrf_score": 0.0,
                    "bm25_rrf": 0.0,
                    "dense_rrf": 0.0,
                    "retrievers": [],
                }
            fused[cid]["weighted_rrf_score"] += score
            fused[cid]["bm25_rrf"] += score
            fused[cid]["retrievers"].append("bm25")

        for item in dense_results:
            cid = item["chunk_id"]
            score = dense_weight * (1.0 / (rrf_k + item["rank"]))
            score *= self._title_boost(item, routed_group)
            if cid not in fused:
                fused[cid] = {
                    **item,
                    "weighted_rrf_score": 0.0,
                    "bm25_rrf": 0.0,
                    "dense_rrf": 0.0,
                    "retrievers": [],
                }
            fused[cid]["weighted_rrf_score"] += score
            fused[cid]["dense_rrf"] += score
            fused[cid]["retrievers"].append("dense")

        ranked = sorted(
            fused.values(),
            key=lambda x: x["weighted_rrf_score"],
            reverse=True,
        )

        filtered = [
            r for r in ranked
            if r["weighted_rrf_score"] >= MIN_WEIGHTED_RRF_SCORE
        ]

        return filtered[:top_k] or ranked[:top_k]

    def _rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k_final: int,
    ) -> list[dict[str, Any]]:
        if not candidates or self.reranker is None:
            return candidates[:top_k_final]

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        reranked = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k_final]

    @staticmethod
    def _to_documents(items: list[dict[str, Any]]) -> list[Document]:
        return [
            Document(
                page_content=item["text"],
                metadata={
                    "chunk_id": item["chunk_id"],
                    "source": item["source"],
                    "page": item["page"],
                    "title": item["title"],
                    "score": item.get(
                        "rerank_score",
                        item.get("weighted_rrf_score", item.get("score")),
                    ),
                    "retrievers": item.get(
                        "retrievers", [item.get("retriever", "unknown")]
                    ),
                    "weighted_rrf_score": item.get("weighted_rrf_score"),
                    "rerank_score": item.get("rerank_score"),
                },
            )
            for item in items
        ]

    def _profile_cfg(self) -> dict[str, Any]:
        if self.profile_name not in RETRIEVAL_PROFILES:
            raise ValueError(
                f"Unknown profile_name: {self.profile_name}. "
                f"Available: {list(RETRIEVAL_PROFILES.keys())}"
            )
        return RETRIEVAL_PROFILES[self.profile_name]

    def retrieve_with_diagnostics(
        self, query: str
    ) -> tuple[list[Document], dict[str, Any]]:
        cfg = self._profile_cfg()
        rewritten_query = self._rewrite_query(query)
        routed_group = self._route_group(rewritten_query)

        top_k_bm25 = cfg.get("top_k_bm25", TOP_K_BM25)
        top_k_dense = cfg.get("top_k_dense", TOP_K_DENSE)
        top_k_fused = cfg.get("top_k_fused", TOP_K_FUSED)
        top_k_rerank = cfg.get("top_k_rerank", TOP_K_RERANK)
        top_k_final = cfg.get("top_k_final", TOP_K_FINAL)
        bm25_weight = cfg.get("bm25_weight", BM25_WEIGHT)
        dense_weight = cfg.get("dense_weight", DENSE_WEIGHT)
        enable_reranker = bool(cfg.get("enable_reranker", False)) and ENABLE_RERANK

        bm25_results: list[dict[str, Any]] = []
        dense_results: list[dict[str, Any]] = []
        final_results: list[dict[str, Any]] = []
        fused_results: list[dict[str, Any]] = []

        if self.profile_name == "bm25_only":
            bm25_results = self._bm25_search(rewritten_query, top_k=top_k_final)
            final_results = bm25_results[:top_k_final]

        elif self.profile_name == "dense_only":
            dense_results = self._dense_search(rewritten_query, top_k=top_k_final)
            final_results = dense_results[:top_k_final]


        elif self.profile_name == "hybrid":
            bm25_results = self._bm25_search(rewritten_query, top_k=top_k_bm25)
            dense_results = self._dense_search(rewritten_query, top_k=top_k_dense)
            fused_results = self._weighted_rrf_fusion(
                bm25_results=bm25_results,
                dense_results=dense_results,
                routed_group=routed_group,
                top_k=top_k_fused,
                rrf_k=RRF_K,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
            )
            final_results = fused_results[:top_k_final]

        elif self.profile_name == "hybrid_rerank":
            bm25_results = self._bm25_search(rewritten_query, top_k=top_k_bm25)
            dense_results = self._dense_search(rewritten_query, top_k=top_k_dense)
            fused_results = self._weighted_rrf_fusion(
                bm25_results=bm25_results,
                dense_results=dense_results,
                routed_group=routed_group,
                top_k=top_k_fused,
                rrf_k=RRF_K,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
            )
            if enable_reranker:
                final_results = self._rerank(
                    rewritten_query,
                    fused_results[:top_k_rerank],
                    top_k_final,
                )
            else:
                final_results = fused_results[:top_k_final]

        else:
            raise ValueError(f"Unknown profile_name: {self.profile_name}")

        diagnostics = {
            "profile_name": self.profile_name,
            "original_query": query,
            "rewritten_query": rewritten_query,
            "routed_group": routed_group,
            "num_bm25": len(bm25_results),
            "num_dense": len(dense_results),
            "num_fused": len(fused_results),
            "num_final": len(final_results),
            "final_chunk_ids": [x["chunk_id"] for x in final_results],
            "top_weighted_rrf_score": (
                fused_results[0]["weighted_rrf_score"] if fused_results else None
            ),
            "top_rerank_score": (
                final_results[0].get("rerank_score")
                if final_results and "rerank_score" in final_results[0]
                else None
            ),
            "no_answer_thresholds": {
                "min_rerank_score": NO_ANSWER_MIN_RERANK_SCORE,
                "min_fused_score": NO_ANSWER_MIN_FUSED_SCORE,
            },
        }

        return self._to_documents(final_results), diagnostics

    def _get_relevant_documents(self, query: str) -> list[Document]:
        docs, _ = self.retrieve_with_diagnostics(query)
        return docs