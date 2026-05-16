from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from scripts import config

def load_retrieval_assets() -> Tuple[Dict[str, Any], Chroma, CrossEncoder | None]:
    """Loads and caches the BM25 payload, Chroma vector database, and CrossEncoder."""
    if not config.BM25_FILE.exists():
        raise FileNotFoundError(f"BM25 index not found at {config.BM25_FILE}")
        
    with config.BM25_FILE.open("rb") as f:
        bm25_payload = pickle.load(f)

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )

    reranker = CrossEncoder(config.RERANKER_MODEL) if config.ENABLE_RERANK else None

    return bm25_payload, vectorstore, reranker


def rewrite_query(query: str) -> str:
    """Replaces colloquial phrases with high-value domain keywords."""
    if not config.ENABLE_QUERY_REWRITE:
        return query
        
    q_lower = query.strip().lower()
    for needle, replacement in config.REWRITE_HINTS.items():
        if needle in q_lower:
            q_lower = q_lower.replace(needle, replacement)
    return q_lower


def route_group(query: str) -> str | None:
    """Classifies the query into a predefined routing group based on keyword matching."""
    if not config.ENABLE_ROUTING:
        return None
        
    q_lower = query.lower()
    best_group = None
    best_score = 0
    
    for group, keywords in config.ROUTING_RULES.items():
        score = sum(1 for kw in keywords if kw in q_lower)
        if score > best_score:
            best_score = score
            best_group = group
            
    return best_group


def calculate_title_boost(title: str, routed_group: str | None) -> float:
    """Applies a multiplier to documents matching the predicted intent routing."""
    if not config.ENABLE_ROUTING or not routed_group:
        return 1.0
    preferred_titles = config.ROUTING_GROUP_TO_TITLES.get(routed_group, [])
    return 1.15 if title in preferred_titles else 1.0


def search_bm25(query: str, bm25_payload: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    """Executes sparse lexical retrieval."""
    if top_k <= 0:
        return []
        
    bm25 = bm25_payload["bm25"]
    chunks = bm25_payload["chunks"]
    tokens = config.bm25_tokenize(query)
    scores = bm25.get_scores(tokens)
    
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


def search_dense(query: str, vectorstore: Chroma, top_k: int) -> List[Dict[str, Any]]:
    """Executes dense semantic retrieval."""
    if top_k <= 0:
        return []
        
    prefixed_query = f"query: {query}"
    docs_and_scores = vectorstore.similarity_search_with_score(prefixed_query, k=top_k)

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


def fuse_results(
    bm25_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    routed_group: str | None,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Merges disparate retrieval results using Reciprocal Rank Fusion (RRF)."""
    fused_map: Dict[str, Dict[str, Any]] = {}

    def apply_rrf(results: List[Dict[str, Any]], weight: float, source_name: str):
        for item in results:
            cid = item["chunk_id"]
            base_score = weight * (1.0 / (config.RRF_K + item["rank"]))
            boosted_score = base_score * calculate_title_boost(item.get("title", ""), routed_group)
            
            if cid not in fused_map:
                fused_map[cid] = {
                    **item,
                    "weighted_rrf_score": 0.0,
                    "retrievers": [],
                }
            fused_map[cid]["weighted_rrf_score"] += boosted_score
            fused_map[cid]["retrievers"].append(source_name)

    apply_rrf(bm25_results, config.BM25_WEIGHT, "bm25")
    apply_rrf(dense_results, config.DENSE_WEIGHT, "dense")

    ranked = sorted(fused_map.values(), key=lambda x: x["weighted_rrf_score"], reverse=True)
    filtered = [r for r in ranked if r["weighted_rrf_score"] >= config.MIN_WEIGHTED_RRF_SCORE]

    return filtered[:top_k] if filtered else ranked[:top_k]


def rerank_results(
    query: str, 
    candidates: List[Dict[str, Any]], 
    reranker: CrossEncoder, 
    top_k: int
) -> List[Dict[str, Any]]:
    """Calculates true semantic similarity using a CrossEncoder model."""
    if not candidates or reranker is None:
        return candidates[:top_k]

    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for cand, score in zip(candidates, scores):
        cand["rerank_score"] = float(score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]


def retrieve_documents(query: str, assets: Tuple, profile_name: str) -> Tuple[List[Dict], Dict]:
    """Main pipeline execution for document retrieval."""
    bm25_payload, vectorstore, reranker = assets
    profile_cfg = config.RETRIEVAL_PROFILES[profile_name]
    
    rewritten_query = rewrite_query(query)
    routed_group = route_group(rewritten_query)

    bm25_docs = search_bm25(rewritten_query, bm25_payload, profile_cfg.get("top_k_bm25", 0))
    dense_docs = search_dense(rewritten_query, vectorstore, profile_cfg.get("top_k_dense", 0))

    if profile_name in ["bm25_only", "dense_only"]:
        final_docs = bm25_docs or dense_docs
        fused_docs = final_docs
    else:
        fused_docs = fuse_results(bm25_docs, dense_docs, routed_group, profile_cfg.get("top_k_fused", 10))
        if profile_cfg.get("enable_reranker") and reranker:
            final_docs = rerank_results(rewritten_query, fused_docs, reranker, profile_cfg.get("top_k_final", 5))
        else:
            final_docs = fused_docs[:profile_cfg.get("top_k_final", 5)]

    diagnostics = {
        "profile_name": profile_name,
        "original_query": query,
        "rewritten_query": rewritten_query,
        "routed_group": routed_group,
        "top_weighted_rrf_score": fused_docs[0].get("weighted_rrf_score") if fused_docs else None,
        "top_rerank_score": final_docs[0].get("rerank_score") if final_docs and "rerank_score" in final_docs[0] else None,
    }

    return final_docs, diagnostics