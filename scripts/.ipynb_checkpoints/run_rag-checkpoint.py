from __future__ import annotations

import argparse
import json
from typing import Any

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from scripts import config
from scripts.retriever import load_retrieval_assets, retrieve_documents


def build_answer_chain(model_name: str, request_timeout: int):
    """Constructs the LangChain generation pipeline."""
    llm = ChatOllama(
        model=model_name,
        temperature=0.0,
        num_predict=120,
        request_timeout=request_timeout,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", config.SYSTEM_PROMPT),
        ("human", config.HUMAN_PROMPT),
    ])
    return create_stuff_documents_chain(llm, prompt)


def should_abstain(
    chunks: list[dict[str, Any]], 
    diagnostics: dict[str, Any],
    rerank_threshold: float,
    fused_threshold: float
) -> bool:
    """Evaluates whether the system lacks confidence to generate a safe answer."""
    if not config.ENABLE_NO_ANSWER_GATING or not chunks:
        return True

    profile_name = diagnostics.get("profile_name", "")
    if profile_name == "hybrid_rerank":
        score = diagnostics.get("top_rerank_score")
        return score is None or float(score) < rerank_threshold
        
    if profile_name == "hybrid":
        score = diagnostics.get("top_weighted_rrf_score")
        return score is None or float(score) < fused_threshold

    return False

def answer_question(
    question: str, 
    assets: tuple, 
    profile_name: str = "hybrid_rerank",
    model_name: str = config.LLM_MODEL,
    rerank_threshold: float = config.NO_ANSWER_MIN_RERANK_SCORE,
    fused_threshold: float = config.NO_ANSWER_MIN_FUSED_SCORE
) -> dict[str, Any]:
    """End-to-end execution of the RAG pipeline with thread-safe parameter passing."""
    chunks, diagnostics = retrieve_documents(question, assets, profile_name)

    if should_abstain(chunks, diagnostics, rerank_threshold, fused_threshold):
        return {
            "question": question,
            "answer": config.SAFE_FALLBACK_ANSWER,
            "retrieved_chunks": chunks,
            "profile_name": profile_name,
        }

    langchain_docs = [Document(page_content=c["text"]) for c in chunks]
    chain = build_answer_chain(model_name, request_timeout=120)
    
    result = chain.invoke({"question": question, "context": langchain_docs})
    answer = result.content if hasattr(result, "content") else str(result)

    return {
        "question": question,
        "answer": answer.strip() or config.SAFE_FALLBACK_ANSWER,
        "retrieved_chunks": chunks,
        "profile_name": profile_name,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--profile", type=str, default="hybrid_rerank")
    args = parser.parse_args()

    assets = load_retrieval_assets()
    result = answer_question(args.query, assets, args.profile)
    
    print(f"Question: {result['question']}\nAnswer: {result['answer']}")