from __future__ import annotations

import argparse
import json
from typing import Any

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from scripts.config import (
    ENABLE_NO_ANSWER_GATING,
    HUMAN_PROMPT,
    LLM_MODEL,
    NO_ANSWER_MIN_FUSED_SCORE,
    NO_ANSWER_MIN_RERANK_SCORE,
    SAFE_FALLBACK_ANSWER,
    SYSTEM_PROMPT,
)
from scripts.retriever import HybridRetriever


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])


def build_answer_chain(
    model_name: str = LLM_MODEL,
    request_timeout: int = 120,
):
    llm = ChatOllama(
        model=model_name,
        temperature=0.0,
        num_predict=120,
        request_timeout=request_timeout,
    )
    prompt = build_prompt()
    return create_stuff_documents_chain(llm, prompt)


def normalize_answer(raw_answer: str) -> str:
    answer = (raw_answer or "").strip()
    if not answer:
        return SAFE_FALLBACK_ANSWER
    if SAFE_FALLBACK_ANSWER.lower() in answer.lower():
        return SAFE_FALLBACK_ANSWER
    return answer


def should_abstain(
    retrieved_chunks: list[dict[str, Any]],
    retrieval_diagnostics: dict[str, Any],
) -> bool:
    if not ENABLE_NO_ANSWER_GATING:
        return False

    if not retrieved_chunks:
        return True

    profile_name = retrieval_diagnostics.get("profile_name", "")

    top_rerank_score = retrieval_diagnostics.get("top_rerank_score")
    top_fused_score = retrieval_diagnostics.get("top_weighted_rrf_score")

    if profile_name == "hybrid_rerank":
        if top_rerank_score is None:
            return False
        return float(top_rerank_score) < NO_ANSWER_MIN_RERANK_SCORE

    if profile_name == "hybrid":
        if top_fused_score is None:
            return False
        return float(top_fused_score) < NO_ANSWER_MIN_FUSED_SCORE

    return False


def answer_question(
    question: str,
    retriever: HybridRetriever,
    model_name: str = LLM_MODEL,
    request_timeout: int = 120,
) -> dict[str, Any]:
    docs, diagnostics = retriever.retrieve_with_diagnostics(question)

    retrieved_chunks = [
        {
            "chunk_id": d.metadata.get("chunk_id"),
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "title": d.metadata.get("title"),
            "score": d.metadata.get("score"),
            "retrievers": d.metadata.get("retrievers"),
            "text": d.page_content,
        }
        for d in docs
    ]

    if should_abstain(retrieved_chunks, diagnostics):
        return {
            "question": question,
            "answer": SAFE_FALLBACK_ANSWER,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_diagnostics": diagnostics,
            "profile_name": retriever.profile_name,
        }

    answer_chain = build_answer_chain(
        model_name=model_name,
        request_timeout=request_timeout,
    )
    result = answer_chain.invoke({
    "question": question,
    "context": docs,
    })
    raw = result.content if hasattr(result, "content") else str(result)
    answer = normalize_answer(raw)

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "retrieval_diagnostics": diagnostics,
        "profile_name": retriever.profile_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--model", type=str, default=LLM_MODEL)
    parser.add_argument("--profile", type=str, default="hybrid_rerank")
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--request_timeout", type=int, default=120)
    args = parser.parse_args()

    retriever = HybridRetriever(profile_name=args.profile)
    result = answer_question(
        question=args.query,
        retriever=retriever,
        model_name=args.model,
        request_timeout=args.request_timeout,
    )

    if args.save_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("\nQuestion:")
    print(result["question"])
    print("\nAnswer:")
    print(result["answer"])
    print(f"\nProfile: {result['profile_name']}")
    print("\nTop retrieved chunks:")
    for i, chunk in enumerate(result["retrieved_chunks"], start=1):
        print(
            f"  {i}. {chunk['source']} | page {chunk['page']} | "
            f"score={chunk['score']} | retrievers={chunk['retrievers']}"
        )


if __name__ == "__main__":
    main()