from __future__ import annotations

"""Production RAG pipeline: M1 + M5 + M2 + M3 + M4."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY, RERANK_TOP_K
from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import failure_analysis, evaluate_ragas, load_test_set, save_report
from src.m5_enrichment import enrich_chunks


def build_pipeline():
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)
    timings = {}

    started = time.perf_counter()
    print("\n[1/4] Chunking documents...", flush=True)
    documents = load_documents()
    chunks = []
    for document in documents:
        _, children = chunk_hierarchical(document["text"], metadata=document["metadata"])
        chunks.extend({
            "text": child.text,
            "metadata": {**child.metadata, "parent_id": child.parent_id},
        } for child in children)
    timings["chunking"] = _elapsed_ms(started)
    print(f"  Loaded {len(chunks)} chunks from {len(documents)} documents", flush=True)

    started = time.perf_counter()
    print(f"\n[2/4] Enriching {len(chunks)} chunks (M5)...", flush=True)
    enriched = enrich_chunks(chunks)
    chunks = [{"text": item.enriched_text, "metadata": item.auto_metadata} for item in enriched]
    timings["enrichment"] = _elapsed_ms(started)

    started = time.perf_counter()
    print(f"\n[3/4] Indexing {len(chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(chunks)
    timings["indexing"] = _elapsed_ms(started)

    started = time.perf_counter()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    timings["reranker_load"] = _elapsed_ms(started)
    search.latency_breakdown = timings
    return search, reranker


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    results = search.search(query)
    documents = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, documents, top_k=RERANK_TOP_K)
    contexts = [result.text for result in reranked] or [result.text for result in results[:3]]

    if OPENAI_API_KEY and not os.getenv("RAG_FAST") and contexts:
        try:
            from openai import OpenAI
            context_str = "\n\n".join(contexts)
            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer only from the context. If absent, say not found."},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
                ],
            )
            answer = response.choices[0].message.content or contexts[0]
        except Exception as exc:
            print(f"  LLM generation failed; using retrieved context ({exc})", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Khong tim thay thong tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []
    for item in test_set:
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])

    started = time.perf_counter()
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    failures = failure_analysis(results.get("per_question", []))
    latency = dict(getattr(search, "latency_breakdown", {}))
    latency["evaluation"] = _elapsed_ms(started)
    save_report(results, failures, latency_breakdown=latency)
    return results


if __name__ == "__main__":
    started = time.perf_counter()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {_elapsed_ms(started) / 1000:.1f}s")
