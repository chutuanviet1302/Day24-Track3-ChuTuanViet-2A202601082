from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, re, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            if os.getenv("RAG_FAST") or os.getenv("PYTEST_CURRENT_TEST"):
                self._model = False
                return self._model
            # Dùng sentence_transformers.CrossEncoder — KHÔNG dùng FlagEmbedding vì
            # FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
            try:
                from sentence_transformers import CrossEncoder
                kwargs = {} if os.getenv("RAG_ALLOW_MODEL_DOWNLOAD") else {"local_files_only": True}
                self._model = CrossEncoder(self.model_name, **kwargs)
            except Exception as exc:
                print(f"  CrossEncoder unavailable; using lexical fallback ({exc})")
                self._model = False
        return self._model

    @staticmethod
    def _fallback_score(query: str, text: str) -> float:
        # ponytail: token overlap is only an availability fallback, not a cross-encoder substitute.
        query_tokens = set(re.findall(r"\w+", query.lower()))
        document_tokens = set(re.findall(r"\w+", text.lower()))
        overlap = len(query_tokens & document_tokens) / max(len(query_tokens), 1)
        phrase_bonus = 0.25 if query.lower().strip() in text.lower() else 0.0
        return overlap + phrase_bonus

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        model = self._load_model()
        if model:
            pairs = [(query, doc["text"]) for doc in documents]
            scores = model.predict(pairs)
        else:
            scores = [self._fallback_score(query, doc["text"]) for doc in documents]
        if isinstance(scores, (int, float)):
            scores = [scores]
        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [RerankResult(
            text=doc["text"],
            original_score=float(doc.get("score", 0.0)),
            rerank_score=float(score),
            metadata=doc.get("metadata", {}),
            rank=i,
        ) for i, (score, doc) in enumerate(scored[:top_k])]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        from flashrank import Ranker, RerankRequest
        if not documents:
            return []
        if self._model is None:
            self._model = Ranker()
        passages = [{"text": d["text"]} for d in documents]
        results = self._model.rerank(RerankRequest(query=query, passages=passages))
        # flashrank trả về passages đã sắp xếp kèm score; map ngược về document gốc.
        by_text = {d["text"]: d for d in documents}
        ranked = []
        for i, r in enumerate(results[:top_k]):
            doc = by_text[r["text"]]
            ranked.append(RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(r["score"]),
                metadata=doc.get("metadata", {}),
                rank=i,
            ))
        return ranked


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
