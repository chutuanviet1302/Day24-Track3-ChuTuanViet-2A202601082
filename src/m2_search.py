from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import hashlib
import math
import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        if os.getenv("RAG_FAST") or os.getenv("PYTEST_CURRENT_TEST"):
            raise RuntimeError("fast/offline tokenizer fallback")
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
    except Exception:
        segmented = text
    # underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép") → tách lại thành khoảng
    # trắng để BM25 split(" ") khớp giữa document và query.
    return segmented.replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        if not chunks:
            self.documents, self.corpus_tokens, self.bm25 = [], [], None
            return
        self.documents = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            tokens = segment_vietnamese(chunk["text"]).split()
            self.corpus_tokens.append(tokens)
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append(SearchResult(
                    text=self.documents[i]["text"],
                    score=float(scores[i]),
                    metadata=self.documents[i].get("metadata", {}),
                    method="bm25",
                ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None
        self._qdrant_indexed = False
        self._fallback_documents: list[dict] = []
        self._fallback_vectors: list[list[float]] = []

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            kwargs = {} if os.getenv("RAG_ALLOW_MODEL_DOWNLOAD") else {"local_files_only": True}
            self._encoder = SentenceTransformer(EMBEDDING_MODEL, **kwargs)
        return self._encoder

    @staticmethod
    def _fallback_vector(text: str, size: int = 256) -> list[float]:
        vector = [0.0] * size
        for token in segment_vietnamese(text).lower().split():
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % size
            vector[index] += 1.0
        length = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / length for value in vector]

    def _index_fallback(self, chunks: list[dict]) -> None:
        # ponytail: hashed vectors keep offline runs working; Qdrant+bge-m3 is the production path.
        self._fallback_documents = chunks
        self._fallback_vectors = [self._fallback_vector(c["text"]) for c in chunks]
        self._qdrant_indexed = False

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        self._index_fallback(chunks)
        if os.getenv("RAG_FAST"):
            return
        try:
            self.client.get_collections()
            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            texts = [c["text"] for c in chunks]
            vectors = self._get_encoder().encode(texts, show_progress_bar=True)
            points = [PointStruct(
                id=i,
                vector=v.tolist(),
                payload={**c.get("metadata", {}), "text": c["text"]},
            ) for i, (c, v) in enumerate(zip(chunks, vectors))]
            self.client.upsert(collection, points)
            self._qdrant_indexed = True
        except Exception as exc:
            print(f"  Dense/Qdrant unavailable; using local fallback ({exc})")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if self._qdrant_indexed:
            query_vector = self._get_encoder().encode(query).tolist()
            response = self.client.query_points(collection, query=query_vector, limit=top_k)
            return [SearchResult(
                text=pt.payload["text"],
                score=float(pt.score),
                metadata=dict(pt.payload),
                method="dense",
            ) for pt in response.points]

        query_vector = self._fallback_vector(query)
        scored = []
        for document, vector in zip(self._fallback_documents, self._fallback_vectors):
            score = sum(a * b for a, b in zip(query_vector, vector))
            scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [SearchResult(
            text=document["text"], score=float(score),
            metadata=document.get("metadata", {}), method="dense",
        ) for score, document in scored[:top_k]]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = result.text
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "result": result}
            # rank-based contribution — BM25 score và cosine score không cùng thang đo.
            rrf_scores[key]["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [SearchResult(
        text=item["result"].text,
        score=item["score"],
        metadata=item["result"].metadata,
        method="hybrid",
    ) for item in ranked[:top_k]]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
