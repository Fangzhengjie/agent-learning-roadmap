"""语义缓存 — 相似问题命中缓存，避免重复调用 LLM

两种缓存策略:
  1. 精确缓存: query hash → 完全相同的问题才命中
  2. 语义缓存: query embedding → 语义相似的问题也命中

生产方案: GPTCache / Redis + 向量索引 / LangChain CacheBackedEmbeddings
"""

import hashlib
import math
import time
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    query: str
    response: str
    embedding: list[float] | None = None
    created_at: float = 0.0
    hit_count: int = 0
    ttl_s: float = 3600.0


class ExactCache:
    """精确缓存 — 完全相同的 query 才命中。"""

    def __init__(self, max_size: int = 1000, ttl_s: float = 3600):
        self._store: dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.ttl_s = ttl_s
        self.stats = {"hits": 0, "misses": 0}

    @staticmethod
    def _hash(query: str) -> str:
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get(self, query: str) -> str | None:
        key = self._hash(query)
        entry = self._store.get(key)
        if entry is None:
            self.stats["misses"] += 1
            return None
        if time.time() - entry.created_at > entry.ttl_s:
            del self._store[key]
            self.stats["misses"] += 1
            return None
        entry.hit_count += 1
        self.stats["hits"] += 1
        return entry.response

    def put(self, query: str, response: str):
        if len(self._store) >= self.max_size:
            oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
            del self._store[oldest_key]
        self._store[self._hash(query)] = CacheEntry(
            query=query, response=response, created_at=time.time(), ttl_s=self.ttl_s
        )

    @property
    def hit_rate(self) -> float:
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total > 0 else 0.0


class SemanticCache:
    """语义缓存 — 语义相似的 query 也命中（余弦相似度）。

    生产方案: GPTCache + FAISS / Redis Vector Search
    """

    def __init__(self, threshold: float = 0.85, max_size: int = 1000, ttl_s: float = 3600):
        self.threshold = threshold
        self.max_size = max_size
        self.ttl_s = ttl_s
        self._entries: list[CacheEntry] = []
        self.stats = {"hits": 0, "misses": 0, "saved_cost": 0.0}

    @staticmethod
    def _simple_embed(text: str) -> list[float]:
        """简易嵌入（字符 bigram TF）。生产用 OpenAI Embedding。"""
        tokens: dict[str, int] = {}
        chars = text.lower().replace(" ", "")
        for i in range(len(chars) - 1):
            bg = chars[i:i + 2]
            tokens[bg] = tokens.get(bg, 0) + 1
        for ch in chars:
            tokens[ch] = tokens.get(ch, 0) + 1
        return list(tokens.values()) if tokens else [0.0]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1e-9
        nb = math.sqrt(sum(x * x for x in b)) or 1e-9
        return dot / (na * nb)

    def get(self, query: str, cost_per_call: float = 0.01) -> str | None:
        q_emb = self._simple_embed(query)
        now = time.time()
        best_score, best_entry = 0.0, None
        for entry in self._entries:
            if now - entry.created_at > entry.ttl_s:
                continue
            score = self._cosine(q_emb, entry.embedding)
            if score > best_score:
                best_score, best_entry = score, entry
        if best_entry and best_score >= self.threshold:
            best_entry.hit_count += 1
            self.stats["hits"] += 1
            self.stats["saved_cost"] += cost_per_call
            return best_entry.response
        self.stats["misses"] += 1
        return None

    def put(self, query: str, response: str):
        if len(self._entries) >= self.max_size:
            self._entries.sort(key=lambda e: e.hit_count)
            self._entries = self._entries[len(self._entries) // 4:]
        self._entries.append(CacheEntry(
            query=query, response=response,
            embedding=self._simple_embed(query),
            created_at=time.time(), ttl_s=self.ttl_s,
        ))

    @property
    def hit_rate(self) -> float:
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total > 0 else 0.0
