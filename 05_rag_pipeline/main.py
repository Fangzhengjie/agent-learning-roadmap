"""RAG (Retrieval-Augmented Generation) 完整流程

核心概念：让 LLM 使用私有数据回答问题 — 不是微调，而是检索后注入上下文。

RAG 五步流程：
  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
  │ 1.加载   │ →  │ 2.分片   │ →  │ 3.嵌入   │ →  │ 4.检索   │ →  │ 5.生成   │
  │ Document │    │ Chunking │    │Embedding │    │Retrieval │    │Generate │
  │ Loading  │    │          │    │ + Store  │    │          │    │         │
  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘

本示例不依赖向量数据库，用纯 Python 展示 RAG 每一步的核心原理，包括：
  1. 五步流程：加载 → 分片 → 嵌入 → 检索 → 生成
  2. 向量数据库选型：8 大向量库多维度对比 + 决策树 + 配置示例
  3. RAG 质量评测：Ragas 四大指标 + 检索指标 (MRR / Hit Rate)
  4. 高级 RAG 技巧：混合检索 / Reranker / Graph RAG / Agentic RAG

生产方案：
  - 向量数据库：FAISS / Chroma / Pinecone / Weaviate / pgvector / Milvus
  - 嵌入模型：OpenAI text-embedding-3-small / BGE / E5 / Jina
  - 评测工具：Ragas / DeepEval / TruLens / LangSmith
  - 框架集成：LangChain Retriever / LlamaIndex / Spring AI QuestionAnswerAdvisor
"""

import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# Step 1: 文档加载（Document Loading）
# ═══════════════════════════════════════════════════════════

# 模拟企业知识库文档（实际中从文件/数据库/API 加载）
RAW_DOCUMENTS = [
    {
        "id": "doc-001",
        "source": "产品手册.pdf",
        "content": """SmartFlow 工作流引擎是企业级的流程自动化平台。
支持可视化流程设计、表单构建、规则引擎和消息通知。
系统要求：Java 17+，MySQL 8.0+，Redis 6.0+。
部署方式支持 Docker 容器化部署和 K8s 集群部署。
默认端口 8080，管理后台端口 9090。
许可证类型分为社区版（免费，最多 10 个流程）和企业版（按用户数计费）。"""
    },
    {
        "id": "doc-002",
        "source": "API文档.md",
        "content": """SmartFlow REST API 使用指南。
认证方式：Bearer Token，通过 POST /api/auth/login 获取。
创建流程：POST /api/workflows，请求体包含 name 和 definition 字段。
启动流程实例：POST /api/workflows/{id}/start。
查询流程状态：GET /api/workflows/{id}/instances/{instanceId}。
API 限流：免费版 100 次/分钟，企业版 10000 次/分钟。
所有 API 返回 JSON 格式，错误码遵循 HTTP 标准状态码。"""
    },
    {
        "id": "doc-003",
        "source": "常见问题.md",
        "content": """Q: 如何重置管理员密码？
A: 执行命令 smartflow admin reset-password --user admin，然后按提示设置新密码。

Q: 流程实例卡住怎么办？
A: 检查日志 /var/log/smartflow/engine.log，常见原因是外部服务超时。
可以通过管理后台的"实例管理"页面手动重试或终止。

Q: 支持哪些数据库？
A: 主数据库支持 MySQL 8.0+ 和 PostgreSQL 14+。
缓存支持 Redis 6.0+ 或 Valkey。

Q: 如何升级到企业版？
A: 联系 sales@smartflow.com，或在管理后台点击"升级许可证"输入企业版 License Key。"""
    },
    {
        "id": "doc-004",
        "source": "部署指南.md",
        "content": """Docker 部署步骤：
1. 拉取镜像：docker pull smartflow/engine:latest
2. 创建配置文件 application.yml
3. 启动容器：docker run -d -p 8080:8080 -v ./config:/app/config smartflow/engine
4. 访问 http://localhost:8080 验证

K8s 部署步骤：
1. 创建 namespace: kubectl create namespace smartflow
2. 部署 MySQL: kubectl apply -f mysql-statefulset.yaml
3. 部署引擎: kubectl apply -f smartflow-deployment.yaml
4. 配置 Ingress: kubectl apply -f smartflow-ingress.yaml

生产环境建议：至少 3 个引擎副本，MySQL 主从复制，Redis Sentinel。"""
    },
    {
        "id": "doc-005",
        "source": "更新日志.md",
        "content": """v3.2.0 (2024-12)
- 新增：支持子流程调用和并行网关
- 新增：WebSocket 实时通知
- 优化：流程编辑器性能提升 60%
- 修复：MySQL 8.0.35 兼容性问题

v3.1.0 (2024-09)
- 新增：REST API v2（向后兼容）
- 新增：LDAP/SSO 集成
- 优化：大批量实例查询性能
- 修复：Redis 连接池泄漏问题"""
    },
]


# ═══════════════════════════════════════════════════════════
# Step 2: 文档分片（Chunking）
# ═══════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """一个文档片段。"""
    chunk_id: str
    doc_id: str
    source: str
    content: str
    metadata: dict = field(default_factory=dict)


class Chunker:
    """文档分片器。

    分片策略：
    1. 固定大小：按字符数切分（最简单）
    2. 语义分段：按段落/标题切分（本示例使用）
    3. 递归分割：先按大分隔符切，不够再按小分隔符切（LangChain 默认）
    4. 滑动窗口：固定大小 + 重叠（保留上下文）
    """

    @staticmethod
    def by_paragraph(doc: dict, max_chunk_size: int = 300, overlap: int = 50) -> list[Chunk]:
        """按段落分片，长段落进一步切分。"""
        content = doc["content"]
        paragraphs = [p.strip() for p in content.split("\n") if p.strip()]

        chunks = []
        current = ""
        chunk_idx = 0

        for para in paragraphs:
            if len(current) + len(para) > max_chunk_size and current:
                chunks.append(Chunk(
                    chunk_id=f"{doc['id']}-{chunk_idx}",
                    doc_id=doc["id"],
                    source=doc["source"],
                    content=current.strip(),
                    metadata={"chunk_index": chunk_idx, "char_count": len(current)},
                ))
                # 保留重叠部分
                if overlap > 0 and len(current) > overlap:
                    current = current[-overlap:] + "\n" + para
                else:
                    current = para
                chunk_idx += 1
            else:
                current = current + "\n" + para if current else para

        if current.strip():
            chunks.append(Chunk(
                chunk_id=f"{doc['id']}-{chunk_idx}",
                doc_id=doc["id"],
                source=doc["source"],
                content=current.strip(),
                metadata={"chunk_index": chunk_idx, "char_count": len(current)},
            ))

        return chunks


# ═══════════════════════════════════════════════════════════
# Step 3: 嵌入 + 向量存储（Embedding + Vector Store）
# ═══════════════════════════════════════════════════════════

class SimpleEmbedding:
    """简易嵌入模型（TF-IDF 近似）。

    生产中应使用：
    - OpenAI text-embedding-3-small（1536 维）
    - BGE-M3（多语言，开源）
    - Jina Embeddings v3
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.dim = 0

    def _tokenize(self, text: str) -> list[str]:
        """简单分词（中文按字，英文按词）。"""
        tokens = []
        for char in text.lower():
            if '\u4e00' <= char <= '\u9fff':
                tokens.append(char)
            elif char.isalnum():
                tokens.append(char)
            elif tokens and tokens[-1] != ' ':
                tokens.append(' ')
        # 合并英文单词
        merged = []
        current = ""
        for t in tokens:
            if t == ' ':
                if current:
                    merged.append(current)
                    current = ""
            elif t.isascii():
                current += t
            else:
                if current:
                    merged.append(current)
                    current = ""
                merged.append(t)
        if current:
            merged.append(current)
        return merged

    def fit(self, texts: list[str]):
        """构建词汇表和 IDF。"""
        doc_count = len(texts)
        df = Counter()  # document frequency

        for text in texts:
            unique_tokens = set(self._tokenize(text))
            for token in unique_tokens:
                df[token] += 1
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        self.dim = len(self.vocab)
        self.idf = {token: math.log(doc_count / (freq + 1)) + 1 for token, freq in df.items()}

    def embed(self, text: str) -> list[float]:
        """生成文本的向量表示（TF-IDF）。"""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1

        vec = [0.0] * self.dim
        for token, count in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                tf_val = count / total
                idf_val = self.idf.get(token, 1.0)
                vec[idx] = tf_val * idf_val

        # L2 归一化
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class SimpleVectorStore:
    """简易向量数据库。

    生产中应使用：
    - FAISS（Meta，单机最快）
    - Chroma（轻量，适合原型）
    - Pinecone（全托管，免运维）
    - Weaviate（功能丰富，开源）
    - pgvector（PostgreSQL 扩展，复用现有 DB）
    - Milvus（大规模分布式）
    """

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []

    def add(self, chunk: Chunk, vector: list[float]):
        self.chunks.append(chunk)
        self.vectors.append(vector)

    def search(self, query_vec: list[float], top_k: int = 3) -> list[tuple[Chunk, float]]:
        """余弦相似度检索。"""
        scores = []
        for i, vec in enumerate(self.vectors):
            # cosine similarity（已归一化，点积即可）
            score = sum(a * b for a, b in zip(query_vec, vec))
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.chunks[i], score) for i, score in scores[:top_k]]

    @property
    def size(self) -> int:
        return len(self.chunks)


# ═══════════════════════════════════════════════════════════
# Step 3b: 主流向量数据库接口实现
# ═══════════════════════════════════════════════════════════

# ---------- 统一抽象接口 ----------

class VectorStoreBase:
    """向量数据库统一接口 — 所有实现遵循此协议。"""

    def add_documents(self, chunks: list[Chunk], vectors: list[list[float]]):
        raise NotImplementedError

    def search(self, query_vec: list[float], top_k: int = 3,
               filter_metadata: dict | None = None) -> list[tuple[Chunk, float]]:
        raise NotImplementedError

    def delete(self, chunk_ids: list[str]):
        raise NotImplementedError

    @property
    def count(self) -> int:
        raise NotImplementedError


# ---------- 1. FAISS 实现 ----------

class FAISSVectorStore(VectorStoreBase):
    """FAISS 向量存储（Meta 开源，单机最快）。

    生产代码:
        import faiss
        index = faiss.IndexFlatIP(dim)           # 内积（归一化=余弦）
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)  # IVF 加速
        index = faiss.IndexHNSWFlat(dim, M=32)   # HNSW 图索引
        index.add(np.array(vectors, dtype='float32'))
        D, I = index.search(query_vec, k=5)

    特点: 纯库模式(无服务器)、十亿级、GPU 加速、无元数据过滤
    pip install faiss-cpu  # 或 faiss-gpu
    """

    def __init__(self, dim: int, index_type: str = "Flat"):
        self.dim = dim
        self.index_type = index_type  # Flat / IVF / HNSW
        self._vectors: list[list[float]] = []
        self._chunks: list[Chunk] = []
        self._id_map: dict[str, int] = {}

    def add_documents(self, chunks: list[Chunk], vectors: list[list[float]]):
        for chunk, vec in zip(chunks, vectors):
            self._id_map[chunk.chunk_id] = len(self._chunks)
            self._chunks.append(chunk)
            self._vectors.append(vec)

    def search(self, query_vec: list[float], top_k: int = 3,
               filter_metadata: dict | None = None) -> list[tuple[Chunk, float]]:
        # FAISS 原生不支持元数据过滤，需要后过滤
        scores = []
        for i, vec in enumerate(self._vectors):
            score = sum(a * b for a, b in zip(query_vec, vec))
            chunk = self._chunks[i]
            if filter_metadata:
                if not all(chunk.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self._chunks[i], s) for i, s in scores[:top_k]]

    def delete(self, chunk_ids: list[str]):
        indices = {self._id_map[cid] for cid in chunk_ids if cid in self._id_map}
        self._chunks = [c for i, c in enumerate(self._chunks) if i not in indices]
        self._vectors = [v for i, v in enumerate(self._vectors) if i not in indices]
        self._id_map = {c.chunk_id: i for i, c in enumerate(self._chunks)}

    @property
    def count(self) -> int:
        return len(self._chunks)


# ---------- 2. Chroma 实现 ----------

class ChromaVectorStore(VectorStoreBase):
    """Chroma 向量存储（最简单，3 行代码即用）。

    生产代码:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"}
        )
        collection.add(
            ids=["d1", "d2"],
            documents=["text1", "text2"],
            embeddings=[[0.1, ...], [0.2, ...]],
            metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}]
        )
        results = collection.query(
            query_embeddings=[[0.15, ...]],
            n_results=3,
            where={"source": "a.pdf"}  # 元数据过滤
        )

    特点: 嵌入式/CS模式、百万级、内置元数据过滤、pip install 即用
    pip install chromadb
    """

    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self._docs: dict[str, dict] = {}  # id → {chunk, vector, metadata}

    def add_documents(self, chunks: list[Chunk], vectors: list[list[float]]):
        for chunk, vec in zip(chunks, vectors):
            self._docs[chunk.chunk_id] = {
                "chunk": chunk,
                "vector": vec,
                "metadata": {"source": chunk.source, **chunk.metadata},
            }

    def search(self, query_vec: list[float], top_k: int = 3,
               filter_metadata: dict | None = None) -> list[tuple[Chunk, float]]:
        scored = []
        for doc_id, doc in self._docs.items():
            # where 过滤（模拟 Chroma 的 where 参数）
            if filter_metadata:
                if not all(doc["metadata"].get(k) == v for k, v in filter_metadata.items()):
                    continue
            score = sum(a * b for a, b in zip(query_vec, doc["vector"]))
            scored.append((doc["chunk"], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete(self, chunk_ids: list[str]):
        for cid in chunk_ids:
            self._docs.pop(cid, None)

    @property
    def count(self) -> int:
        return len(self._docs)


# ---------- 3. pgvector 实现 ----------

class PgVectorStore(VectorStoreBase):
    """pgvector 向量存储（PostgreSQL 扩展，复用现有数据库）。

    生产代码 (SQL):
        CREATE EXTENSION vector;
        CREATE TABLE documents (
            id       TEXT PRIMARY KEY,
            content  TEXT NOT NULL,
            source   TEXT,
            metadata JSONB DEFAULT '{}',
            embedding VECTOR(1536)
        );
        -- 创建索引（二选一）
        CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

    生产代码 (Python):
        import psycopg2
        from pgvector.psycopg2 import register_vector
        conn = psycopg2.connect("postgresql://user:pass@localhost/mydb")
        register_vector(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, content, 1 - (embedding <=> %s) AS score "
            "FROM documents WHERE metadata->>'source' = %s "
            "ORDER BY embedding <=> %s LIMIT %s",
            (query_vec, 'a.pdf', query_vec, 5)
        )

    特点: 复用PG、SQL过滤+向量检索、RLS多租户、事务一致性
    pip install pgvector psycopg2-binary
    """

    def __init__(self, table_name: str = "documents"):
        self.table_name = table_name
        self._rows: list[dict] = []  # 模拟 SQL 表行

    def add_documents(self, chunks: list[Chunk], vectors: list[list[float]]):
        for chunk, vec in zip(chunks, vectors):
            self._rows.append({
                "id": chunk.chunk_id,
                "content": chunk.content,
                "source": chunk.source,
                "metadata": chunk.metadata,
                "embedding": vec,
            })

    def search(self, query_vec: list[float], top_k: int = 3,
               filter_metadata: dict | None = None) -> list[tuple[Chunk, float]]:
        # 模拟 SQL: SELECT ... ORDER BY embedding <=> query LIMIT k
        scored = []
        for row in self._rows:
            if filter_metadata:
                if not all(row["metadata"].get(k) == v for k, v in filter_metadata.items()):
                    continue
            # cosine similarity (normalized dot product)
            score = sum(a * b for a, b in zip(query_vec, row["embedding"]))
            chunk = Chunk(chunk_id=row["id"], doc_id=row["id"].rsplit("-", 1)[0],
                          source=row["source"], content=row["content"], metadata=row["metadata"])
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete(self, chunk_ids: list[str]):
        self._rows = [r for r in self._rows if r["id"] not in set(chunk_ids)]

    def execute_sql(self, sql: str) -> str:
        """模拟 SQL 查询（展示 pgvector 的 SQL 能力）。"""
        return f"[pgvector] 执行: {sql}  (影响 {len(self._rows)} 行)"

    @property
    def count(self) -> int:
        return len(self._rows)


# ---------- 4. Milvus 实现 ----------

class MilvusVectorStore(VectorStoreBase):
    """Milvus 向量存储（大规模分布式，百亿级向量）。

    生产代码:
        from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

        connections.connect("default", host="localhost", port="19530")

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
        ]
        schema = CollectionSchema(fields, description="RAG documents")
        collection = Collection("documents", schema)

        # 创建索引
        collection.create_index("embedding", {
            "index_type": "IVF_FLAT",    # 或 HNSW, IVF_SQ8, IVF_PQ
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        })
        collection.load()

        # 检索
        results = collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=5,
            expr='source == "a.pdf"',  # 标量过滤
            output_fields=["content", "source"]
        )

    特点: 分布式、百亿级、GPU索引、丰富索引类型、标量+向量混合过滤
    pip install pymilvus
    """

    def __init__(self, collection_name: str = "documents", dim: int = 1536):
        self.collection_name = collection_name
        self.dim = dim
        self._entities: list[dict] = []
        self._index_type = "IVF_FLAT"

    def add_documents(self, chunks: list[Chunk], vectors: list[list[float]]):
        for chunk, vec in zip(chunks, vectors):
            self._entities.append({
                "id": chunk.chunk_id,
                "content": chunk.content,
                "source": chunk.source,
                "metadata": chunk.metadata,
                "embedding": vec,
            })

    def search(self, query_vec: list[float], top_k: int = 3,
               filter_metadata: dict | None = None) -> list[tuple[Chunk, float]]:
        # 模拟 collection.search() + expr 过滤
        scored = []
        for ent in self._entities:
            if filter_metadata:
                if not all(ent["metadata"].get(k) == v for k, v in filter_metadata.items()):
                    continue
            score = sum(a * b for a, b in zip(query_vec, ent["embedding"]))
            chunk = Chunk(chunk_id=ent["id"], doc_id=ent["id"].rsplit("-", 1)[0],
                          source=ent["source"], content=ent["content"], metadata=ent["metadata"])
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete(self, chunk_ids: list[str]):
        ids_set = set(chunk_ids)
        self._entities = [e for e in self._entities if e["id"] not in ids_set]

    @property
    def count(self) -> int:
        return len(self._entities)


# ---------- 5. Qdrant 实现 ----------

class QdrantVectorStore(VectorStoreBase):
    """Qdrant 向量存储（Rust 实现，高性能，丰富过滤）。

    生产代码:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

        client = QdrantClient(host="localhost", port=6333)

        client.create_collection(
            collection_name="documents",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        # 插入
        client.upsert(
            collection_name="documents",
            points=[
                PointStruct(id=1, vector=[0.1, ...], payload={"content": "...", "source": "a.pdf"}),
            ]
        )

        # 检索（支持丰富的过滤条件）
        results = client.search(
            collection_name="documents",
            query_vector=[0.15, ...],
            limit=5,
            query_filter=Filter(must=[
                FieldCondition(key="source", match=MatchValue(value="a.pdf"))
            ])
        )

    特点: Rust高性能、丰富过滤(must/should/must_not)、payload索引、推荐系统
    pip install qdrant-client
    """

    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self._points: list[dict] = []
        self._next_id = 1

    def add_documents(self, chunks: list[Chunk], vectors: list[list[float]]):
        for chunk, vec in zip(chunks, vectors):
            self._points.append({
                "id": self._next_id,
                "chunk_id": chunk.chunk_id,
                "vector": vec,
                "payload": {
                    "content": chunk.content,
                    "source": chunk.source,
                    **chunk.metadata,
                },
            })
            self._next_id += 1

    def search(self, query_vec: list[float], top_k: int = 3,
               filter_metadata: dict | None = None) -> list[tuple[Chunk, float]]:
        scored = []
        for pt in self._points:
            if filter_metadata:
                if not all(pt["payload"].get(k) == v for k, v in filter_metadata.items()):
                    continue
            score = sum(a * b for a, b in zip(query_vec, pt["vector"]))
            chunk = Chunk(
                chunk_id=pt["chunk_id"],
                doc_id=pt["chunk_id"].rsplit("-", 1)[0],
                source=pt["payload"]["source"],
                content=pt["payload"]["content"],
            )
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete(self, chunk_ids: list[str]):
        ids_set = set(chunk_ids)
        self._points = [p for p in self._points if p["chunk_id"] not in ids_set]

    @property
    def count(self) -> int:
        return len(self._points)


# ---------- 6. 混合检索 (BM25 + 向量) ----------

class BM25Index:
    """BM25 关键词检索索引（模拟 Elasticsearch / Weaviate 的 BM25）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[tuple[Chunk, list[str]]] = []
        self._avg_dl = 0.0
        self._df: dict[str, int] = {}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = []
        for ch in text.lower():
            if '\u4e00' <= ch <= '\u9fff':
                tokens.append(ch)
            elif ch.isalnum():
                if tokens and tokens[-1][-1].isascii() and ch.isascii():
                    tokens[-1] += ch
                else:
                    tokens.append(ch)
        return tokens

    def add(self, chunks: list[Chunk]):
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            self._docs.append((chunk, tokens))
            for t in set(tokens):
                self._df[t] = self._df.get(t, 0) + 1
        total = sum(len(toks) for _, toks in self._docs)
        self._avg_dl = total / len(self._docs) if self._docs else 1.0

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        q_tokens = self._tokenize(query)
        n = len(self._docs)
        scored = []
        for chunk, doc_tokens in self._docs:
            dl = len(doc_tokens)
            tf_map = Counter(doc_tokens)
            score = 0.0
            for qt in q_tokens:
                if qt not in self._df:
                    continue
                df = self._df[qt]
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                tf = tf_map.get(qt, 0)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl))
                score += idf * tf_norm
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class HybridRetriever:
    """混合检索 — BM25 关键词 + 向量语义检索 + 分数融合。

    Reciprocal Rank Fusion (RRF) 将两路结果合并排序。
    生产中 Weaviate / Qdrant / Elasticsearch 原生支持混合检索。
    """

    def __init__(self, vector_store: VectorStoreBase, bm25_index: BM25Index,
                 embedding: SimpleEmbedding, alpha: float = 0.5):
        self.vector_store = vector_store
        self.bm25 = bm25_index
        self.embedding = embedding
        self.alpha = alpha  # 向量权重, 1-alpha = BM25 权重

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """混合检索 + RRF 融合。"""
        # 向量检索
        q_vec = self.embedding.embed(query)
        vec_results = self.vector_store.search(q_vec, top_k=top_k * 2)

        # BM25 检索
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        # RRF 融合
        k = 60  # RRF 常数
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, (chunk, _) in enumerate(vec_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + self.alpha / (k + rank + 1)
            chunk_map[chunk.chunk_id] = chunk

        for rank, (chunk, _) in enumerate(bm25_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + (1 - self.alpha) / (k + rank + 1)
            chunk_map[chunk.chunk_id] = chunk

        fused = [(chunk_map[cid], score) for cid, score in rrf_scores.items()]
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:top_k]


# ═══════════════════════════════════════════════════════════
# Step 4 & 5: 检索 + 生成（Retrieval + Generation）
# ═══════════════════════════════════════════════════════════

class RAGPipeline:
    """完整的 RAG 流水线。"""

    def __init__(self, embedding: SimpleEmbedding, store: SimpleVectorStore):
        self.embedding = embedding
        self.store = store

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """检索相关文档片段。"""
        query_vec = self.embedding.embed(query)
        return self.store.search(query_vec, top_k)

    def build_prompt(self, query: str, retrieved: list[tuple[Chunk, float]]) -> str:
        """构建 RAG prompt（检索结果注入上下文）。"""
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved):
            context_parts.append(
                f"[来源: {chunk.source} | 相关度: {score:.2f}]\n{chunk.content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        return f"""基于以下参考资料回答用户的问题。如果参考资料中没有相关信息，请明确说明。

## 参考资料

{context}

## 用户问题

{query}

## 回答要求
- 仅基于参考资料回答，不要编造
- 引用来源文档
- 用中文回答"""

    def query(self, question: str, top_k: int = 3) -> dict:
        """执行 RAG 查询（检索 + 构建 prompt）。"""
        retrieved = self.retrieve(question, top_k)
        prompt = self.build_prompt(question, retrieved)
        return {
            "question": question,
            "retrieved_chunks": [(c.chunk_id, c.source, score) for c, score in retrieved],
            "prompt": prompt,
            "prompt_tokens_estimate": int(len(prompt) * 1.5),
        }


# ═══════════════════════════════════════════════════════════
# 向量数据库选型
# ═══════════════════════════════════════════════════════════

def show_vector_db_selection():
    """展示向量数据库深度选型指南。"""
    print(f"\n\n▶ 向量数据库选型指南")
    print("─" * 60)

    print(f"""
  选型维度对比:
  ────────────┬──────────┬──────────┬──────────┬──────────┬──────────
  数据库       │ 部署方式  │ 数据规模  │ 混合检索  │ 多租户   │ 开源
  ────────────┼──────────┼──────────┼──────────┼──────────┼──────────
  FAISS        │ 库/嵌入式 │ 十亿级   │ ❌        │ ❌       │ ✅ MIT
  Chroma       │ 嵌入式/CS │ 百万级   │ ❌        │ ❌       │ ✅ Apache
  pgvector     │ PG 扩展   │ 千万级   │ ✅ SQL    │ ✅ RLS   │ ✅
  Weaviate     │ 独立服务  │ 亿级     │ ✅ BM25F  │ ✅       │ ✅ BSD
  Milvus       │ 分布式    │ 百亿级   │ ✅ Sparse │ ✅       │ ✅ Apache
  Qdrant       │ 独立服务  │ 亿级     │ ✅ Sparse │ ✅       │ ✅ Apache
  Pinecone     │ 全托管    │ 十亿级   │ ✅ Sparse │ ✅ NS    │ ❌ 商业
  Elasticsearch│ 独立集群  │ 百亿级   │ ✅ 原生   │ ✅       │ ⚠️ SSPL

  按场景选型决策树:
  ────────────────────────────────────────────────────────
  你的场景？
  │
  ├─ 快速原型 / 单文件嵌入 ────────→ Chroma（pip install 即用）
  ├─ 单机高性能 / 离线检索 ────────→ FAISS（Meta，最快）
  ├─ 已有 PostgreSQL ─────────────→ pgvector（零额外运维）
  ├─ 需要混合检索（向量+关键词）──→ Weaviate / Qdrant
  ├─ 大规模分布式（>1亿向量）────→ Milvus / Elasticsearch
  ├─ 不想自运维 ──────────────────→ Pinecone（全托管）
  └─ 已有 ES 集群 ────────────────→ Elasticsearch + kNN

  生产配置示例:
  ────────────────────────────────────────────────────────""")

    print("  pgvector (最常见 — 复用已有 PostgreSQL):")
    print("    CREATE EXTENSION vector;")
    print("    CREATE TABLE documents (")
    print("      id       SERIAL PRIMARY KEY,")
    print("      content  TEXT,")
    print("      metadata JSONB,")
    print("      embedding VECTOR(1536)  -- OpenAI 维度")
    print("    );")
    print("    CREATE INDEX ON documents")
    print("      USING ivfflat (embedding vector_cosine_ops)")
    print("      WITH (lists = 100);")
    print()
    print("  Chroma (最简单 — 三行代码):")
    print("    import chromadb")
    print("    client = chromadb.Client()")
    print("    col = client.create_collection('docs')")
    print("    col.add(ids=['d1'], documents=['...'], embeddings=[[0.1, ...]])")
    print("    results = col.query(query_embeddings=[[0.2, ...]], n_results=3)")
    print()
    print("  FAISS (单机最快 — 无服务器):")
    print("    import faiss")
    print("    index = faiss.IndexFlatIP(1536)     # 内积（余弦）")
    print("    index.add(np.array(embeddings))")
    print("    D, I = index.search(query_vec, k=5) # 返回距离和索引")

    print(f"""
  索引类型与性能:
  ──────────────┬───────────────┬──────────────┬───────────────
  索引类型       │ 检索速度       │ 构建速度      │ 内存占用
  ──────────────┼───────────────┼──────────────┼───────────────
  Flat (暴力)    │ ⭐ 精确但慢    │ ⭐⭐⭐ 无需构建 │ ⭐ 全量存储
  IVF            │ ⭐⭐ 快         │ ⭐⭐ 需训练     │ ⭐⭐ 聚类索引
  HNSW           │ ⭐⭐⭐ 最快     │ ⭐ 慢          │ ⭐ 图索引
  PQ (量化)      │ ⭐⭐ 快         │ ⭐ 慢          │ ⭐⭐⭐ 压缩

  嵌入模型选型:
  ──────────────────┬───────────┬───────────┬───────────────
  模型               │ 维度       │ 中文支持   │ 适用场景
  ──────────────────┼───────────┼───────────┼───────────────
  text-embedding-3-s │ 1536      │ ✅         │ 通用（OpenAI）
  text-embedding-3-l │ 3072      │ ✅         │ 高精度
  BGE-M3             │ 1024      │ ✅ 原生    │ 多语言 + 开源
  Jina v3            │ 1024      │ ✅         │ 长文本 8K
  E5-mistral-7b      │ 4096      │ ⚠️        │ 最高精度""")


# ═══════════════════════════════════════════════════════════
# RAG 质量评测
# ═══════════════════════════════════════════════════════════

@dataclass
class RAGEvalCase:
    """RAG 评测用例。"""
    question: str
    relevant_doc_ids: list[str]  # 标注：哪些文档是正确答案来源
    expected_keywords: list[str]  # 回答中应包含的关键信息


# RAG 评测数据集（标注了每个问题的正确来源文档）
RAG_EVAL_DATASET = [
    RAGEvalCase(
        question="SmartFlow 系统要求是什么？",
        relevant_doc_ids=["doc-001"],
        expected_keywords=["Java 17", "MySQL 8.0", "Redis 6.0"],
    ),
    RAGEvalCase(
        question="如何用 Docker 部署？",
        relevant_doc_ids=["doc-004"],
        expected_keywords=["docker pull", "8080", "application.yml"],
    ),
    RAGEvalCase(
        question="API 限流是多少？怎么认证？",
        relevant_doc_ids=["doc-002"],
        expected_keywords=["Bearer Token", "100 次", "10000 次"],
    ),
    RAGEvalCase(
        question="如何重置管理员密码？",
        relevant_doc_ids=["doc-003"],
        expected_keywords=["reset-password", "admin"],
    ),
    RAGEvalCase(
        question="最新版本有什么新功能？",
        relevant_doc_ids=["doc-005"],
        expected_keywords=["子流程", "WebSocket", "v3.2.0"],
    ),
]


class RAGEvaluator:
    """RAG 质量评测器。

    Ragas 四大核心指标（RAG 评测的行业标准）：
    ─────────────────────────────────────────────────────────
    1. Faithfulness（忠实度）
       回答是否忠实于检索到的上下文？有没有编造内容？
       = 回答中的每个声明都能在 context 中找到依据的比例
       → 衡量：幻觉程度

    2. Answer Relevancy（答案相关性）
       回答是否切中用户的问题？有没有答非所问？
       = 从回答反向生成问题，与原问题的语义相似度
       → 衡量：答案质量

    3. Context Precision（上下文精确度）
       检索到的上下文中，排名靠前的是否都是有用的？
       = 相关上下文在结果列表中的加权排名
       → 衡量：检索排序质量（有用的排前面）

    4. Context Recall（上下文召回率）
       回答所需的全部信息是否都被检索到了？
       = ground truth 中每个声明能在 context 中找到依据的比例
       → 衡量：检索完整性

    本示例用关键词匹配模拟以上指标。
    生产中使用 Ragas / DeepEval 通过 LLM-as-Judge 精确计算。

    辅助检索指标：MRR / Hit Rate
    """

    # ── Ragas 四大指标（关键词模拟） ──────────────────────

    @staticmethod
    def faithfulness(context: str, expected_keywords: list[str]) -> float:
        """Faithfulness（忠实度）：上下文中包含多少回答所需的关键信息。

        真实实现：LLM 将回答拆成声明列表，逐条检查是否有 context 依据。
        模拟实现：检查期望关键词在 context 中的覆盖率。
        """
        if not expected_keywords:
            return 1.0
        context_lower = context.lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in context_lower)
        return hits / len(expected_keywords)

    @staticmethod
    def answer_relevancy(question: str, expected_keywords: list[str], context: str) -> float:
        """Answer Relevancy（答案相关性）：上下文是否与问题直接相关。

        真实实现：从回答反向生成问题，与原问题计算嵌入相似度。
        模拟实现：检查问题关键词在检索上下文中的出现率。
        """
        q_tokens = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', question.lower()))
        if not q_tokens:
            return 1.0
        context_lower = context.lower()
        hits = sum(1 for t in q_tokens if t in context_lower)
        return hits / len(q_tokens)

    @staticmethod
    def context_precision(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
        """Context Precision（上下文精确度）：相关文档是否排在前面。

        真实实现：LLM 判断每个 context chunk 是否对回答有用，计算加权精确度。
        模拟实现：相关文档在结果列表中的加权排名（排名越靠前得分越高）。
        """
        if not retrieved_doc_ids:
            return 0.0
        precision_sum = 0.0
        relevant_count = 0
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in relevant_doc_ids:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)  # precision@i
        if relevant_count == 0:
            return 0.0
        return precision_sum / relevant_count

    @staticmethod
    def context_recall(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
        """Context Recall（上下文召回率）：正确文档有多少被检索到。

        真实实现：LLM 将 ground truth 拆成声明，逐条检查是否能从 context 中推导。
        模拟实现：标注的正确文档在检索结果中的召回比例。
        """
        if not relevant_doc_ids:
            return 1.0
        hits = sum(1 for doc_id in relevant_doc_ids if doc_id in retrieved_doc_ids)
        return hits / len(relevant_doc_ids)

    # ── 辅助检索指标 ────────────────────────────────────

    @staticmethod
    def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
        """MRR：第一个正确结果排在第几位（取倒数）。"""
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in relevant_doc_ids:
                return 1.0 / (i + 1)
        return 0.0


def run_rag_evaluation(rag: RAGPipeline, top_k: int = 3):
    """运行 RAG 质量评测。"""
    print(f"\n\n▶ RAG 质量评测 (Ragas 四大指标 + 检索指标)")
    print("─" * 60)

    # 先展示 Ragas 四大指标框架
    print(f"""
  Ragas 四大核心指标 — RAG 评测的行业标准:
  ─────────────────────────────────────────────────────────
  ┌──────────────────┬───────────────────────────────────┐
  │ 指标              │ 衡量什么                          │
  ├──────────────────┼───────────────────────────────────┤
  │ Faithfulness     │ 回答是否忠实于上下文？（幻觉检测） │
  │ Answer Relevancy │ 回答是否切中问题？（答非所问检测）│
  │ Context Precision│ 检索排序是否合理？（有用的排前面）│
  │ Context Recall   │ 该检索的是否都检索到了？（完整性）│
  └──────────────────┴───────────────────────────────────┘

                       ┌───────────────────┐
                       │    User Question  │
                       └────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │   Retrieved Context  │
                    │   ┌─────────────────┐│
                    │   │Context Precision││  检索到的是否有用？
                    │   │Context Recall   ││  该检索的是否齐全？
                    │   └─────────────────┘│
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │   Generated Answer   │
                    │   ┌─────────────────┐│
                    │   │Faithfulness     ││  回答是否有据可依？
                    │   │Answer Relevancy ││  回答是否切题？
                    │   └─────────────────┘│
                    └──────────────────────┘""")

    evaluator = RAGEvaluator()

    all_faithfulness = []
    all_answer_rel = []
    all_ctx_precision = []
    all_ctx_recall = []
    all_rr = []
    hit_count = 0

    for case in RAG_EVAL_DATASET:
        result = rag.query(case.question, top_k=top_k)

        # 提取检索到的 doc_id（从 chunk_id 还原）
        retrieved_doc_ids = []
        for chunk_id, source, score in result["retrieved_chunks"]:
            doc_id = "-".join(chunk_id.split("-")[:2])  # "doc-001-0" → "doc-001"
            if doc_id not in retrieved_doc_ids:
                retrieved_doc_ids.append(doc_id)

        # Ragas 四大指标
        faith = evaluator.faithfulness(result["prompt"], case.expected_keywords)
        ans_rel = evaluator.answer_relevancy(case.question, case.expected_keywords, result["prompt"])
        ctx_prec = evaluator.context_precision(retrieved_doc_ids, case.relevant_doc_ids)
        ctx_rec = evaluator.context_recall(retrieved_doc_ids, case.relevant_doc_ids)

        # 辅助检索指标
        rr = evaluator.reciprocal_rank(retrieved_doc_ids, case.relevant_doc_ids)
        hit = 1 if ctx_rec > 0 else 0

        all_faithfulness.append(faith)
        all_answer_rel.append(ans_rel)
        all_ctx_precision.append(ctx_prec)
        all_ctx_recall.append(ctx_rec)
        all_rr.append(rr)
        hit_count += hit

        icon = "✅" if ctx_rec == 1.0 and faith >= 0.6 else ("⚠️" if ctx_rec > 0 else "❌")
        print(f"  {icon} Q: {case.question}")
        print(f"     来源: {case.relevant_doc_ids} → 检索: {retrieved_doc_ids}")
        print(f"     Faithfulness: {faith:.2f}  "
              f"AnswerRel: {ans_rel:.2f}  "
              f"CtxPrec: {ctx_prec:.2f}  "
              f"CtxRecall: {ctx_rec:.2f}")
        if faith < 1.0:
            context_lower = result["prompt"].lower()
            missing = [kw for kw in case.expected_keywords if kw.lower() not in context_lower]
            if missing:
                print(f"     ⚠️ 上下文缺少关键信息: {missing}")

    # 汇总
    n = len(RAG_EVAL_DATASET)
    print(f"\n  {'═' * 56}")
    print(f"  RAG 质量评测汇总 (top_k={top_k}, {n} 个用例):")
    print(f"  ─────────────────────────────────────────")

    metrics = [
        ("Faithfulness", sum(all_faithfulness) / n),
        ("Answer Relevancy", sum(all_answer_rel) / n),
        ("Context Precision", sum(all_ctx_precision) / n),
        ("Context Recall", sum(all_ctx_recall) / n),
        ("MRR", sum(all_rr) / n),
        ("Hit Rate", hit_count / n),
    ]

    for name, value in metrics:
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        status = "✅" if value >= 0.8 else ("⚠️" if value >= 0.5 else "❌")
        print(f"  {status} {name:20s} {bar} {value:.0%}")

    print(f"""
  Ragas 四大指标详解:
  ─────────────────────────────────────────────────────────
  Faithfulness      │ 回答中的声明能否在 context 中找到依据
                    │ 低 → 模型在编造（幻觉）
                    │ 生产计算：LLM 拆分回答为声明 → 逐条验证
  ──────────────────┼─────────────────────────────────────
  Answer Relevancy  │ 回答是否切中用户的问题
                    │ 低 → 答非所问、跑题
                    │ 生产计算：从回答反向生成问题 → 与原问题算相似度
  ──────────────────┼─────────────────────────────────────
  Context Precision │ 检索结果中有用的是否排在前面
                    │ 低 → 检索排序差，噪声多
                    │ 生产计算：LLM 判断每条 context 是否有用 → 加权精确度
  ──────────────────┼─────────────────────────────────────
  Context Recall    │ 回答所需的信息是否都被检索到了
                    │ 低 → 检索不全，遗漏关键信息
                    │ 生产计算：LLM 拆分 ground truth → 逐条检查 context 覆盖

  生产评测工具对比:
  ──────────────┬──────────────────────────────────────
  Ragas          │ RAG 标准：四大指标 + LLM-as-Judge
  DeepEval       │ 通用 LLM：hallucination / bias / toxicity
  LangSmith      │ 在线评测 + 数据集管理 + 人工标注
  TruLens        │ 三层评估：input / retrieval / output
  Phoenix        │ 可视化追踪 + 嵌入空间分析""")


# ═══════════════════════════════════════════════════════════
# 主流向量数据库对比实测
# ═══════════════════════════════════════════════════════════

def demo_vector_db_implementations(all_chunks: list[Chunk], embedding: SimpleEmbedding):
    """用同一份数据 + 同一个查询，对比 5 大向量数据库的检索结果。"""
    print(f"\n\n▶ 主流向量数据库对比实测")
    print("─" * 60)

    # 预计算所有向量
    vectors = [embedding.embed(c.content) for c in all_chunks]
    dim = len(vectors[0])
    query = "如何用 Docker 部署？"
    q_vec = embedding.embed(query)

    stores: list[tuple[str, VectorStoreBase]] = [
        ("FAISS", FAISSVectorStore(dim=dim)),
        ("Chroma", ChromaVectorStore("rag_demo")),
        ("pgvector", PgVectorStore("documents")),
        ("Milvus", MilvusVectorStore("documents", dim=dim)),
        ("Qdrant", QdrantVectorStore("documents")),
    ]

    print(f"  查询: '{query}'")
    print(f"  数据: {len(all_chunks)} 个片段, {dim} 维向量")
    print()

    for name, store in stores:
        import time as _t
        t0 = _t.time()
        store.add_documents(all_chunks, vectors)
        insert_ms = (_t.time() - t0) * 1000

        t0 = _t.time()
        results = store.search(q_vec, top_k=3)
        search_ms = (_t.time() - t0) * 1000

        print(f"  [{name}] 插入 {store.count} 条 ({insert_ms:.1f}ms) → 检索 ({search_ms:.1f}ms):")
        for chunk, score in results:
            preview = chunk.content[:50].replace("\n", " ")
            print(f"    [{score:.3f}] {chunk.source}: {preview}...")

        # 演示删除
        store.delete([all_chunks[0].chunk_id])
        print(f"    删除后: {store.count} 条")
        print()

    # 演示 pgvector 的 SQL 能力
    pg = stores[2][1]
    print(f"  pgvector SQL 能力演示:")
    print(f"    {pg.execute_sql('SELECT * FROM documents WHERE source = %s ORDER BY embedding <=> %s LIMIT 3')}")

    # 演示元数据过滤（Chroma）
    chroma = stores[1][1]
    filtered = chroma.search(q_vec, top_k=2, filter_metadata={"source": "部署指南.md"})
    print(f"\n  Chroma 元数据过滤 (source='部署指南.md'):")
    for chunk, score in filtered:
        preview = chunk.content[:50].replace("\n", " ")
        print(f"    [{score:.3f}] {chunk.source}: {preview}...")

    print(f"""
  对比总结:
  ──────────────┬──────────────────────────────────────────
  FAISS          │ 最快(纯内存), 无元数据过滤, 适合离线批量
  Chroma         │ 最易用(pip install), where过滤, 适合原型
  pgvector       │ 复用PG, SQL+向量, 事务一致, 适合已有PG的团队
  Milvus         │ 分布式, 百亿级, GPU索引, 适合大规模生产
  Qdrant         │ Rust高性能, 丰富过滤, 适合推荐+搜索混合场景""")


def demo_hybrid_retrieval(all_chunks: list[Chunk], embedding: SimpleEmbedding):
    """演示 BM25 + 向量混合检索 + RRF 融合。"""
    print(f"\n\n▶ 混合检索 — BM25 + 向量 + RRF 融合")
    print("─" * 60)

    # 构建向量库 + BM25 索引
    vectors = [embedding.embed(c.content) for c in all_chunks]
    dim = len(vectors[0])

    vec_store = FAISSVectorStore(dim=dim)
    vec_store.add_documents(all_chunks, vectors)

    bm25 = BM25Index()
    bm25.add(all_chunks)

    hybrid = HybridRetriever(vec_store, bm25, embedding, alpha=0.5)

    queries = [
        "docker pull smartflow",       # 精确关键词 → BM25 强
        "如何部署系统到生产环境",       # 语义 → 向量强
        "API 认证 Bearer Token 限流",  # 混合 → 两路互补
    ]

    for query in queries:
        print(f"\n  查询: '{query}'")

        # 纯向量
        q_vec = embedding.embed(query)
        vec_results = vec_store.search(q_vec, top_k=3)
        print(f"    向量检索 Top-1: [{vec_results[0][1]:.3f}] "
              f"{vec_results[0][0].source}: {vec_results[0][0].content[:40]}...")

        # 纯 BM25
        bm25_results = bm25.search(query, top_k=3)
        print(f"    BM25 检索 Top-1: [{bm25_results[0][1]:.3f}] "
              f"{bm25_results[0][0].source}: {bm25_results[0][0].content[:40]}...")

        # 混合 RRF
        hybrid_results = hybrid.search(query, top_k=3)
        print(f"    混合 RRF  Top-1: [{hybrid_results[0][1]:.4f}] "
              f"{hybrid_results[0][0].source}: {hybrid_results[0][0].content[:40]}...")

    print(f"""
  混合检索原理:
  ─────────────────────────────────────────────────────────
  向量检索 → 语义相近的结果（理解意思，但可能漏掉精确词）
  BM25 检索 → 关键词匹配的结果（精确匹配，但不理解同义词）
  RRF 融合  → Reciprocal Rank Fusion: 1/(k+rank) 分数相加

  公式: RRF_score = α/(k+rank_vec) + (1-α)/(k+rank_bm25)
    k=60 (常数), α=0.5 (向量与BM25各占一半)

  生产方案:
  - Weaviate: 原生 hybrid search (BM25F + 向量)
  - Qdrant: sparse + dense 向量
  - Elasticsearch: kNN + BM25
  - LangChain: EnsembleRetriever""")


# ═══════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════

def main():
    print("=== RAG 完整流程 ===\n")

    # ── Step 1: 文档加载 ─────────────────────────────────
    print("▶ Step 1: 文档加载 (Document Loading)")
    print("─" * 60)
    print(f"  加载了 {len(RAW_DOCUMENTS)} 篇文档:")
    for doc in RAW_DOCUMENTS:
        print(f"    📄 {doc['source']} ({len(doc['content'])} 字)")

    # ── Step 2: 文档分片 ─────────────────────────────────
    print(f"\n\n▶ Step 2: 文档分片 (Chunking)")
    print("─" * 60)

    all_chunks = []
    for doc in RAW_DOCUMENTS:
        chunks = Chunker.by_paragraph(doc, max_chunk_size=200, overlap=30)
        all_chunks.extend(chunks)
        print(f"  📄 {doc['source']} → {len(chunks)} 个片段")

    print(f"\n  总计 {len(all_chunks)} 个片段")
    print(f"\n  片段示例:")
    for chunk in all_chunks[:3]:
        preview = chunk.content[:80].replace("\n", " ")
        print(f"    [{chunk.chunk_id}] {preview}...")

    # 展示分片策略对比
    print(f"\n  分片策略对比:")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  固定大小     │ 按字符数切 (512)    │ 最简单，可能切断句子")
    print(f"  段落分割     │ 按 \\n 切分          │ 语义完整，片段大小不均")
    print(f"  递归分割     │ 先按大→再按小分隔符  │ LangChain 默认，平衡")
    print(f"  滑动窗口     │ 固定大小+重叠        │ 保留上下文，有冗余")
    print(f"  语义分割     │ 按嵌入相似度断句      │ 效果最好，成本最高")

    # ── Step 3: 嵌入 + 存储 ──────────────────────────────
    print(f"\n\n▶ Step 3: 嵌入 + 向量存储 (Embedding + Vector Store)")
    print("─" * 60)

    # 构建嵌入模型
    embedding = SimpleEmbedding()
    embedding.fit([c.content for c in all_chunks])
    print(f"  词汇表大小: {embedding.dim}")

    # 存入向量数据库
    store = SimpleVectorStore()
    for chunk in all_chunks:
        vec = embedding.embed(chunk.content)
        store.add(chunk, vec)
    print(f"  已存入 {store.size} 个向量")

    # 展示向量存储对比
    print(f"\n  向量数据库对比:")
    print(f"  ─────────────┬───────────┬──────────┬──────────────")
    print(f"  数据库        │ 类型       │ 规模     │ 适用场景")
    print(f"  ─────────────┼───────────┼──────────┼──────────────")
    print(f"  FAISS         │ 库(Meta)   │ 十亿级   │ 单机最快")
    print(f"  Chroma        │ 嵌入式DB   │ 百万级   │ 原型开发")
    print(f"  Pinecone      │ 全托管     │ 十亿级   │ 免运维")
    print(f"  Weaviate      │ 开源DB     │ 亿级     │ 功能丰富")
    print(f"  pgvector      │ PG扩展     │ 千万级   │ 复用现有PG")
    print(f"  Milvus        │ 分布式     │ 百亿级   │ 大规模生产")

    # ── Step 4 & 5: 检索 + 生成 ──────────────────────────
    print(f"\n\n▶ Step 4 & 5: 检索 + 生成 (Retrieval + Generation)")
    print("─" * 60)

    rag = RAGPipeline(embedding, store)

    test_questions = [
        "SmartFlow 系统要求是什么？需要什么数据库？",
        "如何用 Docker 部署 SmartFlow？",
        "API 限流是多少？怎么认证？",
        "管理员密码忘了怎么办？",
        "最新版本有什么新功能？",
    ]

    for q in test_questions:
        print(f"\n  {'═' * 56}")
        print(f"  ❓ 问题: {q}")

        result = rag.query(q, top_k=3)

        print(f"  📎 检索到 {len(result['retrieved_chunks'])} 个相关片段:")
        for chunk_id, source, score in result["retrieved_chunks"]:
            print(f"     [{chunk_id}] {source} (相关度: {score:.3f})")

        print(f"  📝 Prompt 长度: ~{result['prompt_tokens_estimate']} tokens")

        # 展示生成的 prompt（截取前几行）
        prompt_preview = result["prompt"].split("\n")
        print(f"  📤 发送给 LLM 的 prompt（前 5 行）:")
        for line in prompt_preview[:5]:
            print(f"     {line}")
        print(f"     ...")

    # ── 向量数据库选型 ──────────────────────────────────
    show_vector_db_selection()

    # ── RAG 质量评测 ──────────────────────────────────────
    run_rag_evaluation(rag, top_k=3)

    # ── 主流向量数据库对比实测 ──────────────────────────
    demo_vector_db_implementations(all_chunks, embedding)

    # ── 混合检索 ──────────────────────────────────────────
    demo_hybrid_retrieval(all_chunks, embedding)

    # ── RAG 高级技巧 ─────────────────────────────────────
    print(f"\n\n▶ RAG 高级技巧")
    print("─" * 60)
    print(f"""
  基础 RAG                    │ 高级 RAG
  ───────────────────────────┼────────────────────────────
  单次检索                    │ 多轮检索（检索→追问→再检索）
  纯文本分片                  │ 语义分片 + 元数据过滤
  单一嵌入模型                │ 混合检索（向量 + 关键词 BM25）
  直接拼接上下文              │ Reranker 重排序（Cohere/BGE）
  无验证                      │ 忠实度检查（幻觉检测）
  固定 top_k                  │ 自适应检索（根据置信度决定）

  高级架构模式:
  ─────────────────────────────────────────────
  Naive RAG       │ 检索 → 生成（本示例）
  Advanced RAG    │ 预处理 → 检索 → 重排 → 生成
  Modular RAG     │ 可插拔的检索/重排/生成模块
  Graph RAG       │ 知识图谱 + 向量检索
  Agentic RAG     │ Agent 自主决定何时检索、检索什么
    """)

    # ── 架构总结 ──────────────────────────────────────────
    print("=" * 60)
    print("📊 RAG 完整流程总结:")
    print()
    print("  Step 1 加载  │ PDF/MD/DB/API → 原始文本")
    print("  Step 2 分片  │ 原始文本 → 200-500 字的片段")
    print("  Step 3 嵌入  │ 片段 → 向量 → 存入向量数据库")
    print("  Step 4 检索  │ 用户问题 → 向量 → 相似度搜索 → top_k 片段")
    print("  Step 5 生成  │ 问题 + 检索片段 → LLM → 基于证据的回答")
    print()
    print("  框架支持:")
    print("  - LangChain: 最成熟的 RAG 生态（200+ 加载器，20+ 向量库）")
    print("  - LlamaIndex: RAG 专用框架（索引优化最强）")
    print("  - Spring AI: QuestionAnswerAdvisor（Java 原生 RAG）")
    print("  - Dify: 可视化 RAG 编排（零代码）")
    print()
    print("  RAG 质量保障 Checklist:")
    print("  ────────────────────────────────────────────")
    print("  □ 标注评测集（问题 + 正确来源文档 + 期望关键词）")
    print("  □ Ragas 四大指标基线: Faithfulness ≥ 0.8, Context Recall ≥ 0.9")
    print("  □ 分片策略调优: chunk_size 和 overlap 对比实验")
    print("  □ 向量库索引选择: HNSW (速度) vs IVF (内存)")
    print("  □ 混合检索: 向量 + BM25 提升长尾查询召回")
    print("  □ Reranker 二次排序: BGE-reranker / Cohere")
    print("  □ 幻觉检测: Faithfulness 低于阈值时自动拒答")


if __name__ == "__main__":
    main()
