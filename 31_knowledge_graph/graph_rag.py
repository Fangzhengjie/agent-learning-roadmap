"""Graph RAG — 知识图谱增强的检索生成

Graph RAG vs Vector RAG:
  Vector RAG: 用户问题 → 向量化 → 相似度搜索 → 片段注入
  Graph RAG:  用户问题 → 实体识别 → 图遍历/子图提取 → 结构化上下文注入

Graph RAG 优势:
  1. 多跳推理: "张三的经理的部门有多少人" → 需要 2 跳图遍历
  2. 关系感知: 向量搜索找不到隐含关系
  3. 全局摘要: 对整个图谱做社区检测 + 摘要（Microsoft GraphRAG）
"""

import re
from dataclasses import dataclass, field

try:
    from .graph_store import KnowledgeGraph, Entity, Relation
except ImportError:
    from graph_store import KnowledgeGraph, Entity, Relation


@dataclass
class GraphRAGResult:
    """Graph RAG 检索结果。"""
    query: str
    entities_found: list[str]
    subgraph_entities: int
    subgraph_relations: int
    context: str
    path_info: list[str] = field(default_factory=list)


class GraphRAGRetriever:
    """Graph RAG 检索器。"""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def retrieve(self, query: str, depth: int = 2) -> GraphRAGResult:
        """Graph RAG 检索流程:
        1. 从 query 中识别实体
        2. 在图中查找匹配实体
        3. 提取子图 (depth 跳)
        4. 将子图转成自然语言上下文
        """
        # Step 1: 从 query 中提取实体名
        query_entities = self._extract_query_entities(query)

        # Step 2: 在图中匹配
        matched_ids = []
        for name in query_entities:
            for eid, entity in self.kg.entities.items():
                if name in entity.name or entity.name in name:
                    matched_ids.append(eid)

        # Step 3: 提取子图
        merged_sub = KnowledgeGraph()
        for eid in matched_ids:
            sub = self.kg.subgraph(eid, depth=depth)
            for e in sub.entities.values():
                merged_sub.add_entity(e)
            for r in sub.relations:
                merged_sub.add_relation(r)

        # Step 4: 转成上下文
        context = self._subgraph_to_context(merged_sub)

        # 路径信息
        paths = []
        if len(matched_ids) >= 2:
            for i in range(len(matched_ids) - 1):
                found = self.kg.find_path(matched_ids[i], matched_ids[i + 1])
                for p in found:
                    paths.append(" → ".join(p))

        return GraphRAGResult(
            query=query,
            entities_found=query_entities,
            subgraph_entities=len(merged_sub.entities),
            subgraph_relations=len(merged_sub.relations),
            context=context,
            path_info=paths,
        )

    def _extract_query_entities(self, query: str) -> list[str]:
        """从查询中提取可能的实体名（简单规则）。"""
        entities = []
        # 中文人名
        for m in re.finditer(r'([张李王赵刘陈杨黄周吴][a-zA-Z\u4e00-\u9fff]{1,3})', query):
            entities.append(m.group(1))
        # 订单号
        for m in re.finditer(r'(ORD[-\w]+)', query):
            entities.append(m.group(1))
        # 如果没有匹配到，用关键词做模糊匹配
        if not entities:
            for word in query.replace("？", "").replace("?", "").split():
                if len(word) >= 2:
                    entities.append(word)
        return entities

    def _subgraph_to_context(self, sub: KnowledgeGraph) -> str:
        """将子图转成自然语言上下文（注入给 LLM）。"""
        if not sub.entities:
            return "未在知识图谱中找到相关信息。"
        lines = ["以下是从知识图谱中检索到的相关信息：\n"]

        # 实体描述
        for e in sub.entities.values():
            props = ", ".join(f"{k}={v}" for k, v in e.properties.items())
            line = f"- {e.type} [{e.name}]"
            if props:
                line += f" ({props})"
            lines.append(line)

        # 关系描述
        if sub.relations:
            lines.append("\n关系：")
            seen = set()
            for r in sub.relations:
                src = sub.entities.get(r.source_id)
                tgt = sub.entities.get(r.target_id)
                if src and tgt:
                    key = f"{src.name}-{r.relation}-{tgt.name}"
                    if key not in seen:
                        seen.add(key)
                        lines.append(f"- {src.name} --[{r.relation}]--> {tgt.name}")

        return "\n".join(lines)
