"""知识图谱存储 + 实体抽取 + 图遍历

核心概念:
  - 三元组: (主体, 关系, 客体)  e.g. (张三, 管理, 订单ORD-001)
  - 实体抽取: 从非结构化文本中提取实体和关系
  - 图遍历: BFS / 路径查找 / 子图抽取
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Entity:
    """知识图谱实体。"""
    id: str
    name: str
    type: str  # Person / Product / Order / Company / ...
    properties: dict = field(default_factory=dict)


@dataclass
class Relation:
    """知识图谱关系（三元组的"边"）。"""
    source_id: str
    relation: str
    target_id: str
    properties: dict = field(default_factory=dict)


class KnowledgeGraph:
    """纯 Python 知识图谱（生产用 Neo4j / Amazon Neptune）。"""

    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self._adj: dict[str, list[Relation]] = defaultdict(list)  # 邻接表
        self._rev: dict[str, list[Relation]] = defaultdict(list)  # 反向邻接

    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity

    def add_relation(self, rel: Relation):
        self.relations.append(rel)
        self._adj[rel.source_id].append(rel)
        self._rev[rel.target_id].append(rel)

    def get_entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def get_neighbors(self, entity_id: str, direction: str = "out") -> list[tuple[Relation, Entity]]:
        """获取邻居节点。direction: out / in / both。"""
        results = []
        if direction in ("out", "both"):
            for rel in self._adj.get(entity_id, []):
                target = self.entities.get(rel.target_id)
                if target:
                    results.append((rel, target))
        if direction in ("in", "both"):
            for rel in self._rev.get(entity_id, []):
                source = self.entities.get(rel.source_id)
                if source:
                    results.append((rel, source))
        return results

    def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> list[list[str]]:
        """BFS 查找两个实体之间的路径。"""
        if start_id not in self.entities or end_id not in self.entities:
            return []
        queue = [(start_id, [start_id])]
        visited = {start_id}
        paths = []
        while queue:
            current, path = queue.pop(0)
            if len(path) > max_depth + 1:
                break
            if current == end_id:
                paths.append(path)
                continue
            for rel, neighbor in self.get_neighbors(current, "both"):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor.id, path + [f"--{rel.relation}-->", neighbor.id]))
        return paths

    def subgraph(self, center_id: str, depth: int = 2) -> "KnowledgeGraph":
        """提取以某实体为中心的子图。"""
        sub = KnowledgeGraph()
        visited = set()
        queue = [(center_id, 0)]
        while queue:
            eid, d = queue.pop(0)
            if eid in visited or d > depth:
                continue
            visited.add(eid)
            entity = self.entities.get(eid)
            if entity:
                sub.add_entity(entity)
            for rel, neighbor in self.get_neighbors(eid, "both"):
                sub.add_entity(neighbor)
                sub.add_relation(rel)
                if neighbor.id not in visited and d + 1 <= depth:
                    queue.append((neighbor.id, d + 1))
        return sub

    def query_by_type(self, entity_type: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.type == entity_type]

    def query_by_relation(self, relation: str) -> list[Relation]:
        return [r for r in self.relations if r.relation == relation]

    @property
    def stats(self) -> dict:
        types = defaultdict(int)
        for e in self.entities.values():
            types[e.type] += 1
        rel_types = defaultdict(int)
        for r in self.relations:
            rel_types[r.relation] += 1
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "entity_types": dict(types),
            "relation_types": dict(rel_types),
        }


# ═══════════════════════════════════════════════════════════
# 实体抽取器（规则式 — 生产用 LLM）
# ═══════════════════════════════════════════════════════════

class EntityExtractor:
    """从文本中抽取实体和关系（规则 + 模式匹配）。

    生产方案: GPT-4 / Claude + 结构化输出 → (entity, relation, entity)
    """

    PERSON_PATTERN = re.compile(r'([张李王赵刘陈杨黄周吴][a-zA-Z\u4e00-\u9fff]{1,3})')
    ORDER_PATTERN = re.compile(r'(ORD[-\w]+)')
    AMOUNT_PATTERN = re.compile(r'[¥￥$]?([\d,]+(?:\.\d+)?)\s*(?:元|万|美元)?')

    RELATION_PATTERNS = [
        (re.compile(r'(\S+)\s*(?:负责|管理|处理)\s*(\S+)'), "manages"),
        (re.compile(r'(\S+)\s*(?:购买|下单|订购)\s*(\S+)'), "purchased"),
        (re.compile(r'(\S+)\s*(?:属于|隶属|归属)\s*(\S+)'), "belongs_to"),
        (re.compile(r'(\S+)\s*(?:包含|含有)\s*(\S+)'), "contains"),
    ]

    def extract(self, text: str) -> tuple[list[Entity], list[Relation]]:
        """从文本中抽取实体和关系。"""
        entities = {}
        relations = []

        # 抽取人名
        for m in self.PERSON_PATTERN.finditer(text):
            name = m.group(1)
            eid = f"person:{name}"
            entities[eid] = Entity(eid, name, "Person")

        # 抽取订单号
        for m in self.ORDER_PATTERN.finditer(text):
            order_id = m.group(1)
            eid = f"order:{order_id}"
            entities[eid] = Entity(eid, order_id, "Order")

        # 抽取关系
        for pattern, rel_type in self.RELATION_PATTERNS:
            for m in pattern.finditer(text):
                src_name, tgt_name = m.group(1), m.group(2)
                src_id = self._find_entity_id(entities, src_name)
                tgt_id = self._find_entity_id(entities, tgt_name)
                if src_id and tgt_id:
                    relations.append(Relation(src_id, rel_type, tgt_id))

        return list(entities.values()), relations

    @staticmethod
    def _find_entity_id(entities: dict, name: str) -> str | None:
        for eid, e in entities.items():
            if e.name in name or name in e.name:
                return eid
        return None
