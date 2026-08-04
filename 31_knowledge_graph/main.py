"""知识图谱 + Graph RAG — 演示入口"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from graph_store import KnowledgeGraph, Entity, Relation, EntityExtractor
    from graph_rag import GraphRAGRetriever
except ImportError:
    from .graph_store import KnowledgeGraph, Entity, Relation, EntityExtractor
    from .graph_rag import GraphRAGRetriever


def build_demo_graph() -> KnowledgeGraph:
    """构建演示用知识图谱。"""
    kg = KnowledgeGraph()

    # 实体
    entities = [
        Entity("p:zhangsan", "张三", "Person", {"role": "销售经理", "department": "销售部"}),
        Entity("p:lisi", "李四", "Person", {"role": "技术总监", "department": "技术部"}),
        Entity("p:wangwu", "王五", "Person", {"role": "CEO", "department": "管理层"}),
        Entity("c:acme", "ACME公司", "Company", {"industry": "科技", "size": "500人"}),
        Entity("c:beta", "Beta科技", "Company", {"industry": "互联网", "size": "200人"}),
        Entity("o:001", "ORD-001", "Order", {"amount": 45000, "status": "已完成"}),
        Entity("o:002", "ORD-002", "Order", {"amount": 128000, "status": "审核中"}),
        Entity("pr:server", "企业服务器X1", "Product", {"price": 15000, "category": "硬件"}),
        Entity("pr:license", "数据库许可证", "Product", {"price": 50000, "category": "软件"}),
        Entity("d:sales", "销售部", "Department", {"head": "张三"}),
        Entity("d:tech", "技术部", "Department", {"head": "李四"}),
    ]
    for e in entities:
        kg.add_entity(e)

    # 关系
    relations = [
        Relation("p:zhangsan", "works_at", "c:acme"),
        Relation("p:lisi", "works_at", "c:acme"),
        Relation("p:wangwu", "manages", "c:acme"),
        Relation("p:zhangsan", "manages", "d:sales"),
        Relation("p:lisi", "manages", "d:tech"),
        Relation("p:zhangsan", "created", "o:001"),
        Relation("p:lisi", "created", "o:002"),
        Relation("o:001", "contains", "pr:server"),
        Relation("o:002", "contains", "pr:license"),
        Relation("o:001", "belongs_to", "c:beta"),
        Relation("o:002", "belongs_to", "c:acme"),
        Relation("p:wangwu", "approves", "o:002"),
        Relation("d:sales", "part_of", "c:acme"),
        Relation("d:tech", "part_of", "c:acme"),
    ]
    for r in relations:
        kg.add_relation(r)

    return kg


def demo_graph_basics():
    print("▶ 1. 知识图谱基础操作")
    print("─" * 60)

    kg = build_demo_graph()
    stats = kg.stats
    print(f"  图谱统计: {stats['entities']} 实体, {stats['relations']} 关系")
    print(f"  实体类型: {stats['entity_types']}")
    print(f"  关系类型: {stats['relation_types']}")

    # 邻居查询
    print(f"\n  张三的关系:")
    for rel, neighbor in kg.get_neighbors("p:zhangsan", "both"):
        print(f"    --[{rel.relation}]--> {neighbor.name} ({neighbor.type})")

    # 路径查找
    print(f"\n  张三 → ORD-002 的路径:")
    paths = kg.find_path("p:zhangsan", "o:002")
    for p in paths:
        print(f"    {' '.join(p)}")

    # 子图提取
    sub = kg.subgraph("o:001", depth=2)
    print(f"\n  ORD-001 的 2 跳子图: {sub.stats['entities']} 实体, {sub.stats['relations']} 关系")

    return kg


def demo_entity_extraction():
    print(f"\n\n▶ 2. 实体抽取")
    print("─" * 60)
    extractor = EntityExtractor()

    texts = [
        "张三负责管理销售部的 ORD-001 订单",
        "李四购买了一台服务器，王五负责审批",
        "ACME公司属于科技行业，包含销售部和技术部",
    ]

    for text in texts:
        entities, relations = extractor.extract(text)
        print(f"\n  文本: {text}")
        print(f"    实体: {[(e.name, e.type) for e in entities]}")
        print(f"    关系: {[(r.source_id, r.relation, r.target_id) for r in relations]}")


def demo_graph_rag(kg: KnowledgeGraph):
    print(f"\n\n▶ 3. Graph RAG 检索")
    print("─" * 60)
    retriever = GraphRAGRetriever(kg)

    queries = [
        "张三负责哪些订单？",
        "ORD-002 包含什么产品？谁创建的？",
        "ACME公司有哪些部门和员工？",
        "张三和李四之间有什么关系？",
    ]

    for q in queries:
        result = retriever.retrieve(q, depth=2)
        print(f"\n  Q: {q}")
        print(f"    识别实体: {result.entities_found}")
        print(f"    子图: {result.subgraph_entities} 实体, {result.subgraph_relations} 关系")
        if result.path_info:
            print(f"    路径: {result.path_info[0]}")
        # 显示上下文前 3 行
        ctx_lines = result.context.split("\n")[:4]
        for line in ctx_lines:
            print(f"    {line}")
        if len(result.context.split("\n")) > 4:
            print(f"    ...")


def demo_comparison():
    print(f"\n\n▶ 4. Graph RAG vs Vector RAG 对比")
    print("─" * 60)
    print(f"""
  维度            │ Vector RAG              │ Graph RAG
  ───────────────┼────────────────────────┼────────────────────────
  检索方式         │ 语义相似度搜索           │ 实体识别 + 图遍历
  多跳推理         │ ❌ 只能检索相似片段       │ ✅ 自然支持 N 跳关系
  关系感知         │ ❌ 丢失结构信息          │ ✅ 保留完整关系
  全局视图         │ ❌ 局部片段             │ ✅ 社区摘要/全图聚合
  构建成本         │ ⭐ 低（嵌入即可）        │ ⭐⭐⭐ 高（实体+关系抽取）
  维护成本         │ ⭐ 低（增量添加）        │ ⭐⭐ 中（图谱更新）
  适用场景         │ 文档问答 / 知识库       │ 多跳推理 / 关系分析
  生产方案         │ FAISS/Chroma/pgvector  │ Neo4j / Amazon Neptune

  最佳实践: 混合使用
  ─────────────────────────────────────────────────────
  1. Vector RAG 做初筛 → 召回相关文档
  2. Graph RAG 做精排 → 关系增强 + 多跳补全
  3. 两路结果融合 → 注入 LLM 上下文

  生产技术栈:
  - Microsoft GraphRAG: 社区检测 + 分层摘要
  - LlamaIndex PropertyGraph: 属性图 + 向量混合
  - Neo4j + LangChain: Cypher 查询生成
  - LightRAG: 轻量级 Graph RAG""")


def main():
    print("=== 知识图谱 + Graph RAG ===\n")
    kg = demo_graph_basics()
    demo_entity_extraction()
    demo_graph_rag(kg)
    demo_comparison()


if __name__ == "__main__":
    main()
