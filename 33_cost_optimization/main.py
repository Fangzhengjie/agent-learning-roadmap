"""LLM 成本优化 — 演示入口"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from semantic_cache import ExactCache, SemanticCache
    from prompt_compress import PromptCompressor, ModelCascade, TokenBudgetController
except ImportError:
    from .semantic_cache import ExactCache, SemanticCache
    from .prompt_compress import PromptCompressor, ModelCascade, TokenBudgetController


def demo_exact_cache():
    print("▶ 1. 精确缓存")
    print("─" * 60)
    cache = ExactCache(ttl_s=60)

    queries = [
        ("什么是 RAG？", "RAG 是检索增强生成..."),
        ("什么是 RAG？", None),          # 命中
        ("什么是RAG？", None),           # 命中（normalize 后相同）
        ("什么是 Agent？", "Agent 是..."),
        ("什么是 RAG？", None),          # 再次命中
    ]

    for q, expected_response in queries:
        cached = cache.get(q)
        if cached:
            print(f"  ✅ HIT  '{q}' → {cached[:30]}...")
        else:
            if expected_response:
                cache.put(q, expected_response)
            print(f"  ❌ MISS '{q}' → {'已缓存' if expected_response else '无响应'}")

    print(f"  命中率: {cache.hit_rate:.0%} ({cache.stats})")


def demo_semantic_cache():
    print(f"\n\n▶ 2. 语义缓存")
    print("─" * 60)
    cache = SemanticCache(threshold=0.6, ttl_s=60)

    # 存入
    pairs = [
        ("什么是 RAG？", "RAG (Retrieval-Augmented Generation) 是检索增强生成技术..."),
        ("如何部署 Docker？", "Docker 部署步骤: 1. 拉取镜像 2. 运行容器..."),
    ]
    for q, r in pairs:
        cache.put(q, r)
        print(f"  存入: '{q}'")

    # 查询（语义相似也命中）
    test_queries = [
        "RAG 是什么？",              # 语义相似 → 应命中
        "请解释一下 RAG 技术",       # 语义相似
        "Docker 怎么部署？",          # 语义相似
        "如何训练一个大模型？",       # 不相似 → miss
    ]
    for q in test_queries:
        result = cache.get(q, cost_per_call=0.01)
        if result:
            print(f"  ✅ HIT  '{q}' → {result[:30]}...")
        else:
            print(f"  ❌ MISS '{q}'")

    print(f"  命中率: {cache.hit_rate:.0%}, 节省费用: ${cache.stats['saved_cost']:.2f}")


def demo_prompt_compression():
    print(f"\n\n▶ 3. Prompt 压缩")
    print("─" * 60)
    compressor = PromptCompressor()

    verbose_prompt = """
    你是一个专业的技术助手。请根据以下上下文回答用户的问题。

    上下文信息：

    SmartFlow 工作流引擎是企业级的流程自动化平台。
    支持可视化流程设计、表单构建、规则引擎和消息通知。
    主要功能包括：审批流、数据流、集成流。


    例如：用户问"如何创建审批流"，你应该回答具体步骤。

    示例：
    问：如何创建数据流？
    答：1. 进入流程设计器 2. 选择数据流模板 3. 配置数据源

    示例：
    问：如何配置通知？
    答：1. 进入通知设置 2. 选择通知渠道 3. 配置触发条件

    示例：
    问：如何导出报表？
    答：1. 进入报表中心 2. 选择报表类型 3. 点击导出

    示例：
    问：如何管理用户？
    答：1. 进入用户管理 2. 添加或编辑用户 3. 分配角色


    请基于以上信息回答用户问题。回答要简洁专业。
    """

    result = compressor.compress(verbose_prompt, target_ratio=0.5)
    print(f"  原始: {result['original_chars']} 字符")
    print(f"  压缩: {result['compressed_chars']} 字符")
    print(f"  节省: {result['saved_pct']}")
    print(f"  压缩后 (前100字):")
    print(f"    {result['text'][:100]}...")


def demo_model_cascade():
    print(f"\n\n▶ 4. 模型级联")
    print("─" * 60)
    cascade = ModelCascade()

    queries = [
        "你好",                                    # 简单 → mini
        "什么是 RAG？",                            # 中等
        "对比分析 FAISS、Chroma、Milvus 在十亿级向量场景下的性能差异和成本结构", # 复杂 → 大模型
        "1+1=?",                                   # 简单
        "请综合评估这三个方案的可行性并推理最优选择",  # 复杂
    ]

    for q in queries:
        result = cascade.route(q)
        print(f"  [{result['model']:20s}] 复杂度={result['complexity']:.1f} "
              f"成本≈{result['estimated_cost']} 延迟={result['latency_ms']}ms ← '{q[:30]}'")

    report = cascade.cost_report()
    print(f"\n  报告: {report['total_calls']} 次调用, 模型分布: {report['model_distribution']}")


def demo_budget_controller():
    print(f"\n\n▶ 5. Token 预算控制器")
    print("─" * 60)
    budget = TokenBudgetController(daily_budget=0.05, monthly_budget=1.0)

    # 模拟多次调用
    calls = [
        (500, "gpt-4o-mini", "team-a", 0.00015),
        (1000, "gpt-4o-mini", "team-a", 0.00015),
        (2000, "gpt-4o", "team-b", 0.0025),
        (500, "gpt-4o-mini", "team-a", 0.00015),
        (3000, "gpt-4o", "team-b", 0.0025),
        (10000, "gpt-4o", "team-b", 0.0025),  # 这个会触发预算告警
    ]

    for tokens, model, tenant, cost in calls:
        budget.record(tokens, model, tenant, cost)
        can = "✅" if budget.can_proceed() else "🚨"
        print(f"  {can} {model:15s} {tokens:5d} tokens (tenant={tenant}) → 推荐: {budget.suggest_model()}")

    report = budget.report
    print(f"\n  日预算: {report['daily']}")
    print(f"  月预算: {report['monthly']}")
    print(f"  租户分布: {report['tenant_usage']}")

    if budget.alerts:
        print(f"  告警:")
        for a in budget.alerts[-2:]:
            print(f"    [{a.level}] {a.message}")


def main():
    print("=== LLM 成本优化 ===\n")
    demo_exact_cache()
    demo_semantic_cache()
    demo_prompt_compression()
    demo_model_cascade()
    demo_budget_controller()

    print(f"\n\n{'=' * 60}")
    print("📊 成本优化总结:")
    print(f"""
  策略           │ 节省比例      │ 实现复杂度  │ 生产方案
  ──────────────┼─────────────┼───────────┼────────────────
  精确缓存       │ 30-50%       │ ⭐          │ Redis / Memcached
  语义缓存       │ 40-60%       │ ⭐⭐        │ GPTCache / Redis VS
  Prompt 压缩    │ 10-30%       │ ⭐          │ LLMLingua / 手动
  模型级联       │ 60-80%       │ ⭐⭐        │ LiteLLM / 自建路由
  Token 预算     │ 避免超支      │ ⭐⭐        │ Helicone / 自建

  成本公式:
    总成本 = Σ(tokens × 单价) × (1 - 缓存命中率) × 级联折扣""")


if __name__ == "__main__":
    main()
