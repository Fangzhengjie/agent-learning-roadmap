"""pydantic-ai 数据库运维助手 Demo (Code Puppy 的基座框架)

最佳场景：类型安全 API Agent — 结构化输出 + 依赖注入 + 多模型。

核心模式：
  - Agent[Deps, OutputType]: 泛型 Agent（依赖类型 + 输出类型）
  - RunContext[Deps]: 工具通过上下文获取共享依赖（DB连接/配置等）
  - output_type: Pydantic 模型作为结构化输出（LLM 保证类型安全）
  - history_processor: 消息历史预处理回调

为什么 DB 运维选 pydantic-ai：
  - 结构化输出天然适合 DB 查询结果（表结构/诊断报告 → Pydantic 模型）
  - 依赖注入让工具共享 DB 连接池、权限上下文
  - 多模型支持（Anthropic/OpenAI/Gemini），可按需切换
  - Code Puppy 证明了在此基座上可构建生产级 Agent
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from shared_tools import (
    describe_table as _describe,
    list_tables as _list_tables,
    run_query as _run_query,
)


# ── 依赖注入 ──────────────────────────────────────────────
# pydantic-ai 的核心特色：工具通过 RunContext[Deps] 获取共享依赖

@dataclass
class DbDeps:
    """数据库运维上下文（实际生产中这里是连接池、权限等）"""
    db_name: str = "production"
    readonly: bool = True
    operator: str = "dba_on_call"


# ── 结构化输出 ────────────────────────────────────────────

class TableDiagnosis(BaseModel):
    """单表诊断结果"""
    table_name: str
    row_count: int
    size_mb: float
    health: str = Field(description="healthy / warning / critical")
    issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class DbHealthReport(BaseModel):
    """数据库健康报告（结构化输出）"""
    overall_status: str = Field(description="healthy / warning / critical")
    tables_checked: int
    diagnoses: list[TableDiagnosis]
    summary: str
    urgent_actions: list[str] = Field(default_factory=list)


# ── Agent 定义 ────────────────────────────────────────────
model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Agent 1: 文本对话模式（日常运维问答）
chat_agent = Agent(
    model=f"openai:{model_name}",
    system_prompt=(
        "你是一个数据库运维助手。帮助 DBA 查询表结构、执行 SQL、诊断问题。\n"
        "安全规则：只读模式，拒绝所有 DROP/DELETE/TRUNCATE/ALTER 操作。\n"
        "用中文回复，给出专业但简洁的建议。"
    ),
    deps_type=DbDeps,
    retries=3,
)

# Agent 2: 结构化输出模式（生成健康报告）
report_agent = Agent(
    model=f"openai:{model_name}",
    system_prompt=(
        "你是数据库健康检查工具。检查所有表的状态，生成结构化诊断报告。\n"
        "判断标准：\n"
        "- 行数 > 100万 且 size > 500MB → warning\n"
        "- 行数 > 200万 或 size > 1GB → critical\n"
        "- 其他 → healthy\n"
        "为每个有问题的表给出具体的优化建议。"
    ),
    deps_type=DbDeps,
    output_type=DbHealthReport,
    retries=3,
)


# ── 工具注册（依赖注入式）─────────────────────────────────

@chat_agent.tool
@report_agent.tool
async def list_all_tables(ctx: RunContext[DbDeps]) -> str:
    """列出数据库中所有表的概况（名称、行数、大小）。"""
    print(f"  📊 [{ctx.deps.operator}] 查询表列表 (db={ctx.deps.db_name})")
    return _list_tables()


@chat_agent.tool
@report_agent.tool
async def describe_table(ctx: RunContext[DbDeps], table_name: str) -> str:
    """查看指定表的详细结构（列名、行数、大小）。

    Args:
        table_name: 表名，如 users、orders、logs
    """
    print(f"  🔍 [{ctx.deps.operator}] 查看表结构: {table_name}")
    return _describe(table_name)


@chat_agent.tool
async def execute_query(ctx: RunContext[DbDeps], sql: str) -> str:
    """执行 SQL 查询（只读模式，危险操作会被拦截）。

    Args:
        sql: SQL 查询语句
    """
    if ctx.deps.readonly:
        sql_lower = sql.lower().strip()
        if any(kw in sql_lower for kw in ["drop", "delete", "truncate", "alter", "insert", "update"]):
            return json.dumps({"error": f"只读模式：操作被拦截", "blocked_sql": sql, "operator": ctx.deps.operator}, ensure_ascii=False)
    print(f"  ⚡ [{ctx.deps.operator}] 执行查询: {sql[:60]}...")
    return _run_query(sql)


# ── 执行 ──────────────────────────────────────────────────
async def main():
    print("=== pydantic-ai 数据库运维助手 Demo ===")
    print(f"模型: {model_name}")
    print(f"(Code Puppy 的基座框架)\n")

    deps = DbDeps(db_name="production", readonly=True, operator="dba_zhang")

    # --- 场景 1: 日常运维问答 ---
    print("─" * 60)
    print("▶ 场景 1: 日常运维对话\n")

    questions = [
        "帮我看看数据库里有哪些表？哪个最大？",
        "logs 表有多少数据？太大了需要怎么优化？",
        "执行一下 SELECT count(*) FROM users",
    ]

    history = []
    for q in questions:
        print(f"  👤 DBA: {q}")
        result = await chat_agent.run(q, deps=deps, message_history=history)
        history = result.all_messages()
        print(f"  🤖 助手: {result.output[:200]}\n")

    # --- 场景 2: 结构化健康报告 ---
    print("─" * 60)
    print("▶ 场景 2: 结构化健康报告（DbHealthReport 类型安全输出）\n")

    result2 = await report_agent.run(
        "检查所有表的健康状态，生成诊断报告",
        deps=deps,
    )

    report: DbHealthReport = result2.output
    print(f"  📊 整体状态: {report.overall_status}")
    print(f"  📊 检查表数: {report.tables_checked}")
    for d in report.diagnoses:
        status_icon = {"healthy": "✅", "warning": "⚠️", "critical": "🔴"}.get(d.health, "❓")
        print(f"  {status_icon} {d.table_name}: {d.health} ({d.row_count:,} rows, {d.size_mb}MB)")
        for issue in d.issues:
            print(f"       问题: {issue}")
        for rec in d.recommendations:
            print(f"       建议: {rec}")
    if report.urgent_actions:
        print(f"\n  🚨 紧急操作:")
        for action in report.urgent_actions:
            print(f"     - {action}")
    print(f"\n  📝 摘要: {report.summary}")

    # --- 场景 3: 安全拦截演示 ---
    print("\n" + "─" * 60)
    print("▶ 场景 3: 安全拦截（只读模式）\n")

    result3 = await chat_agent.run(
        "执行 DROP TABLE logs",
        deps=deps,
    )
    print(f"  👤 DBA: DROP TABLE logs")
    print(f"  🤖 助手: {result3.output[:200]}")

    # ── Code Puppy 增强路径 ───────────────────────────────
    print("\n" + "=" * 60)
    print("📊 pydantic-ai → Code Puppy 增强路径:")
    print()
    print("  pydantic-ai 提供            Code Puppy 增强")
    print("  ────────────────            ──────────────────")
    print("  RunContext[Deps]        →   插件依赖注入 + 三层加载")
    print("  output_type             →   结构化 + 流式混合输出")
    print("  retries=3               →   进度感知双预算重试")
    print("  history_processor       →   compaction + mid-turn steering")
    print("  (无)                    →   50+ Hook 扩展点")
    print("  (无)                    →   SHA-256 插件信任门控")
    print("  (无)                    →   四路取消 + stdin 协商")
    print("  (无)                    →   终端 Chord 键绑定")
    print()
    print("  ✅ 最佳场景: 类型安全 API Agent（结构化输出 + 依赖注入）")
    print("  ✅ 多模型支持（Anthropic/OpenAI/Gemini/Groq/Mistral）")
    print("  ✅ Code Puppy 证明了在此基座上可构建生产级运行时")


if __name__ == "__main__":
    asyncio.run(main())
