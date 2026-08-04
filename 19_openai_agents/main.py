"""OpenAI Agents SDK 智能客服分流 Demo

最佳场景：客服分流与 Agent 移交 — Handoff 机制天然匹配客服分诊场景。

核心模式：
  - Agent + instructions: 每个 Agent 专注一个领域
  - Handoff: Agent 间移交（分诊 → 技术支持 / 退款处理）
  - @function_tool: 类型安全工具注册
  - Runner.run(): 自动工具调用循环

为什么客服分流选 OpenAI Agents SDK：
  - Handoff 是所有框架中最优雅的 Agent 移交机制
  - 分诊 Agent 根据问题类型移交给专业 Agent
  - API 最简洁，2 个概念（Agent + Runner）即可上手
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import Agent, Runner, function_tool, ModelSettings

from shared_tools import (
    check_system_status as _check_status,
    escalate_ticket as _escalate,
    lookup_ticket as _lookup,
    process_refund as _refund,
)


# ── 工具定义 ──────────────────────────────────────────────

@function_tool
def lookup_ticket(ticket_id: str) -> str:
    """查询客服工单详情。

    Args:
        ticket_id: 工单编号，如 T-001
    """
    return _lookup(ticket_id)


@function_tool
def check_system_status(service: str) -> str:
    """检查后端服务状态（auth/payment/api/web）。

    Args:
        service: 服务名称
    """
    return _check_status(service)


@function_tool
def process_refund(ticket_id: str, amount: float, reason: str) -> str:
    """处理退款申请。

    Args:
        ticket_id: 工单编号
        amount: 退款金额
        reason: 退款原因
    """
    return _refund(ticket_id, amount, reason)


@function_tool
def escalate_ticket(ticket_id: str, target_team: str, notes: str) -> str:
    """将工单升级到指定团队。

    Args:
        ticket_id: 工单编号
        target_team: 目标团队（engineering/billing/management）
        notes: 升级说明
    """
    return _escalate(ticket_id, target_team, notes)


# ── Agent 定义（分诊 → 专业 Agent 移交）─────────────────────

async def main():
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    print("=== OpenAI Agents SDK 智能客服分流 Demo ===")
    print(f"模型: {model_name}\n")

    # Agent 1: 技术支持专员
    tech_agent = Agent(
        name="技术支持专员",
        instructions=(
            "你是技术支持专员。处理技术类问题（页面白屏、功能异常、连接问题等）。\n"
            "步骤：1. 用 lookup_ticket 查看工单详情\n"
            "      2. 用 check_system_status 检查相关服务状态\n"
            "      3. 给出诊断和解决方案\n"
            "      4. 如果是系统问题，用 escalate_ticket 升级给 engineering\n"
            "用中文回复。"
        ),
        tools=[lookup_ticket, check_system_status, escalate_ticket],
        model=model_name,
    )

    # Agent 2: 退款处理专员
    refund_agent = Agent(
        name="退款处理专员",
        instructions=(
            "你是退款处理专员。处理退款和计费类问题。\n"
            "步骤：1. 用 lookup_ticket 查看工单详情\n"
            "      2. 核实退款金额和原因\n"
            "      3. 用 process_refund 处理退款\n"
            "      4. 告知用户退款预计到账时间\n"
            "用中文回复。"
        ),
        tools=[lookup_ticket, process_refund],
        model=model_name,
    )

    # Agent 3: 分诊 Agent（入口）— 根据问题类型移交
    triage_agent = Agent(
        name="客服分诊台",
        instructions=(
            "你是客服分诊台。根据用户描述的问题类型，将工单移交给合适的专员：\n"
            "- 技术问题（白屏、报错、连接问题）→ 移交给 技术支持专员\n"
            "- 退款/计费问题 → 移交给 退款处理专员\n"
            "- 其他问题 → 先用 lookup_ticket 查看详情，再决定\n"
            "移交时简要说明问题类型。用中文回复。"
        ),
        tools=[lookup_ticket],
        handoffs=[tech_agent, refund_agent],  # 可以移交给两个专业 Agent
        model=model_name,
    )

    # ── 模拟三种不同类型的客服请求 ─────────────────────────
    tickets = [
        ("T-001", "我的工单 T-001，登录后页面白屏，什么都看不到"),
        ("T-002", "工单 T-002，上个月扣了我 99 元但功能根本用不了，要求退款"),
        ("T-003", "工单 T-003，我想了解如何升级到企业版"),
    ]

    for ticket_id, user_message in tickets:
        print(f"\n{'═' * 60}")
        print(f"👤 用户: {user_message}")
        print(f"{'═' * 60}")

        result = await Runner.run(triage_agent, input=user_message)

        print(f"\n🤖 [{result.last_agent.name}] 回复:")
        print(f"{result.final_output[:400]}")
        print(f"\n  📍 处理路径: 客服分诊台 → {result.last_agent.name}")

    # ── 架构观察 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 OpenAI Agents SDK 客服分流架构观察:")
    print()
    print("  ✅ 最佳场景: 客服分流 / Agent 移交（Handoff 最优雅）")
    print("  ✅ 分诊 → 专业 Agent 移交链路清晰")
    print("  ✅ API 最简洁（Agent + Runner.run 两个概念）")
    print("  ✅ 内置 Trace（无需额外追踪工具）")
    print("  ⚠️  Handoff 仅支持线性移交（不支持多方对话）")
    print("  ⚠️  扩展点有限（无 pre/post tool hook）")
    print("  ❌ 仅支持 OpenAI 模型")
    print("  ❌ 无取消/暂停/重试机制")


if __name__ == "__main__":
    asyncio.run(main())
