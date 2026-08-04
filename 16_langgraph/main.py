"""LangGraph 订单审批流 Demo（Human-in-the-loop）

最佳场景：有状态多步工作流 — 需要 checkpoint、条件路由、人工审批的业务流程。

核心模式：
  - StateGraph 定义节点和边
  - TypedDict 状态 schema（节点间状态流转）
  - Checkpointer 持久化（中断后可恢复）
  - interrupt() 暂停等待人工审批
  - 条件路由（金额 / 风险分级）

为什么审批流选 LangGraph：
  - checkpoint 天然支持"暂停等人审批 → 恢复继续"
  - 图结构可视化审批路径
  - 状态机精确控制每一步
"""

import json
import os
import sys
from typing import Annotated, Any, Literal, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from shared_tools import (
    approve_order as _approve,
    check_credit as _check_credit,
    check_inventory as _check_inventory,
    get_order as _get_order,
)


# ── 状态定义 ──────────────────────────────────────────────
class OrderState(TypedDict):
    order_id: str
    order_data: Optional[dict]
    inventory_ok: Optional[bool]
    credit_ok: Optional[bool]
    risk_level: Optional[str]       # low / medium / high
    human_decision: Optional[str]   # approved / rejected
    final_result: Optional[str]
    step_log: list[str]


# ── 图节点 ────────────────────────────────────────────────
def fetch_order(state: OrderState) -> dict:
    """节点: 获取订单详情"""
    data = json.loads(_get_order(state["order_id"]))
    print(f"  📦 订单: {data.get('customer')} | ¥{data.get('amount'):,} | {data.get('items')}")
    return {"order_data": data, "step_log": state["step_log"] + ["fetch_order"]}


def check_inventory(state: OrderState) -> dict:
    """节点: 检查库存"""
    items = state["order_data"]["items"]
    result = json.loads(_check_inventory(items))
    all_ok = all(v.get("available", False) for v in result.values())
    print(f"  📦 库存检查: {'✅ 全部有货' if all_ok else '❌ 部分缺货'}")
    return {"inventory_ok": all_ok, "step_log": state["step_log"] + ["check_inventory"]}


def check_credit(state: OrderState) -> dict:
    """节点: 检查客户信用"""
    customer = state["order_data"]["customer"]
    credit = json.loads(_check_credit(customer))
    amount = state["order_data"]["amount"]
    remaining = credit["limit"] - credit["used"]
    ok = remaining >= amount
    print(f"  💳 信用检查: {customer} | 等级{credit['level']} | 余额¥{remaining:,} | {'✅ 通过' if ok else '❌ 额度不足'}")
    return {"credit_ok": ok, "step_log": state["step_log"] + ["check_credit"]}


def assess_risk(state: OrderState) -> dict:
    """节点: 风险评估"""
    amount = state["order_data"]["amount"]
    risk_score = state["order_data"].get("risk_score", 0.5)

    if amount < 10000 and risk_score < 0.5:
        level = "low"
    elif amount > 100000 or risk_score > 0.7:
        level = "high"
    else:
        level = "medium"

    print(f"  🎯 风险评估: {level}（金额¥{amount:,}, 风险分{risk_score}）")
    return {"risk_level": level, "step_log": state["step_log"] + ["assess_risk"]}


def auto_approve(state: OrderState) -> dict:
    """节点: 自动审批（低风险）"""
    result = _approve(state["order_id"], "approved", "自动审批: 低风险订单")
    print(f"  ✅ 自动审批通过")
    return {"final_result": result, "human_decision": "auto_approved", "step_log": state["step_log"] + ["auto_approve"]}


def human_review(state: OrderState) -> dict:
    """节点: 人工审批（中/高风险）— 模拟 interrupt"""
    # 在真实 LangGraph 应用中，这里用 interrupt() 暂停图执行
    # 外部系统（Web UI / Slack bot）调用 Command.resume() 恢复
    order = state["order_data"]
    risk = state["risk_level"]

    print(f"\n  ⏸️  [HUMAN-IN-THE-LOOP] 等待人工审批...")
    print(f"     订单: {state['order_id']}")
    print(f"     客户: {order['customer']} | 金额: ¥{order['amount']:,}")
    print(f"     风险: {risk} | 库存: {'✅' if state['inventory_ok'] else '❌'} | 信用: {'✅' if state['credit_ok'] else '❌'}")

    # 模拟人工决策（实际中由 interrupt + resume 实现）
    if state["credit_ok"] and state["inventory_ok"]:
        decision = "approved"
        print(f"     👤 审批人决策: ✅ 批准")
    else:
        decision = "rejected"
        reason = "信用额度不足" if not state["credit_ok"] else "库存不足"
        print(f"     👤 审批人决策: ❌ 驳回（{reason}）")

    result = _approve(state["order_id"], decision, f"人工审批: {risk}风险订单")
    return {"final_result": result, "human_decision": decision, "step_log": state["step_log"] + ["human_review"]}


def auto_reject(state: OrderState) -> dict:
    """节点: 自动驳回（前置检查失败）"""
    reasons = []
    if not state["inventory_ok"]:
        reasons.append("库存不足")
    if not state["credit_ok"]:
        reasons.append("信用额度不足")
    reason = "自动驳回: " + ", ".join(reasons)
    result = _approve(state["order_id"], "rejected", reason)
    print(f"  ❌ 自动驳回: {reason}")
    return {"final_result": result, "human_decision": "auto_rejected", "step_log": state["step_log"] + ["auto_reject"]}


# ── 条件路由 ──────────────────────────────────────────────
def route_after_checks(state: OrderState) -> str:
    """前置检查后路由"""
    if not state["inventory_ok"] or not state["credit_ok"]:
        if state["risk_level"] == "high":
            return "human_review"  # 高风险即使不合格也给人看一眼
        return "auto_reject"
    if state["risk_level"] == "low":
        return "auto_approve"
    return "human_review"


# ── 构建图 ────────────────────────────────────────────────
def build_graph():
    """
    审批流程图：

    ┌─────────┐
    │  START  │
    └────┬────┘
         │
    ┌────▼────────┐
    │ fetch_order  │
    └────┬────────┘
         │
    ┌────▼────────────┐    ┌─────────────┐
    │ check_inventory  │───▶│ check_credit │
    └─────────────────┘    └──────┬──────┘
                                  │
                           ┌──────▼──────┐
                           │ assess_risk  │
                           └──────┬──────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼              ▼
              ┌──────────┐ ┌───────────┐ ┌─────────────┐
              │auto_approv│ │human_revie│ │ auto_reject  │
              └─────┬─────┘└─────┬─────┘ └──────┬──────┘
                    │            │               │
                    └────────────┴───────────────┘
                                 │
                            ┌────▼────┐
                            │   END   │
                            └─────────┘
    """
    graph = StateGraph(OrderState)

    graph.add_node("fetch_order", fetch_order)
    graph.add_node("check_inventory", check_inventory)
    graph.add_node("check_credit", check_credit)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("auto_approve", auto_approve)
    graph.add_node("human_review", human_review)
    graph.add_node("auto_reject", auto_reject)

    graph.set_entry_point("fetch_order")
    graph.add_edge("fetch_order", "check_inventory")
    graph.add_edge("check_inventory", "check_credit")
    graph.add_edge("check_credit", "assess_risk")
    graph.add_conditional_edges("assess_risk", route_after_checks, {
        "auto_approve": "auto_approve",
        "human_review": "human_review",
        "auto_reject": "auto_reject",
    })
    graph.add_edge("auto_approve", END)
    graph.add_edge("human_review", END)
    graph.add_edge("auto_reject", END)

    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ── 执行 ──────────────────────────────────────────────────
def main():
    print("=== LangGraph 订单审批流 Demo ===")
    print(f"模型: 无需 LLM（纯状态机）\n")

    app = build_graph()

    # 三个订单，分别走不同的审批路径
    test_orders = [
        ("ORD-2024-003", "低风险 → 自动审批"),
        ("ORD-2024-001", "中风险 → 人工审批"),
        ("ORD-2024-002", "高风险 + 信用紧张 → 人工审批"),
    ]

    for order_id, description in test_orders:
        print(f"\n{'═' * 60}")
        print(f"📋 {description}")
        print(f"{'═' * 60}")

        config = {"configurable": {"thread_id": order_id}}

        for step, event in enumerate(app.stream(
            {"order_id": order_id, "step_log": []},
            config=config,
            stream_mode="updates",
        )):
            for node_name in event:
                print(f"  📍 Step {step}: {node_name}")

        final = app.get_state(config)
        print(f"\n  📊 审批路径: {' → '.join(final.values['step_log'])}")
        print(f"  📊 最终决策: {final.values['human_decision']}")

    # ── 架构观察 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 LangGraph 审批流架构观察:")
    print()
    print("  ✅ 最佳场景: 有状态多步工作流（审批/管道/流程引擎）")
    print("  ✅ Checkpoint 天然支持暂停-恢复（人工审批）")
    print("  ✅ 条件路由实现业务分级（金额/风险 → 不同路径）")
    print("  ✅ 图结构可视化、可审计（每一步都有记录）")
    print("  ✅ stream_mode='updates' 实时观测节点执行")
    print("  ⚠️  图定义冗长（vs 简单 if/else 代码）")
    print("  ⚠️  interrupt 需要在图中预埋（vs Code Puppy 随时 Ctrl+T）")
    print("  ❌ 纯库，无终端 UX（需要自建 Web UI）")


if __name__ == "__main__":
    main()
