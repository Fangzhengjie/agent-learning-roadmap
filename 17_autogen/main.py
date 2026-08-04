"""AutoGen (AG2) 代码审查三人组 Demo

最佳场景：多 Agent 自主协作 — Agent 间对话、角色分工、动态发言选择。

核心模式：
  - AssistantAgent: 带工具和角色的 AI Agent
  - RoundRobinGroupChat / SelectorGroupChat: 多 Agent 编排
  - TextMentionTermination: 基于关键词的对话终止
  - Agent 间通过消息自主协作

为什么代码审查选 AutoGen：
  - 代码审查天然需要多角色（作者解释 → 审查者质疑 → 讨论 → 共识）
  - GroupChat 让 Agent 间自由对话，而非简单的调用-返回
  - SelectorGroupChat 可以让 LLM 动态决定"谁来说下一句"
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from shared_tools import CODE_SNIPPET, analyze_code, suggest_fix


# ── 三个 Agent：作者、安全审查者、测试者 ─────────────────────

async def main():
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    print("=== AutoGen 代码审查三人组 Demo ===")
    print(f"模型: {model_name}\n")

    model_client = OpenAIChatCompletionClient(model=model_name)

    # Agent 1: 代码作者 — 解释代码意图，回应审查意见
    code_author = AssistantAgent(
        name="code_author",
        description="代码作者，解释代码设计意图，回应审查意见并提供修复方案。",
        system_message=(
            "你是这段代码的作者。你的职责是：\n"
            "1. 解释代码的设计意图\n"
            "2. 针对审查者发现的问题，使用 suggest_fix 工具提供修复方案\n"
            "3. 对于合理的批评，接受并给出改进方案\n"
            "4. 对于误解，礼貌地解释原因\n"
            "讨论结束后回复 'REVIEW_COMPLETE' 表示审查完成。\n"
            "用中文讨论。"
        ),
        model_client=model_client,
        tools=[suggest_fix],
    )

    # Agent 2: 安全审查者 — 专注安全和性能问题
    security_reviewer = AssistantAgent(
        name="security_reviewer",
        description="安全审查专家，专注发现安全漏洞和性能问题。",
        system_message=(
            "你是资深安全审查专家。你的职责是：\n"
            "1. 使用 analyze_code 工具分析代码的 security 和 performance 方面\n"
            "2. 按严重程度排序报告发现的问题\n"
            "3. 对作者的修复方案给出评价\n"
            "4. CRITICAL 和 HIGH 级别问题必须修复后才能通过\n"
            "用中文讨论。"
        ),
        model_client=model_client,
        tools=[analyze_code],
    )

    # Agent 3: 代码规范审查者 — 专注代码风格和可维护性
    style_reviewer = AssistantAgent(
        name="style_reviewer",
        description="代码规范审查者，专注代码风格、可读性和可维护性。",
        system_message=(
            "你是代码规范审查者。你的职责是：\n"
            "1. 使用 analyze_code 工具分析代码的 style 方面\n"
            "2. 检查命名规范、类型注解、日志使用等\n"
            "3. 风格问题是建议性的，不阻塞合并\n"
            "用中文讨论。"
        ),
        model_client=model_client,
        tools=[analyze_code],
    )

    # ── GroupChat 编排 ────────────────────────────────────
    termination = TextMentionTermination("REVIEW_COMPLETE") | MaxMessageTermination(15)

    team = RoundRobinGroupChat(
        participants=[security_reviewer, style_reviewer, code_author],
        termination_condition=termination,
    )

    # ── 执行 ──────────────────────────────────────────────
    task = f"请审查以下代码，找出所有安全、性能和风格问题：\n```python{CODE_SNIPPET}```"

    print(f"📝 待审查代码:{CODE_SNIPPET}")
    print("─" * 60)
    print("🔄 开始多 Agent 代码审查...\n")

    result = await Console(team.run_stream(task=task))

    # ── 结果分析 ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"\n📊 审查统计:")
    print(f"  对话轮次: {len(result.messages)}")
    print(f"  终止原因: {result.stop_reason}")

    for i, msg in enumerate(result.messages):
        src = getattr(msg, "source", "?")
        content_preview = str(getattr(msg, "content", ""))[:80]
        print(f"  [{i}] {src}: {content_preview}...")

    # ── 架构观察 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 AutoGen 代码审查架构观察:")
    print()
    print("  ✅ 最佳场景: 多 Agent 自主协作（审查/辩论/头脑风暴）")
    print("  ✅ Agent 间自然对话（审查者质疑 → 作者回应 → 再讨论）")
    print("  ✅ RoundRobin 保证每个角色都发言")
    print("  ✅ SelectorGroupChat 可让 LLM 决定下一个发言者")
    print("  ✅ 终止条件可组合（关键词 | 最大轮次）")
    print("  ⚠️  对话轮次不可控（可能跑偏）")
    print("  ⚠️  单 Agent 内部不可观测（无 pre/post tool hook）")
    print("  ❌ 无内置取消/重试机制")

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
