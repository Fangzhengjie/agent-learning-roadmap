"""Agent 通信协议：Function Calling / MCP / A2A / ACP

核心问题：Agent 如何与外部世界通信？

四种通信模式：

  1. Function Calling（Agent ↔ LLM 内部）
     LLM 输出结构化工具调用 → 框架执行 → 结果喂回 LLM
     所有 Agent 框架的基础

  2. MCP (Model Context Protocol)（Agent ↔ 工具）
     Anthropic 提出，Agent 调用外部工具的标准协议
     类比：USB 接口（一个工具，所有 Agent 都能用）
     → 见 11_mcp_tools/main.py 的完整实现

  3. A2A (Agent-to-Agent Protocol)（Agent ↔ Agent）
     Google 提出，跨组织/跨框架的 Agent 间通信协议
     类比：HTTP 协议（任何客户端与任何服务器通信）

  4. ACP (Agent Communication Protocol)（Agent ↔ Agent 替代方案）
     IBM/BeeAI 提出，比 A2A 更轻量的 Agent 间通信
     特点：基于 HTTP+SSE，无需 Agent Card 发现

本示例展示四种协议的工作原理和对比。

协议关系图：
  ┌──────────────────────────────────────────────────┐
  │                  Agent 生态                       │
  │                                                  │
  │  ┌─────────┐  A2A 协议  ┌─────────┐             │
  │  │ Agent A  │◄─────────►│ Agent B  │             │
  │  │(LangGraph│           │(Spring AI│             │
  │  └────┬─────┘           └────┬─────┘             │
  │       │ MCP                  │ MCP               │
  │       ▼                      ▼                   │
  │  ┌─────────┐           ┌─────────┐              │
  │  │MCP Server│          │MCP Server│              │
  │  │(工单系统) │          │(支付系统) │              │
  │  └─────────┘           └─────────┘              │
  │                                                  │
  │  Agent 内部：Function Calling（LLM ↔ 工具循环）   │
  └──────────────────────────────────────────────────┘
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ═══════════════════════════════════════════════════════════
# Part 1: Function Calling（Agent 内部通信）
# ═══════════════════════════════════════════════════════════

class FunctionCallingDemo:
    """展示 Function Calling 的工作原理。

    这是所有 Agent 框架的基础 — LLM 输出结构化的工具调用请求。
    """

    @staticmethod
    def show():
        print("▶ 1. Function Calling（Agent ↔ LLM 内部通信）")
        print("─" * 60)
        print()

        # 工具定义（发给 LLM 的 JSON Schema）
        tool_definition = {
            "type": "function",
            "function": {
                "name": "lookup_ticket",
                "description": "查询工单详情",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticket_id": {"type": "string", "description": "工单编号"}
                    },
                    "required": ["ticket_id"]
                }
            }
        }

        # LLM 返回的工具调用请求
        llm_response = {
            "role": "assistant",
            "content": None,  # 不输出文本
            "tool_calls": [{
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "lookup_ticket",
                    "arguments": '{"ticket_id": "T-001"}'
                }
            }]
        }

        # 工具执行结果
        tool_result = {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": '{"ticket_id": "T-001", "status": "open", "type": "technical"}'
        }

        print("  📤 发给 LLM 的工具定义:")
        print(f"     {json.dumps(tool_definition['function']['name'], ensure_ascii=False)}: "
              f"{tool_definition['function']['description']}")
        print()
        print("  📥 LLM 返回工具调用请求:")
        print(f"     {json.dumps(llm_response['tool_calls'][0]['function'], ensure_ascii=False, indent=6)}")
        print()
        print("  📤 工具执行结果喂回 LLM:")
        print(f"     {tool_result['content']}")
        print()

        # 流程图
        print("  完整循环:")
        print("  ┌────────┐   tools schema   ┌────────┐")
        print("  │ 框架   │ ───────────────→ │  LLM   │")
        print("  │        │                  │        │")
        print("  │        │ ◄─────────────── │        │")
        print("  │        │   tool_calls     │        │")
        print("  │        │                  │        │")
        print("  │ 执行工具│                  │        │")
        print("  │        │   tool result    │        │")
        print("  │        │ ───────────────→ │        │")
        print("  │        │                  │ 最终回复│")
        print("  │        │ ◄─────────────── │        │")
        print("  └────────┘   content        └────────┘")
        print()
        print("  支持 Function Calling 的 LLM:")
        print("  - OpenAI: GPT-4o / GPT-4o-mini（最成熟）")
        print("  - Anthropic: Claude 3.5 Sonnet（tool_use 格式）")
        print("  - Google: Gemini 1.5 Pro")
        print("  - 开源: Llama 3.1+, Qwen 2+, DeepSeek V2+")


# ═══════════════════════════════════════════════════════════
# Part 2: MCP (Model Context Protocol)（Agent ↔ 工具）
# ═══════════════════════════════════════════════════════════

class MCPProtocolDemo:
    """展示 MCP 协议的核心概念。

    完整实现见 11_mcp_tools/main.py
    """

    @staticmethod
    def show():
        print("\n\n▶ 2. MCP — Model Context Protocol（Agent ↔ 工具）")
        print("─" * 60)
        print()

        # MCP 消息格式（基于 JSON-RPC 2.0）
        messages = {
            "初始化": {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}, "resources": {}},
                    "clientInfo": {"name": "my-agent", "version": "1.0"},
                },
                "id": 1,
            },
            "列出工具": {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 2,
            },
            "调用工具": {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "lookup_ticket",
                    "arguments": {"ticket_id": "T-001"},
                },
                "id": 3,
            },
            "读取资源": {
                "jsonrpc": "2.0",
                "method": "resources/read",
                "params": {"uri": "config://routing-rules"},
                "id": 4,
            },
        }

        print("  MCP 消息示例（JSON-RPC 2.0）:")
        for label, msg in messages.items():
            method = msg.get("method", "")
            print(f"\n  📨 {label}: {method}")
            if "params" in msg:
                params_str = json.dumps(msg["params"], ensure_ascii=False, indent=6)
                for line in params_str.split("\n")[:4]:
                    print(f"     {line}")
                if params_str.count("\n") > 4:
                    print("     ...")

        print(f"""

  MCP 三种能力:
  ─────────────┬───────────────────────────
  Tool (工具)   │ Agent 可调用的函数（RPC）
  Resource (资源)│ Agent 可读取的数据（只读）
  Prompt (提示) │ 预定义的 prompt 模板

  MCP 传输方式:
  ─────────────┬───────────────────────────
  stdio        │ 子进程通信（本地，最简单）
  SSE          │ HTTP Server-Sent Events
  Streamable   │ HTTP 双向流（新标准）

  MCP 生态:
  - Claude Desktop: 原生支持 MCP
  - VS Code Copilot: 支持 MCP 工具
  - LangChain: MCP adapter
  - Spring AI: MCP client/server
  - 社区: 1000+ MCP Server（GitHub/Slack/DB/文件系统等）
  → 完整实现见 11_mcp_tools/main.py""")


# ═══════════════════════════════════════════════════════════
# Part 3: A2A (Agent-to-Agent Protocol)（Agent ↔ Agent）
# ═══════════════════════════════════════════════════════════

class TaskState(Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentCard:
    """A2A Agent 名片 — 描述 Agent 的能力，供其他 Agent 发现。

    发布在 /.well-known/agent.json（类似 robots.txt）。
    """
    name: str
    description: str
    url: str
    skills: list[dict]
    version: str = "1.0"
    protocol: str = "a2a/1.0"

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "protocol": self.protocol,
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
            },
            "skills": self.skills,
        }


@dataclass
class A2ATask:
    """A2A 任务 — Agent 间传递的工作单元。"""
    task_id: str
    state: TaskState
    messages: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class A2AServer:
    """模拟 A2A Agent 服务端。"""

    def __init__(self, card: AgentCard):
        self.card = card
        self.tasks: dict[str, A2ATask] = {}

    def handle_send_task(self, request: dict) -> dict:
        """处理 tasks/send 请求。"""
        task_id = request.get("id", str(uuid.uuid4())[:8])
        message = request.get("message", {})

        task = A2ATask(
            task_id=task_id,
            state=TaskState.WORKING,
            messages=[message],
        )

        # 模拟处理
        user_text = ""
        for part in message.get("parts", []):
            if part.get("type") == "text":
                user_text = part["text"]

        # 生成响应
        response_text = f"[{self.card.name}] 已收到任务: {user_text}。正在处理..."
        task.state = TaskState.COMPLETED
        task.artifacts.append({
            "parts": [{"type": "text", "text": response_text}],
            "index": 0,
        })

        self.tasks[task_id] = task

        return {
            "id": task.task_id,
            "status": {"state": task.state.value},
            "artifacts": task.artifacts,
        }

    def get_agent_card(self) -> dict:
        """返回 Agent 名片（/.well-known/agent.json）。"""
        return self.card.to_json()


class A2AClient:
    """模拟 A2A Agent 客户端。"""

    def __init__(self, name: str):
        self.name = name

    def send_task(self, server: A2AServer, text: str) -> dict:
        """向另一个 Agent 发送任务。"""
        request = {
            "id": str(uuid.uuid4())[:8],
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": text}],
            }
        }
        return server.handle_send_task(request)


class A2AProtocolDemo:
    """展示 A2A 协议的完整流程。"""

    @staticmethod
    def show():
        print("\n\n▶ 3. A2A — Agent-to-Agent Protocol（Agent ↔ Agent）")
        print("─" * 60)
        print()

        # Step 1: Agent 名片（服务发现）
        print("  Step 1: Agent 名片（服务发现）")
        print("  每个 Agent 发布 /.well-known/agent.json 描述自己的能力:\n")

        ticket_agent = A2AServer(AgentCard(
            name="工单处理 Agent",
            description="处理客服工单的分类、路由和回复",
            url="https://ticket-agent.example.com",
            skills=[
                {"id": "classify", "name": "工单分类", "description": "判断工单类型和优先级"},
                {"id": "route", "name": "工单路由", "description": "将工单路由到合适的团队"},
            ],
        ))

        payment_agent = A2AServer(AgentCard(
            name="支付处理 Agent",
            description="处理退款、对账和支付查询",
            url="https://payment-agent.example.com",
            skills=[
                {"id": "refund", "name": "退款处理", "description": "处理用户退款申请"},
                {"id": "query", "name": "支付查询", "description": "查询支付状态"},
            ],
        ))

        for agent in [ticket_agent, payment_agent]:
            card = agent.get_agent_card()
            print(f"  🤖 {card['name']}: {card['url']}")
            print(f"     能力: {', '.join(s['name'] for s in card['skills'])}")
        print()

        # Step 2: Agent 间任务发送
        print("  Step 2: Agent 间任务通信\n")

        supervisor = A2AClient(name="主管 Agent")

        # 主管 Agent → 工单 Agent
        print(f"  📨 {supervisor.name} → {ticket_agent.card.name}:")
        result1 = supervisor.send_task(ticket_agent, "分析工单 T-001，判断类型和优先级")
        print(f"     状态: {result1['status']['state']}")
        print(f"     响应: {result1['artifacts'][0]['parts'][0]['text']}")
        print()

        # 主管 Agent → 支付 Agent
        print(f"  📨 {supervisor.name} → {payment_agent.card.name}:")
        result2 = supervisor.send_task(payment_agent, "处理工单 T-002 的退款，金额 99 元")
        print(f"     状态: {result2['status']['state']}")
        print(f"     响应: {result2['artifacts'][0]['parts'][0]['text']}")

        # A2A 核心 API
        print(f"""

  A2A 核心 API:
  ──────────────────┬──────────────────────────────
  POST tasks/send    │ 发送任务给另一个 Agent（同步）
  POST tasks/sendSubscribe │ 发送任务 + 订阅结果（流式）
  GET  tasks/get     │ 查询任务状态
  POST tasks/cancel  │ 取消任务
  GET  .well-known/agent.json │ 获取 Agent 名片

  A2A 任务状态流转:
  submitted → working → completed
                  ↓
            input-required → working → completed
                  ↓
                failed

  A2A 消息格式 (类似 ChatML):
  ┌──────────────────────────────────────────┐
  │ {{"role": "user",                         │
  │  "parts": [                              │
  │    {{"type": "text", "text": "..."}},      │
  │    {{"type": "file", "data": "base64..."}} │
  │  ]}}                                      │
  └──────────────────────────────────────────┘""")


# ═══════════════════════════════════════════════════════════
# Part 4: ACP (Agent Communication Protocol)
# ═══════════════════════════════════════════════════════════

class ACPProtocolDemo:
    """展示 ACP 协议的核心概念。

    ACP 是 IBM/BeeAI 提出的 Agent 间通信协议，与 A2A 定位相同但设计理念不同。
    """

    @staticmethod
    def show():
        print("\n\n▶ 4. ACP — Agent Communication Protocol（Agent ↔ Agent 替代方案）")
        print("─" * 60)

        # ACP 核心 API
        print(f"""
  ACP vs A2A: 两种 Agent 间通信协议的竞争

  ACP (IBM/BeeAI)                │ A2A (Google)
  ──────────────────────────────┼───────────────────────────
  更轻量，HTTP + SSE             │ 更完整，HTTP + JSON-RPC
  无 Agent Card（直接调用）       │ Agent Card 服务发现
  简单的 run/status/cancel       │ Task 状态机 + 多种消息类型
  面向内部 Agent 编排             │ 面向跨组织 Agent 协作
  IBM/BeeAI 主导                 │ Google 主导
  开源实现先行                    │ 规范先行

  ACP 核心 API:
  ─────────────────────────────────────────────────
  POST   /agents                │ 列出可用 Agent
  POST   /agents/{{id}}/run      │ 调用 Agent 执行任务
  GET    /agents/{{id}}/status   │ 查询执行状态
  POST   /agents/{{id}}/cancel   │ 取消执行""")

        # 模拟 ACP 消息
        acp_run_request = {
            "input": [
                {"type": "text", "text": "分析工单 T-001 并分类"}
            ],
            "config": {
                "timeout": 30,
                "stream": True,
            }
        }

        acp_run_response = {
            "run_id": "run-abc123",
            "status": "completed",
            "output": [
                {"type": "text", "text": "工单 T-001 分类为技术问题，优先级 P1"}
            ],
        }

        print(f"\n  ACP 请求示例:")
        print(f"  POST /agents/ticket-classifier/run")
        print(f"  {json.dumps(acp_run_request, ensure_ascii=False, indent=4)[:200]}")

        print(f"\n  ACP 响应示例:")
        print(f"  {json.dumps(acp_run_response, ensure_ascii=False, indent=4)}")

        print(f"""

  ACP 流式响应（SSE）:
  ─────────────────────────────────────────────────
  POST /agents/ticket-classifier/run (stream=true)

  data: {{"type": "progress", "message": "正在分析工单..."}}
  data: {{"type": "progress", "message": "检测到技术问题关键词..."}}
  data: {{"type": "result", "output": [{{"type": "text", "text": "分类完成"}}]}}
  data: [DONE]

  ACP 的设计哲学:
  - 简单优先：3 个 API 足够（run/status/cancel）
  - 无发现：不需要 Agent Card，直接知道 Agent URL
  - 流式原生：SSE 支持实时进度反馈
  - 适合内部编排：Supervisor Agent 调度多个子 Agent""")


# ═══════════════════════════════════════════════════════════
# Part 5: 四种协议对比
# ═══════════════════════════════════════════════════════════

def show_comparison():
    print("\n\n▶ 5. 四种协议对比")
    print("─" * 60)
    print(f"""
  维度          │ Function Calling │ MCP             │ A2A             │ ACP
  ─────────────┼─────────────────┼────────────────┼────────────────┼──────────────
  通信方向      │ LLM ↔ 框架内部  │ Agent → 工具   │ Agent ↔ Agent  │ Agent ↔ Agent
  提出者        │ OpenAI (2023)   │ Anthropic(2024)│ Google (2025)  │ IBM/BeeAI
  类比          │ 方法调用         │ USB 接口       │ HTTP 协议      │ RPC 调用
  传输层        │ LLM API 响应    │ JSON-RPC/stdio │ HTTP + JSON    │ HTTP + SSE
  发现机制      │ tools 参数      │ list_tools()   │ agent.json     │ 无（直接调用）
  状态管理      │ 无              │ 无             │ Task 状态机    │ run_id
  流式支持      │ 部分            │ SSE            │ sendSubscribe  │ SSE 原生
  复杂度        │ 低              │ 中             │ 高             │ 低
  成熟度        │ ✅ 成熟         │ ✅ 快速扩展    │ ⚠️ 早期        │ ⚠️ 早期

  什么时候用什么:
  ─────────────────────────────────────────────────────────
  Function Calling │ 单 Agent 调用工具（所有场景的基础）
  MCP              │ 工具标准化（一个工具服务多个 Agent）
  A2A              │ 跨组织/跨框架的 Agent 协作（有发现需求）
  ACP              │ 内部 Agent 编排（简单直接，无需发现）
  MCP + A2A/ACP    │ Agent 通过 MCP 调工具，通过 A2A/ACP 互相协作

  A2A vs ACP 选型:
  ─────────────────────────────────────────────────────────
  跨组织协作（Agent 需要被发现） → A2A（Agent Card 机制）
  内部编排（Agent 地址已知）     → ACP（更简单，3 个 API）
  需要复杂状态流转               → A2A（Task 状态机）
  需要流式进度反馈               → ACP（SSE 原生）

  协议分层:
  ┌────────────────────────────────────────────┐
  │  A2A / ACP（Agent ↔ Agent 协作层）        │
  ├────────────────────────────────────────────┤
  │  MCP（Agent ↔ Tool 工具层）               │
  ├────────────────────────────────────────────┤
  │  Function Calling（LLM ↔ 框架 基础层）    │
  └────────────────────────────────────────────┘""")


# ═══════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════

def main():
    print("=== Agent 通信协议：Function Calling / MCP / A2A / ACP ===\n")

    FunctionCallingDemo.show()
    MCPProtocolDemo.show()
    A2AProtocolDemo.show()
    ACPProtocolDemo.show()
    show_comparison()

    # ── 架构总结 ──────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 Agent 通信协议总结:")
    print()
    print("  1. Function Calling = Agent 的神经系统（LLM 与工具的内部通信）")
    print("  2. MCP = Agent 的手（Agent 操控外部工具的标准接口）")
    print("  3. A2A = Agent 的嘴（跨组织 Agent 协作，有服务发现）")
    print("  4. ACP = Agent 的对讲机（内部 Agent 编排，简单直接）")
    print()
    print("  当前生态状态:")
    print("  - Function Calling: ✅ 所有主流 LLM 已支持")
    print("  - MCP: ✅ 生态爆发中（Claude/Copilot/1000+ Server）")
    print("  - A2A: ⚠️ 早期阶段（Google 主导）")
    print("  - ACP: ⚠️ 早期阶段（IBM/BeeAI 主导）")
    print()
    print("  建议:")
    print("  - 现在：Function Calling + MCP 是最实用的组合")
    print("  - 内部编排：ACP 更简单（3 个 API）")
    print("  - 跨组织：关注 A2A（Agent Card 服务发现）")


if __name__ == "__main__":
    main()
