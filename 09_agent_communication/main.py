"""Agent 通信方式 — SSE + WebSocket

核心概念：Agent 需要实时推送中间状态给前端 — SSE 是首选，WebSocket 用于双向交互。

Agent 通信方式全景:
  ┌──────────────┬──────────┬──────────┬──────────┬───────────┐
  │ 方式          │ 流式输出  │ 中间状态  │ 双向通信  │ 适用场景   │
  ├──────────────┼──────────┼──────────┼──────────┼───────────┤
  │ HTTP 同步     │ ❌       │ ❌       │ ❌       │ 简单问答   │
  │ HTTP SSE      │ ✅       │ ✅(单向) │ ❌       │ 流式聊天   │
  │ WebSocket     │ ✅       │ ✅       │ ✅       │ 交互式Agent│
  │ gRPC 流       │ ✅       │ ✅       │ ✅       │ 微服务间   │
  └──────────────┴──────────┴──────────┴──────────┴───────────┘

本示例展示:
  Part A: SSE（Server-Sent Events）— Agent 流式输出首选
    1. SSE 协议基础 — 格式/字段/浏览器 API
    2. OpenAI 流式格式 — stream=True 的真实 SSE 输出
    3. Agent SSE 事件协议 — 多事件类型设计
    4. SSE 客户端解析器 — 模拟 EventSource 行为
    5. 生产环境注意事项 — 代理缓冲/超时/连接数/资源泄漏
    6. 服务端 SSE 实现 — FastAPI/Spring AI/Vercel AI SDK
    7. MCP Streamable HTTP — 新一代 MCP 传输

  Part B: WebSocket — 双向交互 Agent
    8. WebSocket 消息协议 — 类型定义 + 消息格式
    9. WebSocket Agent 交互演示 — 流式输出 + 工具调用 + HITL
    10. WebSocket 架构 — 网关 + 心跳 + 重连
    11. SSE vs WebSocket 选型指南
"""

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ╔═══════════════════════════════════════════════════════════╗
# ║  Part A: SSE (Server-Sent Events)                        ║
# ╚═══════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════
# 1. SSE 协议基础
# ═══════════════════════════════════════════════════════════

def show_sse_basics():
    """SSE 协议格式说明。"""
    print("▶ 1. SSE 协议基础")
    print("─" * 60)

    print(f"""
  SSE (Server-Sent Events) 格式:
  ─────────────────────────────────────────────
  HTTP/1.1 200 OK
  Content-Type: text/event-stream
  Cache-Control: no-cache
  Connection: keep-alive

  data: 第一条消息\\n\\n

  event: custom_event
  data: {{"type": "progress"}}\\n\\n

  id: msg-001
  data: 带 ID 的消息（用于重连恢复）\\n\\n

  SSE 格式规则:
  ─────────────────────────────────────────────
  data:    │ 消息内容（必需）
  event:   │ 事件类型（可选，默认 "message"）
  id:      │ 消息 ID（可选，断线重连用 Last-Event-ID）
  retry:   │ 重连间隔毫秒数（可选）
  \\n\\n     │ 消息结束标记（两个换行）

  浏览器 API:
  ─────────────────────────────────────────────
  const es = new EventSource('/api/chat/stream');
  es.onmessage = (e) => console.log(e.data);      // 默认事件
  es.addEventListener('tool', (e) => {{ ... }});    // 自定义事件
  es.onerror = (e) => {{ ... }};                    // 自动重连""")


# ═══════════════════════════════════════════════════════════
# 2. OpenAI 流式 API 的 SSE 格式
# ═══════════════════════════════════════════════════════════

def show_openai_sse():
    """展示 OpenAI API 的 SSE 流式格式。"""
    print("\n\n▶ 2. OpenAI 流式 API 的 SSE 格式")
    print("─" * 60)

    openai_chunks = [
        {"id": "chatcmpl-abc123", "object": "chat.completion.chunk", "model": "gpt-4o-mini",
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "我"}}]},
        {"choices": [{"delta": {"content": "来"}}]},
        {"choices": [{"delta": {"content": "帮"}}]},
        {"choices": [{"delta": {"content": "你"}}]},
        {"choices": [{"delta": {"content": "查"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_abc", "type": "function",
         "function": {"name": "lookup_ticket", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"tick'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'et_id": "T-001"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]

    print("\n  OpenAI stream=True 的 SSE 输出:")
    print("  ─────────────────────────────────────────────")
    for chunk in openai_chunks:
        chunk_json = json.dumps(chunk, ensure_ascii=False)
        if len(chunk_json) > 80:
            chunk_json = chunk_json[:77] + "..."
        print(f"  data: {chunk_json}")
    print("  data: [DONE]")
    print()
    print("  关键点:")
    print("  - 每个 chunk 是 JSON，包在 'data: ' 前缀中")
    print("  - 文本在 delta.content 中逐字到达")
    print("  - 工具调用在 delta.tool_calls 中逐段到达")
    print("  - data: [DONE] 标记流结束")
    print("  - finish_reason: 'stop'(文本) / 'tool_calls'(工具)")


# ═══════════════════════════════════════════════════════════
# 3. Agent SSE 事件协议设计
# ═══════════════════════════════════════════════════════════

@dataclass
class SSEEvent:
    """一个 SSE 事件。"""
    event: str = "message"
    data: str = ""
    id: str | None = None
    retry: int | None = None

    def encode(self) -> str:
        lines = []
        if self.event != "message":
            lines.append(f"event: {self.event}")
        if self.id:
            lines.append(f"id: {self.id}")
        if self.retry is not None:
            lines.append(f"retry: {self.retry}")
        for line in self.data.split("\n"):
            lines.append(f"data: {line}")
        lines.append("")
        return "\n".join(lines) + "\n"


class AgentSSEProtocol:
    """Agent SSE 多事件类型协议。"""

    @staticmethod
    async def generate_stream(user_input: str) -> AsyncGenerator[SSEEvent, None]:
        msg_id = 0

        def next_id():
            nonlocal msg_id
            msg_id += 1
            return f"evt-{msg_id:04d}"

        yield SSEEvent(event="thinking",
                       data=json.dumps({"message": "分析用户问题..."}, ensure_ascii=False), id=next_id())
        await asyncio.sleep(0.05)

        for token in list("工单 T-001 是技术问题，"):
            yield SSEEvent(event="token",
                           data=json.dumps({"token": token}, ensure_ascii=False), id=next_id())
            await asyncio.sleep(0.02)

        yield SSEEvent(event="tool_start",
                       data=json.dumps({"tool": "lookup_ticket", "args": {"ticket_id": "T-001"}}, ensure_ascii=False),
                       id=next_id())
        await asyncio.sleep(0.1)

        yield SSEEvent(event="tool_end",
                       data=json.dumps({"tool": "lookup_ticket",
                                        "result": {"status": "open", "type": "technical"},
                                        "duration_ms": 95}, ensure_ascii=False), id=next_id())
        await asyncio.sleep(0.05)

        for token in list("已路由到 engineering 团队。"):
            yield SSEEvent(event="token",
                           data=json.dumps({"token": token}, ensure_ascii=False), id=next_id())
            await asyncio.sleep(0.02)

        yield SSEEvent(event="done",
                       data=json.dumps({"summary": "工单 T-001 已路由到 engineering",
                                        "tokens_used": 420, "tools_called": ["lookup_ticket"]},
                                       ensure_ascii=False), id=next_id())


async def demo_sse_stream():
    """演示 Agent SSE 事件流。"""
    print("\n\n▶ 3. Agent SSE 多事件类型协议")
    print("─" * 60)
    print("\n  模拟 SSE 事件流:\n")

    collected_text = ""
    event_count = 0
    async for sse_event in AgentSSEProtocol.generate_stream("查工单 T-001"):
        event_count += 1
        encoded = sse_event.encode()
        if sse_event.event == "token":
            token_data = json.loads(sse_event.data)
            collected_text += token_data["token"]
            if event_count <= 3 or token_data["token"] in ("，", "。"):
                print(f"  {encoded.strip()}")
        else:
            print(f"  {encoded.strip()}")

    print(f"\n  收到 {event_count} 个 SSE 事件")
    print(f"  拼接后的完整回复: {collected_text}")


# ═══════════════════════════════════════════════════════════
# 4. SSE 客户端解析器
# ═══════════════════════════════════════════════════════════

class SSEParser:
    """SSE 客户端解析器（模拟浏览器 EventSource 行为）。"""

    def __init__(self):
        self.last_event_id: str | None = None
        self.retry_ms: int = 3000

    def parse(self, raw_stream: str) -> list[dict]:
        events = []
        current = {"event": "message", "data": [], "id": None, "retry": None}
        for line in raw_stream.split("\n"):
            if line.startswith("event: "):
                current["event"] = line[7:]
            elif line.startswith("data: "):
                current["data"].append(line[6:])
            elif line.startswith("id: "):
                current["id"] = line[4:]
            elif line.startswith("retry: "):
                current["retry"] = int(line[7:])
                self.retry_ms = current["retry"]
            elif line == "":
                if current["data"]:
                    event = {"event": current["event"], "data": "\n".join(current["data"])}
                    if current["id"]:
                        event["id"] = current["id"]
                        self.last_event_id = current["id"]
                    events.append(event)
                current = {"event": "message", "data": [], "id": None, "retry": None}
        return events


async def demo_sse_parser():
    """演示 SSE 客户端解析器。"""
    print("\n\n▶ 4. SSE 客户端解析器")
    print("─" * 60)

    raw_parts = []
    async for sse_event in AgentSSEProtocol.generate_stream("查工单"):
        raw_parts.append(sse_event.encode())
    raw_stream = "".join(raw_parts)

    parser = SSEParser()
    events = parser.parse(raw_stream)

    print(f"\n  原始 SSE 流: {len(raw_stream)} 字节 → {len(events)} 个事件")
    print(f"  Last-Event-ID: {parser.last_event_id}")

    type_counts: dict[str, int] = {}
    for evt in events:
        t = evt["event"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  事件分布: {dict(sorted(type_counts.items()))}")


# ═══════════════════════════════════════════════════════════
# 5. SSE 生产环境注意事项
# ═══════════════════════════════════════════════════════════

def show_sse_production():
    """SSE 生产环境陷阱和应对。"""
    print("\n\n▶ 5. SSE 生产环境注意事项")
    print("─" * 60)

    print(f"""
  ┌─ 陷阱 1: 代理/CDN 缓冲 ─────────────────────────────────┐
  │  Nginx/CloudFlare 默认缓冲响应 → SSE 不实时              │
  │  修复: proxy_buffering off; X-Accel-Buffering: no        │
  └──────────────────────────────────────────────────────────┘

  ┌─ 陷阱 2: 连接超时 ──────────────────────────────────────┐
  │  负载均衡器 60s 超时 → 长时间无 token 断连                │
  │  修复: 定期发 SSE 心跳注释行  : heartbeat\\n\\n            │
  └──────────────────────────────────────────────────────────┘

  ┌─ 陷阱 3: 浏览器连接数限制 ──────────────────────────────┐
  │  HTTP/1.1 每域名最多 6 个并发 → SSE 占一个                │
  │  修复: 使用 HTTP/2 多路复用                               │
  └──────────────────────────────────────────────────────────┘

  ┌─ 陷阱 4: EventSource 只支持 GET ────────────────────────┐
  │  Agent 需要 POST 发 messages → EventSource 不行          │
  │  修复: fetch + ReadableStream（主流） / @microsoft/fetch-event-source │
  └──────────────────────────────────────────────────────────┘

  ┌─ 陷阱 5: 服务端资源泄漏 ────────────────────────────────┐
  │  客户端断开后 generator 继续运行                          │
  │  修复: 捕获 asyncio.CancelledError 清理资源              │
  └──────────────────────────────────────────────────────────┘

  断线重连:
  ─────────────────────────────────────────────────────
  1. 浏览器 EventSource 自动重连（默认 3s）
  2. 重连时带 Last-Event-ID 请求头
  3. 服务端从该 ID 之后恢复推送
  4. 指数退避: 1s → 2s → 4s → 8s → max 30s""")


# ═══════════════════════════════════════════════════════════
# 6. SSE 服务端实现（各框架）
# ═══════════════════════════════════════════════════════════

def show_sse_implementations():
    """各框架 SSE 实现。"""
    print("\n\n▶ 6. SSE 服务端实现（各框架）")
    print("─" * 60)

    print("""
  ┌─ FastAPI (Python) ──────────────────────────────────────┐
  │  @app.post("/api/chat")                                 │
  │  async def chat(request: ChatRequest):                  │
  │      async def generate():                              │
  │          async for chunk in agent.stream(request):      │
  │              yield f"data: {chunk}\\n\\n"                 │
  │          yield "data: [DONE]\\n\\n"                       │
  │      return StreamingResponse(generate(),               │
  │          media_type="text/event-stream")                 │
  └─────────────────────────────────────────────────────────┘

  ┌─ Spring AI (Java) ─────────────────────────────────────┐
  │  @PostMapping(value="/chat/stream",                     │
  │      produces=MediaType.TEXT_EVENT_STREAM_VALUE)         │
  │  public Flux<String> chatStream(@RequestBody req) {     │
  │      return chatClient.prompt().user(req).stream()      │
  │          .content();                                     │
  │  }                                                      │
  └─────────────────────────────────────────────────────────┘

  ┌─ Vercel AI SDK (TypeScript) ───────────────────────────┐
  │  export async function POST(req: Request) {             │
  │    const result = streamText({ model, messages, tools })│
  │    return result.toDataStreamResponse(); // 自动 SSE    │
  │  }                                                      │
  └─────────────────────────────────────────────────────────┘

  各框架的通信选择:
  ──────────────────┬────────────────────────────────
  OpenAI / Claude   │ HTTP SSE（stream=True）
  Vercel AI SDK     │ HTTP SSE（streamText / useChat）
  Spring AI         │ WebFlux SSE（Reactor Flux）
  LangGraph Cloud   │ HTTP SSE + WebSocket
  Dify              │ HTTP SSE（API）/ WebSocket（Web UI）""")


# ═══════════════════════════════════════════════════════════
# 7. MCP Streamable HTTP
# ═══════════════════════════════════════════════════════════

def show_mcp_streamable():
    """MCP Streamable HTTP 传输。"""
    print("\n\n▶ 7. MCP Streamable HTTP（新一代 MCP 传输）")
    print("─" * 60)

    print(f"""
  MCP 传输层演进:
  ─────────────────────────────────────────────────────
  stdio          │ 子进程通信（本地开发）
  HTTP + SSE     │ 旧版远程传输（已弃用）
  Streamable HTTP│ 新标准（2025）— POST 请求 + SSE 响应

  工作方式:
  1. 客户端 POST JSON-RPC body → 服务端返回 SSE 流
  2. 同一个端点处理所有 MCP 方法
  3. 比旧方案更简单（一个端点 vs 两个端点）""")


# ╔═══════════════════════════════════════════════════════════╗
# ║  Part B: WebSocket（双向交互 Agent）                       ║
# ╚═══════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════
# 8. WebSocket 消息协议
# ═══════════════════════════════════════════════════════════

class MessageType(Enum):
    """Agent WebSocket 消息类型。"""
    USER_MESSAGE = "user_message"
    USER_INTERRUPT = "user_interrupt"
    USER_APPROVAL = "user_approval"
    PING = "ping"
    TOKEN = "token"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    AGENT_THINKING = "agent_thinking"
    APPROVAL_REQUEST = "approval_request"
    TASK_COMPLETE = "task_complete"
    ERROR = "error"
    PONG = "pong"


@dataclass
class WSMessage:
    """WebSocket 消息。"""
    type: str
    data: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({"type": self.type, "data": self.data,
                           "id": self.id, "timestamp": self.timestamp}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════
# 9. WebSocket Agent 交互演示
# ═══════════════════════════════════════════════════════════

class WebSocketAgentServer:
    """模拟 WebSocket Agent 服务端。"""

    async def handle_message(self, user_msg: str) -> AsyncGenerator[WSMessage, None]:
        yield WSMessage(type=MessageType.AGENT_THINKING.value,
                        data={"message": "正在分析您的问题..."})
        await asyncio.sleep(0.1)

        for i, char in enumerate("我来帮您查一下工单状态。"):
            yield WSMessage(type=MessageType.TOKEN.value, data={"token": char, "index": i})
            await asyncio.sleep(0.02)

        yield WSMessage(type=MessageType.TOOL_CALL_START.value,
                        data={"tool": "lookup_ticket", "args": {"ticket_id": "T-001"},
                              "message": "正在查询工单 T-001..."})
        await asyncio.sleep(0.3)

        yield WSMessage(type=MessageType.TOOL_CALL_END.value,
                        data={"tool": "lookup_ticket",
                              "result": {"ticket_id": "T-001", "type": "technical", "status": "open"},
                              "duration_ms": 280})
        await asyncio.sleep(0.1)

        yield WSMessage(type=MessageType.APPROVAL_REQUEST.value,
                        data={"action": "escalate_ticket",
                              "args": {"ticket_id": "T-001", "team": "engineering"},
                              "message": "是否确认将工单路由到 engineering 团队？",
                              "options": ["approve", "reject"]})
        await asyncio.sleep(0.2)

        for i, char in enumerate("已完成！工单已路由到 engineering 团队。"):
            yield WSMessage(type=MessageType.TOKEN.value, data={"token": char, "index": i})
            await asyncio.sleep(0.02)

        yield WSMessage(type=MessageType.TASK_COMPLETE.value,
                        data={"summary": "工单 T-001 已路由", "tools_called": ["lookup_ticket"],
                              "total_tokens": 850})


class WebSocketAgentClient:
    """模拟 WebSocket Agent 客户端。"""

    def __init__(self):
        self.received: list[WSMessage] = []
        self.current_text = ""

    async def connect_and_chat(self, server: WebSocketAgentServer, user_input: str):
        print(f"  📤 用户: {user_input}\n")
        async for msg in server.handle_message(user_input):
            self.received.append(msg)
            self._render(msg)

    def _render(self, msg: WSMessage):
        mt = msg.type
        if mt == MessageType.AGENT_THINKING.value:
            print(f"  💭 {msg.data['message']}")
        elif mt == MessageType.TOKEN.value:
            self.current_text += msg.data["token"]
            print(f"\r  🤖 {self.current_text}", end="", flush=True)
        elif mt == MessageType.TOOL_CALL_START.value:
            print(f"\n  🔧 [{msg.data['tool']}] {msg.data['message']}")
        elif mt == MessageType.TOOL_CALL_END.value:
            result = json.dumps(msg.data["result"], ensure_ascii=False)[:60]
            print(f"  ✅ [{msg.data['tool']}] 完成 ({msg.data['duration_ms']}ms) → {result}")
        elif mt == MessageType.APPROVAL_REQUEST.value:
            print(f"\n  ⏸️  [审批请求] {msg.data['message']}")
            print(f"     ✅ 自动批准（模拟）")
        elif mt == MessageType.TASK_COMPLETE.value:
            print(f"\n  ✅ 任务完成: {msg.data['summary']}")


# ═══════════════════════════════════════════════════════════
# 10. WebSocket 架构
# ═══════════════════════════════════════════════════════════

def show_ws_architecture():
    """WebSocket Agent 架构。"""
    print(f"""
  WebSocket Agent 架构:
  ┌──────────────┐   WebSocket    ┌──────────────────────┐
  │   前端 UI     │◄─────────────►│   Agent Gateway      │
  │ - 逐字显示    │   全双工       │ - 认证/限流/会话管理   │
  │ - 工具进度    │               └─────────┬────────────┘
  │ - 审批交互    │                         │
  └──────────────┘               ┌─────────▼────────────┐
                                 │   Agent + LLM + 工具  │
                                 └──────────────────────┘

  协议设计:
  ─────────────────────────────────────────────────
  1. 统一消息格式: type + data + id + timestamp
  2. 心跳: ping/pong 每 30s，3 次无响应判断断连
  3. 重连: 指数退避 1s → 2s → 4s → max 30s
  4. 背压: 客户端处理不过来时的流控

  技术栈:
  ─────────────────────────────────────────────────
  Python:   FastAPI WebSocket / websockets / Socket.IO
  Java:     Spring WebSocket / Netty
  Node.js:  ws / Socket.IO
  网关:     Nginx proxy_pass / Kong / Traefik""")


# ═══════════════════════════════════════════════════════════
# 11. SSE vs WebSocket 选型指南
# ═══════════════════════════════════════════════════════════

def show_selection_guide():
    """SSE vs WebSocket 选型。"""
    print("\n\n▶ 11. SSE vs WebSocket 选型指南")
    print("─" * 60)

    print(f"""
  场景                      │ 推荐方式    │ 原因
  ─────────────────────────┼────────────┼──────────────────
  纯流式 LLM 输出           │ SSE        │ 单向够用，更简单
  工具调用进度推送           │ SSE        │ event 字段区分类型
  需要用户中途中断           │ WebSocket  │ 需要客户端→服务端
  Human-in-the-loop 审批    │ WebSocket  │ 双向交互
  多 Agent 实时广播          │ WebSocket  │ 多方通信
  MCP 远程工具调用           │ SSE        │ Streamable HTTP

  为什么 SSE 是 Agent 流式输出首选:
  ─────────────────────────────────────────────────────
  ✅ HTTP 原生 — 任何基础设施直接支持
  ✅ 浏览器 EventSource 内置断线重连
  ✅ CDN/反向代理/负载均衡器兼容
  ✅ OpenAI/Claude/Gemini 都用 SSE
  ✅ 90% Agent 场景只需单向推送

  什么时候必须用 WebSocket:
  ─────────────────────────────────────────────────────
  🔄 双向交互（用户中断/追问/审批）
  🔄 多 Agent 实时状态广播
  🔄 长任务心跳保活（>30s）
  🔄 实时协作编辑场景""")


# ═══════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════

async def async_main():
    print("=== Agent 通信方式 — SSE + WebSocket ===\n")

    # Part A: SSE
    print("=" * 60)
    print("Part A: SSE (Server-Sent Events)")
    print("=" * 60)

    show_sse_basics()
    show_openai_sse()
    await demo_sse_stream()
    await demo_sse_parser()
    show_sse_production()
    show_sse_implementations()
    show_mcp_streamable()

    # Part B: WebSocket
    print("\n\n" + "=" * 60)
    print("Part B: WebSocket（双向交互）")
    print("=" * 60)

    print("\n\n▶ 8~9. WebSocket Agent 交互演示")
    print("─" * 60)

    server = WebSocketAgentServer()
    client = WebSocketAgentClient()
    await client.connect_and_chat(server, "帮我查工单 T-001，如果是技术问题路由到 engineering")

    msg_counts: dict[str, int] = {}
    for msg in client.received:
        msg_counts[msg.type] = msg_counts.get(msg.type, 0) + 1
    print(f"\n  WebSocket 消息统计:")
    for mt, count in sorted(msg_counts.items()):
        print(f"    {mt:25s} × {count}")
    print(f"    总计: {len(client.received)} 条消息")

    print(f"\n\n▶ 10. WebSocket 架构")
    print("─" * 60)
    show_ws_architecture()

    # 选型指南
    show_selection_guide()

    # 总结
    print("\n\n" + "=" * 60)
    print("📊 Agent 通信方式总结:")
    print()
    print("  方式       │ 定位           │ 选型")
    print("  ──────────┼───────────────┼──────────────────")
    print("  SSE       │ 单向流式推送    │ 90% 场景首选")
    print("  WebSocket │ 双向实时交互    │ HITL / 多 Agent")
    print()
    print("  生产 Checklist:")
    print("  ────────────────────────────────────────────")
    print("  □ SSE: 关 Nginx 缓冲 + 心跳 + HTTP/2")
    print("  □ WS:  心跳 + 重连 + 消息 ID + 背压")
    print("  □ 消息格式: type + data + id + timestamp")
    print("  □ 重连: 指数退避 + Last-Event-ID 恢复")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
