"""MCP (Model Context Protocol) 工具服务器 + 客户端

核心概念：Agent ↔ 工具的标准通信协议。

MCP 解决什么问题：
  - 以前：每个 Agent 框架各自定义工具格式，工具不可复用
  - 现在：MCP 定义统一协议，一个工具服务器可以被任何 Agent 调用
  - 类比：MCP 之于 Agent 工具 = USB 之于外设

MCP 架构：
  ┌──────────┐   MCP 协议    ┌────────────────┐
  │  Agent   │ ◄───────────► │  MCP Server    │
  │ (客户端) │   JSON-RPC    │ (工具/资源/提示) │
  └──────────┘               └────────────────┘

本示例包含：
  1. MCP Server：暴露企业工具（工单查询、系统状态）
  2. MCP Client：连接 Server 并调用工具
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# Part 1: MCP Server（工具提供方）
# ═══════════════════════════════════════════════════════════

# 在真实场景中，MCP Server 是一个独立进程
# 这里用 mcp 库的 FastMCP 创建服务器

def create_mcp_server():
    """创建 MCP 工具服务器。

    核心概念：
    - @mcp.tool(): 注册工具（Agent 可调用的函数）
    - @mcp.resource(): 注册资源（Agent 可读取的数据）
    - @mcp.prompt(): 注册提示模板
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("enterprise-tools", version="1.0.0")

    # ── 工具：Agent 可调用的函数 ──────────────────────────

    @mcp.tool()
    def lookup_ticket(ticket_id: str) -> str:
        """查询工单详情。

        Args:
            ticket_id: 工单编号，如 T-001
        """
        tickets = {
            "T-001": {"user": "alice", "type": "technical", "issue": "登录后页面白屏", "status": "open"},
            "T-002": {"user": "bob", "type": "billing", "issue": "订阅扣费但功能无法使用", "amount": 99.0},
        }
        ticket = tickets.get(ticket_id)
        if not ticket:
            return json.dumps({"error": f"工单 {ticket_id} 不存在"}, ensure_ascii=False)
        return json.dumps({"ticket_id": ticket_id, **ticket}, ensure_ascii=False)

    @mcp.tool()
    def check_system_status(service: str) -> str:
        """检查后端服务状态。

        Args:
            service: 服务名称（auth/payment/api/web）
        """
        statuses = {"auth": "healthy", "payment": "degraded", "api": "healthy", "web": "healthy"}
        return json.dumps({"service": service, "status": statuses.get(service, "unknown")}, ensure_ascii=False)

    @mcp.tool()
    def route_ticket(ticket_id: str, team: str, reason: str) -> str:
        """将工单路由到指定团队。

        Args:
            ticket_id: 工单编号
            team: 目标团队（engineering/billing/support）
            reason: 路由原因
        """
        return json.dumps({
            "ticket_id": ticket_id, "routed_to": team,
            "reason": reason, "status": "routed"
        }, ensure_ascii=False)

    # ── 资源：Agent 可读取的数据 ──────────────────────────

    @mcp.resource("config://routing-rules")
    def get_routing_rules() -> str:
        """获取工单路由规则配置。"""
        rules = {
            "technical": "engineering",
            "billing": "billing",
            "general": "support",
            "escalation_threshold": "P0/P1 自动升级",
        }
        return json.dumps(rules, ensure_ascii=False)

    # ── 提示模板：预定义的 prompt 片段 ────────────────────

    @mcp.prompt()
    def triage_prompt(ticket_id: str) -> str:
        """生成工单分诊提示。"""
        return f"""请分析工单 {ticket_id}：
1. 查询工单详情
2. 根据路由规则判断目标团队
3. 如果是技术问题，检查相关系统状态
4. 执行路由"""

    return mcp


# ═══════════════════════════════════════════════════════════
# Part 2: MCP Client（工具消费方）
# ═══════════════════════════════════════════════════════════

async def demo_mcp_client():
    """演示 MCP 客户端如何连接服务器并调用工具。

    在真实场景中：
    - Server 运行在独立进程/容器中
    - Client 通过 stdio/SSE/HTTP 连接
    - Agent 框架（LangChain/pydantic-ai/Claude）内置 MCP 客户端
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    print("=== MCP 工具服务器 + 客户端 ===\n")

    # 连接到 MCP Server（通过 stdio 传输）
    # 实际部署中也可以用 SSE/HTTP 传输
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[__file__, "--server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ── 列出可用工具 ──────────────────────────────
            tools = await session.list_tools()
            print("📋 可用工具:")
            for tool in tools.tools:
                print(f"  🔧 {tool.name}: {tool.description}")

            # ── 列出可用资源 ──────────────────────────────
            resources = await session.list_resources()
            print(f"\n📦 可用资源:")
            for res in resources.resources:
                print(f"  📄 {res.uri}: {res.name}")

            # ── 列出可用提示模板 ──────────────────────────
            prompts = await session.list_prompts()
            print(f"\n💬 可用提示模板:")
            for p in prompts.prompts:
                print(f"  📝 {p.name}: {p.description}")

            # ── 调用工具 ──────────────────────────────────
            print(f"\n{'─' * 60}")
            print("▶ 调用工具演示\n")

            # 1. 查询工单
            result = await session.call_tool("lookup_ticket", {"ticket_id": "T-001"})
            print(f"  🔧 lookup_ticket('T-001'):")
            print(f"     → {result.content[0].text}\n")

            # 2. 检查系统状态
            result = await session.call_tool("check_system_status", {"service": "payment"})
            print(f"  🔧 check_system_status('payment'):")
            print(f"     → {result.content[0].text}\n")

            # 3. 路由工单
            result = await session.call_tool("route_ticket", {
                "ticket_id": "T-001",
                "team": "engineering",
                "reason": "技术问题：页面白屏",
            })
            print(f"  🔧 route_ticket('T-001', 'engineering'):")
            print(f"     → {result.content[0].text}\n")

            # ── 读取资源 ──────────────────────────────────
            resource = await session.read_resource("config://routing-rules")
            print(f"  📄 读取路由规则:")
            print(f"     → {resource.contents[0].text}\n")

    # ── 架构观察 ──────────────────────────────────────────
    print("=" * 60)
    print("📊 MCP 架构观察:")
    print()
    print("  ✅ 统一协议：一个 MCP Server 可被任何 Agent 框架调用")
    print("  ✅ 三种能力：工具（tool）+ 资源（resource）+ 提示（prompt）")
    print("  ✅ 多种传输：stdio / SSE / HTTP（适配不同部署场景）")
    print("  ✅ 安全隔离：工具运行在独立进程，Agent 只能通过协议调用")
    print("  ✅ 类比 USB：写一次工具，所有 Agent 都能用")
    print()
    print("  MCP vs 直接 function_tool:")
    print("  ─────────────────────────")
    print("  @function_tool    →  工具和 Agent 在同一进程（耦合）")
    print("  MCP Server        →  工具独立部署，Agent 远程调用（解耦）")
    print("  @function_tool    →  只能被一个框架使用")
    print("  MCP Server        →  LangChain/Claude/pydantic-ai 都能调用")


# ── 入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    if "--server" in sys.argv:
        # 作为 MCP Server 运行（被客户端 stdio 启动）
        mcp_server = create_mcp_server()
        mcp_server.run(transport="stdio")
    else:
        # 作为客户端运行
        asyncio.run(demo_mcp_client())
