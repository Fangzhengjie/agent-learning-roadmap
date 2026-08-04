"""纯 Python MCP 模拟 — 不依赖 mcp 库，手写 JSON-RPC 协议

MCP 协议核心:
  - 传输层: JSON-RPC 2.0 over stdio / SSE / HTTP
  - 三种能力: tools (工具) / resources (资源) / prompts (提示模板)
  - 生命周期: initialize → list_tools → call_tool → ... → shutdown

本文件可独立运行: python 07_mcp_tools/mcp_pure.py
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


# ═══════════════════════════════════════════════════════════
# JSON-RPC 2.0 基础
# ═══════════════════════════════════════════════════════════

@dataclass
class JsonRpcRequest:
    method: str
    params: dict = field(default_factory=dict)
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]

    def to_dict(self) -> dict:
        return {"jsonrpc": "2.0", "method": self.method, "params": self.params, "id": self.id}


@dataclass
class JsonRpcResponse:
    id: str
    result: Any = None
    error: dict | None = None

    def to_dict(self) -> dict:
        d = {"jsonrpc": "2.0", "id": self.id}
        if self.error:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d


# ═══════════════════════════════════════════════════════════
# MCP Server（纯 Python 实现）
# ═══════════════════════════════════════════════════════════

@dataclass
class MCPToolDef:
    name: str
    description: str
    input_schema: dict
    fn: Callable


@dataclass
class MCPResource:
    uri: str
    name: str
    description: str
    mime_type: str
    content_fn: Callable[[], str]


@dataclass
class MCPPrompt:
    name: str
    description: str
    arguments: list[dict]
    template: str


class MCPServer:
    """纯 Python MCP Server — 模拟 MCP 协议的完整交互。"""

    SERVER_INFO = {
        "name": "enterprise-tools",
        "version": "1.0.0",
        "protocolVersion": "2024-11-05",
    }

    def __init__(self, name: str = "mcp-server"):
        self.name = name
        self._tools: dict[str, MCPToolDef] = {}
        self._resources: dict[str, MCPResource] = {}
        self._prompts: dict[str, MCPPrompt] = {}
        self._log: list[dict] = []

    def tool(self, name: str, description: str, input_schema: dict):
        """装饰器: 注册工具。"""
        def decorator(fn):
            self._tools[name] = MCPToolDef(name, description, input_schema, fn)
            return fn
        return decorator

    def resource(self, uri: str, name: str, description: str, mime_type: str = "text/plain"):
        def decorator(fn):
            self._resources[uri] = MCPResource(uri, name, description, mime_type, fn)
            return fn
        return decorator

    def prompt(self, name: str, description: str, arguments: list[dict], template: str):
        self._prompts[name] = MCPPrompt(name, description, arguments, template)

    def handle_request(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """处理 JSON-RPC 请求（模拟 MCP Server 的请求路由）。"""
        self._log.append({"method": request.method, "params": request.params})
        handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_list_tools,
            "tools/call": self._handle_call_tool,
            "resources/list": self._handle_list_resources,
            "resources/read": self._handle_read_resource,
            "prompts/list": self._handle_list_prompts,
            "prompts/get": self._handle_get_prompt,
        }
        handler = handlers.get(request.method)
        if not handler:
            return JsonRpcResponse(request.id, error={"code": -32601, "message": f"Unknown method: {request.method}"})
        try:
            result = handler(request.params)
            return JsonRpcResponse(request.id, result=result)
        except Exception as e:
            return JsonRpcResponse(request.id, error={"code": -32000, "message": str(e)})

    def _handle_initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": self.SERVER_INFO["protocolVersion"],
            "serverInfo": self.SERVER_INFO,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
        }

    def _handle_list_tools(self, params: dict) -> dict:
        return {"tools": [
            {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
            for t in self._tools.values()
        ]}

    def _handle_call_tool(self, params: dict) -> dict:
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        result = tool.fn(**arguments)
        return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}

    def _handle_list_resources(self, params: dict) -> dict:
        return {"resources": [
            {"uri": r.uri, "name": r.name, "description": r.description, "mimeType": r.mime_type}
            for r in self._resources.values()
        ]}

    def _handle_read_resource(self, params: dict) -> dict:
        uri = params.get("uri", "")
        res = self._resources.get(uri)
        if not res:
            raise ValueError(f"Resource not found: {uri}")
        return {"contents": [{"uri": uri, "mimeType": res.mime_type, "text": res.content_fn()}]}

    def _handle_list_prompts(self, params: dict) -> dict:
        return {"prompts": [
            {"name": p.name, "description": p.description, "arguments": p.arguments}
            for p in self._prompts.values()
        ]}

    def _handle_get_prompt(self, params: dict) -> dict:
        name = params.get("name", "")
        p = self._prompts.get(name)
        if not p:
            raise ValueError(f"Prompt not found: {name}")
        text = p.template
        for arg in p.arguments:
            key = arg["name"]
            text = text.replace(f"{{{key}}}", params.get("arguments", {}).get(key, f"<{key}>"))
        return {"messages": [{"role": "user", "content": {"type": "text", "text": text}}]}


# ═══════════════════════════════════════════════════════════
# MCP Client（纯 Python 实现）
# ═══════════════════════════════════════════════════════════

class MCPClient:
    """纯 Python MCP Client — 直接调用 Server（同进程模拟 stdio 传输）。"""

    def __init__(self, server: MCPServer):
        self.server = server
        self.initialized = False

    def _call(self, method: str, params: dict | None = None) -> Any:
        req = JsonRpcRequest(method=method, params=params or {})
        resp = self.server.handle_request(req)
        if resp.error:
            raise RuntimeError(f"MCP Error: {resp.error}")
        return resp.result

    def initialize(self) -> dict:
        result = self._call("initialize", {"clientInfo": {"name": "demo-client", "version": "1.0"}})
        self.initialized = True
        return result

    def list_tools(self) -> list[dict]:
        return self._call("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._call("tools/call", {"name": name, "arguments": arguments})
        contents = result.get("content", [])
        return contents[0]["text"] if contents else ""

    def list_resources(self) -> list[dict]:
        return self._call("resources/list").get("resources", [])

    def read_resource(self, uri: str) -> str:
        result = self._call("resources/read", {"uri": uri})
        contents = result.get("contents", [])
        return contents[0]["text"] if contents else ""

    def list_prompts(self) -> list[dict]:
        return self._call("prompts/list").get("prompts", [])

    def get_prompt(self, name: str, arguments: dict | None = None) -> str:
        result = self._call("prompts/get", {"name": name, "arguments": arguments or {}})
        messages = result.get("messages", [])
        return messages[0]["content"]["text"] if messages else ""


# ═══════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════

def build_demo_server() -> MCPServer:
    """构建演示用 MCP Server。"""
    server = MCPServer("enterprise-tools")

    # 注册工具
    @server.tool("query_ticket", "查询工单详情", {
        "type": "object",
        "properties": {"ticket_id": {"type": "string", "description": "工单编号"}},
        "required": ["ticket_id"],
    })
    def query_ticket(ticket_id: str) -> dict:
        db = {"TK-001": {"status": "处理中", "assignee": "张三", "priority": "高"}}
        return db.get(ticket_id, {"error": f"工单 {ticket_id} 不存在"})

    @server.tool("system_status", "查询系统状态", {
        "type": "object", "properties": {}, "required": [],
    })
    def system_status() -> dict:
        return {"cpu": "45%", "memory": "62%", "disk": "38%", "status": "healthy"}

    @server.tool("search_kb", "搜索知识库", {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["query"],
    })
    def search_kb(query: str, limit: int = 3) -> list:
        docs = [
            {"title": "密码重置流程", "relevance": 0.95},
            {"title": "VPN 配置指南", "relevance": 0.82},
            {"title": "报销制度说明", "relevance": 0.71},
        ]
        return [d for d in docs if query.lower() in d["title"].lower() or True][:limit]

    # 注册资源
    @server.resource("config://app/settings", "应用配置", "当前应用运行配置", "application/json")
    def app_settings():
        return json.dumps({"env": "production", "version": "2.1.0", "features": ["rag", "agent"]})

    # 注册提示模板
    server.prompt("ticket_analysis", "工单分析提示", [
        {"name": "ticket_id", "description": "工单编号", "required": True},
    ], "请分析工单 {ticket_id} 的问题原因，并提出解决方案。")

    return server


def main():
    print("=== MCP 纯 Python 模拟（不依赖 mcp 库） ===\n")

    server = build_demo_server()
    client = MCPClient(server)

    # 1. 初始化
    print("▶ 1. 初始化握手 (initialize)")
    print("─" * 60)
    info = client.initialize()
    print(f"  协议版本: {info['protocolVersion']}")
    print(f"  服务信息: {info['serverInfo']}")
    print(f"  能力: {list(info['capabilities'].keys())}")

    # 2. 列出工具
    print(f"\n\n▶ 2. 列出工具 (tools/list)")
    print("─" * 60)
    tools = client.list_tools()
    for t in tools:
        params = list(t["inputSchema"].get("properties", {}).keys())
        print(f"  🔧 {t['name']}: {t['description']} (params: {params})")

    # 3. 调用工具
    print(f"\n\n▶ 3. 调用工具 (tools/call)")
    print("─" * 60)
    calls = [
        ("query_ticket", {"ticket_id": "TK-001"}),
        ("system_status", {}),
        ("search_kb", {"query": "密码", "limit": 2}),
    ]
    for name, args in calls:
        result = client.call_tool(name, args)
        print(f"  {name}({args}) → {result}")

    # 4. 读取资源
    print(f"\n\n▶ 4. 读取资源 (resources/read)")
    print("─" * 60)
    resources = client.list_resources()
    for r in resources:
        content = client.read_resource(r["uri"])
        print(f"  📄 {r['uri']}: {content[:60]}")

    # 5. 获取提示
    print(f"\n\n▶ 5. 获取提示模板 (prompts/get)")
    print("─" * 60)
    prompts = client.list_prompts()
    for p in prompts:
        text = client.get_prompt(p["name"], {"ticket_id": "TK-001"})
        print(f"  📝 {p['name']}: {text}")

    # 6. JSON-RPC 协议细节
    print(f"\n\n▶ 6. JSON-RPC 协议细节")
    print("─" * 60)
    req = JsonRpcRequest(method="tools/call", params={"name": "system_status", "arguments": {}})
    resp = server.handle_request(req)
    print(f"  请求: {json.dumps(req.to_dict(), ensure_ascii=False)}")
    print(f"  响应: {json.dumps(resp.to_dict(), ensure_ascii=False)}")

    # 错误处理
    bad_req = JsonRpcRequest(method="unknown/method")
    bad_resp = server.handle_request(bad_req)
    print(f"\n  错误请求: {json.dumps(bad_req.to_dict(), ensure_ascii=False)}")
    print(f"  错误响应: {json.dumps(bad_resp.to_dict(), ensure_ascii=False)}")

    print(f"\n  Server 日志 ({len(server._log)} 次调用):")
    for log in server._log[:5]:
        print(f"    → {log['method']}")

    print(f"\n\n{'=' * 60}")
    print("📊 MCP 协议总结:")
    print(f"""
  协议层           │ 说明
  ────────────────┼──────────────────────────────
  传输层           │ stdio / SSE / Streamable HTTP
  消息格式         │ JSON-RPC 2.0
  能力发现         │ initialize → capabilities
  工具调用         │ tools/list → tools/call
  资源访问         │ resources/list → resources/read
  提示模板         │ prompts/list → prompts/get

  本文件 vs main.py:
  - main.py:      使用 mcp 库（需要 pip install mcp）
  - mcp_pure.py:  纯 Python 手写（零依赖，可直接运行）""")


if __name__ == "__main__":
    main()
