"""Function Calling 原理 — LLM 结构化工具调用的底层机制

核心流程：
  1. 定义工具 → 转成 JSON Schema → 注入 system prompt / tools 参数
  2. LLM 输出 tool_calls JSON → 框架解析 → 执行函数 → 结果喂回
  3. LLM 基于结果继续生成

本模块展示：
  - 手写 Function Calling 的完整流程（不依赖 OpenAI SDK）
  - 工具定义 → Schema 生成 → 模拟 LLM 调用 → 结果注入
"""

import json
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints


# ═══════════════════════════════════════════════════════════
# 工具定义与注册
# ═══════════════════════════════════════════════════════════

_TYPE_MAP = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}


@dataclass
class ToolParam:
    name: str
    type: str
    description: str = ""
    required: bool = True
    enum: list[str] | None = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: list[ToolParam] = field(default_factory=list)
    fn: Callable | None = None

    def to_openai_schema(self) -> dict:
        """转成 OpenAI Function Calling 格式。"""
        props = {}
        required = []
        for p in self.parameters:
            prop = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            props[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    """工具注册表 — 管理所有可用工具。"""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, fn: Callable, description: str = "",
                 param_descriptions: dict[str, str] | None = None) -> ToolDefinition:
        """从函数签名自动生成工具定义。"""
        hints = get_type_hints(fn)
        sig = inspect.signature(fn)
        params = []
        for name, param in sig.parameters.items():
            type_hint = hints.get(name, str)
            type_str = _TYPE_MAP.get(type_hint.__name__, "string")
            desc = (param_descriptions or {}).get(name, "")
            required = param.default is inspect.Parameter.empty
            params.append(ToolParam(name=name, type=type_str, description=desc, required=required))

        tool_def = ToolDefinition(
            name=fn.__name__,
            description=description or fn.__doc__ or "",
            parameters=params,
            fn=fn,
        )
        self._tools[fn.__name__] = tool_def
        return tool_def

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def to_openai_tools(self) -> list[dict]:
        """生成 OpenAI API 的 tools 参数。"""
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> Any:
        """执行工具调用。"""
        tool = self._tools.get(name)
        if not tool or not tool.fn:
            raise ValueError(f"Unknown tool: {name}")
        return tool.fn(**arguments)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())


# ═══════════════════════════════════════════════════════════
# 模拟 Function Calling 流程
# ═══════════════════════════════════════════════════════════

@dataclass
class ToolCall:
    """模拟 OpenAI 的 tool_calls 响应。"""
    id: str
    function_name: str
    arguments: dict


class FunctionCallingSimulator:
    """模拟完整的 Function Calling 流程。

    真实流程:
      User → LLM(tools=[...]) → tool_calls → 执行 → 结果注入 → LLM → 最终回答

    本模拟:
      User → 规则匹配决定调用哪个工具 → 执行 → 构建 messages → 生成回答
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def decide_tool_call(self, user_message: str) -> ToolCall | None:
        """模拟 LLM 决定是否调用工具（规则匹配代替 LLM）。"""
        msg = user_message.lower()
        for name in self.registry.names:
            if name.replace("_", " ") in msg or name.replace("_", "") in msg:
                return ToolCall(id=f"call_{name}_1", function_name=name, arguments={})
        return None

    def build_messages(self, user_msg: str, tool_call: ToolCall | None,
                       tool_result: Any = None) -> list[dict]:
        """构建完整的 messages 列表（OpenAI 格式）。"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": user_msg},
        ]
        if tool_call:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function_name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, ensure_ascii=False) if tool_result else "{}",
            })
        return messages

    def run(self, user_message: str, tool_args: dict | None = None) -> dict:
        """执行完整的 Function Calling 流程。"""
        tool_call = self.decide_tool_call(user_message)
        tool_result = None

        if tool_call:
            if tool_args:
                tool_call.arguments = tool_args
            try:
                tool_result = self.registry.execute(tool_call.function_name, tool_call.arguments)
            except Exception as e:
                tool_result = {"error": str(e)}

        messages = self.build_messages(user_message, tool_call, tool_result)

        return {
            "tool_called": tool_call.function_name if tool_call else None,
            "tool_args": tool_call.arguments if tool_call else None,
            "tool_result": tool_result,
            "messages": messages,
            "messages_count": len(messages),
        }
