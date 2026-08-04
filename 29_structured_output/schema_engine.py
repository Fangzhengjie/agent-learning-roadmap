"""JSON Schema 约束引擎

核心能力：
  1. 从 Python dataclass / dict 生成 JSON Schema
  2. 验证 LLM 输出是否符合 Schema
  3. 自动修复常见的 JSON 格式错误
"""

import json
import re
from dataclasses import dataclass, field, fields, MISSING
from typing import Any, get_type_hints


# ═══════════════════════════════════════════════════════════
# JSON Schema 生成器
# ═══════════════════════════════════════════════════════════

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class SchemaGenerator:
    """从 Python 类型生成 JSON Schema。"""

    @staticmethod
    def from_dataclass(cls) -> dict:
        """从 dataclass 生成 JSON Schema。"""
        hints = get_type_hints(cls)
        properties = {}
        required = []

        for f in fields(cls):
            prop = SchemaGenerator._type_to_schema(hints[f.name])
            # 添加 description（如果字段有 metadata）
            if f.metadata and "description" in f.metadata:
                prop["description"] = f.metadata["description"]
            properties[f.name] = prop
            if f.default is MISSING and f.default_factory is MISSING:
                required.append(f.name)

        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        if cls.__doc__:
            schema["description"] = cls.__doc__.strip()
        return schema

    @staticmethod
    def from_dict(spec: dict[str, str]) -> dict:
        """从简单字典 {name: type} 生成 Schema。

        示例: {"name": "string", "age": "integer", "tags": "array"}
        """
        properties = {}
        for name, type_str in spec.items():
            properties[name] = {"type": type_str}
        return {
            "type": "object",
            "properties": properties,
            "required": list(spec.keys()),
        }

    @staticmethod
    def _type_to_schema(tp) -> dict:
        origin = getattr(tp, "__origin__", None)
        if origin is list:
            args = getattr(tp, "__args__", ())
            items = SchemaGenerator._type_to_schema(args[0]) if args else {}
            return {"type": "array", "items": items}
        if origin is dict:
            return {"type": "object"}
        if tp in _TYPE_MAP:
            return {"type": _TYPE_MAP[tp]}
        # Enum 支持
        if hasattr(tp, "__members__"):
            return {"type": "string", "enum": list(tp.__members__.keys())}
        return {"type": "string"}


# ═══════════════════════════════════════════════════════════
# JSON Schema 验证器
# ═══════════════════════════════════════════════════════════

class SchemaValidator:
    """轻量 JSON Schema 验证器（不依赖 jsonschema 库）。"""

    def __init__(self, schema: dict):
        self.schema = schema

    def validate(self, data: Any) -> list[str]:
        """验证数据，返回错误列表。空列表 = 通过。"""
        return self._check(data, self.schema, path="$")

    def _check(self, data: Any, schema: dict, path: str) -> list[str]:
        errors = []
        expected_type = schema.get("type")

        # 类型检查
        if expected_type:
            if not self._type_match(data, expected_type):
                errors.append(f"{path}: 期望 {expected_type}，实际 {type(data).__name__}")
                return errors

        # object 属性检查
        if expected_type == "object" and isinstance(data, dict):
            props = schema.get("properties", {})
            required = schema.get("required", [])
            for req in required:
                if req not in data:
                    errors.append(f"{path}.{req}: 必填字段缺失")
            for key, val in data.items():
                if key in props:
                    errors.extend(self._check(val, props[key], f"{path}.{key}"))
                elif schema.get("additionalProperties") is False:
                    errors.append(f"{path}.{key}: 不允许的额外字段")

        # array 元素检查
        if expected_type == "array" and isinstance(data, list):
            items_schema = schema.get("items", {})
            for i, item in enumerate(data):
                errors.extend(self._check(item, items_schema, f"{path}[{i}]"))

        # enum 检查
        if "enum" in schema and data not in schema["enum"]:
            errors.append(f"{path}: 值 '{data}' 不在允许列表 {schema['enum']} 中")

        return errors

    @staticmethod
    def _type_match(data: Any, expected: str) -> bool:
        mapping = {
            "string": str, "integer": int, "number": (int, float),
            "boolean": bool, "array": list, "object": dict, "null": type(None),
        }
        return isinstance(data, mapping.get(expected, object))


# ═══════════════════════════════════════════════════════════
# JSON 修复器
# ═══════════════════════════════════════════════════════════

class JSONRepair:
    """修复 LLM 输出中常见的 JSON 格式错误。"""

    @staticmethod
    def repair(raw: str) -> str:
        """尝试修复并返回合法 JSON 字符串。"""
        text = raw.strip()

        # 1. 提取 ```json ... ``` 代码块
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if m:
            text = m.group(1).strip()

        # 2. 提取第一个 { ... } 或 [ ... ]
        for opener, closer in [('{', '}'), ('[', ']')]:
            start = text.find(opener)
            if start != -1:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == opener:
                        depth += 1
                    elif text[i] == closer:
                        depth -= 1
                    if depth == 0:
                        text = text[start:i + 1]
                        break

        # 3. 修复尾部逗号: {a: 1,} → {a: 1}
        text = re.sub(r',\s*([}\]])', r'\1', text)

        # 4. 修复单引号 → 双引号
        text = text.replace("'", '"')

        # 5. 修复没有引号的 key: {name: "v"} → {"name": "v"}
        text = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', text)

        # 6. 修复 Python 布尔值
        text = text.replace("True", "true").replace("False", "false").replace("None", "null")

        return text

    @staticmethod
    def safe_parse(raw: str) -> tuple[Any | None, str]:
        """安全解析：先原样 parse，失败后修复再 parse。"""
        try:
            return json.loads(raw), ""
        except json.JSONDecodeError:
            pass
        repaired = JSONRepair.repair(raw)
        try:
            return json.loads(repaired), "repaired"
        except json.JSONDecodeError as e:
            return None, f"parse_error: {e}"
