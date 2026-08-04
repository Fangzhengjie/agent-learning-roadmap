"""结构化输出解析器 — 从 LLM 原始文本中提取结构化数据

支持模式：
  1. JSON Mode — 约束 LLM 输出 JSON，解析 + 验证
  2. Pydantic 解析 — 用 dataclass 定义结构，自动校验字段
  3. 重试修复 — 解析失败后构建修复 prompt 让 LLM 重试
  4. 多格式抽取 — 从 Markdown/XML/YAML 混合输出中提取
"""

import json
import re
from dataclasses import dataclass, fields, MISSING
from typing import Any

try:
    from .schema_engine import SchemaValidator, JSONRepair, SchemaGenerator
except ImportError:
    from schema_engine import SchemaValidator, JSONRepair, SchemaGenerator


# ═══════════════════════════════════════════════════════════
# 输出解析器
# ═══════════════════════════════════════════════════════════

class OutputParser:
    """通用输出解析器。"""

    def __init__(self, schema: dict):
        self.schema = schema
        self.validator = SchemaValidator(schema)

    def parse(self, raw: str) -> tuple[Any | None, list[str]]:
        """解析 LLM 输出 → (结果, 错误列表)。"""
        data, repair_status = JSONRepair.safe_parse(raw)
        if data is None:
            return None, [repair_status]
        errors = self.validator.validate(data)
        if errors:
            return data, errors
        return data, []

    def to_prompt_instruction(self) -> str:
        """生成约束 LLM 输出格式的 prompt 指令。"""
        schema_str = json.dumps(self.schema, indent=2, ensure_ascii=False)
        return f"""请严格按以下 JSON Schema 格式输出，不要添加其他文字：

```json
{schema_str}
```

要求：
- 输出纯 JSON，不要包裹在 markdown 代码块中
- 所有 required 字段必须提供
- 字段类型必须匹配"""


class DataclassParser:
    """Dataclass 解析器 — 将 LLM 输出解析为 Python dataclass 实例。"""

    def __init__(self, cls):
        self.cls = cls
        self.schema = SchemaGenerator.from_dataclass(cls)
        self.parser = OutputParser(self.schema)

    def parse(self, raw: str) -> tuple[Any | None, list[str]]:
        """解析 → dataclass 实例。"""
        data, errors = self.parser.parse(raw)
        if errors or data is None:
            return None, errors
        try:
            # 只传 dataclass 定义的字段
            valid_fields = {f.name for f in fields(self.cls)}
            filtered = {k: v for k, v in data.items() if k in valid_fields}
            instance = self.cls(**filtered)
            return instance, []
        except (TypeError, ValueError) as e:
            return None, [f"dataclass 构造失败: {e}"]

    def to_prompt_instruction(self) -> str:
        return self.parser.to_prompt_instruction()


# ═══════════════════════════════════════════════════════════
# 重试修复器
# ═══════════════════════════════════════════════════════════

class RetryParser:
    """解析失败后自动构建修复 prompt 让 LLM 重试。"""

    def __init__(self, parser: OutputParser | DataclassParser, max_retries: int = 3):
        self.parser = parser
        self.max_retries = max_retries
        self.attempts: list[dict] = []

    def parse_with_retry(self, raw: str, mock_fix_fn=None) -> tuple[Any | None, list[dict]]:
        """尝试解析，失败则构建修复 prompt。

        Args:
            raw: LLM 原始输出
            mock_fix_fn: 模拟 LLM 修复的函数 (errors, raw) → fixed_str
        """
        current = raw
        for attempt in range(self.max_retries + 1):
            result, errors = self.parser.parse(current)
            self.attempts.append({
                "attempt": attempt,
                "input_preview": current[:100],
                "success": len(errors) == 0,
                "errors": errors,
            })
            if not errors:
                return result, self.attempts
            if attempt < self.max_retries and mock_fix_fn:
                fix_prompt = self._build_fix_prompt(current, errors)
                current = mock_fix_fn(errors, current, fix_prompt)
        return None, self.attempts

    def _build_fix_prompt(self, raw: str, errors: list[str]) -> str:
        schema_str = json.dumps(
            self.parser.schema if hasattr(self.parser, 'schema') else self.parser.parser.schema,
            indent=2, ensure_ascii=False
        )
        return f"""你的上一次输出有格式错误，请修正：

错误信息：
{chr(10).join(f'  - {e}' for e in errors)}

你的输出：
{raw[:500]}

请严格按此 Schema 重新输出：
{schema_str}

只输出 JSON，不要其他文字。"""


# ═══════════════════════════════════════════════════════════
# 多格式抽取器
# ═══════════════════════════════════════════════════════════

class MultiFormatExtractor:
    """从混合格式文本中抽取结构化数据。"""

    @staticmethod
    def extract_json(text: str) -> list[dict]:
        """抽取文本中所有 JSON 对象。"""
        results = []
        # 先找 code block
        for m in re.finditer(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL):
            data, _ = JSONRepair.safe_parse(m.group(1))
            if data:
                results.append(data)
        # 再找裸 JSON
        if not results:
            data, _ = JSONRepair.safe_parse(text)
            if data:
                results.append(data)
        return results

    @staticmethod
    def extract_xml_tags(text: str, tag: str) -> list[str]:
        """抽取 XML 标签内容。"""
        return re.findall(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)

    @staticmethod
    def extract_key_value(text: str) -> dict[str, str]:
        """抽取 key: value 格式。"""
        result = {}
        for line in text.strip().split("\n"):
            m = re.match(r'^\s*[•\-*]?\s*\*?\*?(\w[\w\s]*?)\*?\*?\s*[:：]\s*(.+)', line)
            if m:
                result[m.group(1).strip()] = m.group(2).strip()
        return result
