"""流式 Token 解析 + 结构化数据流式提取

问题: LLM 流式输出是一个个 token，如何在流式过程中:
  1. 实时显示文本
  2. 检测并提取 JSON / 工具调用
  3. 在流中途做安全检查

本模块模拟 SSE 流式输出并展示实时解析技术。
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Generator


# ═══════════════════════════════════════════════════════════
# 流式 Token 模拟器
# ═══════════════════════════════════════════════════════════

class StreamSimulator:
    """模拟 LLM 流式输出（一个个 token 生成）。"""

    def __init__(self, text: str, tokens_per_second: int = 20):
        self.text = text
        self.tps = tokens_per_second

    def stream(self) -> Generator[str, None, None]:
        """逐 token 生成。"""
        # 简单按字符切（中文按字，英文按词）
        tokens = self._tokenize(self.text)
        for token in tokens:
            yield token

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = []
        buf = ""
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff' or ch in '，。！？；：、\n':
                if buf:
                    tokens.append(buf)
                    buf = ""
                tokens.append(ch)
            elif ch == ' ':
                if buf:
                    tokens.append(buf + ' ')
                    buf = ""
                else:
                    tokens.append(' ')
            else:
                buf += ch
        if buf:
            tokens.append(buf)
        return tokens


# ═══════════════════════════════════════════════════════════
# 流式 JSON 检测器
# ═══════════════════════════════════════════════════════════

@dataclass
class StreamEvent:
    type: str  # text / json_start / json_complete / tool_call / warning
    content: str
    data: dict | None = None


class StreamParser:
    """流式解析器 — 实时检测流中的 JSON / 工具调用 / 安全问题。"""

    def __init__(self):
        self.buffer = ""
        self.json_depth = 0
        self.in_json = False
        self.json_start_pos = 0
        self.events: list[StreamEvent] = []

    def feed(self, token: str) -> list[StreamEvent]:
        """喂入一个 token，返回检测到的事件。"""
        events = []
        self.buffer += token

        for ch in token:
            if ch == '{' and not self.in_json:
                self.in_json = True
                self.json_depth = 1
                self.json_start_pos = len(self.buffer) - 1
                events.append(StreamEvent("json_start", "{"))
            elif ch == '{' and self.in_json:
                self.json_depth += 1
            elif ch == '}' and self.in_json:
                self.json_depth -= 1
                if self.json_depth == 0:
                    json_str = self.buffer[self.json_start_pos:]
                    try:
                        data = json.loads(json_str)
                        events.append(StreamEvent("json_complete", json_str, data))
                        # 检测是否是工具调用
                        if "function" in data or "tool_call" in data or "name" in data:
                            events.append(StreamEvent("tool_call", json_str, data))
                    except json.JSONDecodeError:
                        pass
                    self.in_json = False

        if not events:
            events.append(StreamEvent("text", token))

        self.events.extend(events)
        return events

    def reset(self):
        self.buffer = ""
        self.json_depth = 0
        self.in_json = False
        self.events.clear()


# ═══════════════════════════════════════════════════════════
# 流式 Guardrail
# ═══════════════════════════════════════════════════════════

class StreamGuardrail:
    """流式安全护栏 — 在流式输出过程中实时检测。

    挑战: 不能等全部输出完再检查（用户已经看到了！）
    策略: 滑动窗口检测 + 即时中断
    """

    DANGER_PATTERNS = [
        (re.compile(r'(?:sk-|api[_-]?key)[a-zA-Z0-9]{10,}'), "api_key_leak"),
        (re.compile(r'(?:密码|password)\s*[:=：]\s*\S+', re.I), "credential"),
        (re.compile(r'\b(?:rm\s+-rf|DROP\s+TABLE)\b', re.I), "dangerous_cmd"),
    ]

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.buffer = ""
        self.violations: list[dict] = []
        self.interrupted = False

    def check(self, token: str) -> dict | None:
        """检查新 token，返回违规信息或 None。"""
        self.buffer += token
        # 保持滑动窗口大小
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]

        for pattern, category in self.DANGER_PATTERNS:
            if pattern.search(self.buffer):
                violation = {"category": category, "position": len(self.buffer), "action": "interrupt"}
                self.violations.append(violation)
                self.interrupted = True
                return violation
        return None

    def reset(self):
        self.buffer = ""
        self.violations.clear()
        self.interrupted = False


# ═══════════════════════════════════════════════════════════
# Backpressure 控制
# ═══════════════════════════════════════════════════════════

class BackpressureController:
    """背压控制 — 当消费者处理不过来时减慢生产者。

    场景: 流式输出 → 前端渲染太慢 → 需要降速或缓冲
    """

    def __init__(self, max_buffer: int = 100, high_water: float = 0.8, low_water: float = 0.3):
        self.max_buffer = max_buffer
        self.high_water = high_water
        self.low_water = low_water
        self._buffer: list[str] = []
        self._paused = False
        self.stats = {"total_tokens": 0, "paused_count": 0, "dropped_count": 0}

    def produce(self, token: str) -> bool:
        """生产者推入 token。返回 True=接受, False=被拒绝。"""
        self.stats["total_tokens"] += 1
        usage = len(self._buffer) / self.max_buffer
        if usage >= self.high_water:
            self._paused = True
            self.stats["paused_count"] += 1
        if len(self._buffer) >= self.max_buffer:
            self.stats["dropped_count"] += 1
            return False
        self._buffer.append(token)
        return True

    def consume(self, count: int = 1) -> list[str]:
        """消费者取出 token。"""
        result = self._buffer[:count]
        self._buffer = self._buffer[count:]
        if len(self._buffer) / self.max_buffer < self.low_water:
            self._paused = False
        return result

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def buffer_usage(self) -> float:
        return len(self._buffer) / self.max_buffer
