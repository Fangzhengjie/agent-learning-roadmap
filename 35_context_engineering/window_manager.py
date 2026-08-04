"""上下文窗口管理 — 对话历史压缩 + 滑动窗口 + 动态截断

当对话越来越长时，如何管理上下文窗口？

三种策略:
  1. 滑动窗口 — 只保留最近 N 轮
  2. 摘要压缩 — 旧消息压缩成摘要
  3. 重要性过滤 — 保留关键消息，丢弃闲聊
"""

import re
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str
    turn: int = 0
    importance: float = 0.5  # 0-1
    token_estimate: int = 0

    def __post_init__(self):
        if self.token_estimate == 0:
            self.token_estimate = max(1, int(len(self.content) * 0.7))


# ═══════════════════════════════════════════════════════════
# 1. 滑动窗口
# ═══════════════════════════════════════════════════════════

class SlidingWindowManager:
    """滑动窗口 — 只保留最近 N 轮。

    最简单的策略，但会丢失早期重要信息。
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns

    def trim(self, messages: list[Message]) -> list[Message]:
        if len(messages) <= self.max_turns * 2:
            return messages
        # 保留 system + 最近 N 轮
        system = [m for m in messages if m.role == "system"]
        others = [m for m in messages if m.role != "system"]
        return system + others[-(self.max_turns * 2):]


# ═══════════════════════════════════════════════════════════
# 2. 摘要压缩
# ═══════════════════════════════════════════════════════════

class SummaryCompressor:
    """摘要压缩 — 将旧消息压缩成一段摘要。

    生产方案: 调 LLM 生成摘要
    本模拟: 规则提取关键句
    """

    def __init__(self, keep_recent: int = 6, summary_max_chars: int = 300):
        self.keep_recent = keep_recent
        self.summary_max_chars = summary_max_chars

    def compress(self, messages: list[Message]) -> list[Message]:
        non_system = [m for m in messages if m.role != "system"]
        system = [m for m in messages if m.role == "system"]

        if len(non_system) <= self.keep_recent:
            return messages

        # 需要压缩的旧消息
        old = non_system[:-self.keep_recent]
        recent = non_system[-self.keep_recent:]

        summary = self._summarize(old)
        summary_msg = Message(
            role="system",
            content=f"[对话摘要] {summary}",
            importance=0.8,
        )

        return system + [summary_msg] + recent

    def _summarize(self, messages: list[Message]) -> str:
        """模拟摘要（生产中调 LLM）。"""
        key_points = []
        for m in messages:
            # 提取关键信息: 问题/结论/操作
            if m.role == "user":
                key_points.append(f"用户问: {m.content[:30]}")
            elif m.role == "assistant":
                # 取第一句
                first_line = m.content.split("。")[0].split("\n")[0]
                key_points.append(f"回答: {first_line[:30]}")

        summary = "；".join(key_points[:5])
        return summary[:self.summary_max_chars]


# ═══════════════════════════════════════════════════════════
# 3. 重要性过滤
# ═══════════════════════════════════════════════════════════

class ImportanceFilter:
    """重要性过滤 — 评估每条消息的重要性，丢弃低分的。

    重要性信号:
      - 包含工具调用/结果 → 高
      - 包含关键决策/数字/代码 → 高
      - 纯闲聊/致谢 → 低
      - 最近的消息 → 高（时间衰减）
    """

    HIGH_IMPORTANCE_PATTERNS = [
        re.compile(r'(?:结论|总结|决定|确认|同意|action)', re.I),
        re.compile(r'\d{3,}'),  # 包含数字
        re.compile(r'```'),     # 代码块
        re.compile(r'(?:错误|异常|失败|error|fail)', re.I),
    ]

    LOW_IMPORTANCE_PATTERNS = [
        re.compile(r'^(?:好的|谢谢|OK|Thanks|明白了|收到)\s*$', re.I),
        re.compile(r'^(?:嗯|是的|对)\s*$'),
    ]

    def score(self, message: Message, position: int, total: int) -> float:
        """评估消息重要性 0-1。"""
        base = 0.5

        # 高重要性模式
        for p in self.HIGH_IMPORTANCE_PATTERNS:
            if p.search(message.content):
                base += 0.15

        # 低重要性模式
        for p in self.LOW_IMPORTANCE_PATTERNS:
            if p.search(message.content):
                base -= 0.3

        # 消息长度（长消息通常更重要）
        if len(message.content) > 200:
            base += 0.1
        elif len(message.content) < 10:
            base -= 0.1

        # 时间衰减（越近越重要）
        recency = position / total if total > 0 else 0.5
        base += recency * 0.2

        # 工具相关
        if message.role == "tool":
            base += 0.2

        return max(0.0, min(1.0, base))

    def filter(self, messages: list[Message], token_budget: int) -> list[Message]:
        """按重要性筛选消息，控制在 token 预算内。"""
        # 评分
        scored = []
        for i, m in enumerate(messages):
            if m.role == "system":
                scored.append((m, 1.0))  # system 始终保留
            else:
                scored.append((m, self.score(m, i, len(messages))))

        # 按重要性排序（但保持原始相对顺序）
        scored_with_idx = [(m, s, i) for i, (m, s) in enumerate(scored)]
        scored_with_idx.sort(key=lambda x: x[1], reverse=True)

        # 贪心填充
        selected_indices = set()
        used_tokens = 0
        for m, s, idx in scored_with_idx:
            if used_tokens + m.token_estimate <= token_budget:
                selected_indices.add(idx)
                used_tokens += m.token_estimate

        # 按原始顺序输出
        return [messages[i] for i in sorted(selected_indices)]


# ═══════════════════════════════════════════════════════════
# 4. 组合策略
# ═══════════════════════════════════════════════════════════

class AdaptiveWindowManager:
    """自适应窗口管理 — 组合多种策略。

    策略:
      1. 先用重要性评分标记
      2. 对低重要性的旧消息做摘要压缩
      3. 最终确保不超预算
    """

    def __init__(self, token_budget: int = 4000, keep_recent: int = 6):
        self.token_budget = token_budget
        self.keep_recent = keep_recent
        self.compressor = SummaryCompressor(keep_recent=keep_recent)
        self.importance = ImportanceFilter()

    def manage(self, messages: list[Message]) -> dict:
        """管理上下文窗口。"""
        original_tokens = sum(m.token_estimate for m in messages)

        if original_tokens <= self.token_budget:
            return {
                "messages": messages,
                "strategy": "none",
                "original_tokens": original_tokens,
                "final_tokens": original_tokens,
            }

        # Step 1: 摘要压缩
        compressed = self.compressor.compress(messages)
        compressed_tokens = sum(m.token_estimate for m in compressed)

        if compressed_tokens <= self.token_budget:
            return {
                "messages": compressed,
                "strategy": "summary_compression",
                "original_tokens": original_tokens,
                "final_tokens": compressed_tokens,
            }

        # Step 2: 重要性过滤
        filtered = self.importance.filter(compressed, self.token_budget)
        final_tokens = sum(m.token_estimate for m in filtered)

        return {
            "messages": filtered,
            "strategy": "summary + importance_filter",
            "original_tokens": original_tokens,
            "final_tokens": final_tokens,
        }
