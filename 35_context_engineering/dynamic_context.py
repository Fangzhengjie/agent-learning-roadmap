"""动态上下文选择 — 按任务类型/用户画像动态组装上下文

核心思想: 不是所有信息都应该放进上下文 — 选对的信息比选多的信息更重要。

技术:
  1. Few-shot 动态选择 — 根据当前 query 选最相关的示例
  2. 工具定义裁剪 — 只暴露当前任务需要的工具
  3. 用户画像注入 — 根据用户身份定制 system prompt
  4. 指令缓存 (Anthropic) — 不变的前缀 prompt 可缓存
"""

import math
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════
# 1. 动态 Few-shot 选择
# ═══════════════════════════════════════════════════════════

@dataclass
class Example:
    """Few-shot 示例。"""
    query: str
    response: str
    category: str = ""
    embedding: list[float] | None = None


class DynamicFewShotSelector:
    """根据当前 query 动态选择最相关的 few-shot 示例。

    比固定 few-shot 更好:
      - 固定: 每次都用同样 3 个示例 → 可能不相关，浪费 token
      - 动态: 从示例库选与当前 query 最相似的 → 更准确，更省 token
    """

    def __init__(self, examples: list[Example] | None = None):
        self.examples = examples or []

    def add(self, example: Example):
        if example.embedding is None:
            example.embedding = self._simple_embed(example.query)
        self.examples.append(example)

    def select(self, query: str, top_k: int = 3, category: str | None = None) -> list[Example]:
        """选最相关的示例。"""
        q_emb = self._simple_embed(query)
        candidates = self.examples
        if category:
            candidates = [e for e in candidates if e.category == category]

        scored = []
        for ex in candidates:
            emb = ex.embedding or self._simple_embed(ex.query)
            sim = self._cosine(q_emb, emb)
            scored.append((ex, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, _ in scored[:top_k]]

    @staticmethod
    def _simple_embed(text: str) -> list[float]:
        chars = text.lower().replace(" ", "")
        tokens: dict[str, int] = {}
        for i in range(len(chars) - 1):
            bg = chars[i:i + 2]
            tokens[bg] = tokens.get(bg, 0) + 1
        return list(tokens.values()) if tokens else [0.0]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        min_len = min(len(a), len(b))
        if min_len == 0:
            return 0.0
        a, b = a[:min_len], b[:min_len]
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1e-9
        nb = math.sqrt(sum(x * x for x in b)) or 1e-9
        return dot / (na * nb)


# ═══════════════════════════════════════════════════════════
# 2. 工具定义裁剪
# ═══════════════════════════════════════════════════════════

@dataclass
class ToolDef:
    name: str
    description: str
    category: str
    schema_tokens: int = 50  # 每个工具定义占的 token


class ToolPruner:
    """工具定义裁剪 — 只暴露当前任务需要的工具。

    问题: 注册了 50 个工具 → 全部放进 context → 占 5000 tokens + 分散 LLM 注意力
    方案: 根据 query 只暴露相关的 5-10 个工具

    Anthropic 建议: 超过 10 个工具时性能下降，应做裁剪。
    """

    def __init__(self, tools: list[ToolDef] | None = None):
        self.tools = tools or []

    def prune(self, query: str, max_tools: int = 10) -> list[ToolDef]:
        """裁剪工具列表。"""
        query_lower = query.lower()
        scored = []
        for tool in self.tools:
            score = 0
            for word in tool.name.lower().split("_"):
                if word in query_lower:
                    score += 2
            for word in tool.description.lower().split():
                if len(word) > 2 and word in query_lower:
                    score += 1
            scored.append((tool, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        # 至少返回 category 匹配的
        selected = [t for t, s in scored[:max_tools] if s > 0]

        # 如果没有匹配的，返回最常用的
        if not selected:
            selected = [t for t, _ in scored[:max_tools // 2]]

        return selected

    @property
    def total_tokens(self) -> int:
        return sum(t.schema_tokens for t in self.tools)


# ═══════════════════════════════════════════════════════════
# 3. 用户画像上下文
# ═══════════════════════════════════════════════════════════

@dataclass
class UserProfile:
    user_id: str
    role: str       # admin / developer / customer / ...
    language: str   # zh-CN / en-US
    expertise: str  # beginner / intermediate / expert
    preferences: dict = field(default_factory=dict)


class PersonalizedContext:
    """根据用户画像定制 system prompt。

    同一个 Agent，不同用户看到不同的回复风格:
      - 专家: 简洁专业，可以用术语
      - 新手: 详细解释，避免术语
      - 管理员: 可以执行敏感操作
      - 普通用户: 受限操作
    """

    EXPERTISE_INSTRUCTIONS = {
        "beginner": "用户是初学者。回答要详细，避免术语，多给示例。",
        "intermediate": "用户有一定基础。回答要清晰，可以使用常见术语。",
        "expert": "用户是专家。回答要简洁专业，直接给结论和代码。",
    }

    ROLE_INSTRUCTIONS = {
        "admin": "用户是管理员，有完整操作权限。",
        "developer": "用户是开发者，关注技术细节和 API 用法。",
        "customer": "用户是终端客户，关注产品使用和问题解决。",
    }

    def build_context(self, profile: UserProfile) -> str:
        """生成个性化的 system prompt 片段。"""
        parts = []

        if profile.expertise in self.EXPERTISE_INSTRUCTIONS:
            parts.append(self.EXPERTISE_INSTRUCTIONS[profile.expertise])

        if profile.role in self.ROLE_INSTRUCTIONS:
            parts.append(self.ROLE_INSTRUCTIONS[profile.role])

        if profile.language == "en-US":
            parts.append("Please respond in English.")
        elif profile.language == "zh-CN":
            parts.append("请用中文回答。")

        if profile.preferences.get("concise"):
            parts.append("用户偏好简洁回答，不要过长。")

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════
# 4. Prompt 缓存（Anthropic 风格）
# ═══════════════════════════════════════════════════════════

class PromptCache:
    """Prompt 缓存 — 将不变的前缀缓存起来，减少重复计算。

    Anthropic Prompt Caching:
      - 标记 cache_control: {type: "ephemeral"}
      - 首次调用写入缓存，后续调用命中 → 输入 token 费用降 90%
      - TTL: 5 分钟

    适用: System prompt + 工具定义 + 长文档 → 这些每次都一样
    不适用: 用户消息 → 每次都不一样
    """

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self.stats = {"hits": 0, "misses": 0, "saved_tokens": 0}

    def build_cached_messages(self, system_prompt: str, tool_definitions: str,
                               user_message: str) -> list[dict]:
        """构建带缓存标记的 messages（Anthropic 格式）。"""
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},  # ← 缓存标记
                    },
                    {
                        "type": "text",
                        "text": tool_definitions,
                        "cache_control": {"type": "ephemeral"},  # ← 缓存标记
                    },
                ],
            },
            {"role": "user", "content": user_message},  # 不缓存
        ]

    def estimate_savings(self, cached_tokens: int, calls: int) -> dict:
        """估算缓存节省的费用。"""
        # Anthropic 定价: 输入 $3/M, 缓存写入 $3.75/M, 缓存读取 $0.30/M
        no_cache_cost = cached_tokens * calls * 3 / 1_000_000
        cache_write_cost = cached_tokens * 3.75 / 1_000_000
        cache_read_cost = cached_tokens * (calls - 1) * 0.30 / 1_000_000
        cached_cost = cache_write_cost + cache_read_cost

        return {
            "cached_tokens": cached_tokens,
            "calls": calls,
            "no_cache_cost": f"${no_cache_cost:.4f}",
            "cached_cost": f"${cached_cost:.4f}",
            "savings": f"${no_cache_cost - cached_cost:.4f}",
            "savings_pct": f"{(1 - cached_cost / no_cache_cost) * 100:.0f}%" if no_cache_cost > 0 else "0%",
        }
