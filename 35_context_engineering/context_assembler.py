"""上下文组装器 — 动态构建最优 LLM 输入

上下文工程 vs 提示工程:
  提示工程 (Prompt Engineering): "怎么写一个好 prompt"
  上下文工程 (Context Engineering): "怎么把正确的信息在正确的时间喂给 LLM"

上下文的六大来源:
  ┌──────────────────────────────────────────────────────┐
  │                    LLM 上下文窗口                      │
  │                                                      │
  │  ┌──────────┐  System Prompt（角色 + 规则）            │
  │  │ 指令层    │  ← 静态，每次都带                        │
  │  ├──────────┤                                        │
  │  │ 知识层    │  RAG 检索结果 / 知识图谱                 │
  │  │          │  ← 动态，按 query 检索                   │
  │  ├──────────┤                                        │
  │  │ 记忆层    │  对话历史 / 用户偏好 / 长期记忆            │
  │  │          │  ← 动态，按会话状态                      │
  │  ├──────────┤                                        │
  │  │ 工具层    │  工具定义 / 工具执行结果                   │
  │  │          │  ← 动态，按可用工具                      │
  │  ├──────────┤                                        │
  │  │ 任务层    │  Few-shot 示例 / 输出格式约束              │
  │  │          │  ← 按任务类型选择                        │
  │  ├──────────┤                                        │
  │  │ 用户层    │  当前用户消息                             │
  │  │          │  ← 实时                                │
  │  └──────────┘                                        │
  └──────────────────────────────────────────────────────┘

核心挑战: 上下文窗口有限 → 如何在有限空间内塞入最有价值的信息？
"""

from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════
# 上下文块（Context Block）
# ═══════════════════════════════════════════════════════════

@dataclass
class ContextBlock:
    """上下文中的一个信息块。"""
    name: str
    content: str
    source: str       # instruction / knowledge / memory / tool / example / user
    priority: int     # 1-10, 越高越重要
    token_estimate: int = 0
    is_required: bool = False  # 是否必须包含（如 system prompt）

    def __post_init__(self):
        if self.token_estimate == 0:
            # 粗估: 中文约 1.5 token/字, 英文约 1.3 token/词
            self.token_estimate = max(1, int(len(self.content) * 0.7))


# ═══════════════════════════════════════════════════════════
# 上下文预算管理器
# ═══════════════════════════════════════════════════════════

class ContextBudget:
    """上下文 Token 预算管理 — 确保不超过模型窗口限制。"""

    # 常见模型上下文窗口
    MODEL_WINDOWS = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "claude-3.5-sonnet": 200000,
        "claude-3-haiku": 200000,
        "gemini-1.5-pro": 2000000,
        "llama-3.1-70b": 128000,
        "deepseek-v3": 128000,
    }

    def __init__(self, model: str = "gpt-4o", reserve_output: int = 4096):
        self.model = model
        self.max_window = self.MODEL_WINDOWS.get(model, 128000)
        self.reserve_output = reserve_output  # 留给输出的 token
        self.available = self.max_window - reserve_output

    def allocate(self, blocks: list[ContextBlock]) -> list[ContextBlock]:
        """按优先级分配 Token 预算，返回能放进去的块。

        策略:
          1. 必须包含的块先放（system prompt）
          2. 剩余空间按 priority 排序填充
          3. 超出的块被丢弃（或截断）
        """
        # 分离必须 vs 可选
        required = [b for b in blocks if b.is_required]
        optional = sorted([b for b in blocks if not b.is_required],
                          key=lambda b: b.priority, reverse=True)

        selected = []
        used = 0

        # 先放必须的
        for b in required:
            if used + b.token_estimate <= self.available:
                selected.append(b)
                used += b.token_estimate

        # 再按优先级填充
        for b in optional:
            if used + b.token_estimate <= self.available:
                selected.append(b)
                used += b.token_estimate

        return selected

    @property
    def utilization_report(self) -> dict:
        return {
            "model": self.model,
            "max_window": self.max_window,
            "reserve_output": self.reserve_output,
            "available_input": self.available,
        }


# ═══════════════════════════════════════════════════════════
# 上下文组装器
# ═══════════════════════════════════════════════════════════

class ContextAssembler:
    """上下文组装器 — 将多来源信息组装成最优 LLM 输入。

    流程:
      1. 收集所有候选上下文块
      2. 按优先级 + Token 预算筛选
      3. 按固定顺序排列
      4. 输出 messages 列表
    """

    # 排列顺序（System → Knowledge → Memory → Tools → Examples → User）
    SOURCE_ORDER = ["instruction", "knowledge", "memory", "tool", "example", "user"]

    def __init__(self, budget: ContextBudget):
        self.budget = budget
        self._blocks: list[ContextBlock] = []

    def add(self, block: ContextBlock):
        self._blocks.append(block)

    def add_instruction(self, content: str, priority: int = 10):
        self._blocks.append(ContextBlock("system_prompt", content, "instruction", priority, is_required=True))

    def add_knowledge(self, name: str, content: str, priority: int = 7):
        self._blocks.append(ContextBlock(name, content, "knowledge", priority))

    def add_memory(self, name: str, content: str, priority: int = 6):
        self._blocks.append(ContextBlock(name, content, "memory", priority))

    def add_tool_def(self, name: str, content: str, priority: int = 8):
        self._blocks.append(ContextBlock(name, content, "tool", priority))

    def add_tool_result(self, name: str, content: str, priority: int = 9):
        self._blocks.append(ContextBlock(name, content, "tool", priority))

    def add_example(self, name: str, content: str, priority: int = 5):
        self._blocks.append(ContextBlock(name, content, "example", priority))

    def add_user_message(self, content: str):
        self._blocks.append(ContextBlock("user_message", content, "user", 10, is_required=True))

    def assemble(self) -> dict:
        """组装上下文 → 返回 {messages, stats}。"""
        selected = self.budget.allocate(self._blocks)
        dropped = [b for b in self._blocks if b not in selected]

        # 按来源排序
        selected.sort(key=lambda b: self.SOURCE_ORDER.index(b.source)
                       if b.source in self.SOURCE_ORDER else 99)

        # 构建 messages
        messages = []
        system_parts = []
        for b in selected:
            if b.source == "instruction":
                system_parts.append(b.content)
            elif b.source == "user":
                messages.append({"role": "user", "content": b.content})
            elif b.source in ("knowledge", "memory", "example"):
                system_parts.append(f"\n## {b.name}\n{b.content}")
            elif b.source == "tool":
                system_parts.append(f"\n## 工具: {b.name}\n{b.content}")

        if system_parts:
            messages.insert(0, {"role": "system", "content": "\n".join(system_parts)})

        total_tokens = sum(b.token_estimate for b in selected)
        return {
            "messages": messages,
            "stats": {
                "total_blocks": len(self._blocks),
                "selected_blocks": len(selected),
                "dropped_blocks": len(dropped),
                "estimated_tokens": total_tokens,
                "budget_usage": f"{total_tokens / self.budget.available:.0%}",
                "dropped_names": [b.name for b in dropped],
            },
        }

    def reset(self):
        self._blocks.clear()
