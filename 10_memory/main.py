"""Agent 记忆体系

核心概念：Agent 如何记住信息 — 短期/长期/工作记忆/Compaction。

记忆类型：
  ┌─────────────┬────────────────┬──────────────────┐
  │ 短期记忆     │ 当前对话历史    │ message_history  │
  │ 工作记忆     │ 当前任务状态    │ state dict       │
  │ 长期记忆     │ 跨会话知识      │ 向量数据库        │
  │ 情景记忆     │ 历史案例经验    │ 案例检索          │
  └─────────────┴────────────────┴──────────────────┘

Compaction（记忆压缩）：
  对话过长 → 上下文窗口放不下 → 需要压缩历史消息
  策略：摘要压缩 / 滑动窗口 / 重要性排序
"""

import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# 1. 短期记忆（Short-term Memory）
# ═══════════════════════════════════════════════════════════

class ShortTermMemory:
    """当前对话的消息历史 — 所有 Agent 框架的基础。"""

    def __init__(self, max_messages: int = 100):
        self.messages: list[dict] = []
        self.max_messages = max_messages

    def add(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        # 简单截断（生产中应使用 Compaction）
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_history(self) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def token_estimate(self) -> int:
        """粗略估算 token 数（中文约 1 字 = 1.5 token）"""
        return sum(int(len(m["content"]) * 1.5) for m in self.messages)


# ═══════════════════════════════════════════════════════════
# 2. Compaction（记忆压缩）
# ═══════════════════════════════════════════════════════════

class MemoryCompactor:
    """当对话过长时压缩历史消息。

    三种策略：
    1. 滑动窗口：只保留最近 N 条
    2. 摘要压缩：将旧消息总结为一段摘要
    3. 重要性排序：保留包含工具调用和关键决策的消息
    """

    @staticmethod
    def sliding_window(messages: list[dict], keep_last: int = 10) -> list[dict]:
        """策略 1: 滑动窗口 — 保留最近 N 条消息。"""
        if len(messages) <= keep_last:
            return messages
        return messages[-keep_last:]

    @staticmethod
    def summarize(messages: list[dict], max_keep: int = 5) -> list[dict]:
        """策略 2: 摘要压缩 — 将旧消息压缩为摘要。

        生产中应调用 LLM 生成摘要，这里用简单提取。
        """
        if len(messages) <= max_keep:
            return messages

        old_messages = messages[:-max_keep]
        recent_messages = messages[-max_keep:]

        # 提取旧消息的关键信息
        key_points = []
        for m in old_messages:
            content = m["content"]
            if len(content) > 50:
                key_points.append(f"[{m['role']}] {content[:50]}...")
            else:
                key_points.append(f"[{m['role']}] {content}")

        summary = {
            "role": "system",
            "content": f"[历史摘要 - {len(old_messages)} 条消息]\n" + "\n".join(key_points[-5:]),
        }

        return [summary] + recent_messages

    @staticmethod
    def importance_filter(messages: list[dict], max_keep: int = 15) -> list[dict]:
        """策略 3: 重要性过滤 — 保留关键消息。"""
        if len(messages) <= max_keep:
            return messages

        important_keywords = ["决定", "确认", "结果", "错误", "工具", "tool", "approved", "rejected"]

        scored = []
        for i, m in enumerate(messages):
            score = 0
            if m["role"] == "system":
                score += 10  # 系统消息始终保留
            if any(kw in m["content"].lower() for kw in important_keywords):
                score += 5   # 包含关键词的消息
            score += i * 0.1  # 越新的消息越重要
            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        kept = [m for _, m in scored[:max_keep]]
        # 按原始顺序排列
        return sorted(kept, key=lambda m: messages.index(m))


# ═══════════════════════════════════════════════════════════
# 3. 长期记忆（Long-term Memory）
# ═══════════════════════════════════════════════════════════

class LongTermMemory:
    """跨会话的持久化记忆。

    生产方案：
    - 向量数据库（FAISS / Chroma / Pinecone）存储嵌入
    - Mem0 等专用记忆层

    本示例用简单的字典模拟。
    """

    def __init__(self):
        self.facts: list[dict] = []
        self.preferences: dict[str, str] = {}

    def remember(self, fact: str, category: str = "general", importance: float = 0.5):
        """记住一个事实。"""
        self.facts.append({
            "fact": fact,
            "category": category,
            "importance": importance,
            "remembered_at": datetime.now().isoformat(),
        })

    def set_preference(self, key: str, value: str):
        """记住用户偏好。"""
        self.preferences[key] = value

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        """检索相关记忆（模拟向量搜索）。"""
        query_lower = query.lower()
        scored = []
        for fact in self.facts:
            score = sum(1 for w in query_lower.split() if w in fact["fact"].lower())
            score += fact["importance"]
            if score > 0:
                scored.append((score, fact))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k]]

    def get_context(self) -> str:
        """生成注入到 system prompt 的记忆上下文。"""
        parts = []
        if self.preferences:
            parts.append("用户偏好: " + ", ".join(f"{k}={v}" for k, v in self.preferences.items()))
        if self.facts:
            recent = sorted(self.facts, key=lambda x: x["importance"], reverse=True)[:5]
            parts.append("已知事实: " + "; ".join(f["fact"] for f in recent))
        return "\n".join(parts) if parts else "无历史记忆"


# ═══════════════════════════════════════════════════════════
# 4. 工作记忆（Working Memory）
# ═══════════════════════════════════════════════════════════

@dataclass
class WorkingMemory:
    """当前任务的中间状态 — LangGraph State 的简化版。"""
    task: str = ""
    current_step: str = ""
    collected_data: dict = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)
    pending_actions: list[str] = field(default_factory=list)

    def to_context(self) -> str:
        parts = [f"当前任务: {self.task}", f"当前步骤: {self.current_step}"]
        if self.collected_data:
            parts.append(f"已收集数据: {json.dumps(self.collected_data, ensure_ascii=False)}")
        if self.decisions:
            parts.append(f"已做决策: {', '.join(self.decisions)}")
        if self.pending_actions:
            parts.append(f"待处理: {', '.join(self.pending_actions)}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════
# 5. ConversationMemoryManager（集成管理器）
# ═══════════════════════════════════════════════════════════

class ConversationMemoryManager:
    """对话记忆统一管理器 — 集成 STM + LTM + WM + 压缩 + 持久化。

    生产中用 Redis/PostgreSQL 替代 JSON，这里用文件演示。
    """

    def __init__(self, persist_dir: str | None = None, compact_threshold: int = 15):
        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory()
        self.wm = WorkingMemory()
        self.compactor = MemoryCompactor()
        self.compact_threshold = compact_threshold
        self.persist_dir = persist_dir
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._load()

    def add_message(self, role: str, content: str):
        """添加消息，超过阈值时自动压缩。"""
        self.stm.add(role, content)
        if len(self.stm.messages) > self.compact_threshold:
            self.stm.messages = self.compactor.summarize(
                self.stm.get_history(), max_keep=self.compact_threshold // 2
            )

    def remember_fact(self, fact: str, **kwargs):
        self.ltm.remember(fact, **kwargs)

    def build_context(self) -> dict:
        """组装完整上下文（注入 LLM prompt）。"""
        return {
            "long_term": self.ltm.get_context(),
            "working": self.wm.to_context() if self.wm.task else "",
            "short_term": self.stm.get_history(),
            "token_estimate": self.stm.token_estimate(),
        }

    def save(self):
        """持久化到 JSON。"""
        if not self.persist_dir:
            return
        data = {
            "stm_messages": self.stm.messages,
            "ltm_facts": self.ltm.facts,
            "ltm_preferences": self.ltm.preferences,
        }
        with open(os.path.join(self.persist_dir, "memory.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        """从 JSON 恢复。"""
        path = os.path.join(self.persist_dir, "memory.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.stm.messages = data.get("stm_messages", [])
        self.ltm.facts = data.get("ltm_facts", [])
        self.ltm.preferences = data.get("ltm_preferences", {})


# ═══════════════════════════════════════════════════════════
# 5. VectorMemory（向量记忆 — 模拟 FAISS 余弦相似度）
# ═══════════════════════════════════════════════════════════

class VectorMemory:
    """向量记忆 — 用字符级 TF 向量 + 余弦相似度模拟语义检索。

    生产中替换为: FAISS / Chroma / Pinecone + OpenAI Embeddings。
    """

    def __init__(self):
        self._entries: list[dict] = []  # {"text": str, "vec": dict, "tags": list}

    @staticmethod
    def _tokenize(text: str) -> dict[str, int]:
        """简易分词 — 按字符 bigram 统计词频。"""
        tokens: dict[str, int] = {}
        chars = text.replace(" ", "")
        for i in range(len(chars) - 1):
            bigram = chars[i:i+2]
            tokens[bigram] = tokens.get(bigram, 0) + 1
        # 也加入单字
        for ch in chars:
            tokens[ch] = tokens.get(ch, 0) + 1
        return tokens

    @staticmethod
    def _cosine_sim(a: dict[str, int], b: dict[str, int]) -> float:
        """余弦相似度。"""
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        norm_a = math.sqrt(sum(v * v for v in a.values())) or 1e-9
        norm_b = math.sqrt(sum(v * v for v in b.values())) or 1e-9
        return dot / (norm_a * norm_b)

    def add(self, text: str, tags: list[str] | None = None):
        self._entries.append({
            "text": text,
            "vec": self._tokenize(text),
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
        })

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[tuple[str, float]]:
        """语义搜索 — 返回 [(text, score), ...]。"""
        q_vec = self._tokenize(query)
        scored = []
        for entry in self._entries:
            score = self._cosine_sim(q_vec, entry["vec"])
            if score >= min_score:
                scored.append((entry["text"], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def size(self) -> int:
        return len(self._entries)


# ═══════════════════════════════════════════════════════════
# 6. TTLCache（带过期的记忆缓存）
# ═══════════════════════════════════════════════════════════

class TTLCache:
    """TTL 缓存 — 记忆条目自动过期。

    企业用途:
    - 会话上下文缓存（30 分钟过期）
    - 用户偏好缓存（24 小时过期）
    - LLM 语义缓存（1 小时过期）
    """

    def __init__(self, default_ttl_s: float = 3600):
        self.default_ttl_s = default_ttl_s
        self._store: dict[str, tuple[Any, float]] = {}  # key → (value, expire_at)

    def set(self, key: str, value: Any, ttl_s: float | None = None):
        expire_at = time.time() + (ttl_s if ttl_s is not None else self.default_ttl_s)
        self._store[key] = (value, expire_at)

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expire_at = entry
        if time.time() > expire_at:
            del self._store[key]
            return None
        return value

    def delete(self, key: str):
        self._store.pop(key, None)

    def cleanup(self):
        """清理所有过期条目。"""
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]

    def size(self) -> int:
        return len(self._store)


# ═══════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════

def main():
    print("=== Agent 记忆体系 ===\n")

    # ── 1. 短期记忆 + Compaction ─────────────────────────
    print("▶ 1. 短期记忆 + Compaction")
    print("─" * 60)

    stm = ShortTermMemory()

    # 模拟长对话
    conversations = [
        ("user", "你好，我想查一下订单"),
        ("assistant", "好的，请提供订单号"),
        ("user", "ORD-2024-001"),
        ("assistant", "查到了，订单金额 ¥45,000，状态待审批"),
        ("user", "库存够吗？"),
        ("assistant", "库存充足，服务器x2和交换机x1都有货"),
        ("user", "信用怎么样？"),
        ("assistant", "张三信用等级A，可用额度 ¥80,000，足够"),
        ("user", "那就批准吧"),
        ("assistant", "好的，已将订单标记为批准。风险评估为低风险。"),
        ("user", "再帮我看看 ORD-2024-002"),
        ("assistant", "这个订单金额 ¥128,000，风险评分 0.8，建议人工审核"),
        ("user", "什么原因风险高？"),
        ("assistant", "李四信用额度已用 ¥150,000/200,000，且数据库许可证属于高价值商品"),
    ]

    for role, content in conversations:
        stm.add(role, content)

    print(f"  原始消息数: {len(stm.messages)}, 预估 Token: {stm.token_estimate()}")

    # 三种 Compaction 策略对比
    strategies = [
        ("滑动窗口 (keep=5)", MemoryCompactor.sliding_window(stm.get_history(), keep_last=5)),
        ("摘要压缩 (keep=3)", MemoryCompactor.summarize(stm.get_history(), max_keep=3)),
        ("重要性过滤 (keep=6)", MemoryCompactor.importance_filter(stm.get_history(), max_keep=6)),
    ]

    for name, result in strategies:
        print(f"\n  📦 {name}: {len(result)} 条消息")
        for m in result:
            preview = m["content"][:50].replace("\n", " ")
            print(f"     [{m['role']}] {preview}...")

    # ── 2. 长期记忆 ──────────────────────────────────────
    print(f"\n\n▶ 2. 长期记忆 (Long-term Memory)")
    print("─" * 60)

    ltm = LongTermMemory()

    # 跨会话记住信息
    ltm.remember("用户张三偏好邮件通知而非短信", category="preference", importance=0.8)
    ltm.remember("上次订单 ORD-2024-001 因库存问题延迟了 2 天", category="history", importance=0.6)
    ltm.remember("张三是 VIP 客户，合作 3 年", category="profile", importance=0.9)
    ltm.set_preference("通知方式", "邮件")
    ltm.set_preference("语言", "中文")

    # 检索相关记忆
    query = "张三的订单"
    results = ltm.recall(query)
    print(f"  检索 '{query}' 相关记忆:")
    for r in results:
        print(f"    💡 [{r['category']}] {r['fact']} (重要性: {r['importance']})")

    print(f"\n  注入到 prompt 的上下文:")
    print(f"    {ltm.get_context()}")

    # ── 3. 工作记忆 ──────────────────────────────────────
    print(f"\n\n▶ 3. 工作记忆 (Working Memory)")
    print("─" * 60)

    wm = WorkingMemory(
        task="处理订单 ORD-2024-002 审批",
        current_step="信用检查",
        collected_data={"order_amount": 128000, "customer": "李四", "inventory": "ok"},
        decisions=["库存检查通过"],
        pending_actions=["信用额度确认", "风险评估", "人工审批"],
    )

    print(f"  工作记忆快照:")
    print(f"  {wm.to_context()}")

    # ── 4. ConversationMemoryManager ─────────────────────
    print(f"\n\n▶ 4. ConversationMemoryManager（集成 + 持久化）")
    print("─" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建并使用
        mgr = ConversationMemoryManager(persist_dir=tmpdir, compact_threshold=8)

        mgr.remember_fact("张三偏好邮件通知", category="preference", importance=0.8)
        mgr.remember_fact("张三是 VIP 客户", category="profile", importance=0.9)
        mgr.ltm.set_preference("通知方式", "邮件")

        # 模拟 10 轮对话（触发自动压缩）
        dialog = [
            ("user", "帮我查订单"), ("assistant", "请提供订单号"),
            ("user", "ORD-001"), ("assistant", "查到了，金额 45000"),
            ("user", "库存够吗"), ("assistant", "库存充足"),
            ("user", "信用如何"), ("assistant", "信用等级 A"),
            ("user", "批准吧"), ("assistant", "已批准"),
        ]
        for role, content in dialog:
            mgr.add_message(role, content)

        print(f"  对话 10 轮后（阈值 8）:")
        print(f"    短期记忆剩余: {len(mgr.stm.messages)} 条（已自动压缩）")
        print(f"    Token 估算: {mgr.stm.token_estimate()}")

        # 持久化
        mgr.save()
        saved_path = os.path.join(tmpdir, "memory.json")
        file_size = os.path.getsize(saved_path)
        print(f"    持久化: memory.json ({file_size} bytes)")

        # 从磁盘恢复
        mgr2 = ConversationMemoryManager(persist_dir=tmpdir)
        print(f"    恢复后: {len(mgr2.stm.messages)} 条消息, {len(mgr2.ltm.facts)} 条事实")

        # 组装上下文
        ctx = mgr.build_context()
        print(f"\n  组装的上下文:")
        print(f"    长期记忆: {ctx['long_term'][:60]}...")
        print(f"    短期消息: {len(ctx['short_term'])} 条")
        print(f"    Token 估算: {ctx['token_estimate']}")

    # ── 5. VectorMemory 演示 ──────────────────────────────
    print(f"\n\n▶ 5. VectorMemory — 向量记忆（余弦相似度检索）")
    print("─" * 60)

    vmem = VectorMemory()
    vmem.add("用户喜欢机械键盘", tags=["偏好"])
    vmem.add("用户的订单号是 ORD-2024-001", tags=["订单"])
    vmem.add("用户是 VIP 企业客户", tags=["等级"])
    vmem.add("用户最近咨询过退款政策", tags=["退款"])
    vmem.add("用户的公司在上海浦东", tags=["地址"])

    results = vmem.search("键盘 购买", top_k=3)
    print(f"  查询: '键盘 购买' → 前 3 结果:")
    for text, score in results:
        print(f"    [{score:.3f}] {text}")

    results2 = vmem.search("退款", top_k=2)
    print(f"  查询: '退款' → 前 2 结果:")
    for text, score in results2:
        print(f"    [{score:.3f}] {text}")

    print(f"  总记忆数: {vmem.size()}")

    # ── 6. TTLCache 演示 ─────────────────────────────────
    print(f"\n\n▶ 6. TTLCache — 带过期时间的记忆缓存")
    print("─" * 60)

    cache = TTLCache(default_ttl_s=0.3)
    cache.set("session:user-1", {"name": "张三", "role": "VIP"}, ttl_s=0.5)
    cache.set("session:user-2", {"name": "李四", "role": "普通"}, ttl_s=0.2)
    cache.set("context:order", {"order_id": "ORD-001"})  # 用默认 TTL

    print(f"  写入 3 条, 当前: {cache.size()} 条")
    print(f"  读取 user-1: {cache.get('session:user-1')}")

    time.sleep(0.25)
    cache.cleanup()
    print(f"  等待 0.25s 后清理: {cache.size()} 条 (user-2 和 context 已过期)")
    print(f"  读取 user-2: {cache.get('session:user-2')}  (已过期)")
    print(f"  读取 user-1: {cache.get('session:user-1')}  (未过期)")

    # ── 架构总结 ──────────────────────────────────────────
    print(f"\n\n{'=' * 60}")
    print("📊 Agent 记忆体系总结:")
    print()
    print("  记忆类型        │ 生命周期   │ 生产方案")
    print("  ────────────────┼──────────┼─────────────────────")
    print("  短期记忆         │ 单次对话   │ message_history")
    print("  工作记忆         │ 单个任务   │ LangGraph State / dict")
    print("  长期记忆         │ 跨会话     │ 向量数据库 / Mem0")
    print("  情景记忆         │ 永久       │ 案例库 + 向量检索")
    print()
    print("  Compaction 策略  │ 适用场景")
    print("  ────────────────┼──────────────────────────────")
    print("  滑动窗口         │ 简单对话，丢弃旧消息可接受")
    print("  摘要压缩         │ 需要保留关键信息的长对话")
    print("  重要性过滤       │ 包含大量工具调用的技术对话")
    print()
    print("  框架支持:")
    print("  - LangChain: ConversationBufferMemory / Summary / VectorStore")
    print("  - LangGraph: State + Checkpoint（工作记忆最强）")
    print("  - Code Puppy: compaction_processor（内置压缩）")
    print("  - Mem0: 独立记忆层（可嵌入任何框架）")


if __name__ == "__main__":
    main()
