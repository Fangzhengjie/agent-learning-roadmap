"""Prompt 压缩 + 模型级联 + Token 预算控制

1. Prompt 压缩 — 减少 Token 数量，降低成本
2. 模型级联 — 小模型预筛 → 大模型精做
3. Token 预算控制器 — 实时追踪和限制开支
"""

import re
import time
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════
# Prompt 压缩
# ═══════════════════════════════════════════════════════════

class PromptCompressor:
    """Prompt 压缩器 — 减少 Token 消耗。

    技术:
      1. 去除冗余空白和格式
      2. 截断低相关性的上下文
      3. 摘要替代原文（LLMlingua 风格）
    """

    @staticmethod
    def compress_whitespace(text: str) -> str:
        """去除多余空白（通常节省 10-20% Token）。"""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        return text.strip()

    @staticmethod
    def truncate_context(text: str, max_chars: int = 2000, keep_ratio: float = 0.7) -> str:
        """截断上下文，保留开头和结尾。"""
        if len(text) <= max_chars:
            return text
        keep = int(max_chars * keep_ratio)
        tail = max_chars - keep
        return text[:keep] + f"\n...(省略 {len(text) - max_chars} 字)...\n" + text[-tail:]

    @staticmethod
    def remove_examples(text: str, max_examples: int = 2) -> str:
        """减少 few-shot 示例数量。"""
        example_markers = ["例如：", "示例：", "Example:", "e.g.,"]
        lines = text.split("\n")
        result, count = [], 0
        skip_until_blank = False
        for line in lines:
            if any(m in line for m in example_markers):
                count += 1
                if count > max_examples:
                    skip_until_blank = True
                    continue
            if skip_until_blank:
                if line.strip() == "":
                    skip_until_blank = False
                continue
            result.append(line)
        return "\n".join(result)

    def compress(self, text: str, target_ratio: float = 0.6) -> dict:
        """综合压缩。返回压缩结果和统计。"""
        original_len = len(text)
        compressed = self.compress_whitespace(text)
        compressed = self.remove_examples(compressed)
        target_chars = int(original_len * target_ratio)
        if len(compressed) > target_chars:
            compressed = self.truncate_context(compressed, target_chars)
        return {
            "text": compressed,
            "original_chars": original_len,
            "compressed_chars": len(compressed),
            "ratio": len(compressed) / original_len if original_len > 0 else 1.0,
            "saved_pct": f"{(1 - len(compressed) / original_len) * 100:.1f}%" if original_len else "0%",
        }


# ═══════════════════════════════════════════════════════════
# 模型级联
# ═══════════════════════════════════════════════════════════

@dataclass
class ModelTier:
    name: str
    cost_per_1k: float  # $ per 1k tokens
    quality: float       # 0-1 quality score
    latency_ms: int
    max_tokens: int = 4096


class ModelCascade:
    """模型级联 — 小模型预筛 → 大模型精做。

    流程:
      1. 先用便宜模型（GPT-4o-mini）做初步判断
      2. 如果置信度低 → 升级到贵模型（GPT-4o）
      3. 简单任务不浪费贵模型的额度

    节省: 典型场景下减少 60-80% 成本
    """

    DEFAULT_TIERS = [
        ModelTier("gpt-4o-mini", 0.00015, 0.7, 200),
        ModelTier("gpt-4o", 0.0025, 0.95, 800),
        ModelTier("claude-3.5-sonnet", 0.003, 0.97, 1000),
    ]

    def __init__(self, tiers: list[ModelTier] | None = None, confidence_threshold: float = 0.8):
        self.tiers = tiers or self.DEFAULT_TIERS
        self.threshold = confidence_threshold
        self.call_log: list[dict] = []

    def route(self, query: str, complexity_fn=None) -> dict:
        """根据任务复杂度选择模型。"""
        if complexity_fn:
            complexity = complexity_fn(query)
        else:
            complexity = self._estimate_complexity(query)

        # 选择模型层级
        selected = self.tiers[0]
        for tier in self.tiers:
            if tier.quality >= complexity:
                selected = tier
                break

        estimated_tokens = len(query) * 2  # 粗估
        estimated_cost = estimated_tokens / 1000 * selected.cost_per_1k

        result = {
            "model": selected.name,
            "complexity": complexity,
            "estimated_cost": f"${estimated_cost:.6f}",
            "latency_ms": selected.latency_ms,
        }
        self.call_log.append(result)
        return result

    @staticmethod
    def _estimate_complexity(query: str) -> float:
        """简单启发式估计任务复杂度（0-1）。"""
        score = 0.3  # 基线
        if len(query) > 200:
            score += 0.2
        complex_words = ["分析", "对比", "推理", "evaluate", "compare", "analyze", "synthesize"]
        if any(w in query.lower() for w in complex_words):
            score += 0.3
        if "?" in query and query.count("?") > 2:
            score += 0.1
        return min(score, 1.0)

    def cost_report(self) -> dict:
        """成本报告。"""
        if not self.call_log:
            return {"total_calls": 0}
        model_counts = {}
        for log in self.call_log:
            model_counts[log["model"]] = model_counts.get(log["model"], 0) + 1
        return {
            "total_calls": len(self.call_log),
            "model_distribution": model_counts,
            "avg_latency_ms": sum(l["latency_ms"] for l in self.call_log) / len(self.call_log),
        }


# ═══════════════════════════════════════════════════════════
# Token 预算控制器
# ═══════════════════════════════════════════════════════════

@dataclass
class BudgetAlert:
    level: str  # info / warning / critical
    message: str
    usage_pct: float


class TokenBudgetController:
    """Token 预算控制器 — 实时追踪和限制 LLM 开支。

    功能:
      - 按日/周/月设置预算
      - 按租户/项目分配额度
      - 超预算告警和自动降级
    """

    def __init__(self, daily_budget: float = 10.0, monthly_budget: float = 200.0):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.daily_usage = 0.0
        self.monthly_usage = 0.0
        self.call_count = 0
        self.tenant_usage: dict[str, float] = {}
        self.alerts: list[BudgetAlert] = []

    def record(self, tokens: int, model: str = "gpt-4o-mini",
               tenant: str = "default", cost_per_1k: float = 0.00015):
        """记录一次 LLM 调用。"""
        cost = tokens / 1000 * cost_per_1k
        self.daily_usage += cost
        self.monthly_usage += cost
        self.call_count += 1
        self.tenant_usage[tenant] = self.tenant_usage.get(tenant, 0) + cost
        self._check_alerts()

    def _check_alerts(self):
        daily_pct = self.daily_usage / self.daily_budget if self.daily_budget else 0
        monthly_pct = self.monthly_usage / self.monthly_budget if self.monthly_budget else 0

        if daily_pct >= 1.0:
            self.alerts.append(BudgetAlert("critical", "日预算已超限!", daily_pct))
        elif daily_pct >= 0.8:
            self.alerts.append(BudgetAlert("warning", f"日预算已用 {daily_pct:.0%}", daily_pct))

        if monthly_pct >= 0.9:
            self.alerts.append(BudgetAlert("critical", f"月预算已用 {monthly_pct:.0%}", monthly_pct))

    def can_proceed(self) -> bool:
        """是否还有预算继续调用。"""
        return self.daily_usage < self.daily_budget

    def suggest_model(self) -> str:
        """根据预算剩余推荐模型。"""
        remaining_pct = 1 - (self.daily_usage / self.daily_budget)
        if remaining_pct > 0.5:
            return "gpt-4o"
        elif remaining_pct > 0.2:
            return "gpt-4o-mini"
        else:
            return "cache_only"

    @property
    def report(self) -> dict:
        return {
            "daily": f"${self.daily_usage:.4f} / ${self.daily_budget:.2f}",
            "monthly": f"${self.monthly_usage:.4f} / ${self.monthly_budget:.2f}",
            "calls": self.call_count,
            "tenant_usage": {k: f"${v:.4f}" for k, v in self.tenant_usage.items()},
            "recommended_model": self.suggest_model(),
        }
