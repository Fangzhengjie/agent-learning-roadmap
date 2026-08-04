"""AI Gateway / LLM Router — 大模型网关层

核心概念：在 Agent 和 LLM 之间加一层网关，统一管理路由、降级、成本、安全。

生产环境的 Agent 不应直接调用 LLM API:
  ┌────────┐     ┌──────────────┐     ┌───────────┐
  │ Agent  │ →   │  AI Gateway  │ →   │ LLM API   │
  │        │     │ 路由/降级/限流 │     │ GPT/Claude│
  └────────┘     └──────────────┘     └───────────┘

本示例展示 AI Gateway 的核心能力：
  1. 模型路由 — 按任务/成本/延迟选择模型
  2. Fallback 降级 — 主模型挂了自动切备用
  3. 成本优化 — Token 预算 + 缓存 + 模型降级
  4. 限流与配额 — Rate Limiting + 用户配额
  5. 可观测性 — 请求日志 + 延迟追踪 + 成本统计
  6. 开源方案对比
"""

import hashlib
import hmac
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# 1. 模型路由
# ═══════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """模型配置。"""
    name: str
    provider: str
    cost_per_1k_input: float   # $/1K input tokens
    cost_per_1k_output: float  # $/1K output tokens
    avg_latency_ms: int
    max_context: int
    quality_score: float       # 0~1


MODELS = {
    "gpt-4o": ModelConfig("gpt-4o", "openai", 0.0025, 0.01, 800, 128000, 0.95),
    "gpt-4o-mini": ModelConfig("gpt-4o-mini", "openai", 0.00015, 0.0006, 400, 128000, 0.85),
    "claude-3.5-sonnet": ModelConfig("claude-3.5-sonnet", "anthropic", 0.003, 0.015, 900, 200000, 0.95),
    "claude-3.5-haiku": ModelConfig("claude-3.5-haiku", "anthropic", 0.0008, 0.004, 300, 200000, 0.80),
    "deepseek-v3": ModelConfig("deepseek-v3", "deepseek", 0.00014, 0.00028, 500, 128000, 0.88),
}


class ModelRouter:
    """模型路由器 — 按策略选择最优模型。"""

    def __init__(self, models: dict[str, ModelConfig]):
        self.models = models

    def route_by_task(self, task_type: str) -> ModelConfig:
        """按任务类型路由。"""
        routing_table = {
            "tool_calling": "gpt-4o-mini",      # 工具调用：快速 + 便宜
            "complex_reasoning": "gpt-4o",       # 复杂推理：质量优先
            "code_generation": "claude-3.5-sonnet",  # 代码生成：Claude 最强
            "simple_qa": "deepseek-v3",          # 简单问答：最便宜
            "long_context": "claude-3.5-sonnet", # 长上下文：200K 窗口
            "classification": "gpt-4o-mini",     # 分类：便宜够用
        }
        model_name = routing_table.get(task_type, "gpt-4o-mini")
        return self.models[model_name]

    def route_by_cost(self, max_cost_per_1k: float) -> ModelConfig:
        """按成本上限路由（选最便宜的满足条件的模型）。"""
        candidates = [m for m in self.models.values()
                      if m.cost_per_1k_input <= max_cost_per_1k]
        if not candidates:
            return self.models["deepseek-v3"]
        return min(candidates, key=lambda m: m.cost_per_1k_input)

    def route_by_latency(self, max_latency_ms: int) -> ModelConfig:
        """按延迟上限路由（选最快且质量最高的）。"""
        candidates = [m for m in self.models.values()
                      if m.avg_latency_ms <= max_latency_ms]
        if not candidates:
            return min(self.models.values(), key=lambda m: m.avg_latency_ms)
        return max(candidates, key=lambda m: m.quality_score)


def show_model_routing():
    """展示模型路由。"""
    print("▶ 1. 模型路由 — 按任务/成本/延迟选择模型")
    print("─" * 60)

    router = ModelRouter(MODELS)

    # 按任务路由
    print("\n  按任务类型路由:")
    for task in ["tool_calling", "complex_reasoning", "code_generation", "simple_qa"]:
        model = router.route_by_task(task)
        print(f"    {task:20s} → {model.name:22s} (${model.cost_per_1k_input}/1K)")

    # 按成本路由
    print("\n  按成本上限路由:")
    for budget in [0.001, 0.003, 0.01]:
        model = router.route_by_cost(budget)
        print(f"    ≤${budget}/1K      → {model.name:22s}")

    # 按延迟路由
    print("\n  按延迟上限路由:")
    for latency in [350, 500, 1000]:
        model = router.route_by_latency(latency)
        print(f"    ≤{latency}ms       → {model.name:22s}")

    print(f"""
  路由策略:
  ──────────────┬──────────────────────────────────────
  任务路由       │ 不同任务用不同模型（最常见）
  成本路由       │ 在预算内选最好的模型
  延迟路由       │ 在延迟要求内选质量最高的
  A/B 测试路由   │ 按比例分流到不同模型（评估效果）
  用户等级路由   │ VIP 用 GPT-4o，普通用户用 mini
  地域路由       │ 国内用 DeepSeek，国际用 OpenAI""")


# ═══════════════════════════════════════════════════════════
# 2. Fallback 降级
# ═══════════════════════════════════════════════════════════

class FallbackChain:
    """降级链 — 主模型失败时自动切换备用模型。"""

    def __init__(self, chain: list[str], models: dict[str, ModelConfig]):
        self.chain = chain
        self.models = models

    def call(self, prompt: str) -> dict:
        """模拟调用，按降级链依次尝试。"""
        for i, model_name in enumerate(self.chain):
            model = self.models[model_name]
            # 模拟：第一个模型有 30% 概率失败
            simulated_failure = (i == 0 and hash(prompt) % 10 < 3)

            if simulated_failure:
                print(f"    ❌ {model_name}: 超时/限流/5xx")
                continue

            latency = model.avg_latency_ms
            cost = model.cost_per_1k_input * 0.5  # 假设 500 tokens
            print(f"    ✅ {model_name}: {latency}ms, ${cost:.4f}")
            return {"model": model_name, "latency": latency, "cost": cost}

        print(f"    💀 所有模型均失败")
        return {"model": None, "error": "all_models_failed"}


def show_fallback():
    """展示 Fallback 降级策略。"""
    print(f"\n\n▶ 2. Fallback 降级 — 主模型挂了自动切备用")
    print("─" * 60)

    print(f"""
  降级链配置:
  ─────────────────────────────────────────────────────────
  GPT-4o → Claude 3.5 Sonnet → GPT-4o-mini → DeepSeek V3

  触发降级的条件:
  ──────────────┬──────────────────────────────────────
  超时           │ 请求超过 10s 无响应
  429 限流       │ Rate Limit 触发
  5xx 错误       │ 服务端错误
  模型不可用     │ 维护/下线/地域不可达
  Token 超限     │ 上下文超过模型窗口 → 切长窗口模型
  """)

    chain = FallbackChain(
        ["gpt-4o", "claude-3.5-sonnet", "gpt-4o-mini", "deepseek-v3"],
        MODELS
    )

    print("  模拟降级调用:")
    for prompt in ["查询工单 T-001", "分析错误日志", "生成报告"]:
        print(f"\n    请求: {prompt}")
        chain.call(prompt)


# ═══════════════════════════════════════════════════════════
# 3. 成本优化
# ═══════════════════════════════════════════════════════════

class PromptCache:
    """语义缓存 — 相似请求直接返回缓存结果。"""

    def __init__(self):
        self.cache: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0

    def _hash(self, prompt: str) -> str:
        """简化的语义哈希（生产中用 embedding 相似度）。"""
        normalized = prompt.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()[:8]

    def get(self, prompt: str) -> dict | None:
        key = self._hash(prompt)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, prompt: str, response: dict):
        key = self._hash(prompt)
        self.cache[key] = response

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def show_cost_optimization():
    """展示成本优化策略。"""
    print(f"\n\n▶ 3. 成本优化 — Token 预算 + 缓存 + 模型降级")
    print("─" * 60)

    # 缓存演示
    cache = PromptCache()
    queries = [
        "SmartFlow 系统要求是什么？",
        "SmartFlow 系统要求是什么？",  # 命中缓存
        "如何部署 SmartFlow？",
        "SmartFlow 系统要求是什么？",  # 命中缓存
        "如何部署 SmartFlow？",        # 命中缓存
    ]

    print("\n  语义缓存演示:")
    for q in queries:
        cached = cache.get(q)
        if cached:
            print(f"    🟢 缓存命中: {q[:25]}... (省 ~$0.003)")
        else:
            cache.set(q, {"answer": "模拟回答"})
            print(f"    🔴 缓存未命中: {q[:25]}... (调用 LLM)")

    print(f"    缓存命中率: {cache.hit_rate:.0%}")

    print(f"""
  成本优化策略:
  ─────────────────────────────────────────────────────────

  1. 模型降级（最直接）:
  ──────────────┬──────────────┬──────────────────────
  复杂推理       │ GPT-4o       │ $2.5/M input
  日常任务       │ GPT-4o-mini  │ $0.15/M input (省 94%)
  简单分类       │ DeepSeek V3  │ $0.14/M input (省 94%)

  2. Prompt 压缩:
  ──────────────┬──────────────────────────────────────
  System Prompt │ 精简到 500 token（vs 常见 2000+）
  Few-shot      │ 用 1~2 个示例（vs 5+）
  历史消息       │ 滑动窗口（只保留最近 5 轮）
  RAG 上下文    │ top_k=3（vs top_k=10）

  3. 语义缓存:
  ──────────────┬──────────────────────────────────────
  完全匹配缓存   │ 相同 prompt → 直接返回（TTL: 1h）
  语义相似缓存   │ 语义相似 prompt → 返回（余弦 > 0.95）
  结果缓存       │ 工具调用结果缓存（无副作用的工具）

  4. 批处理:
  ──────────────┬──────────────────────────────────────
  Batch API     │ OpenAI 异步批处理（成本降 50%）
  批量嵌入       │ 合并多个文本一次嵌入

  月度成本估算（日均 10K 次 Agent 调用）:
  ──────────────┬──────────────────────────────────────
  全用 GPT-4o   │ ~$750/月
  智能路由 + 缓存│ ~$150/月 (节省 80%)""")


# ═══════════════════════════════════════════════════════════
# 4. 限流与配额
# ═══════════════════════════════════════════════════════════

class RateLimiter:
    """令牌桶限流器。"""

    def __init__(self, max_rpm: int, max_tpm: int):
        self.max_rpm = max_rpm  # 每分钟最大请求数
        self.max_tpm = max_tpm  # 每分钟最大 Token 数
        self.request_count = 0
        self.token_count = 0

    def check(self, tokens: int) -> tuple[bool, str]:
        if self.request_count >= self.max_rpm:
            return False, f"请求限流: {self.request_count}/{self.max_rpm} RPM"
        if self.token_count + tokens > self.max_tpm:
            return False, f"Token 限流: {self.token_count}/{self.max_tpm} TPM"
        self.request_count += 1
        self.token_count += tokens
        return True, "通过"


def show_rate_limiting():
    """展示限流与配额。"""
    print(f"\n\n▶ 4. 限流与配额 — Rate Limiting + 用户配额")
    print("─" * 60)

    limiter = RateLimiter(max_rpm=5, max_tpm=10000)

    print("\n  令牌桶限流演示 (max 5 RPM, 10K TPM):")
    requests = [
        ("查询工单", 500),
        ("分析报告", 2000),
        ("代码审查", 3000),
        ("生成文档", 4000),
        ("翻译文本", 1500),
        ("总结会议", 800),  # 会被限流
    ]
    for prompt, tokens in requests:
        ok, msg = limiter.check(tokens)
        icon = "✅" if ok else "❌"
        print(f"    {icon} {prompt} ({tokens} tokens): {msg}")

    print(f"""
  限流维度:
  ──────────────┬──────────────────────────────────────
  RPM           │ 每分钟请求数（防突发流量）
  TPM           │ 每分钟 Token 数（防大 prompt）
  RPD           │ 每天请求数（成本控制）
  per-user      │ 每用户独立限额
  per-model     │ 每个模型独立限额

  配额管理:
  ──────────────┬──────────────────────────────────────
  免费用户       │ 100 次/天, GPT-4o-mini only
  付费用户       │ 10K 次/天, 所有模型
  企业用户       │ 无限制, 自定义模型
  内部服务       │ 独立配额池, 高优先级

  超限处理:
  ──────────────┬──────────────────────────────────────
  排队           │ 等待令牌恢复（用户感知延迟）
  降级           │ 自动切换到更便宜的模型
  拒绝           │ 返回 429 + Retry-After
  告警           │ 通知管理员配额即将耗尽""")


# ═══════════════════════════════════════════════════════════
# 5. 可观测性
# ═══════════════════════════════════════════════════════════

def show_observability():
    """展示 Gateway 可观测性。"""
    print(f"\n\n▶ 5. 可观测性 — 请求日志 + 延迟追踪 + 成本统计")
    print("─" * 60)

    # 模拟请求日志
    logs = [
        {"ts": "14:01:03", "model": "gpt-4o-mini", "in": 450, "out": 120, "ms": 380, "cost": 0.0001, "status": "ok"},
        {"ts": "14:01:05", "model": "gpt-4o", "in": 2100, "out": 800, "ms": 1200, "cost": 0.013, "status": "ok"},
        {"ts": "14:01:07", "model": "gpt-4o", "in": 1800, "out": 0, "ms": 10000, "cost": 0, "status": "timeout"},
        {"ts": "14:01:07", "model": "claude-3.5-sonnet", "in": 1800, "out": 650, "ms": 900, "cost": 0.015, "status": "fallback"},
        {"ts": "14:01:10", "model": "gpt-4o-mini", "in": 300, "out": 80, "ms": 250, "cost": 0.0001, "status": "cached"},
    ]

    print("\n  请求日志示例:")
    print(f"    {'时间':8s} {'模型':20s} {'IN':>5s} {'OUT':>5s} {'延迟':>7s} {'费用':>8s} {'状态':8s}")
    print(f"    {'─'*60}")
    for l in logs:
        status_icon = {"ok": "✅", "timeout": "❌", "fallback": "⚠️", "cached": "🟢"}[l["status"]]
        print(f"    {l['ts']:8s} {l['model']:20s} {l['in']:>5d} {l['out']:>5d} {l['ms']:>5d}ms ${l['cost']:<7.4f} {status_icon} {l['status']}")

    total_cost = sum(l["cost"] for l in logs)
    avg_latency = sum(l["ms"] for l in logs) / len(logs)
    print(f"\n    合计: ${total_cost:.4f} | 平均延迟: {avg_latency:.0f}ms | 降级率: 20%")

    print(f"""
  监控面板关键指标:
  ──────────────┬──────────────────────────────────────
  请求量 (QPS)   │ 每秒请求数 + 趋势
  成功率         │ 2xx / 总请求（目标 > 99.5%）
  P50/P95 延迟   │ 中位数和长尾延迟
  Token 消耗     │ 输入/输出 Token 分布
  日/月成本      │ 实时费用 + 预算消耗进度
  模型分布       │ 各模型的请求占比
  降级率         │ fallback 触发频率
  缓存命中率     │ 缓存节省的请求比例""")


# ═══════════════════════════════════════════════════════════
# 6. 开源方案
# ═══════════════════════════════════════════════════════════

def show_solutions():
    """展示开源 Gateway 方案。"""
    print(f"\n\n▶ 6. 开源方案对比")
    print("─" * 60)

    print(f"""
  开源 AI Gateway:
  ──────────────┬──────────────────────────────────────
  LiteLLM       │ 最流行，100+ 模型统一 API
                │ OpenAI 兼容接口，Python SDK
                │ 路由 / Fallback / 限流 / 成本追踪
  ──────────────┼──────────────────────────────────────
  Portkey        │ AI Gateway SaaS + 开源
                │ 路由 / 缓存 / 重试 / 可观测性
                │ 企业级，支持 guardrails
  ──────────────┼──────────────────────────────────────
  AI Gateway    │ Cloudflare 开源
  (Cloudflare)  │ 边缘部署，缓存 + 限流 + 日志
  ──────────────┼──────────────────────────────────────
  Helicone      │ 可观测性优先的 Gateway
                │ 请求日志 + 成本分析 + Prompt 管理
  ──────────────┼──────────────────────────────────────
  OneAPI        │ 国内常用，多模型统一管理
                │ 渠道管理 + 令牌分发 + 额度控制

  LiteLLM 快速接入示例:
  ─────────────────────────────────────────────────────────
  from litellm import completion

  # 统一接口调用任何模型
  response = completion(
      model="gpt-4o-mini",        # 或 "claude-3.5-sonnet"
      messages=[{{"role": "user", "content": "hello"}}],
      fallbacks=["claude-3.5-haiku", "deepseek-v3"],
      timeout=10,
      num_retries=2,
  )

  # 路由配置
  router = litellm.Router(
      model_list=[
          {{"model_name": "fast", "litellm_params": {{"model": "gpt-4o-mini"}}}},
          {{"model_name": "smart", "litellm_params": {{"model": "gpt-4o"}}}},
      ],
      routing_strategy="least-busy",  # 或 "cost-based"
  )""")


# ═══════════════════════════════════════════════════════════
# 7. 认证鉴权 (AuthMiddleware)
# ═══════════════════════════════════════════════════════════

@dataclass
class APIKeyInfo:
    """API Key 信息。"""
    key_id: str
    tenant_id: str
    name: str
    scopes: list[str] = field(default_factory=lambda: ["chat"])
    rate_limit_rpm: int = 60
    active: bool = True


class AuthMiddleware:
    """认证中间件 — 支持 API Key 和 JWT 验证。"""

    def __init__(self):
        self._keys: dict[str, APIKeyInfo] = {}
        self._jwt_secret = "gateway-secret-key"

    def register_key(self, info: APIKeyInfo) -> str:
        """注册 API Key，返回完整 key。"""
        full_key = f"gw-{info.key_id}-{uuid.uuid4().hex[:16]}"
        self._keys[full_key] = info
        return full_key

    def verify_api_key(self, key: str) -> tuple[bool, APIKeyInfo | str]:
        """验证 API Key。"""
        info = self._keys.get(key)
        if info is None:
            return False, "Invalid API key"
        if not info.active:
            return False, "API key disabled"
        return True, info

    def create_jwt(self, tenant_id: str, scopes: list[str], ttl_s: int = 3600) -> str:
        """生成简易 JWT（模拟，生产用 PyJWT）。"""
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "tenant_id": tenant_id,
            "scopes": scopes,
            "exp": int(time.time()) + ttl_s,
            "iat": int(time.time()),
        }
        import base64
        h = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        p = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        sig = hmac.new(self._jwt_secret.encode(), f"{h}.{p}".encode(), "sha256").hexdigest()[:32]
        return f"{h}.{p}.{sig}"

    def verify_jwt(self, token: str) -> tuple[bool, dict | str]:
        """验证 JWT。"""
        import base64
        parts = token.split(".")
        if len(parts) != 3:
            return False, "Invalid JWT format"
        try:
            padding = 4 - len(parts[1]) % 4
            payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * padding))
            if payload.get("exp", 0) < time.time():
                return False, "JWT expired"
            return True, payload
        except Exception as e:
            return False, f"JWT decode error: {e}"


# ═══════════════════════════════════════════════════════════
# 8. 多租户隔离 (TenantIsolation)
# ═══════════════════════════════════════════════════════════

@dataclass
class TenantConfig:
    """租户配置。"""
    tenant_id: str
    name: str
    allowed_models: list[str] = field(default_factory=lambda: ["gpt-4o-mini"])
    max_rpm: int = 60
    max_daily_cost: float = 10.0
    daily_cost: float = 0.0


class TenantIsolation:
    """多租户隔离 — 每个租户独立配额、模型权限、成本限制。"""

    def __init__(self):
        self._tenants: dict[str, TenantConfig] = {}

    def register(self, config: TenantConfig):
        self._tenants[config.tenant_id] = config

    def check_access(self, tenant_id: str, model: str, estimated_cost: float) -> tuple[bool, str]:
        """检查租户是否有权使用指定模型和预算。"""
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return False, f"Unknown tenant: {tenant_id}"
        if model not in tenant.allowed_models:
            return False, f"Model {model} not allowed for tenant {tenant.name}"
        if tenant.daily_cost + estimated_cost > tenant.max_daily_cost:
            return False, f"Daily budget exceeded: ${tenant.daily_cost:.2f}/${tenant.max_daily_cost:.2f}"
        return True, "OK"

    def record_cost(self, tenant_id: str, cost: float):
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.daily_cost += cost

    def get_usage(self, tenant_id: str) -> dict:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return {}
        return {
            "tenant": tenant.name,
            "daily_cost": f"${tenant.daily_cost:.4f}",
            "budget_remaining": f"${tenant.max_daily_cost - tenant.daily_cost:.4f}",
            "allowed_models": tenant.allowed_models,
        }


# ═══════════════════════════════════════════════════════════
# 9. 审计日志 (AuditLogger)
# ═══════════════════════════════════════════════════════════

@dataclass
class AuditEntry:
    """审计日志条目。"""
    timestamp: str
    tenant_id: str
    action: str
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    status: str = "ok"
    ip: str = ""
    request_id: str = ""


class AuditLogger:
    """审计日志 — 记录所有 Gateway 请求（合规必备）。"""

    def __init__(self, max_entries: int = 10000):
        self._entries: list[AuditEntry] = []
        self._max = max_entries

    def log(self, entry: AuditEntry):
        if len(self._entries) >= self._max:
            self._entries = self._entries[-self._max // 2:]
        self._entries.append(entry)

    def query(self, tenant_id: str | None = None, limit: int = 10) -> list[AuditEntry]:
        entries = self._entries
        if tenant_id:
            entries = [e for e in entries if e.tenant_id == tenant_id]
        return entries[-limit:]

    def summary(self, tenant_id: str | None = None) -> dict:
        entries = self.query(tenant_id, limit=len(self._entries))
        return {
            "total_requests": len(entries),
            "total_cost": sum(e.cost for e in entries),
            "models_used": list(set(e.model for e in entries if e.model)),
            "error_count": sum(1 for e in entries if e.status != "ok"),
        }


# ═══════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════

def demo_auth_tenant_audit():
    """演示认证 + 多租户 + 审计。"""
    print(f"\n\n▶ 7. 认证鉴权 + 多租户隔离 + 审计日志")
    print("─" * 60)

    # Auth
    auth = AuthMiddleware()
    key1 = auth.register_key(APIKeyInfo("k1", "tenant-a", "前端团队", scopes=["chat", "embed"]))
    key2 = auth.register_key(APIKeyInfo("k2", "tenant-b", "数据团队", scopes=["chat"]))
    print(f"  API Key 注册:")
    print(f"    前端团队: {key1[:20]}...")
    print(f"    数据团队: {key2[:20]}...")

    ok, info = auth.verify_api_key(key1)
    print(f"  验证 key1: {'✅' if ok else '❌'} tenant={info.tenant_id if ok else info}")

    ok, info = auth.verify_api_key("gw-invalid-key")
    print(f"  验证无效key: {'✅' if ok else '❌'} {info}")

    # JWT
    jwt = auth.create_jwt("tenant-a", ["chat"])
    ok, payload = auth.verify_jwt(jwt)
    print(f"  JWT: {'✅' if ok else '❌'} tenant={payload.get('tenant_id') if ok else payload}")

    # Tenant
    print()
    tenant_mgr = TenantIsolation()
    tenant_mgr.register(TenantConfig("tenant-a", "前端团队", ["gpt-4o-mini", "gpt-4o"], max_daily_cost=5.0))
    tenant_mgr.register(TenantConfig("tenant-b", "数据团队", ["gpt-4o-mini"], max_daily_cost=1.0))

    tests = [
        ("tenant-a", "gpt-4o", 0.5),
        ("tenant-b", "gpt-4o", 0.1),       # 模型不允许
        ("tenant-b", "gpt-4o-mini", 0.8),
    ]
    for tid, model, cost in tests:
        ok, msg = tenant_mgr.check_access(tid, model, cost)
        print(f"  {tid} → {model} (${cost}): {'✅' if ok else '❌'} {msg}")
        if ok:
            tenant_mgr.record_cost(tid, cost)

    # Audit
    print()
    audit = AuditLogger()
    for i, (tid, model, cost) in enumerate(tests):
        audit.log(AuditEntry(
            timestamp=datetime.now().isoformat(),
            tenant_id=tid, action="chat", model=model,
            tokens_in=500, tokens_out=200, cost=cost,
            status="ok" if cost < 0.5 else "blocked",
            request_id=f"req-{i+1}",
        ))

    s = audit.summary()
    print(f"  审计汇总: {s['total_requests']} 请求, ${s['total_cost']:.2f} 成本, "
          f"模型: {s['models_used']}")
    print(f"  tenant-a 审计: {audit.summary('tenant-a')}")


def main():
    print("=== AI Gateway / LLM Router ===\n")

    show_model_routing()
    show_fallback()
    show_cost_optimization()
    show_rate_limiting()
    show_observability()
    show_solutions()
    demo_auth_tenant_audit()

    # ── 架构总结 ──────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 AI Gateway 总结:")
    print()
    print("  核心能力:")
    print("  ────────────────────────────────────────────")
    print("  模型路由   │ 按任务/成本/延迟/用户选择模型")
    print("  Fallback  │ 主模型挂了自动切备用")
    print("  成本优化   │ 智能路由 + 缓存节省 80%")
    print("  限流配额   │ RPM/TPM/per-user 多维度")
    print("  可观测性   │ 日志 + 延迟 + 成本 + 降级率")
    print()
    print("  生产 Checklist:")
    print("  ────────────────────────────────────────────")
    print("  □ 配置 Fallback 降级链（至少 2 个备用模型）")
    print("  □ 开启语义缓存（命中率目标 > 30%）")
    print("  □ 设置用户级 Rate Limit")
    print("  □ 监控面板: 成功率 / P95 延迟 / 日成本")
    print("  □ A/B 测试: 新模型灰度 10% 流量验证")
    print("  □ 推荐工具: LiteLLM（开源）或 Portkey（SaaS）")


if __name__ == "__main__":
    main()
