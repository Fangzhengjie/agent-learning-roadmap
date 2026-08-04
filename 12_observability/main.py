"""Agent 可观测性（Observability）

核心概念：追踪 Agent 的每一步执行，定位问题和优化性能。

三大支柱：
  1. Tracing（追踪）：Agent 执行的完整链路（LLM 调用 → 工具调用 → 结果）
  2. Metrics（指标）：Token 消耗、延迟、工具调用成功率
  3. Logging（日志）：结构化日志，可搜索可聚合

生产工具：
  - LangSmith：LangChain 官方（付费）
  - Langfuse：开源可自部署
  - Phoenix (Arize)：开源，支持 OpenTelemetry
  - OpenTelemetry：通用标准（本示例使用）

本示例展示如何手动构建 Agent 可观测性层，不依赖外部服务。
"""

import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# 1. Trace（追踪）
# ═══════════════════════════════════════════════════════════

@dataclass
class Span:
    """一个执行单元（类似 OpenTelemetry Span）。"""
    span_id: str
    name: str
    span_type: str  # llm / tool / agent / chain
    parent_id: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "running"
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class Trace:
    """一次 Agent 执行的完整追踪。"""
    trace_id: str
    spans: list[Span] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def find_spans(self, span_type: str) -> list[Span]:
        return [s for s in self.spans if s.span_type == span_type]


class Tracer:
    """Agent 追踪器。"""

    def __init__(self):
        self.traces: list[Trace] = []
        self._current_trace: Trace | None = None
        self._span_stack: list[Span] = []

    @contextmanager
    def start_trace(self, name: str, **metadata):
        """开始一次新的追踪。"""
        trace = Trace(
            trace_id=str(uuid.uuid4())[:8],
            metadata={"name": name, "start_time": datetime.now().isoformat(), **metadata},
        )
        self._current_trace = trace
        try:
            yield trace
        finally:
            self.traces.append(trace)
            self._current_trace = None

    @contextmanager
    def start_span(self, name: str, span_type: str, **attributes):
        """开始一个执行单元。"""
        parent = self._span_stack[-1] if self._span_stack else None
        span = Span(
            span_id=str(uuid.uuid4())[:8],
            name=name,
            span_type=span_type,
            parent_id=parent.span_id if parent else None,
            start_time=time.time(),
            attributes=attributes,
        )
        self._span_stack.append(span)
        if self._current_trace:
            self._current_trace.spans.append(span)
        try:
            yield span
        except Exception as e:
            span.status = "error"
            span.events.append({"error": str(e), "time": time.time()})
            raise
        finally:
            span.end_time = time.time()
            if span.status == "running":
                span.status = "ok"
            self._span_stack.pop()


# ═══════════════════════════════════════════════════════════
# 2. Metrics（指标收集）
# ═══════════════════════════════════════════════════════════

@dataclass
class AgentMetrics:
    """Agent 执行指标。"""
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    tool_success: int = 0
    tool_failure: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    # 按模型计价（GPT-4o-mini 参考价格）
    PRICING = {
        "gpt-4o-mini": {"prompt": 0.15 / 1_000_000, "completion": 0.60 / 1_000_000},
        "gpt-4o": {"prompt": 2.50 / 1_000_000, "completion": 10.00 / 1_000_000},
    }

    def record_llm_call(self, model: str, prompt_tokens: int, completion_tokens: int, latency_ms: float):
        self.total_llm_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.latencies_ms.append(latency_ms)

        pricing = self.PRICING.get(model, self.PRICING["gpt-4o-mini"])
        self.total_cost_usd += prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]

    def record_tool_call(self, success: bool):
        self.total_tool_calls += 1
        if success:
            self.tool_success += 1
        else:
            self.tool_failure += 1

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def tool_success_rate(self) -> float:
        total = self.tool_success + self.tool_failure
        return self.tool_success / total if total > 0 else 1.0

    def summary(self) -> dict:
        return {
            "llm_calls": self.total_llm_calls,
            "tool_calls": self.total_tool_calls,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.total_cost_usd, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "tool_success_rate": f"{self.tool_success_rate:.1%}",
        }


# ═══════════════════════════════════════════════════════════
# 演示：模拟一次 Agent 执行并收集可观测性数据
# ═══════════════════════════════════════════════════════════

def simulate_agent_execution(tracer: Tracer, metrics: AgentMetrics):
    """模拟 Agent 处理客服工单的完整流程。"""

    with tracer.start_trace("customer-support", user="alice", task="ticket-T-001"):

        # Step 1: 首次 LLM 调用（理解用户意图）
        with tracer.start_span("llm-intent-analysis", "llm", model="gpt-4o-mini") as span:
            time.sleep(0.05)  # 模拟延迟
            span.attributes["prompt_tokens"] = 350
            span.attributes["completion_tokens"] = 120
            span.attributes["intent"] = "technical_support"
            metrics.record_llm_call("gpt-4o-mini", 350, 120, 52.3)

        # Step 2: 工具调用 — 查询工单
        with tracer.start_span("tool-lookup-ticket", "tool", tool_name="lookup_ticket") as span:
            time.sleep(0.02)
            span.attributes["args"] = {"ticket_id": "T-001"}
            span.attributes["result"] = "found"
            metrics.record_tool_call(success=True)

        # Step 3: 工具调用 — 检查系统状态
        with tracer.start_span("tool-check-status", "tool", tool_name="check_system_status") as span:
            time.sleep(0.01)
            span.attributes["args"] = {"service": "auth"}
            span.attributes["result"] = "healthy"
            metrics.record_tool_call(success=True)

        # Step 4: 第二次 LLM 调用（生成回复）
        with tracer.start_span("llm-generate-response", "llm", model="gpt-4o-mini") as span:
            time.sleep(0.08)
            span.attributes["prompt_tokens"] = 800
            span.attributes["completion_tokens"] = 250
            metrics.record_llm_call("gpt-4o-mini", 800, 250, 83.7)

        # Step 5: 工具调用 — 路由工单
        with tracer.start_span("tool-route-ticket", "tool", tool_name="route_ticket") as span:
            time.sleep(0.01)
            span.attributes["args"] = {"ticket_id": "T-001", "team": "engineering"}
            span.attributes["result"] = "routed"
            metrics.record_tool_call(success=True)

        # Step 6: 模拟一次工具调用失败
        with tracer.start_span("tool-send-notification", "tool", tool_name="send_email") as span:
            time.sleep(0.01)
            span.status = "error"
            span.events.append({"error": "SMTP connection timeout", "time": time.time()})
            metrics.record_tool_call(success=False)

        # Step 7: 最终 LLM 调用（总结）
        with tracer.start_span("llm-summary", "llm", model="gpt-4o-mini") as span:
            time.sleep(0.04)
            span.attributes["prompt_tokens"] = 1200
            span.attributes["completion_tokens"] = 180
            metrics.record_llm_call("gpt-4o-mini", 1200, 180, 41.5)


def main():
    print("=== Agent 可观测性 (Observability) ===\n")

    tracer = Tracer()
    metrics = AgentMetrics()

    # 执行模拟
    simulate_agent_execution(tracer, metrics)

    trace = tracer.traces[0]

    # ── 1. Trace 可视化 ──────────────────────────────────
    print("▶ 1. Trace（执行链路追踪）")
    print("─" * 60)
    print(f"  Trace ID: {trace.trace_id}")
    print(f"  任务: {trace.metadata.get('task')}")
    print(f"  总 Span 数: {len(trace.spans)}\n")

    # 瀑布图
    print("  时间线（瀑布图）:")
    base_time = trace.spans[0].start_time
    for span in trace.spans:
        offset = (span.start_time - base_time) * 1000
        duration = span.duration_ms
        status_icon = "✅" if span.status == "ok" else "❌"
        bar_len = max(1, int(duration / 5))
        bar = "█" * bar_len
        print(f"  {offset:6.1f}ms │{bar} {duration:.1f}ms │ {status_icon} [{span.span_type}] {span.name}")

    # 错误 Span
    errors = [s for s in trace.spans if s.status == "error"]
    if errors:
        print(f"\n  ❌ 错误 Span:")
        for s in errors:
            for e in s.events:
                print(f"     {s.name}: {e.get('error')}")

    # ── 2. Metrics 仪表板 ────────────────────────────────
    print(f"\n\n▶ 2. Metrics（指标仪表板）")
    print("─" * 60)

    summary = metrics.summary()
    print(f"""
  ┌──────────────────────────────────────────┐
  │          Agent 执行指标                   │
  ├──────────────────┬───────────────────────┤
  │ LLM 调用次数      │ {summary['llm_calls']:>20} │
  │ 工具调用次数       │ {summary['tool_calls']:>20} │
  │ 总 Token 消耗     │ {summary['total_tokens']:>20} │
  │   Prompt Token    │ {summary['prompt_tokens']:>20} │
  │   Completion Token│ {summary['completion_tokens']:>20} │
  │ 估算费用 (USD)     │ ${summary['cost_usd']:>19} │
  │ 平均延迟           │ {summary['avg_latency_ms']:>17}ms │
  │ P95 延迟           │ {summary['p95_latency_ms']:>17}ms │
  │ 工具成功率         │ {summary['tool_success_rate']:>20} │
  └──────────────────┴───────────────────────┘""")

    # ── 3. 按类型分析 ────────────────────────────────────
    print(f"\n\n▶ 3. 分类分析")
    print("─" * 60)

    llm_spans = trace.find_spans("llm")
    tool_spans = trace.find_spans("tool")

    print(f"\n  LLM 调用明细:")
    for s in llm_spans:
        pt = s.attributes.get("prompt_tokens", 0)
        ct = s.attributes.get("completion_tokens", 0)
        print(f"    {s.name}: {pt}+{ct}={pt+ct} tokens, {s.duration_ms:.1f}ms")

    print(f"\n  工具调用明细:")
    for s in tool_spans:
        status = "✅" if s.status == "ok" else "❌"
        tool = s.attributes.get("tool_name", "?")
        print(f"    {status} {tool}: {s.duration_ms:.1f}ms")

    # ── 4. OpenTelemetry Span 导出 ─────────────────────────
    demo_otel_export(trace)

    # ── 5. 分布式追踪 ────────────────────────────────────
    demo_distributed_tracing()

    # ── 6. 告警规则引擎 ─────────────────────────────────
    demo_alert_rules(trace, metrics)

    # ── 架构总结 ──────────────────────────────────────────
    print(f"\n\n{'=' * 60}")
    print("📊 Agent 可观测性总结:")
    print()
    print("  三大支柱         │ 本示例              │ 生产工具")
    print("  ────────────────┼────────────────────┼──────────────────")
    print("  Tracing          │ Tracer+Span+OTel   │ LangSmith / Langfuse")
    print("  Metrics          │ AgentMetrics       │ Prometheus + Grafana")
    print("  Logging          │ print              │ 结构化日志 + ELK")
    print("  Distributed     │ TraceContext 传播   │ OpenTelemetry / Jaeger")
    print("  Alerting         │ AlertRuleEngine    │ PagerDuty / Grafana Alert")
    print()
    print("  OpenTelemetry 集成:")
    print("  - 大多数 Agent 框架支持导出 OTLP traces")
    print("  - 可接入 Jaeger / Zipkin / Datadog / New Relic")
    print("  - Phoenix (Arize) 专为 LLM 设计的可观测性平台")


# ═══════════════════════════════════════════════════════════
# 4. OpenTelemetry Span 标准格式导出
# ═══════════════════════════════════════════════════════════

class OTelExporter:
    """将内部 Span 转成 OpenTelemetry 标准格式（OTLP JSON）。"""

    @staticmethod
    def to_otlp(trace_obj: Trace) -> dict:
        """转成 OTLP JSON 格式（可发送到 Collector）。"""
        spans = []
        for s in trace_obj.spans:
            spans.append({
                "traceId": trace_obj.trace_id.replace("-", "")[:32].ljust(32, "0"),
                "spanId": s.span_id.replace("-", "")[:16].ljust(16, "0"),
                "parentSpanId": (s.parent_id or "").replace("-", "")[:16] or None,
                "name": s.name,
                "kind": "SPAN_KIND_INTERNAL",
                "startTimeUnixNano": int(s.start_time * 1e9),
                "endTimeUnixNano": int(s.end_time * 1e9),
                "status": {"code": "STATUS_CODE_OK" if s.status == "ok" else "STATUS_CODE_ERROR"},
                "attributes": [
                    {"key": k, "value": {"stringValue": str(v)}}
                    for k, v in s.attributes.items()
                ],
            })
        return {
            "resourceSpans": [{
                "resource": {"attributes": [
                    {"key": "service.name", "value": {"stringValue": "agent-service"}},
                    {"key": "service.version", "value": {"stringValue": "1.0.0"}},
                ]},
                "scopeSpans": [{"scope": {"name": "agent-tracer"}, "spans": spans}],
            }]
        }


def demo_otel_export(trace_obj: Trace):
    print(f"\n\n▶ 4. OpenTelemetry Span 标准导出")
    print("─" * 60)
    otlp = OTelExporter.to_otlp(trace_obj)
    spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
    print(f"  导出 {len(spans)} 个 Span (OTLP JSON 格式):")
    for s in spans[:3]:
        attrs = {a["key"]: a["value"]["stringValue"] for a in s["attributes"][:2]}
        print(f"    {s['name']:25s} status={s['status']['code']:20s} attrs={attrs}")
    print(f"    ... ({len(spans)} spans total)")
    print(f"\n  生产集成:")
    print(f"    POST /v1/traces → OpenTelemetry Collector → Jaeger / Zipkin / Datadog")


# ═══════════════════════════════════════════════════════════
# 5. 分布式追踪（跨服务 TraceContext 传播）
# ═══════════════════════════════════════════════════════════

class TraceContext:
    """W3C Trace Context — 跨服务传播追踪上下文。

    格式: traceparent: 00-{traceId}-{spanId}-{flags}
    """

    def __init__(self, trace_id: str | None = None, span_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex[:32]
        self.span_id = span_id or uuid.uuid4().hex[:16]
        self.flags = "01"  # sampled

    def to_header(self) -> dict[str, str]:
        """生成 HTTP Header (W3C traceparent)。"""
        return {"traceparent": f"00-{self.trace_id}-{self.span_id}-{self.flags}"}

    @classmethod
    def from_header(cls, headers: dict[str, str]) -> "TraceContext":
        """从 HTTP Header 解析。"""
        tp = headers.get("traceparent", "")
        parts = tp.split("-")
        if len(parts) == 4:
            return cls(trace_id=parts[1], span_id=parts[2])
        return cls()

    def child(self) -> "TraceContext":
        """创建子 Span 上下文（同一 trace_id，新 span_id）。"""
        return TraceContext(trace_id=self.trace_id, span_id=uuid.uuid4().hex[:16])


def demo_distributed_tracing():
    print(f"\n\n▶ 5. 分布式追踪 — 跨服务 TraceContext 传播")
    print("─" * 60)

    # 服务 A: Gateway
    ctx = TraceContext()
    print(f"  Service A (Gateway):")
    print(f"    创建 trace: {ctx.to_header()}")

    # 传播到 Service B: Agent
    child_ctx = ctx.child()
    headers = ctx.to_header()
    print(f"\n  Service B (Agent) — 收到 header:")
    parsed = TraceContext.from_header(headers)
    print(f"    解析 traceId: {parsed.trace_id}")
    print(f"    创建子 span:  {child_ctx.span_id}")

    # 传播到 Service C: Tool
    tool_ctx = child_ctx.child()
    print(f"\n  Service C (Tool):")
    print(f"    同一 traceId: {tool_ctx.trace_id}")
    print(f"    新 spanId:    {tool_ctx.span_id}")

    print(f"\n  追踪链路:")
    print(f"    Gateway({ctx.span_id}) → Agent({child_ctx.span_id}) → Tool({tool_ctx.span_id})")
    print(f"    traceId: {ctx.trace_id} (贯穿全部服务)")


# ═══════════════════════════════════════════════════════════
# 6. 告警规则引擎
# ═══════════════════════════════════════════════════════════

@dataclass
class AlertRule:
    name: str
    condition: str  # 人类可读条件描述
    check_fn: object  # Callable
    severity: str = "warning"  # info / warning / critical


@dataclass
class Alert:
    rule_name: str
    severity: str
    message: str
    value: float


class AlertRuleEngine:
    """告警规则引擎 — 基于指标自动触发告警。"""

    def __init__(self):
        self.rules: list[AlertRule] = []
        self.fired: list[Alert] = []

    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)

    def evaluate(self, metrics_data: dict) -> list[Alert]:
        """评估所有规则。"""
        alerts = []
        for rule in self.rules:
            try:
                result = rule.check_fn(metrics_data)
                if result is not None:
                    alert = Alert(rule.name, rule.severity, result[0], result[1])
                    alerts.append(alert)
            except Exception:
                pass
        self.fired.extend(alerts)
        return alerts


def demo_alert_rules(trace_obj: Trace, metrics):
    print(f"\n\n▶ 6. 告警规则引擎")
    print("─" * 60)

    engine = AlertRuleEngine()

    # 定义规则
    engine.add_rule(AlertRule(
        "high_latency",
        "Agent 总延迟 > 5s",
        lambda m: (f"总延迟 {m.get('total_ms', 0):.0f}ms > 5000ms", m.get('total_ms', 0))
        if m.get('total_ms', 0) > 5000 else None,
        severity="warning",
    ))
    engine.add_rule(AlertRule(
        "high_token_usage",
        "单次 Token > 2000",
        lambda m: (f"Token 用量 {m.get('tokens', 0)} > 2000", m.get('tokens', 0))
        if m.get('tokens', 0) > 2000 else None,
        severity="warning",
    ))
    engine.add_rule(AlertRule(
        "tool_failure",
        "工具调用失败率 > 30%",
        lambda m: (f"失败率 {m.get('fail_rate', 0):.0%} > 30%", m.get('fail_rate', 0))
        if m.get('fail_rate', 0) > 0.3 else None,
        severity="critical",
    ))
    engine.add_rule(AlertRule(
        "cost_spike",
        "单次成本 > $0.10",
        lambda m: (f"成本 ${m.get('cost', 0):.4f} > $0.10", m.get('cost', 0))
        if m.get('cost', 0) > 0.10 else None,
        severity="critical",
    ))

    # 模拟指标数据
    total_ms = sum(s.duration_ms for s in trace_obj.spans)
    total_tokens = sum(s.attributes.get("prompt_tokens", 0) + s.attributes.get("completion_tokens", 0)
                       for s in trace_obj.spans)
    tool_spans = trace_obj.find_spans("tool")
    fail_rate = sum(1 for s in tool_spans if s.status != "ok") / max(len(tool_spans), 1)

    metrics_data = {
        "total_ms": total_ms,
        "tokens": total_tokens,
        "fail_rate": fail_rate,
        "cost": total_tokens * 0.00003,
    }
    print(f"  指标: 延迟={total_ms:.0f}ms, tokens={total_tokens}, 失败率={fail_rate:.0%}, 成本=${metrics_data['cost']:.4f}")

    alerts = engine.evaluate(metrics_data)
    print(f"\n  规则: {len(engine.rules)} 条, 触发: {len(alerts)} 条")
    for a in alerts:
        icon = "🚨" if a.severity == "critical" else "⚠️"
        print(f"    {icon} [{a.severity}] {a.rule_name}: {a.message}")

    if not alerts:
        print(f"    ✅ 所有指标正常")


if __name__ == "__main__":
    main()
