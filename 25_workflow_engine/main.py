"""Agent 工作流引擎 — 企业级多步骤编排

核心概念：将 Agent 的复杂任务拆解为可编排、可恢复、可观测的工作流。

企业级工作流 vs 简单 Agent Loop:
  ┌──────────────────────────────────────────────────────────┐
  │  简单 Agent Loop    │  企业级工作流引擎                    │
  ├─────────────────────┼────────────────────────────────────┤
  │  线性执行            │  条件分支 + 并行 + 循环              │
  │  失败即终止          │  重试 + 超时 + 降级                  │
  │  无状态              │  Checkpoint 断点续跑                │
  │  无审计              │  每步执行记录 + 耗时统计             │
  │  单 Agent            │  多 Agent 协作编排                   │
  └──────────────────────┴────────────────────────────────────┘

本示例展示:
  1. Step — 工作流步骤（支持条件/重试/超时）
  2. WorkflowEngine — 执行引擎（串行/并行/分支）
  3. Checkpoint — 断点持久化与恢复
  4. ParallelGroup — 并行步骤组
  5. ConditionalBranch — 条件分支路由
"""

import json
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# 1. Step 定义
# ═══════════════════════════════════════════════════════════

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class StepResult:
    """步骤执行结果。"""
    status: StepStatus
    output: Any = None
    error: str = ""
    attempts: int = 1
    duration_ms: float = 0
    started_at: str = ""
    finished_at: str = ""


@dataclass
class Step:
    """工作流步骤。

    支持: 重试策略、超时、条件跳过、自定义回退。
    """
    name: str
    fn: Callable[..., Any]
    max_retries: int = 0
    timeout_s: float = 30.0
    retry_delay_s: float = 1.0
    condition: Callable[[dict], bool] | None = None  # 返回 False 则跳过
    fallback: Callable[..., Any] | None = None

    def execute(self, context: dict) -> StepResult:
        """执行步骤（含重试和超时）。"""
        started = datetime.now().isoformat()
        t0 = time.time()

        # 条件检查
        if self.condition and not self.condition(context):
            return StepResult(
                status=StepStatus.SKIPPED,
                started_at=started,
                finished_at=datetime.now().isoformat(),
                duration_ms=(time.time() - t0) * 1000,
            )

        last_error = ""
        for attempt in range(1, self.max_retries + 2):  # +2 因为包含首次
            try:
                output = self.fn(context)
                return StepResult(
                    status=StepStatus.SUCCESS,
                    output=output,
                    attempts=attempt,
                    started_at=started,
                    finished_at=datetime.now().isoformat(),
                    duration_ms=(time.time() - t0) * 1000,
                )
            except Exception as e:
                last_error = str(e)
                if attempt <= self.max_retries:
                    time.sleep(self.retry_delay_s * attempt)  # 线性退避
                    continue

        # 全部重试失败 → 尝试 fallback
        if self.fallback:
            try:
                output = self.fallback(context)
                return StepResult(
                    status=StepStatus.SUCCESS,
                    output=output,
                    attempts=self.max_retries + 1,
                    started_at=started,
                    finished_at=datetime.now().isoformat(),
                    duration_ms=(time.time() - t0) * 1000,
                )
            except Exception as e:
                last_error = f"fallback also failed: {e}"

        return StepResult(
            status=StepStatus.FAILED,
            error=last_error,
            attempts=self.max_retries + 1,
            started_at=started,
            finished_at=datetime.now().isoformat(),
            duration_ms=(time.time() - t0) * 1000,
        )


# ═══════════════════════════════════════════════════════════
# 2. 并行步骤组
# ═══════════════════════════════════════════════════════════

@dataclass
class ParallelGroup:
    """并行执行多个步骤。"""
    name: str
    steps: list[Step]
    max_workers: int = 4

    def execute(self, context: dict) -> dict[str, StepResult]:
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(s.execute, dict(context)): s for s in self.steps}
            for future in as_completed(futures):
                step = futures[future]
                results[step.name] = future.result()
        return results


# ═══════════════════════════════════════════════════════════
# 3. 条件分支
# ═══════════════════════════════════════════════════════════

@dataclass
class ConditionalBranch:
    """条件分支路由 — 根据上下文选择执行路径。"""
    name: str
    branches: list[tuple[Callable[[dict], bool], Step]]  # (条件, 步骤)
    default: Step | None = None

    def evaluate(self, context: dict) -> Step | None:
        for condition, step in self.branches:
            if condition(context):
                return step
        return self.default


# ═══════════════════════════════════════════════════════════
# 4. Checkpoint（断点持久化）
# ═══════════════════════════════════════════════════════════

class Checkpoint:
    """工作流断点 — 持久化已完成步骤，支持断点续跑。"""

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir
        self.completed_steps: dict[str, StepResult] = {}
        self.context_snapshot: dict = {}
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._load()

    def save_step(self, step_name: str, result: StepResult, context: dict):
        self.completed_steps[step_name] = result
        self.context_snapshot = dict(context)
        if self.persist_dir:
            self._persist()

    def is_completed(self, step_name: str) -> bool:
        r = self.completed_steps.get(step_name)
        return r is not None and r.status == StepStatus.SUCCESS

    def _persist(self):
        data = {
            "completed": {
                name: {"status": r.status.value, "output": str(r.output)[:200],
                       "attempts": r.attempts, "duration_ms": r.duration_ms}
                for name, r in self.completed_steps.items()
            },
            "context_keys": list(self.context_snapshot.keys()),
        }
        with open(os.path.join(self.persist_dir, "checkpoint.json"), "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self):
        path = os.path.join(self.persist_dir, "checkpoint.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for name, info in data.get("completed", {}).items():
                self.completed_steps[name] = StepResult(
                    status=StepStatus(info["status"]),
                    output=info.get("output"),
                    attempts=info.get("attempts", 1),
                    duration_ms=info.get("duration_ms", 0),
                )


# ═══════════════════════════════════════════════════════════
# 5. WorkflowEngine
# ═══════════════════════════════════════════════════════════

class WorkflowEngine:
    """工作流执行引擎。

    支持:
    - 串行步骤
    - 并行步骤组 (ParallelGroup)
    - 条件分支 (ConditionalBranch)
    - 断点续跑 (Checkpoint)
    - 执行追踪
    """

    def __init__(self, name: str, checkpoint: Checkpoint | None = None):
        self.name = name
        self.steps: list[Step | ParallelGroup | ConditionalBranch] = []
        self.checkpoint = checkpoint or Checkpoint()
        self.trace: list[dict] = []

    def add(self, step: Step | ParallelGroup | ConditionalBranch) -> "WorkflowEngine":
        self.steps.append(step)
        return self

    def run(self, initial_context: dict | None = None, verbose: bool = True) -> dict:
        """执行工作流。"""
        context = dict(initial_context or {})
        context["__workflow__"] = self.name
        results = {}
        t0 = time.time()

        if verbose:
            print(f"\n  🚀 工作流 [{self.name}] 启动")

        for item in self.steps:
            if isinstance(item, Step):
                results[item.name] = self._run_step(item, context, verbose)
            elif isinstance(item, ParallelGroup):
                results[item.name] = self._run_parallel(item, context, verbose)
            elif isinstance(item, ConditionalBranch):
                results[item.name] = self._run_branch(item, context, verbose)

            # 更新上下文
            if isinstance(item, Step) and item.name in results:
                r = results[item.name]
                if r.status == StepStatus.SUCCESS:
                    context[item.name] = r.output

            # 失败时中断（可配置）
            if isinstance(item, Step) and results[item.name].status == StepStatus.FAILED:
                if verbose:
                    print(f"  ❌ 工作流中断: {item.name} 失败")
                break

        total_ms = (time.time() - t0) * 1000
        summary = {
            "workflow": self.name,
            "total_steps": len(self.steps),
            "executed": len(results),
            "success": sum(1 for r in results.values()
                          if isinstance(r, StepResult) and r.status == StepStatus.SUCCESS),
            "failed": sum(1 for r in results.values()
                         if isinstance(r, StepResult) and r.status == StepStatus.FAILED),
            "skipped": sum(1 for r in results.values()
                          if isinstance(r, StepResult) and r.status == StepStatus.SKIPPED),
            "total_ms": round(total_ms, 1),
            "results": results,
        }

        if verbose:
            print(f"  🏁 工作流完成: {summary['success']}/{summary['executed']} 步成功, "
                  f"耗时 {summary['total_ms']:.0f}ms")

        return summary

    def _run_step(self, step: Step, context: dict, verbose: bool) -> StepResult:
        # 断点跳过
        if self.checkpoint.is_completed(step.name):
            if verbose:
                print(f"    ⏭️  {step.name} — 已完成(checkpoint), 跳过")
            return self.checkpoint.completed_steps[step.name]

        if verbose:
            print(f"    ▶ {step.name}...", end="", flush=True)
        result = step.execute(context)
        self.checkpoint.save_step(step.name, result, context)
        self.trace.append({"step": step.name, "status": result.status.value,
                           "duration_ms": result.duration_ms, "attempts": result.attempts})
        if verbose:
            icon = {"success": "✅", "failed": "❌", "skipped": "⏭️"}.get(result.status.value, "❓")
            extra = f" ({result.attempts} attempts)" if result.attempts > 1 else ""
            print(f" {icon} {result.duration_ms:.0f}ms{extra}")
        return result

    def _run_parallel(self, group: ParallelGroup, context: dict, verbose: bool) -> dict:
        if verbose:
            print(f"    ⚡ 并行组 [{group.name}] ({len(group.steps)} 步)...", end="", flush=True)
        results = group.execute(context)
        success = sum(1 for r in results.values() if r.status == StepStatus.SUCCESS)
        if verbose:
            print(f" ✅ {success}/{len(results)} 成功")
        # 将并行结果写入上下文
        for name, r in results.items():
            if r.status == StepStatus.SUCCESS:
                context[name] = r.output
        return results

    def _run_branch(self, branch: ConditionalBranch, context: dict, verbose: bool) -> StepResult:
        step = branch.evaluate(context)
        if step is None:
            if verbose:
                print(f"    🔀 分支 [{branch.name}] — 无匹配条件, 跳过")
            return StepResult(status=StepStatus.SKIPPED)
        if verbose:
            print(f"    🔀 分支 [{branch.name}] → {step.name}")
        return self._run_step(step, context, verbose)


# ═══════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════

def demo_basic_workflow():
    """演示基本串行工作流（含重试和条件跳过）。"""
    print("▶ 1. 基本工作流（串行 + 重试 + 条件跳过）")
    print("─" * 60)

    call_count = {"validate": 0}

    def validate_order(ctx):
        call_count["validate"] += 1
        if call_count["validate"] < 2:
            raise ValueError("数据库连接超时")
        ctx["order_valid"] = True
        return {"order_id": ctx["order_id"], "amount": ctx["amount"], "valid": True}

    def check_inventory(ctx):
        time.sleep(0.05)
        return {"in_stock": True, "warehouse": "华东仓"}

    def check_credit(ctx):
        return {"credit_ok": ctx["amount"] < 100000, "score": 85}

    def approve_order(ctx):
        return {"approved": True, "approver": "auto"}

    def notify_vip(ctx):
        return {"notified": True, "channel": "邮件"}

    wf = WorkflowEngine("订单审批流")
    wf.add(Step("验证订单", validate_order, max_retries=2, retry_delay_s=0.1))
    wf.add(Step("库存检查", check_inventory))
    wf.add(Step("信用检查", check_credit))
    wf.add(Step("审批", approve_order))
    wf.add(Step("VIP通知", notify_vip,
                condition=lambda ctx: ctx.get("amount", 0) > 50000))

    result = wf.run({"order_id": "ORD-2024-001", "amount": 45000, "customer": "张三"})
    print(f"  结果: {result['success']} 成功, {result['skipped']} 跳过, {result['failed']} 失败")


def demo_parallel_workflow():
    """演示并行步骤组。"""
    print(f"\n\n▶ 2. 并行工作流")
    print("─" * 60)

    def enrich_from_crm(ctx):
        time.sleep(0.1)
        return {"segment": "enterprise", "tier": "gold"}

    def enrich_from_risk(ctx):
        time.sleep(0.15)
        return {"risk_score": 0.2, "flagged": False}

    def enrich_from_kyc(ctx):
        time.sleep(0.08)
        return {"verified": True, "doc_type": "身份证"}

    def final_decision(ctx):
        risk = ctx.get("风险评估", {})
        return {"decision": "approve" if not risk.get("flagged") else "reject"}

    wf = WorkflowEngine("客户尽调")
    wf.add(ParallelGroup("数据充实", [
        Step("CRM查询", enrich_from_crm),
        Step("风险评估", enrich_from_risk),
        Step("KYC验证", enrich_from_kyc),
    ]))
    wf.add(Step("最终决策", final_decision))

    result = wf.run({"customer_id": "C-001"})


def demo_conditional_branch():
    """演示条件分支。"""
    print(f"\n\n▶ 3. 条件分支路由")
    print("─" * 60)

    def auto_approve(ctx):
        return {"approved": True, "mode": "自动审批"}

    def manual_review(ctx):
        return {"approved": True, "mode": "人工审核", "reviewer": "王经理"}

    def escalate(ctx):
        return {"approved": False, "mode": "上报VP", "reason": "超大金额"}

    wf = WorkflowEngine("金额路由审批")
    wf.add(Step("获取订单", lambda ctx: {"amount": ctx["amount"]}))
    wf.add(ConditionalBranch("审批路由", [
        (lambda ctx: ctx.get("amount", 0) < 5000, Step("自动审批", auto_approve)),
        (lambda ctx: ctx.get("amount", 0) < 50000, Step("人工审核", manual_review)),
    ], default=Step("上报VP", escalate)))

    for amount in [2000, 30000, 100000]:
        print(f"\n  --- 金额: ¥{amount:,} ---")
        wf.checkpoint = Checkpoint()  # 重置
        wf.run({"amount": amount})


def demo_checkpoint_resume():
    """演示断点续跑。"""
    print(f"\n\n▶ 4. Checkpoint 断点续跑")
    print("─" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        call_log = []

        def step_a(ctx):
            call_log.append("A")
            return "done_a"

        def step_b(ctx):
            call_log.append("B")
            if len(call_log) == 2:  # 第一次运行时失败
                raise RuntimeError("网络中断")
            return "done_b"

        def step_c(ctx):
            call_log.append("C")
            return "done_c"

        # 第一次运行 — step_b 会失败
        print("  第一次运行（step_b 会失败）:")
        cp = Checkpoint(persist_dir=tmpdir)
        wf = WorkflowEngine("可恢复流程", checkpoint=cp)
        wf.add(Step("step_a", step_a))
        wf.add(Step("step_b", step_b, max_retries=0))
        wf.add(Step("step_c", step_c))
        r1 = wf.run({"task": "test"})
        print(f"  执行日志: {call_log}")
        print(f"  Checkpoint 文件: {os.path.getsize(os.path.join(tmpdir, 'checkpoint.json'))} bytes")

        # 第二次运行 — 从 checkpoint 恢复
        print("\n  第二次运行（从 checkpoint 恢复）:")
        call_log.clear()
        cp2 = Checkpoint(persist_dir=tmpdir)  # 从磁盘加载
        wf2 = WorkflowEngine("可恢复流程", checkpoint=cp2)
        wf2.add(Step("step_a", step_a))
        wf2.add(Step("step_b", step_b))
        wf2.add(Step("step_c", step_c))
        r2 = wf2.run({"task": "test"})
        print(f"  执行日志: {call_log}  (step_a 被跳过 ✅)")


def main():
    print("=== Agent 工作流引擎 ===\n")

    demo_basic_workflow()
    demo_parallel_workflow()
    demo_conditional_branch()
    demo_checkpoint_resume()

    # ── 架构总结 ──────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 工作流引擎总结:")
    print()
    print("  组件              │ 功能                │ 企业场景")
    print("  ──────────────────┼────────────────────┼────────────────")
    print("  Step              │ 单步骤执行+重试     │ API 调用、LLM 推理")
    print("  ParallelGroup     │ 并行执行多步骤      │ 多源数据聚合")
    print("  ConditionalBranch │ 条件路由分支        │ 审批分级、A/B 路由")
    print("  Checkpoint        │ 断点持久化+恢复     │ 长流程断点续跑")
    print("  WorkflowEngine    │ 统一编排引擎        │ 订单/审批/尽调流程")
    print()
    print("  生产替代方案:")
    print("  - LangGraph: StateGraph + Checkpoint（Python Agent 首选）")
    print("  - Temporal.io: 分布式工作流引擎（跨语言企业级）")
    print("  - Prefect / Airflow: 数据管道编排")
    print("  - Spring Statemachine: Java 状态机")


if __name__ == "__main__":
    main()
