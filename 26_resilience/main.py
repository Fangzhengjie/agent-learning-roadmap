"""Agent 容错与韧性 — 企业级错误处理

核心概念：Agent 调用 LLM/工具时必然遇到失败，韧性层让系统在故障中存活。

Agent 故障类型:
  ┌──────────────────────────────────────────────────────────┐
  │  故障类型         │ 示例                                  │
  ├──────────────────┼──────────────────────────────────────┤
  │  瞬时故障         │ LLM API 429/503、网络抖动             │
  │  持续故障         │ API Key 过期、服务下线                │
  │  超时             │ LLM 推理慢、工具调用卡住              │
  │  数据错误         │ LLM 返回非法 JSON、工具返回异常       │
  │  成本爆炸         │ 无限循环调用、Token 超限              │
  └──────────────────┴──────────────────────────────────────┘

本示例展示:
  1. RetryPolicy — 指数退避 + 抖动 + 可重试异常过滤
  2. CircuitBreaker — 熔断器（关闭→打开→半开）
  3. Fallback — 降级策略链
  4. Bulkhead — 并发隔离（限制同时调用数）
  5. DeadLetterQueue — 死信队列（捕获无法处理的请求）
  6. ResiliencePipeline — 组合所有策略的统一管道
"""

import json
import math
import os
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# 1. RetryPolicy — 重试策略
# ═══════════════════════════════════════════════════════════

class RetryPolicy:
    """重试策略 — 指数退避 + 抖动 + 异常类型过滤。

    enterprise pattern: 只对瞬时故障重试，持续故障直接失败。
    """

    def __init__(self, max_retries: int = 3, base_delay_s: float = 1.0,
                 max_delay_s: float = 30.0, jitter: bool = True,
                 retryable_exceptions: tuple = (ConnectionError, TimeoutError, OSError)):
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

    def _delay(self, attempt: int) -> float:
        """计算指数退避延迟。"""
        delay = min(self.base_delay_s * (2 ** attempt), self.max_delay_s)
        if self.jitter:
            delay *= (0.5 + random.random() * 0.5)  # 50%-100% 抖动
        return delay

    def execute(self, fn: Callable, *args, **kwargs) -> Any:
        """执行函数，遇到可重试异常时自动重试。"""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except self.retryable_exceptions as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self._delay(attempt)
                    time.sleep(delay)
                    continue
            except Exception:
                raise  # 非可重试异常，直接抛出
        raise last_error


# ═══════════════════════════════════════════════════════════
# 2. CircuitBreaker — 熔断器
# ═══════════════════════════════════════════════════════════

class CircuitState(Enum):
    CLOSED = "closed"      # 正常通行
    OPEN = "open"          # 熔断（拒绝所有请求）
    HALF_OPEN = "half_open"  # 试探（允许少量请求测试恢复）


class CircuitBreakerOpen(Exception):
    """熔断器打开时抛出。"""
    pass


class CircuitBreaker:
    """熔断器 — 检测连续故障并快速失败。

    状态转换: CLOSED → (连续失败 >= threshold) → OPEN
              OPEN → (等待 recovery_timeout) → HALF_OPEN
              HALF_OPEN → 成功 → CLOSED
              HALF_OPEN → 失败 → OPEN
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout_s: float = 30.0,
                 half_open_max_calls: int = 1):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.half_open_max_calls = half_open_max_calls
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.recovery_timeout_s:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
            return self._state

    def execute(self, fn: Callable, *args, **kwargs) -> Any:
        """通过熔断器执行函数。"""
        state = self.state

        if state == CircuitState.OPEN:
            raise CircuitBreakerOpen(
                f"Circuit open, retry after {self.recovery_timeout_s}s")

        if state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpen("Half-open: max probe calls reached")
                self._half_open_calls += 1

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
            self._success_count += 1

    def _on_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN

    def stats(self) -> dict:
        return {
            "state": self.state.value,
            "failures": self._failure_count,
            "successes": self._success_count,
        }


# ═══════════════════════════════════════════════════════════
# 3. Fallback — 降级策略链
# ═══════════════════════════════════════════════════════════

class Fallback:
    """降级策略链 — 主方案失败时按顺序尝试备选方案。

    企业场景:
    - GPT-4o 失败 → 降级到 GPT-4o-mini → 降级到缓存结果
    - 实时 API 失败 → 降级到本地模型 → 降级到规则引擎
    """

    def __init__(self, strategies: list[tuple[str, Callable]]):
        """strategies: [(名称, 函数), ...]"""
        self.strategies = strategies

    def execute(self, *args, **kwargs) -> tuple[str, Any]:
        """按顺序尝试策略，返回 (策略名, 结果)。"""
        errors = []
        for name, fn in self.strategies:
            try:
                result = fn(*args, **kwargs)
                return name, result
            except Exception as e:
                errors.append((name, str(e)))
        raise RuntimeError(f"All {len(self.strategies)} fallback strategies failed: {errors}")


# ═══════════════════════════════════════════════════════════
# 4. Bulkhead — 并发隔离
# ═══════════════════════════════════════════════════════════

class BulkheadFull(Exception):
    pass


class Bulkhead:
    """并发隔离 — 限制同时执行的调用数量。

    防止一个慢服务拖垮整个系统。
    """

    def __init__(self, max_concurrent: int = 5, max_wait_s: float = 5.0):
        self.max_concurrent = max_concurrent
        self.max_wait_s = max_wait_s
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active = 0
        self._rejected = 0
        self._lock = threading.Lock()

    def execute(self, fn: Callable, *args, **kwargs) -> Any:
        acquired = self._semaphore.acquire(timeout=self.max_wait_s)
        if not acquired:
            with self._lock:
                self._rejected += 1
            raise BulkheadFull(
                f"Bulkhead full: {self.max_concurrent} concurrent calls, "
                f"waited {self.max_wait_s}s")
        try:
            with self._lock:
                self._active += 1
            return fn(*args, **kwargs)
        finally:
            with self._lock:
                self._active -= 1
            self._semaphore.release()

    def stats(self) -> dict:
        return {"active": self._active, "rejected": self._rejected,
                "max": self.max_concurrent}


# ═══════════════════════════════════════════════════════════
# 5. DeadLetterQueue — 死信队列
# ═══════════════════════════════════════════════════════════

@dataclass
class DeadLetter:
    """死信记录。"""
    id: str
    error: str
    payload: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    retry_count: int = 0


class DeadLetterQueue:
    """死信队列 — 捕获所有无法处理的请求，稍后人工或自动重试。"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queue: deque[DeadLetter] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def push(self, letter: DeadLetter):
        with self._lock:
            self._queue.append(letter)

    def pop(self) -> DeadLetter | None:
        with self._lock:
            return self._queue.popleft() if self._queue else None

    def peek(self, n: int = 5) -> list[DeadLetter]:
        with self._lock:
            return list(self._queue)[:n]

    def size(self) -> int:
        return len(self._queue)

    def drain(self, handler: Callable[[DeadLetter], bool]) -> dict:
        """批量处理死信，handler 返回 True 表示成功。"""
        processed, failed = 0, 0
        while True:
            letter = self.pop()
            if letter is None:
                break
            if handler(letter):
                processed += 1
            else:
                letter.retry_count += 1
                self.push(letter)  # 放回队列
                failed += 1
                if failed > 10:  # 防止无限循环
                    break
        return {"processed": processed, "failed": failed, "remaining": self.size()}


# ═══════════════════════════════════════════════════════════
# 6. ResiliencePipeline — 组合管道
# ═══════════════════════════════════════════════════════════

class ResiliencePipeline:
    """韧性管道 — 组合 Retry + CircuitBreaker + Bulkhead + Fallback + DLQ。

    请求流:
    Bulkhead → CircuitBreaker → Retry → fn()
                                     ↓ 全部失败
                                 Fallback
                                     ↓ 全部失败
                                 DeadLetterQueue
    """

    def __init__(self, retry: RetryPolicy | None = None,
                 circuit_breaker: CircuitBreaker | None = None,
                 bulkhead: Bulkhead | None = None,
                 fallback: Fallback | None = None,
                 dlq: DeadLetterQueue | None = None):
        self.retry = retry
        self.circuit_breaker = circuit_breaker
        self.bulkhead = bulkhead
        self.fallback = fallback
        self.dlq = dlq

    def execute(self, fn: Callable, request_id: str = "", *args, **kwargs) -> Any:
        """通过韧性管道执行函数。"""
        call = fn

        # 包装 Retry
        if self.retry:
            original_call = call
            call = lambda *a, **kw: self.retry.execute(original_call, *a, **kw)

        # 包装 CircuitBreaker
        if self.circuit_breaker:
            retried_call = call
            call = lambda *a, **kw: self.circuit_breaker.execute(retried_call, *a, **kw)

        # 包装 Bulkhead
        if self.bulkhead:
            cb_call = call
            call = lambda *a, **kw: self.bulkhead.execute(cb_call, *a, **kw)

        try:
            return call(*args, **kwargs)
        except Exception as primary_error:
            # 尝试 Fallback
            if self.fallback:
                try:
                    name, result = self.fallback.execute(*args, **kwargs)
                    return result
                except Exception:
                    pass

            # 最终失败 → DLQ
            if self.dlq:
                self.dlq.push(DeadLetter(
                    id=request_id or f"req-{time.time_ns()}",
                    error=str(primary_error),
                    payload={"args": str(args)[:200], "kwargs": str(kwargs)[:200]},
                ))
            raise


# ═══════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════

def demo_retry():
    """演示指数退避重试。"""
    print("▶ 1. RetryPolicy — 指数退避重试")
    print("─" * 60)

    call_count = {"n": 0}
    def flaky_api():
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError(f"Connection refused (attempt {call_count['n']})")
        return {"status": "ok", "data": "result"}

    policy = RetryPolicy(max_retries=3, base_delay_s=0.05, max_delay_s=1.0,
                         retryable_exceptions=(ConnectionError,))

    random.seed(42)
    result = policy.execute(flaky_api)
    print(f"  成功: 第 {call_count['n']} 次尝试返回 {result}")

    # 不可重试异常
    def auth_error():
        raise PermissionError("API Key invalid")

    try:
        policy.execute(auth_error)
    except PermissionError as e:
        print(f"  不可重试异常直接抛出: {e}")


def demo_circuit_breaker():
    """演示熔断器状态转换。"""
    print(f"\n\n▶ 2. CircuitBreaker — 熔断器")
    print("─" * 60)

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_s=0.5, half_open_max_calls=1)
    call_log = []

    def unstable_service():
        if len(call_log) < 4:
            call_log.append("fail")
            raise ConnectionError("Service unavailable")
        call_log.append("ok")
        return "success"

    # 连续失败 → 触发熔断
    for i in range(5):
        try:
            result = cb.execute(unstable_service)
            print(f"  调用 #{i+1}: ✅ {result}  [state={cb.state.value}]")
        except CircuitBreakerOpen as e:
            print(f"  调用 #{i+1}: 🔴 熔断  [state={cb.state.value}]")
        except ConnectionError:
            print(f"  调用 #{i+1}: ❌ 失败  [state={cb.state.value}]")

    # 等待恢复
    print(f"  等待 0.6s 恢复...")
    time.sleep(0.6)

    # 半开状态试探
    try:
        result = cb.execute(unstable_service)
        print(f"  恢复调用: ✅ {result}  [state={cb.state.value}]")
    except Exception as e:
        print(f"  恢复调用: ❌ {e}  [state={cb.state.value}]")

    print(f"  最终统计: {cb.stats()}")


def demo_fallback():
    """演示降级策略链。"""
    print(f"\n\n▶ 3. Fallback — 降级策略链")
    print("─" * 60)

    def call_gpt4o():
        raise ConnectionError("GPT-4o API timeout")

    def call_gpt4o_mini():
        raise ConnectionError("GPT-4o-mini also down")

    def use_cache():
        return {"answer": "缓存结果: 工单 T-001 状态为处理中", "source": "cache"}

    fallback = Fallback([
        ("GPT-4o", call_gpt4o),
        ("GPT-4o-mini", call_gpt4o_mini),
        ("缓存", use_cache),
    ])

    strategy_name, result = fallback.execute()
    print(f"  最终使用: {strategy_name}")
    print(f"  结果: {result}")


def demo_bulkhead():
    """演示并发隔离。"""
    print(f"\n\n▶ 4. Bulkhead — 并发隔离")
    print("─" * 60)

    bh = Bulkhead(max_concurrent=2, max_wait_s=0.3)
    results = []

    def slow_task(task_id):
        time.sleep(0.2)
        return f"task-{task_id} done"

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(lambda i=i: bh.execute(slow_task, i)): i for i in range(5)}
        for f in concurrent.futures.as_completed(futures):
            tid = futures[f]
            try:
                r = f.result()
                results.append(("✅", tid, r))
            except BulkheadFull as e:
                results.append(("🚫", tid, "rejected"))

    for icon, tid, r in sorted(results, key=lambda x: x[1]):
        print(f"  {icon} Task {tid}: {r}")
    print(f"  统计: {bh.stats()}")


def demo_dlq():
    """演示死信队列。"""
    print(f"\n\n▶ 5. DeadLetterQueue — 死信队列")
    print("─" * 60)

    dlq = DeadLetterQueue(max_size=100)

    # 模拟失败请求入队
    for i in range(5):
        dlq.push(DeadLetter(
            id=f"req-{i+1}",
            error="LLM API timeout",
            payload={"query": f"查询工单 T-{i+1:03d}"},
        ))

    print(f"  队列大小: {dlq.size()}")
    print(f"  前 3 条: {[l.id for l in dlq.peek(3)]}")

    # 批量重试处理
    retry_count = {"n": 0}
    def retry_handler(letter: DeadLetter) -> bool:
        retry_count["n"] += 1
        return retry_count["n"] <= 3  # 前 3 个成功

    result = dlq.drain(retry_handler)
    print(f"  处理结果: {result}")


def demo_pipeline():
    """演示完整韧性管道。"""
    print(f"\n\n▶ 6. ResiliencePipeline — 组合管道")
    print("─" * 60)

    dlq = DeadLetterQueue()

    pipeline = ResiliencePipeline(
        retry=RetryPolicy(max_retries=2, base_delay_s=0.05,
                          retryable_exceptions=(ConnectionError, TimeoutError)),
        circuit_breaker=CircuitBreaker(failure_threshold=5, recovery_timeout_s=1.0),
        bulkhead=Bulkhead(max_concurrent=3),
        fallback=Fallback([
            ("本地规则引擎", lambda: {"answer": "规则匹配: 默认回复", "source": "rule_engine"}),
        ]),
        dlq=dlq,
    )

    # 场景 1: 正常调用
    result = pipeline.execute(lambda: {"answer": "工单已处理"}, request_id="req-1")
    print(f"  正常调用: {result}")

    # 场景 2: 瞬时故障 → 重试成功
    attempt = {"n": 0}
    def flaky():
        attempt["n"] += 1
        if attempt["n"] < 2:
            raise ConnectionError("timeout")
        return {"answer": "重试后成功"}

    result = pipeline.execute(flaky, request_id="req-2")
    print(f"  重试成功: {result} (尝试 {attempt['n']} 次)")

    # 场景 3: 全部失败 → Fallback
    def always_fail():
        raise ConnectionError("全部失败")

    result = pipeline.execute(always_fail, request_id="req-3")
    print(f"  降级结果: {result}")

    print(f"  死信队列: {dlq.size()} 条")


def main():
    print("=== Agent 容错与韧性 ===\n")

    demo_retry()
    demo_circuit_breaker()
    demo_fallback()
    demo_bulkhead()
    demo_dlq()
    demo_pipeline()

    # ── 架构总结 ──────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 Agent 韧性体系总结:")
    print()
    print("  组件              │ 功能                │ 何时用")
    print("  ──────────────────┼────────────────────┼────────────────")
    print("  RetryPolicy       │ 指数退避重试        │ LLM API 429/503")
    print("  CircuitBreaker    │ 熔断快速失败        │ 服务长时间不可用")
    print("  Fallback          │ 降级策略链          │ 主模型→备模型→缓存")
    print("  Bulkhead          │ 并发隔离            │ 防止慢调用拖垮系统")
    print("  DeadLetterQueue   │ 失败请求暂存        │ 异步重试 + 人工审查")
    print("  ResiliencePipeline│ 组合管道            │ 生产环境统一入口")
    print()
    print("  生产替代方案:")
    print("  - Python: tenacity（重试）、pybreaker（熔断器）")
    print("  - Java: Resilience4j（Spring Boot 集成）")
    print("  - 分布式: Istio/Envoy Sidecar（网格层面）")


if __name__ == "__main__":
    main()
