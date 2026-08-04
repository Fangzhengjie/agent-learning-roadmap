"""Agent 测试体系 — 企业级 Agent 质量保障

核心概念：LLM 输出不确定性要求 Agent 测试远比传统软件复杂。

Agent 测试金字塔:
  ┌──────────────────────────────────────────────────────────┐
  │         🔺 E2E 测试（真实 LLM + 真实工具）               │
  │        ──── 昂贵、慢、必要但少量 ────                     │
  │       🔸 集成测试（Mock LLM + 真实工具）                  │
  │      ──── 验证工具调用链路 ────                           │
  │     🔹 单元测试（Mock LLM + Mock 工具）                   │
  │    ──── 快速、便宜、大量 ────                             │
  └──────────────────────────────────────────────────────────┘

本示例展示:
  1. MockLLM — 可控的 LLM 替身（支持多轮/工具调用）
  2. AgentTestCase — 测试用例定义（输入→期望输出→断言）
  3. TestRunner — 批量运行 + 报告生成
  4. RegressionSuite — 回归测试 + 快照对比
  5. ABTestFramework — A/B 测试（对比不同模型/prompt 效果）
"""

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# 1. MockLLM — 可控 LLM 替身
# ═══════════════════════════════════════════════════════════

class MockLLM:
    """Mock LLM — 按规则返回预设结果，用于快速单元测试。

    支持:
    - 精确匹配 / 关键词匹配 / 顺序返回
    - 模拟工具调用
    - 模拟错误/延迟
    - 调用记录（验证 prompt 内容）
    """

    def __init__(self):
        self._rules: list[tuple[Callable[[str], bool], Any]] = []
        self._sequence: list[Any] = []
        self._seq_idx = 0
        self._default: Any = {"role": "assistant", "content": "我是 MockLLM 的默认回复"}
        self._delay_s: float = 0
        self._error_after: int = -1  # 第 N 次调用后报错
        self.call_log: list[dict] = []

    def add_rule(self, matcher: Callable[[str], bool], response: Any) -> "MockLLM":
        """添加匹配规则。"""
        self._rules.append((matcher, response))
        return self

    def add_keyword_rule(self, keyword: str, response: Any) -> "MockLLM":
        """关键词匹配规则。"""
        return self.add_rule(lambda msg: keyword in msg, response)

    def add_tool_call_rule(self, keyword: str, tool_name: str, arguments: dict) -> "MockLLM":
        """模拟 LLM 返回工具调用。"""
        return self.add_keyword_rule(keyword, {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": tool_name, "arguments": json.dumps(arguments)}}],
        })

    def set_sequence(self, responses: list[Any]) -> "MockLLM":
        """按顺序返回（多轮对话）。"""
        self._sequence = responses
        return self

    def set_delay(self, delay_s: float) -> "MockLLM":
        self._delay_s = delay_s
        return self

    def set_error_after(self, n: int) -> "MockLLM":
        """第 n 次调用后抛出异常。"""
        self._error_after = n
        return self

    def chat(self, messages: list[dict]) -> dict:
        """模拟 chat completion 调用。"""
        call_num = len(self.call_log) + 1
        user_msg = messages[-1].get("content", "") if messages else ""

        self.call_log.append({
            "call_num": call_num,
            "messages": messages,
            "timestamp": datetime.now().isoformat(),
        })

        if self._delay_s > 0:
            time.sleep(self._delay_s)

        if 0 < self._error_after < call_num:
            raise ConnectionError(f"MockLLM: simulated error at call #{call_num}")

        # 顺序模式
        if self._sequence and self._seq_idx < len(self._sequence):
            resp = self._sequence[self._seq_idx]
            self._seq_idx += 1
            return resp

        # 规则匹配
        for matcher, response in self._rules:
            if matcher(user_msg):
                return response

        return self._default

    def reset(self):
        self.call_log.clear()
        self._seq_idx = 0


# ═══════════════════════════════════════════════════════════
# 2. AgentTestCase — 测试用例
# ═══════════════════════════════════════════════════════════

@dataclass
class Assertion:
    """单个断言。"""
    name: str
    check_fn: Callable[[Any], bool]
    message: str = ""


@dataclass
class AgentTestCase:
    """Agent 测试用例。"""
    name: str
    input: dict
    assertions: list[Assertion]
    tags: list[str] = field(default_factory=list)
    timeout_s: float = 10.0
    description: str = ""


@dataclass
class TestResult:
    """测试结果。"""
    test_name: str
    passed: bool
    assertions_passed: int = 0
    assertions_total: int = 0
    failures: list[str] = field(default_factory=list)
    duration_ms: float = 0
    output: Any = None
    error: str = ""


# ═══════════════════════════════════════════════════════════
# 3. TestRunner — 测试运行器
# ═══════════════════════════════════════════════════════════

class TestRunner:
    """批量运行测试用例并生成报告。"""

    def __init__(self, agent_fn: Callable[[dict], Any]):
        """agent_fn: 接受输入 dict，返回 Agent 输出的函数。"""
        self.agent_fn = agent_fn
        self.results: list[TestResult] = []

    def run(self, cases: list[AgentTestCase], verbose: bool = True) -> dict:
        """运行所有测试用例。"""
        self.results.clear()

        for case in cases:
            t0 = time.time()
            result = TestResult(
                test_name=case.name,
                passed=True,
                assertions_total=len(case.assertions),
            )

            try:
                output = self.agent_fn(case.input)
                result.output = output

                # 执行断言
                for assertion in case.assertions:
                    try:
                        if assertion.check_fn(output):
                            result.assertions_passed += 1
                        else:
                            result.passed = False
                            result.failures.append(
                                f"{assertion.name}: {assertion.message or 'assertion failed'}")
                    except Exception as e:
                        result.passed = False
                        result.failures.append(f"{assertion.name}: exception: {e}")

            except Exception as e:
                result.passed = False
                result.error = str(e)

            result.duration_ms = (time.time() - t0) * 1000
            self.results.append(result)

            if verbose:
                icon = "✅" if result.passed else "❌"
                print(f"  {icon} {case.name} ({result.assertions_passed}/{result.assertions_total} "
                      f"assertions, {result.duration_ms:.0f}ms)")
                for f in result.failures:
                    print(f"     ⚠️  {f}")

        return self._summary()

    def _summary(self) -> dict:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": f"{passed/total*100:.0f}%" if total else "N/A",
            "total_ms": sum(r.duration_ms for r in self.results),
        }


# ═══════════════════════════════════════════════════════════
# 4. RegressionSuite — 回归测试 + 快照
# ═══════════════════════════════════════════════════════════

class RegressionSuite:
    """回归测试 — 对比当前输出与历史快照。

    用法:
    1. 首次运行生成 baseline 快照
    2. 后续运行对比输出差异
    3. 差异超过阈值则标记为回归
    """

    def __init__(self, snapshot_dir: str):
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)

    def save_snapshot(self, name: str, output: Any):
        path = os.path.join(self.snapshot_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"output": output, "timestamp": datetime.now().isoformat()}, f,
                      ensure_ascii=False, indent=2)

    def load_snapshot(self, name: str) -> Any | None:
        path = os.path.join(self.snapshot_dir, f"{name}.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("output")

    def compare(self, name: str, current_output: Any) -> dict:
        """对比当前输出与快照。"""
        baseline = self.load_snapshot(name)
        if baseline is None:
            self.save_snapshot(name, current_output)
            return {"status": "new_baseline", "name": name}

        # 比较
        if isinstance(baseline, dict) and isinstance(current_output, dict):
            added = set(current_output.keys()) - set(baseline.keys())
            removed = set(baseline.keys()) - set(current_output.keys())
            changed = {k for k in set(baseline.keys()) & set(current_output.keys())
                       if baseline[k] != current_output[k]}
            is_match = not (added or removed or changed)
            return {
                "status": "match" if is_match else "diff",
                "name": name,
                "added_keys": list(added),
                "removed_keys": list(removed),
                "changed_keys": list(changed),
            }
        else:
            is_match = baseline == current_output
            return {"status": "match" if is_match else "diff", "name": name}


# ═══════════════════════════════════════════════════════════
# 5. ABTestFramework — A/B 测试
# ═══════════════════════════════════════════════════════════

@dataclass
class ABVariant:
    """A/B 测试变体。"""
    name: str
    agent_fn: Callable[[dict], Any]
    description: str = ""


class ABTestFramework:
    """A/B 测试 — 对比不同模型/prompt/配置的效果。"""

    def __init__(self, variants: list[ABVariant], scorer: Callable[[Any, dict], float]):
        """scorer: (output, test_input) → 0~1 分数。"""
        self.variants = variants
        self.scorer = scorer

    def run(self, test_inputs: list[dict], verbose: bool = True) -> dict:
        """运行 A/B 测试。"""
        results = {v.name: {"scores": [], "errors": 0, "total_ms": 0} for v in self.variants}

        for inp in test_inputs:
            for variant in self.variants:
                t0 = time.time()
                try:
                    output = variant.agent_fn(inp)
                    score = self.scorer(output, inp)
                    results[variant.name]["scores"].append(score)
                except Exception:
                    results[variant.name]["errors"] += 1
                results[variant.name]["total_ms"] += (time.time() - t0) * 1000

        # 汇总
        summary = {}
        for name, data in results.items():
            scores = data["scores"]
            summary[name] = {
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "errors": data["errors"],
                "avg_ms": data["total_ms"] / len(test_inputs) if test_inputs else 0,
            }

        if verbose:
            for name, s in summary.items():
                print(f"  [{name}] avg={s['avg_score']:.2f}, "
                      f"range=[{s['min_score']:.2f}, {s['max_score']:.2f}], "
                      f"errors={s['errors']}, avg_ms={s['avg_ms']:.0f}")

        # 胜出者
        winner = max(summary, key=lambda n: summary[n]["avg_score"])
        return {"summary": summary, "winner": winner}


# ═══════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════

def demo_mock_llm():
    """演示 MockLLM。"""
    print("▶ 1. MockLLM — 可控 LLM 替身")
    print("─" * 60)

    llm = MockLLM()
    llm.add_keyword_rule("工单", {"role": "assistant", "content": "工单 T-001 状态为处理中"})
    llm.add_tool_call_rule("查询", "lookup_ticket", {"ticket_id": "T-001"})
    llm.add_keyword_rule("你好", {"role": "assistant", "content": "你好！有什么可以帮助你的？"})

    # 文本回复
    r1 = llm.chat([{"role": "user", "content": "帮我看下工单状态"}])
    print(f"  文本回复: {r1['content']}")

    # 工具调用
    r2 = llm.chat([{"role": "user", "content": "查询 T-001"}])
    print(f"  工具调用: {r2['tool_calls'][0]['function']}")

    # 调用日志
    print(f"  调用次数: {len(llm.call_log)}")


def demo_test_runner():
    """演示测试运行器。"""
    print(f"\n\n▶ 2. TestRunner — 批量测试")
    print("─" * 60)

    # 模拟 Agent
    def my_agent(inp: dict) -> dict:
        query = inp.get("query", "")
        if "工单" in query:
            return {"answer": "工单 T-001 已处理", "tool_used": "lookup_ticket", "confidence": 0.95}
        if "天气" in query:
            return {"answer": "今天晴天", "tool_used": "weather_api", "confidence": 0.8}
        return {"answer": "我不确定", "tool_used": None, "confidence": 0.3}

    cases = [
        AgentTestCase(
            name="工单查询",
            input={"query": "帮我查工单 T-001"},
            assertions=[
                Assertion("包含工单号", lambda o: "T-001" in o["answer"]),
                Assertion("使用了正确工具", lambda o: o["tool_used"] == "lookup_ticket"),
                Assertion("置信度>0.8", lambda o: o["confidence"] > 0.8, "置信度太低"),
            ],
            tags=["ticket", "core"],
        ),
        AgentTestCase(
            name="天气查询",
            input={"query": "今天天气怎么样"},
            assertions=[
                Assertion("有回答", lambda o: len(o["answer"]) > 0),
                Assertion("使用天气工具", lambda o: o["tool_used"] == "weather_api"),
            ],
            tags=["weather"],
        ),
        AgentTestCase(
            name="未知问题",
            input={"query": "量子力学是什么"},
            assertions=[
                Assertion("有回答", lambda o: len(o["answer"]) > 0),
                Assertion("低置信度", lambda o: o["confidence"] < 0.5, "未知问题应低置信度"),
            ],
            tags=["fallback"],
        ),
    ]

    runner = TestRunner(my_agent)
    summary = runner.run(cases)
    print(f"\n  汇总: {summary['passed']}/{summary['total']} 通过 "
          f"({summary['pass_rate']}), 总耗时 {summary['total_ms']:.0f}ms")


def demo_regression():
    """演示回归测试。"""
    print(f"\n\n▶ 3. RegressionSuite — 回归快照测试")
    print("─" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        suite = RegressionSuite(tmpdir)

        # 第一次：生成 baseline
        output_v1 = {"answer": "工单已处理", "tool": "lookup_ticket", "latency_ms": 120}
        r1 = suite.compare("test_ticket_query", output_v1)
        print(f"  首次运行: {r1['status']}")

        # 第二次：相同输出
        r2 = suite.compare("test_ticket_query", output_v1)
        print(f"  相同输出: {r2['status']}")

        # 第三次：输出有变化（回归）
        output_v2 = {"answer": "工单已处理", "tool": "search_tickets", "latency_ms": 250}
        r3 = suite.compare("test_ticket_query", output_v2)
        print(f"  输出变化: {r3['status']}")
        if r3["status"] == "diff":
            print(f"    变更字段: {r3['changed_keys']}")


def demo_ab_test():
    """演示 A/B 测试。"""
    print(f"\n\n▶ 4. ABTestFramework — A/B 对比测试")
    print("─" * 60)

    # 模拟两个不同配置的 Agent
    def agent_simple(inp):
        return {"answer": f"简单回复: {inp['query'][:10]}", "confidence": 0.6}

    def agent_detailed(inp):
        return {"answer": f"详细分析: {inp['query']}", "confidence": 0.85}

    # 评分函数
    def scorer(output, inp):
        score = output.get("confidence", 0)
        if len(output.get("answer", "")) > 15:
            score += 0.1
        return min(score, 1.0)

    ab = ABTestFramework(
        variants=[
            ABVariant("简洁模式", agent_simple, "temperature=0, 简短回复"),
            ABVariant("详细模式", agent_detailed, "temperature=0.3, 详细分析"),
        ],
        scorer=scorer,
    )

    test_inputs = [
        {"query": "查询工单 T-001"},
        {"query": "本月销售数据分析"},
        {"query": "帮我写一封邮件"},
    ]

    result = ab.run(test_inputs)
    print(f"\n  🏆 胜出: {result['winner']}")


def main():
    print("=== Agent 测试体系 ===\n")

    demo_mock_llm()
    demo_test_runner()
    demo_regression()
    demo_ab_test()

    # ── 架构总结 ──────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 Agent 测试体系总结:")
    print()
    print("  层级        │ 组件             │ 速度  │ 成本  │ 覆盖度")
    print("  ────────────┼─────────────────┼──────┼──────┼──────")
    print("  单元测试     │ MockLLM + 断言   │ ⚡快  │ 💰低 │ 高")
    print("  集成测试     │ Mock LLM+真工具  │ 🔸中  │ 💰中 │ 中")
    print("  回归测试     │ 快照对比         │ ⚡快  │ 💰低 │ 中")
    print("  A/B 测试     │ 多变体评分       │ 🔸中  │ 💰中 │ 高")
    print("  E2E 测试     │ 真实 LLM + 工具  │ 🐢慢  │ 💰高 │ 最高")
    print()
    print("  生产工具:")
    print("  - pytest + MockLLM: 单元测试基础")
    print("  - DeepEval: LLM 输出质量评测")
    print("  - Ragas: RAG 管道评测")
    print("  - LangSmith: 在线评测 + A/B 测试")
    print("  - Promptfoo: prompt 评测 + 回归检测")


if __name__ == "__main__":
    main()
