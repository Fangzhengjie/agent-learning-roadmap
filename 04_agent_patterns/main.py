"""Agent 设计模式 — 从 Tool Use Loop 到 Multi-Agent

核心概念：Agent 的行为由设计模式决定 — 选对模式比选对框架更重要。

本示例用纯 Python 模拟 Agent 核心设计模式：
  1. Tool Use Loop — 所有 Agent 的基础循环
  2. ReAct — Reasoning + Acting 交替
  3. Plan-and-Execute — 先规划后执行
  4. Reflexion — 执行后自我反思修正
  5. Multi-Agent — Supervisor / Debate / Handoff
  6. 模式选型指南
  7. Agent 工程范式 — Loop / Harness / Graph Engineering
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_tools import lookup_ticket, check_system_status, escalate_ticket


# ═══════════════════════════════════════════════════════════
# 1. Tool Use Loop（工具调用循环）
# ═══════════════════════════════════════════════════════════

def demo_tool_use_loop():
    """模拟 Agent 最基础的工具调用循环。"""
    print("▶ 1. Tool Use Loop — 所有 Agent 的基础")
    print("─" * 60)

    print("""
  核心循环（每个 Agent 框架的底层都是这个）:
  ─────────────────────────────────────────────────────────

  while True:
      response = LLM(messages)       # 1. 调用 LLM
      if response.has_tool_calls:     # 2. 需要调工具？
          for tool_call in response.tool_calls:
              result = execute(tool_call)   # 3. 执行工具
              messages.append(result)       # 4. 结果喂回
      else:
          return response.text        # 5. 不需要工具 → 返回

  流程图:
  ─────────────────────────────────────────────────────────

       ┌─────────┐
       │  用户输入 │
       └────┬────┘
            ▼
  ┌──→ ┌────────┐     否     ┌──────────┐
  │    │ 调 LLM  │ ──────→  │ 返回文本   │
  │    └────┬────┘          └──────────┘
  │         │ 有 tool_calls
  │         ▼
  │    ┌──────────┐
  │    │ 执行工具  │
  │    └────┬─────┘
  │         │ 结果
  └─────────┘  (喂回 LLM)
  """)

    # 模拟一次工具调用循环
    print("  模拟执行:")
    user_input = "查一下工单 T-001"

    # 模拟 LLM 决定调用工具
    print(f"    User: {user_input}")
    print(f"    LLM → tool_call: lookup_ticket(ticket_id='T-001')")

    result = json.loads(lookup_ticket("T-001"))
    print(f"    Tool Result: {json.dumps(result, ensure_ascii=False)}")

    # 模拟 LLM 生成最终回复
    print(f"    LLM → 工单 T-001 是 {result['type']} 类型问题，状态为 {result['status']}。")

    print(f"""
  各框架的 Tool Use Loop 实现:
  ──────────────┬──────────────────────────────────────
  OpenAI SDK    │ while tool_calls: execute → append → 再调 LLM
  LangChain     │ AgentExecutor.invoke() (内置循环)
  pydantic-ai   │ agent.run() (内置 max_result_retries)
  Vercel AI     │ maxSteps 参数控制最大循环次数
  Spring AI     │ ChatClient + FunctionCallback 自动循环""")


# ═══════════════════════════════════════════════════════════
# 2. ReAct 引擎（可复用）
# ═══════════════════════════════════════════════════════════

@dataclass
class ToolDef:
    """工具定义（名称 + 函数 + 描述）。"""
    name: str
    fn: Callable[..., str]
    description: str = ""


@dataclass
class TraceStep:
    """一步执行记录。"""
    step_type: str  # "thought" | "action" | "observation" | "answer"
    content: str


class ReactEngine:
    """ReAct 引擎 — Thought → Action → Observation 循环。

    用确定性规则模拟 LLM 的推理和决策，展示 ReAct 核心机制。
    生产中用 LLM 替代 `_default_decide` 方法。
    """

    def __init__(self, tools: list[ToolDef], max_steps: int = 10):
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.trace: list[TraceStep] = []

    def _record(self, step_type: str, content: str):
        self.trace.append(TraceStep(step_type, content))
        icons = {"thought": "💭", "action": "🔧", "observation": "👁", "answer": "💬"}
        print(f"    {icons.get(step_type, '')} {step_type:12s}: {content[:90]}")

    def call_tool(self, name: str, kwargs: dict) -> str:
        """执行工具调用。"""
        tool = self.tools.get(name)
        if not tool:
            return json.dumps({"error": f"工具 '{name}' 不存在"}, ensure_ascii=False)
        try:
            return tool.fn(**kwargs)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def run(self, task: str, decide_fn: Callable | None = None) -> str:
        """运行 ReAct 循环。

        decide_fn: (task, trace, tools) -> (thought, action_name, kwargs) or None
        如果不传，用内置的确定性规则演示。
        """
        print(f"\n  任务: {task}")
        self.trace = []

        for step in range(self.max_steps):
            if decide_fn:
                result = decide_fn(task, self.trace, self.tools)
            else:
                result = self._default_decide(task)

            if result is None:
                break

            thought, action_name, kwargs = result
            self._record("thought", thought)
            self._record("action", f"{action_name}({json.dumps(kwargs, ensure_ascii=False)})")

            observation = self.call_tool(action_name, kwargs)
            self._record("observation", observation[:90])

        answer = self._summarize()
        self._record("answer", answer)
        return answer

    def _default_decide(self, task: str) -> tuple[str, str, dict] | None:
        """内置决策逻辑（模拟 LLM）。"""
        step_count = len([s for s in self.trace if s.step_type == "action"])

        if step_count == 0:
            ticket_id = "T-001"
            for word in task.split():
                if word.startswith("T-"):
                    ticket_id = word.rstrip("，。,")
                    break
            return ("用户需要查询工单信息，先查工单。", "lookup_ticket", {"ticket_id": ticket_id})

        if step_count == 1:
            obs = self.trace[-1].content
            if "technical" in obs:
                return ("工单是技术问题，检查相关服务状态。", "check_system_status", {"service": "auth"})
            return None

        if step_count == 2:
            return ("服务正常，路由工单到 engineering。", "escalate_ticket",
                    {"ticket_id": "T-001", "target_team": "engineering", "notes": "前端问题，服务正常"})

        return None

    def _summarize(self) -> str:
        obs_list = [s for s in self.trace if s.step_type == "observation"]
        return f"已完成任务。共执行 {len(obs_list)} 次工具调用。"


def demo_react():
    """用 ReactEngine 运行真实工具调用。"""
    print(f"\n\n▶ 2. ReAct 引擎 — Reasoning + Acting 交替")
    print("─" * 60)

    print("""
  ReAct (Yao et al., 2022):
  ─────────────────────────────────────────────────────────
  在 Tool Use Loop 基础上，显式要求 LLM 先推理再行动。
  模式: Thought → Action → Observation → Thought → ...
  """)

    engine = ReactEngine(tools=[
        ToolDef("lookup_ticket", lookup_ticket, "查询工单详情"),
        ToolDef("check_system_status", check_system_status, "检查服务状态"),
        ToolDef("escalate_ticket", escalate_ticket, "升级工单"),
    ], max_steps=5)

    engine.run("查工单 T-001 检查服务状态 路由到正确团队")

    print(f"""
  ReAct vs Tool Use Loop:
  ──────────────┬──────────────────────────────────────
  Tool Use Loop │ LLM 直接决定调工具，无显式推理
  ReAct         │ LLM 先说"我在想什么" → 再调工具
                │ → 推理更透明，可审计
                │ → 复杂任务准确率更高""")


# ═══════════════════════════════════════════════════════════
# 3. Plan-and-Execute 引擎（可复用）
# ═══════════════════════════════════════════════════════════

@dataclass
class PlanStep:
    """计划步骤。"""
    description: str
    tool_name: str | None = None
    tool_kwargs: dict = field(default_factory=dict)
    status: str = "pending"  # pending | done | failed
    result: str = ""


class PlanExecuteEngine:
    """Plan-and-Execute 引擎 — 先规划再执行。

    1. Planner: 生成步骤列表
    2. Executor: 逐步执行（工具调用 + 判断）
    3. Re-planner: 执行中发现问题可重新规划
    """

    def __init__(self, tools: list[ToolDef]):
        self.tools = {t.name: t for t in tools}
        self.plan: list[PlanStep] = []

    def set_plan(self, steps: list[PlanStep]):
        self.plan = steps

    def execute(self, verbose: bool = True) -> list[PlanStep]:
        """执行计划中所有步骤。"""
        for i, step in enumerate(self.plan):
            if verbose:
                print(f"    Step {i+1}: {step.description}")

            if step.tool_name:
                tool = self.tools.get(step.tool_name)
                if tool:
                    try:
                        step.result = tool.fn(**step.tool_kwargs)
                        step.status = "done"
                    except Exception as e:
                        step.result = str(e)
                        step.status = "failed"
                else:
                    step.result = f"工具 '{step.tool_name}' 不存在"
                    step.status = "failed"
            else:
                step.status = "done"
                step.result = "判断完成"

            status_icon = "✅" if step.status == "done" else "❌"
            if verbose:
                display = step.result[:60] if step.result else ""
                print(f"    {status_icon} 结果: {display}")

        return self.plan

    def summary(self) -> dict:
        done = sum(1 for s in self.plan if s.status == "done")
        failed = sum(1 for s in self.plan if s.status == "failed")
        return {"total": len(self.plan), "done": done, "failed": failed}


def demo_plan_and_execute():
    """用 PlanExecuteEngine 运行真实计划。"""
    print(f"\n\n▶ 3. Plan-and-Execute 引擎 — 先规划后执行")
    print("─" * 60)

    print("""
  核心思想:
  ─────────────────────────────────────────────────────────
  1. Planner: LLM 先生成完整执行计划（步骤列表）
  2. Executor: 按计划逐步执行（每步可调用工具）
  3. Re-planner: 执行中发现问题时重新规划

  vs ReAct: ReAct 是"边想边做"，P&E 是"先想好再做"
  """)

    engine = PlanExecuteEngine(tools=[
        ToolDef("lookup_ticket", lookup_ticket, "查询工单详情"),
        ToolDef("check_system_status", check_system_status, "检查服务状态"),
        ToolDef("escalate_ticket", escalate_ticket, "升级工单"),
    ])

    # 设置计划
    print("  📋 Planner 生成计划:")
    engine.set_plan([
        PlanStep("查询工单 T-001 详情", "lookup_ticket", {"ticket_id": "T-001"}),
        PlanStep("根据工单类型判断需检查服务"),
        PlanStep("检查 auth 服务状态", "check_system_status", {"service": "auth"}),
        PlanStep("根据检查结果判断路由目标"),
        PlanStep("路由工单到 engineering", "escalate_ticket",
                 {"ticket_id": "T-001", "target_team": "engineering", "notes": "前端白屏"}),
    ])

    # 执行计划
    print("\n  ⚙️ Executor 逐步执行:")
    engine.execute()

    # 汇总
    s = engine.summary()
    print(f"\n  汇总: {s['total']} 步, {s['done']} 完成, {s['failed']} 失败")

    print(f"""
  适用场景:
  ─────────────────────────────────────────────────────────
  ✅ 复杂多步任务（>5 步）
  ✅ 需要全局规划的任务（先做什么后做什么有依赖）
  ✅ 需要让用户确认计划再执行的场景

  代表实现:
  - Semantic Kernel Planner
  - LangGraph Plan-and-Execute template
  - BabyAGI""")


# ═══════════════════════════════════════════════════════════
# 4. Reflexion 模式
# ═══════════════════════════════════════════════════════════

def demo_reflexion():
    """模拟 Reflexion 模式。"""
    print(f"\n\n▶ 4. Reflexion — 执行后自我反思")
    print("─" * 60)

    print("""
  核心思想:
  ─────────────────────────────────────────────────────────
  Execute → Evaluate → Reflect → Retry

  LLM 执行任务后，检查结果是否满意:
  - 满意 → 返回
  - 不满意 → 反思哪里出错 → 修正后重试

  流程:
  ─────────────────────────────────────────────────────────
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ 首次执行  │ → │ 评估结果  │ → │ 满意？    │ → Yes → 返回
  └──────────┘     └──────────┘     └────┬─────┘
                                         │ No
                                    ┌────▼─────┐
                                    │ 反思错误  │
                                    └────┬─────┘
                                         │
                                    ┌────▼─────┐
                                    │ 修正重试  │ → 回到评估
                                    └──────────┘
  """)

    # 模拟 Reflexion 过程
    print("  模拟: 生成 SQL 查询任务")
    print()

    attempts = [
        {
            "attempt": 1,
            "output": "SELECT * FROM tickets WHERE id = 'T-001'",
            "evaluation": "❌ 表名错误: 应该是 support_tickets",
            "reflection": "表名使用了 tickets，但实际表名是 support_tickets",
        },
        {
            "attempt": 2,
            "output": "SELECT * FROM support_tickets WHERE id = 'T-001'",
            "evaluation": "⚠️ 字段名不精确: id 应该是 ticket_id",
            "reflection": "主键字段名是 ticket_id 不是 id",
        },
        {
            "attempt": 3,
            "output": "SELECT * FROM support_tickets WHERE ticket_id = 'T-001'",
            "evaluation": "✅ 正确",
            "reflection": None,
        },
    ]

    for a in attempts:
        print(f"    Attempt {a['attempt']}: {a['output']}")
        print(f"    评估: {a['evaluation']}")
        if a["reflection"]:
            print(f"    反思: {a['reflection']}")
        print()

    print(f"""  适用场景:
  ─────────────────────────────────────────────────────────
  ✅ 代码生成（生成 → 运行测试 → 反思 → 修复）
  ✅ 数学推理（解题 → 验算 → 纠错）
  ✅ 搜索任务（查询 → 评估结果 → 改进查询）
  ❌ 简单问答（不需要多轮修正）""")


# ═══════════════════════════════════════════════════════════
# 5. Multi-Agent 模式
# ═══════════════════════════════════════════════════════════

def demo_multi_agent():
    """展示多 Agent 协作模式。"""
    print(f"\n\n▶ 5. Multi-Agent 协作模式")
    print("─" * 60)

    print(f"""
  三种主流多 Agent 模式:
  ─────────────────────────────────────────────────────────

  ┌─ Supervisor（主管分配）────────────────────────────┐
  │                                                    │
  │         ┌──────────┐                               │
  │         │ Supervisor│                               │
  │         │  (主管)   │                               │
  │         └──┬──┬──┬─┘                               │
  │            │  │  │                                  │
  │       ┌────┘  │  └────┐                             │
  │       ▼       ▼       ▼                             │
  │  ┌────────┐┌───────┐┌────────┐                     │
  │  │研究Agent││写作Agent││审核Agent│                    │
  │  └────────┘└───────┘└────────┘                     │
  │                                                    │
  │  代表: CrewAI hierarchical, LangGraph supervisor   │
  │  适用: 任务分配明确，需要集中调度                    │
  └────────────────────────────────────────────────────┘

  ┌─ Debate（辩论共识）───────────────────────────────┐
  │                                                    │
  │  ┌────────┐  ←→  ┌────────┐  ←→  ┌────────┐      │
  │  │ Agent A │      │ Agent B │      │ Agent C │      │
  │  │ (正方)  │      │ (反方)  │      │ (评审)  │      │
  │  └────────┘      └────────┘      └────────┘      │
  │                                                    │
  │  代表: AutoGen GroupChat                           │
  │  适用: 需要多角度分析，提高准确性                    │
  └────────────────────────────────────────────────────┘

  ┌─ Handoff（接力移交）──────────────────────────────┐
  │                                                    │
  │  ┌────────┐    ┌────────┐    ┌────────┐           │
  │  │ 分诊    │ →  │ 技术    │    │ 退款    │          │
  │  │ Agent   │ →  │ Agent  │    │ Agent  │          │
  │  └────────┘    └────────┘    └────────┘           │
  │       ↓ 移交                                       │
  │  根据用户问题类型，移交给专门的 Agent                │
  │                                                    │
  │  代表: OpenAI Agents SDK Handoff                   │
  │  适用: 客服分流，各 Agent 各司其职                   │
  └────────────────────────────────────────────────────┘

  模式对比:
  ──────────────┬────────────┬──────────────┬──────────
  模式           │ 通信方式    │ 决策方式      │ 适用场景
  ──────────────┼────────────┼──────────────┼──────────
  Supervisor    │ 主管分配    │ 集中式        │ 流水线任务
  Debate        │ 自由对话    │ 共识/投票      │ 提高准确性
  Handoff       │ 单向移交    │ 路由式        │ 客服分流""")


# ═══════════════════════════════════════════════════════════
# 6. 模式选型
# ═══════════════════════════════════════════════════════════

def show_pattern_selection():
    """展示设计模式选型指南。"""
    print(f"\n\n▶ 6. 模式选型指南")
    print("─" * 60)

    print(f"""
  选型决策树:
  ─────────────────────────────────────────────────────────
  你的任务？
  │
  ├─ 单步工具调用（查数据/发消息）──→ Tool Use Loop
  │
  ├─ 多步推理 + 工具调用 ───────────→ ReAct
  │    └─ 需要透明推理过程？ ────────→ ✅ ReAct 必选
  │
  ├─ 复杂任务（>5步，有依赖）──────→ Plan-and-Execute
  │    └─ 需要用户确认计划？ ────────→ ✅ P&E + HITL
  │
  ├─ 需要自我修正（代码/数学）────→ Reflexion
  │    └─ 有自动评估方式？ ──────────→ ✅ Reflexion
  │
  └─ 多角色协作 ─────────────────────┐
       ├─ 流水线（分工明确）──────────→ Supervisor
       ├─ 讨论/辩论（提高质量）───────→ Debate
       └─ 客服分流（各司其职）────────→ Handoff

  复杂度排序:
  ─────────────────────────────────────────────────────────
  Tool Use Loop < ReAct < Plan-Execute < Reflexion < Multi-Agent

  模式组合（生产中常见）:
  ─────────────────────────────────────────────────────────
  ReAct + Reflexion     │ 推理+修正（代码生成 Agent）
  P&E + HITL            │ 规划+人工确认（审批工作流）
  Supervisor + ReAct    │ 主管分配 + 子 Agent 独立推理
  Handoff + RAG         │ 分流 + 知识检索（智能客服）""")


# ═══════════════════════════════════════════════════════════
# 7. Agent 工程范式
# ═══════════════════════════════════════════════════════════

def show_engineering_paradigms():
    """展示 Agent 工程范式：Loop / Harness / Graph Engineering。"""
    print(f"\n\n▶ 7. Agent 工程范式 — Loop / Harness / Graph Engineering")
    print("─" * 60)

    print(f"""
  上面展示了 Agent 的“设计模式”，这里展示 Agent 的“工程范式” —
  即如何组织和控制 Agent 的运行斶行为。

  三大工程范式:
  ═════════════════════════════════════════════════════════

  ┌─ 1. Loop Engineering（循环工程）────────────────────┐
  │                                                    │
  │  核心思想: Agent 的本质是一个“循环”               │
  │  工程重点: 设计和控制 Agent 的迭代循环行为         │
  │                                                    │
  │  while not done:                                   │
  │      observation = perceive(env)    # 感知        │
  │      thought = reason(observation)  # 推理        │
  │      action = decide(thought)       # 决策        │
  │      result = act(action)           # 行动        │
  │      done = evaluate(result)        # 评估        │
  │                                                    │
  │  关键变量:                                          │
  │  ──────────────┬───────────────────────────│
  │  循环终止条件   │ max_turns / 置信度阈值 / done│
  │  重试策略       │ 指数退避 / 固定次数 / 随机     │
  │  状态累积       │ 每轮结果加入 context         │
  │  刮耍检测       │ 连续相同输出时强制跳出       │
  │  ──────────────┴───────────────────────────│
  │                                                    │
  │  典型实现:                                          │
  │  - ReAct Loop (Thought→Action→Obs 循环)            │
  │  - Reflexion Loop (执行→评估→反思 循环)            │
  │  - Tool Use Loop (调工具→喝回结果 循环)           │
  │                                                    │
  │  代表框架: pydantic-ai (max_result_retries),        │
  │           Vercel AI SDK (maxSteps),                │
  │           OpenAI Agents SDK (max_turns)             │
  └────────────────────────────────────────────────────┘

  ┌─ 2. Harness Engineering（外挂工程 / “驾具”工程）─────┐
  │                                                    │
  │  核心思想: 不信任 LLM，用确定性代码包裹它       │
  │  工程重点: 在 LLM 周围建“外壳”，约束其行为     │
  │                                                    │
  │  ┌────────────────────────────────────────┐  │
  │  │  Harness (确定性代码)                    │  │
  │  │  ┌────────────────────────────────────┐  │  │
  │  │  │  ① 输入验证 (格式/长度/安全)          │  │  │
  │  │  │  ② Prompt 构建 (模板 + 变量注入)       │  │  │
  │  │  │  ③ ┌─────────────┐                    │  │  │
  │  │  │    │ LLM (不可控) │ ← 唯一的“黑盒”   │  │  │
  │  │  │  ⑤ └─────────────┘                    │  │  │
  │  │  │  ④ 输出解析 (JSON/Schema 验证)       │  │  │
  │  │  │  ⑤ 护栏检查 (安全/合规/PII)            │  │  │
  │  │  │  ⑥ 重试逻辑 (解析失败时自动重试)       │  │  │
  │  │  └────────────────────────────────────┘  │  │
  │  └────────────────────────────────────────┘  │
  │                                                    │
  │  关键组件:                                          │
  │  ──────────────┬───────────────────────────│
  │  输入护栏       │ 格式校验 / 注入检测 / 长度限制  │
  │  Prompt 模板  │ Jinja2 / f-string / 框架内置  │
  │  输出解析器     │ JSON Schema / Pydantic / Zod   │
  │  输出护栏       │ PII脱敏 / 内容审核 / 语气检查  │
  │  重试层         │ 解析失败自动重试 N 次         │
  │  日志/追踪      │ 每次调用的入参/出参/延迟     │
  │  ──────────────┴───────────────────────────│
  │                                                    │
  │  代表框架: pydantic-ai (output_type + RunContext),   │
  │           Guardrails AI (RAIL spec),                │
  │           Instructor (structured output),           │
  │           NeMo Guardrails (Colang 规则)              │
  └────────────────────────────────────────────────────┘

  ┌─ 3. Graph Engineering（图工程）───────────────────┐
  │                                                    │
  │  核心思想: 用有向图建模 Agent 的工作流             │
  │  工程重点: 节点 + 边 + 状态 + 条件路由             │
  │                                                    │
  │  ┌─────────┐     ┌─────────┐     ┌─────────┐  │
  │  │ 分诊节点  │ ─→─│ 工具节点  │ ─→─│ 回复节点  │  │
  │  └────┬────┘     └────┬────┘     └─────────┘  │
  │       │ 条件路由      │ 循环                      │
  │       └───────────┘                              │
  │                                                    │
  │  核心元素:                                          │
  │  ──────────────┬───────────────────────────│
  │  Node (节点)   │ 一个处理步骤（LLM调用/工具/逻辑） │
  │  Edge (边)     │ 节点间的连接（条件/无条件）      │
  │  State (状态)  │ 图的全局状态（TypedDict）        │
  │  Checkpoint   │ 状态快照（断点续跑/回滚）          │
  │  Router (路由) │ 条件边（根据状态选择下一个节点） │
  │  Interrupt    │ 暂停点（等待人工审批）            │
  │  ──────────────┴───────────────────────────│
  │                                                    │
  │  vs 纯代码工作流:                                   │
  │  - 图可视化、可调试（看得见执行路径）              │
  │  - 状态可持久化（故障恢复、时间旅行调试）          │
  │  - 条件路由声明式（而非嵌套 if-else）              │
  │  - 天然支持并行、循环、中断                      │
  │                                                    │
  │  代表框架: LangGraph (StateGraph),                  │
  │           Mastra (Workflow + Step),                 │
  │           AutoGen (GroupChat 消息图),                │
  │           Semantic Kernel (Process Framework)       │
  └────────────────────────────────────────────────────┘

  三者关系（不是三选一，而是三层嵌套）:
  ═════════════════════════════════════════════════════════

  ┌────────────────────────────────────────────────┐
  │  Graph Engineering (图编排层)                       │
  │  ┌────────────────────────────────────────────┐│
  │  │  Harness Engineering (每个节点的护栏包裹)        ││
  │  │  ┌────────────────────────────────────────┐││
  │  │  │  Loop Engineering (单个 LLM 调用循环)       │││
  │  │  │  while not done:                           │││
  │  │  │      response = harness.call(llm, prompt) │││
  │  │  │      done = evaluate(response)             │││
  │  │  └────────────────────────────────────────┘││
  │  └────────────────────────────────────────────┘│
  └────────────────────────────────────────────────┘

  各框架的工程范式定位:
  ──────────────┬──────────┬──────────┬───────────────
  框架           │ Loop     │ Harness  │ Graph
  ──────────────┼──────────┼──────────┼───────────────
  LangChain     │ ✅        │ ✅        │ ❌ (→ LangGraph)
  LangGraph     │ ✅        │ ✅        │ ⭐ 核心定位
  pydantic-ai   │ ✅        │ ⭐ 核心   │ ❌
  OpenAI SDK    │ ✅        │ ⚠️ 基础   │ ❌
  CrewAI        │ ✅        │ ✅        │ ⚠️ (流水线图)
  AutoGen       │ ✅        │ ⚠️ 基础   │ ⚠️ (消息图)
  Spring AI     │ ✅        │ ✅ Advisor│ ❌
  Vercel AI SDK │ ✅ maxSteps│ ✅ Zod    │ ❌
  Mastra        │ ✅        │ ✅        │ ✅ Workflow
  Sem. Kernel   │ ✅        │ ✅ Plugin │ ✅ Process

  实践建议:
  ─────────────────────────────────────────────────────────
  1. 所有 Agent 都从 Loop Engineering 开始
     → 控制好 max_turns / 重试 / 刮耍检测
  2. 生产环境必须加 Harness Engineering
     → 输入护栏 + 输出解析 + 安全检查
  3. 复杂工作流才需要 Graph Engineering
     → 多节点 + 条件路由 + 状态持久化
  4. 三者可以嵌套: Graph 的每个节点是一个 Harness，
     Harness 内部是一个 Loop""")


# ═══════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════

def main():
    print("=== Agent 设计模式 ===\n")

    demo_tool_use_loop()
    demo_react()
    demo_plan_and_execute()
    demo_reflexion()
    demo_multi_agent()
    show_pattern_selection()
    show_engineering_paradigms()

    # ── 架构总结 ──────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 Agent 设计模式总结:")
    print()
    print("  设计模式         │ 一句话         │ 代表框架")
    print("  ───────────────┼───────────────┼──────────────")
    print("  Tool Use Loop  │ 调工具循环     │ 所有框架")
    print("  ReAct          │ 想了再做       │ LangChain/pydantic-ai")
    print("  Plan-Execute   │ 先规划后执行   │ Semantic Kernel")
    print("  Reflexion      │ 做完再反思     │ LangGraph 循环")
    print("  Supervisor     │ 主管分配       │ CrewAI/LangGraph")
    print("  Debate         │ 多人讨论       │ AutoGen")
    print("  Handoff        │ 接力移交       │ OpenAI Agents SDK")
    print()
    print("  工程范式         │ 一句话         │ 关注点")
    print("  ───────────────┼───────────────┼──────────────")
    print("  Loop Eng.      │ 控制循环行为     │ 终止/重试/刮耍")
    print("  Harness Eng.   │ 确定性包裹 LLM  │ 护栏/解析/重试")
    print("  Graph Eng.     │ 图编排工作流   │ 节点/边/状态/路由")
    print()
    print("  核心原则:")
    print("  ────────────────────────────────────────────")
    print("  1. Loop: 控制好 max_turns、重试、刮耍检测")
    print("  2. Harness: 生产环境必须加护栏 + 输出解析")
    print("  3. Graph: 复杂工作流才需要图编排")
    print("  4. 三者嵌套: Graph ⊃ Harness ⊃ Loop")
    print("  5. 设计模式 + 工程范式组合使用")


if __name__ == "__main__":
    main()
