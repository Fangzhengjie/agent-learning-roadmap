"""五种多 Agent 编排模式

1. Pipeline    — 线性流水线: A → B → C
2. Supervisor  — 主管分配: Supervisor 决定下一个 Agent
3. Debate      — 辩论达成共识: 多个 Agent 互相评审
4. Voting      — 投票表决: 多个 Agent 独立回答 → 投票选最优
5. Marketplace — 自由市场: Agent 发布能力 → 按需匹配
"""

import random
from dataclasses import dataclass, field
from typing import Any, Callable
from collections import Counter


# ═══════════════════════════════════════════════════════════
# Agent 基类
# ═══════════════════════════════════════════════════════════

@dataclass
class AgentMessage:
    sender: str
    content: str
    metadata: dict = field(default_factory=dict)


class SimpleAgent:
    """简单 Agent — 模拟 LLM 决策（规则替代）。"""

    def __init__(self, name: str, role: str, respond_fn: Callable[[str], str] | None = None):
        self.name = name
        self.role = role
        self._respond_fn = respond_fn
        self.history: list[AgentMessage] = []

    def respond(self, input_text: str) -> str:
        if self._respond_fn:
            output = self._respond_fn(input_text)
        else:
            output = f"[{self.name}/{self.role}] 处理: {input_text[:50]}"
        self.history.append(AgentMessage(self.name, output))
        return output


# ═══════════════════════════════════════════════════════════
# 1. Pipeline（线性流水线）
# ═══════════════════════════════════════════════════════════

class PipelineOrchestrator:
    """线性流水线: A → B → C，每个 Agent 的输出是下一个的输入。

    类比: CrewAI sequential process
    适用: 内容生产（调研 → 写作 → 审核 → 发布）
    """

    def __init__(self, agents: list[SimpleAgent]):
        self.agents = agents
        self.trace: list[dict] = []

    def run(self, initial_input: str) -> str:
        current = initial_input
        for agent in self.agents:
            output = agent.respond(current)
            self.trace.append({"agent": agent.name, "input": current[:80], "output": output[:80]})
            current = output
        return current


# ═══════════════════════════════════════════════════════════
# 2. Supervisor（主管分配）
# ═══════════════════════════════════════════════════════════

class SupervisorOrchestrator:
    """主管模式: Supervisor 决定将任务分配给哪个 Agent。

    类比: CrewAI hierarchical, LangGraph supervisor node
    适用: 客服分流、任务分类后路由
    """

    def __init__(self, supervisor: SimpleAgent, workers: dict[str, SimpleAgent]):
        self.supervisor = supervisor
        self.workers = workers
        self.trace: list[dict] = []

    def run(self, task: str, route_fn: Callable[[str], str] | None = None) -> dict:
        """route_fn: 模拟 Supervisor 路由决策。"""
        if route_fn:
            target = route_fn(task)
        else:
            target = list(self.workers.keys())[0]

        self.trace.append({"step": "route", "supervisor": self.supervisor.name, "target": target})

        worker = self.workers.get(target)
        if not worker:
            return {"error": f"Unknown worker: {target}", "trace": self.trace}

        result = worker.respond(task)
        self.trace.append({"step": "execute", "agent": worker.name, "result": result[:80]})

        # Supervisor 审核
        review = self.supervisor.respond(f"审核 {worker.name} 的结果: {result}")
        self.trace.append({"step": "review", "review": review[:80]})

        return {"result": result, "review": review, "worker": target, "trace": self.trace}


# ═══════════════════════════════════════════════════════════
# 3. Debate（辩论达成共识）
# ═══════════════════════════════════════════════════════════

class DebateOrchestrator:
    """辩论模式: 多个 Agent 对同一问题给出观点，互相评审，达成共识。

    类比: AutoGen GroupChat with critic
    适用: 提高准确性、决策质量
    """

    def __init__(self, agents: list[SimpleAgent], max_rounds: int = 3):
        self.agents = agents
        self.max_rounds = max_rounds
        self.rounds: list[dict] = []

    def run(self, question: str) -> dict:
        opinions = {}
        # Round 1: 各自回答
        for agent in self.agents:
            opinions[agent.name] = agent.respond(question)
        self.rounds.append({"round": 1, "type": "initial", "opinions": dict(opinions)})

        # Round 2+: 看到其他人的答案后修正
        for round_num in range(2, self.max_rounds + 1):
            all_opinions = "\n".join(f"  {k}: {v}" for k, v in opinions.items())
            new_opinions = {}
            for agent in self.agents:
                prompt = f"其他人的观点:\n{all_opinions}\n\n请结合上述观点，修正你的回答:"
                new_opinions[agent.name] = agent.respond(prompt)
            opinions = new_opinions
            self.rounds.append({"round": round_num, "type": "revision", "opinions": dict(opinions)})

        return {
            "final_opinions": opinions,
            "consensus": self._find_consensus(opinions),
            "rounds": len(self.rounds),
        }

    @staticmethod
    def _find_consensus(opinions: dict[str, str]) -> str:
        """简单共识: 取最长的（生产中用 LLM 合并）。"""
        return max(opinions.values(), key=len)


# ═══════════════════════════════════════════════════════════
# 4. Voting（投票表决）
# ═══════════════════════════════════════════════════════════

class VotingOrchestrator:
    """投票模式: 多个 Agent 独立回答 → 多数表决。

    类比: Self-consistency (SC) prompting
    适用: 数学推理、分类任务（同问题多次采样取多数）
    """

    def __init__(self, agents: list[SimpleAgent]):
        self.agents = agents

    def run(self, question: str, extract_answer_fn: Callable[[str], str] | None = None) -> dict:
        responses = {}
        answers = []

        for agent in self.agents:
            resp = agent.respond(question)
            responses[agent.name] = resp
            answer = extract_answer_fn(resp) if extract_answer_fn else resp.strip()
            answers.append(answer)

        vote_count = Counter(answers)
        winner = vote_count.most_common(1)[0]

        return {
            "responses": responses,
            "votes": dict(vote_count),
            "winner": winner[0],
            "confidence": winner[1] / len(answers),
            "total_voters": len(answers),
        }


# ═══════════════════════════════════════════════════════════
# 5. Marketplace（能力市场）
# ═══════════════════════════════════════════════════════════

@dataclass
class AgentCapability:
    agent_name: str
    skill: str
    description: str
    cost: float = 1.0
    rating: float = 5.0


class MarketplaceOrchestrator:
    """能力市场: Agent 注册能力 → 任务来了按需匹配。

    类比: A2A Agent Card, Google ADK service discovery
    适用: 大规模 Agent 生态、跨组织协作
    """

    def __init__(self):
        self.registry: list[AgentCapability] = []
        self.agents: dict[str, SimpleAgent] = {}

    def register(self, agent: SimpleAgent, capabilities: list[AgentCapability]):
        self.agents[agent.name] = agent
        self.registry.extend(capabilities)

    def find_agent(self, task: str, top_k: int = 1) -> list[AgentCapability]:
        """简单关键词匹配（生产用向量相似度）。"""
        scored = []
        task_lower = task.lower()
        for cap in self.registry:
            score = sum(1 for w in cap.skill.lower().split() if w in task_lower)
            score += sum(1 for w in cap.description.lower().split() if w in task_lower)
            scored.append((cap, score + cap.rating * 0.1))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [cap for cap, _ in scored[:top_k]]

    def dispatch(self, task: str) -> dict:
        """自动匹配并分派任务。"""
        matches = self.find_agent(task, top_k=1)
        if not matches:
            return {"error": "No matching agent found"}
        cap = matches[0]
        agent = self.agents.get(cap.agent_name)
        if not agent:
            return {"error": f"Agent {cap.agent_name} not available"}
        result = agent.respond(task)
        return {
            "matched_agent": cap.agent_name,
            "skill": cap.skill,
            "result": result,
            "cost": cap.cost,
        }
