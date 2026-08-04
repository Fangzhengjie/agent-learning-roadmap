"""多 Agent 编排模式 — 演示入口"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from patterns import (SimpleAgent, PipelineOrchestrator, SupervisorOrchestrator,
                          DebateOrchestrator, VotingOrchestrator,
                          MarketplaceOrchestrator, AgentCapability)
except ImportError:
    from .patterns import (SimpleAgent, PipelineOrchestrator, SupervisorOrchestrator,
                           DebateOrchestrator, VotingOrchestrator,
                           MarketplaceOrchestrator, AgentCapability)


def demo_pipeline():
    print("▶ 1. Pipeline（线性流水线）")
    print("─" * 60)

    researcher = SimpleAgent("调研员", "research",
                             lambda x: f"调研结果: {x[:20]}... 的市场规模约500亿，增长率15%")
    writer = SimpleAgent("写手", "writing",
                         lambda x: f"文章草稿: 基于{x[:30]}...，AI市场正在快速扩张。")
    editor = SimpleAgent("编辑", "editing",
                         lambda x: f"终稿: [已润色] {x[:40]}...")

    pipeline = PipelineOrchestrator([researcher, writer, editor])
    result = pipeline.run("AI Agent 市场分析")

    print(f"  输入: AI Agent 市场分析")
    for step in pipeline.trace:
        print(f"    [{step['agent']}] → {step['output'][:60]}")
    print(f"  最终输出: {result[:60]}")


def demo_supervisor():
    print(f"\n\n▶ 2. Supervisor（主管分配）")
    print("─" * 60)

    supervisor = SimpleAgent("主管", "supervisor", lambda x: f"审核通过: {x[:40]}")
    workers = {
        "tech": SimpleAgent("技术Agent", "tech_support",
                            lambda x: f"技术回答: 请检查网络设置和DNS配置"),
        "billing": SimpleAgent("账务Agent", "billing",
                               lambda x: f"账务回答: 您的账单金额为¥299，已发送到邮箱"),
        "general": SimpleAgent("通用Agent", "general",
                               lambda x: f"通用回答: 感谢咨询，已记录您的问题"),
    }

    def route(task):
        if any(w in task for w in ["网络", "故障", "无法连接"]):
            return "tech"
        if any(w in task for w in ["账单", "费用", "退款"]):
            return "billing"
        return "general"

    orch = SupervisorOrchestrator(supervisor, workers)
    tasks = ["网络无法连接怎么办？", "查一下我的账单", "你们的营业时间是什么？"]
    for task in tasks:
        result = orch.run(task, route_fn=route)
        print(f"  任务: {task}")
        print(f"    → 路由: {result['worker']} → 结果: {result['result'][:40]}")
        orch.trace.clear()


def demo_debate():
    print(f"\n\n▶ 3. Debate（辩论达成共识）")
    print("─" * 60)

    agents = [
        SimpleAgent("乐观派", "optimist",
                     lambda x: f"AI 将大幅提升效率，预计3年内普及，ROI > 300%"),
        SimpleAgent("谨慎派", "cautious",
                     lambda x: f"需要考虑合规风险和数据安全，建议先小规模试点"),
        SimpleAgent("技术派", "technical",
                     lambda x: f"当前 GPT-4o 已足够成熟，关键是 RAG 和工具集成的工程质量"),
    ]

    debate = DebateOrchestrator(agents, max_rounds=2)
    result = debate.run("企业是否应该全面采用 AI Agent？")

    print(f"  问题: 企业是否应该全面采用 AI Agent？")
    for round_info in debate.rounds:
        print(f"  第 {round_info['round']} 轮 ({round_info['type']}):")
        for name, opinion in round_info["opinions"].items():
            print(f"    {name}: {opinion[:50]}...")
    print(f"  共识: {result['consensus'][:60]}...")


def demo_voting():
    print(f"\n\n▶ 4. Voting（投票表决）")
    print("─" * 60)

    import random
    random.seed(42)
    answers_pool = ["A: 北京", "A: 北京", "A: 北京", "A: 上海", "A: 北京"]
    idx = [0]

    def make_voter(name, pool_idx):
        def fn(q):
            ans = answers_pool[pool_idx]
            return ans
        return SimpleAgent(name, "voter", fn)

    agents = [make_voter(f"模型{i+1}", i) for i in range(5)]
    voting = VotingOrchestrator(agents)
    result = voting.run("中国的首都是哪里？", extract_answer_fn=lambda r: r.split(": ")[-1])

    print(f"  问题: 中国的首都是哪里？")
    print(f"  投票: {result['votes']}")
    print(f"  结果: {result['winner']} (置信度: {result['confidence']:.0%})")


def demo_marketplace():
    print(f"\n\n▶ 5. Marketplace（能力市场）")
    print("─" * 60)

    market = MarketplaceOrchestrator()

    # 注册 Agent 和能力
    translator = SimpleAgent("翻译Agent", "translator",
                              lambda x: f"翻译结果: This is a translation of '{x[:20]}'")
    coder = SimpleAgent("代码Agent", "coder",
                         lambda x: f"代码: def solution(): return sorted(data)")
    analyst = SimpleAgent("分析Agent", "analyst",
                           lambda x: f"分析报告: 数据显示增长趋势明显，建议加大投入")

    market.register(translator, [
        AgentCapability("翻译Agent", "翻译 translation", "中英文翻译", cost=0.5, rating=4.8),
    ])
    market.register(coder, [
        AgentCapability("代码Agent", "代码 编程 coding", "Python代码生成", cost=1.0, rating=4.5),
    ])
    market.register(analyst, [
        AgentCapability("分析Agent", "数据分析 报告", "商业数据分析和报告生成", cost=2.0, rating=4.9),
    ])

    tasks = ["请翻译这段文字", "帮我写段排序代码", "分析上季度销售数据"]
    for task in tasks:
        result = market.dispatch(task)
        print(f"  任务: {task}")
        print(f"    → 匹配: {result['matched_agent']} (技能: {result['skill']}, 成本: ${result['cost']})")
        print(f"    → 结果: {result['result'][:50]}")


def main():
    print("=== 多 Agent 编排模式 ===\n")
    demo_pipeline()
    demo_supervisor()
    demo_debate()
    demo_voting()
    demo_marketplace()

    print(f"\n\n{'=' * 60}")
    print("📊 多 Agent 编排模式总结:")
    print(f"""
  模式          │ 适用场景              │ 框架支持
  ─────────────┼─────────────────────┼──────────────────
  Pipeline      │ 内容生产/ETL         │ CrewAI sequential
  Supervisor    │ 客服分流/任务路由     │ LangGraph/CrewAI hierarchical
  Debate        │ 提高准确性/决策质量   │ AutoGen GroupChat
  Voting        │ 分类/推理一致性       │ Self-consistency
  Marketplace   │ 大规模Agent生态       │ A2A / Agent Card""")


if __name__ == "__main__":
    main()
