"""CrewAI 新闻内容生产流水线 Demo

最佳场景：内容生产 Pipeline — 调研 → 写作 → 编辑，角色分工 + 任务依赖。

核心模式：
  - Agent: role + goal + backstory（角色扮演式 Agent）
  - Task: description + expected_output + context（任务依赖链）
  - Crew: 编排容器（sequential / hierarchical 执行策略）
  - Process.sequential: 按任务依赖顺序执行

为什么内容生产选 CrewAI：
  - 角色扮演（调研员/记者/编辑）天然匹配内容生产团队
  - Task 依赖链完美表达流水线（调研 → 写作 → 编辑）
  - 最简单直观的多 Agent 框架，非技术人员也能理解
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

from shared_tools import search_knowledge as _search, write_file as _write_file


# ── 工具定义 ──────────────────────────────────────────────

@tool
def research_topic(query: str) -> str:
    """调研特定话题，搜索相关资料和数据。query: 调研关键词。"""
    return _search(query)


@tool
def save_article(filename: str, content: str) -> str:
    """保存文章到文件。filename: 文件名。content: 文章内容。"""
    return _write_file(filename, content)


# ── 角色定义 ──────────────────────────────────────────────
# CrewAI 的核心特色：role + goal + backstory 构成 Agent 的"人格"

def create_crew():
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

    # Agent 1: 调研员 — 收集素材和数据
    researcher = Agent(
        role="资深新闻调研员",
        goal="深入调研话题，收集关键数据、事实和多方观点",
        backstory=(
            "你是一位有15年经验的调研记者，曾在财新、澎湃工作。"
            "你以事实准确、数据翔实著称，总是从多个角度审视问题。"
            "你的调研报告是写作团队最信赖的素材来源。"
        ),
        tools=[research_topic],
        llm=f"openai/{model_name}",
        verbose=True,
    )

    # Agent 2: 记者 — 根据调研素材写作
    writer = Agent(
        role="深度报道记者",
        goal="将调研素材转化为引人入胜、信息量大的深度文章",
        backstory=(
            "你是一位获奖记者，擅长将复杂话题转化为大众可读的深度报道。"
            "你的文章结构严谨，善用数据可视化、案例对比和专家引述。"
            "你坚持'用事实说话'的原则，从不添加未经证实的信息。"
        ),
        llm=f"openai/{model_name}",
        verbose=True,
    )

    # Agent 3: 编辑 — 审校、优化、定稿
    editor = Agent(
        role="主编",
        goal="确保文章事实准确、逻辑清晰、语言流畅，达到发表标准",
        backstory=(
            "你是一位资深主编，有20年编辑经验。你对事实核查极其严格，"
            "对文章结构和语言有敏锐的判断力。你的编辑意见总是切中要害，"
            "既能指出问题，又能给出具体的改进建议。"
        ),
        tools=[save_article],
        llm=f"openai/{model_name}",
        verbose=True,
    )

    # ── 任务流水线 ────────────────────────────────────────
    # 关键：context 参数定义任务依赖，形成流水线

    research_task = Task(
        description=(
            "调研'企业员工福利制度'话题。需要覆盖：\n"
            "1. 使用 research_topic 工具搜索'请假'和'报销'相关政策\n"
            "2. 整理关键数据点和政策要点\n"
            "3. 列出值得深入报道的角度\n"
            "输出结构化的调研备忘录。"
        ),
        expected_output="包含政策要点、关键数据和报道角度的调研备忘录",
        agent=researcher,
    )

    writing_task = Task(
        description=(
            "基于调研备忘录，撰写一篇企业员工福利制度分析文章。要求：\n"
            "1. 标题吸引人，副标题说明核心观点\n"
            "2. 分节讨论不同福利类型（假期、报销、IT支持等）\n"
            "3. 每节包含具体数据和政策引用\n"
            "4. 结尾给出员工实用建议\n"
            "文章约 500-800 字，中文 Markdown 格式。"
        ),
        expected_output="一篇结构完整、数据翔实的中文深度分析文章",
        agent=writer,
        context=[research_task],  # 依赖调研结果
    )

    editing_task = Task(
        description=(
            "作为主编审校文章：\n"
            "1. 核查所有数据引用是否与调研备忘录一致\n"
            "2. 检查文章结构是否清晰，逻辑是否连贯\n"
            "3. 优化语言表达，确保通俗易懂\n"
            "4. 使用 save_article 工具保存最终稿到 article.md\n"
            "在文章末尾添加'[编辑注：经主编审校]'。"
        ),
        expected_output="一篇经过审校的最终版文章，已保存到 article.md",
        agent=editor,
        context=[research_task, writing_task],  # 同时参考调研和初稿
    )

    # ── Crew 编排 ─────────────────────────────────────────
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, writing_task, editing_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew


# ── 执行 ──────────────────────────────────────────────────
def main():
    print("=== CrewAI 新闻内容生产流水线 Demo ===")
    print(f"模型: {os.getenv('MODEL_NAME', 'gpt-4o-mini')}")
    print(f"流水线: 调研员 → 记者 → 主编\n")
    print("─" * 60)

    crew = create_crew()
    result = crew.kickoff()

    print("\n" + "─" * 60)
    print(f"\n📋 最终发表稿（前500字）:\n{result.raw[:500]}...")

    # ── 架构观察 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 CrewAI 内容生产架构观察:")
    print()
    print("  ✅ 最佳场景: 内容生产流水线（调研→写作→编辑）")
    print("  ✅ 角色扮演直觉化（backstory 赋予 Agent '人格'）")
    print("  ✅ Task.context 自然表达流水线依赖")
    print("  ✅ 上手最简单，非技术人员也能理解")
    print("  ✅ Process.hierarchical 支持'主编分配'模式")
    print("  ⚠️  角色模板限制灵活性（必须 role+goal+backstory）")
    print("  ⚠️  深度定制需要 hack 框架内部")
    print("  ❌ 无运行时干预（无法中途修改方向）")
    print("  ❌ 无安全模型 / 无取消机制")


if __name__ == "__main__":
    main()
