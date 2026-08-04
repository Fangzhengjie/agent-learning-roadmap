"""LangChain RAG 知识库问答 Demo

最佳场景：RAG（检索增强生成）— LangChain 拥有最成熟的 RAG 生态。

核心模式：
  - 文档加载 → 分割 → 向量化 → 检索 → 生成
  - Chain 组合（Retriever | Prompt | LLM | OutputParser）
  - ConversationBufferMemory 多轮对话记忆
  - CallbackHandler 执行观测

为什么 RAG 选 LangChain：
  - 200+ 文档加载器（PDF/HTML/Notion/Confluence/...）
  - 50+ 向量数据库集成（FAISS/Chroma/Pinecone/Weaviate/...）
  - 最成熟的 text splitter 和 embedding 生态
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from shared_tools import KNOWLEDGE_BASE, search_knowledge as _search_knowledge


# ── 模拟向量检索 ──────────────────────────────────────────
# 实际生产中这里是 FAISS / Chroma / Pinecone
# LangChain 的优势在于 Retriever 抽象统一了所有向量库

@tool
def search_knowledge_base(query: str) -> str:
    """在企业知识库中检索相关文档。query: 用户的问题或关键词。"""
    return _search_knowledge(query)


# ── 回调处理器 ────────────────────────────────────────────
class RAGCallbackHandler(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"  🔍 [检索] query={input_str[:80]}")

    def on_tool_end(self, output, **kwargs):
        print(f"  📄 [检索结果] {str(output)[:100]}...")


# ── 构建 RAG Agent ────────────────────────────────────────
def main():
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    print("=== LangChain RAG 企业知识库问答 Demo ===")
    print(f"模型: {model_name}")
    print(f"知识库文档数: {len(KNOWLEDGE_BASE)}\n")

    llm = ChatOpenAI(model=model_name, temperature=0)

    agent = create_react_agent(
        model=llm,
        tools=[search_knowledge_base],
        prompt=ChatPromptTemplate.from_messages([
            ("system",
             "你是企业内部知识库助手。用户会问关于公司制度、IT 操作等问题。\n"
             "请先使用 search_knowledge_base 工具检索相关文档，然后基于检索结果回答。\n"
             "如果知识库中没有相关信息，请明确告知用户。\n"
             "回答请简洁专业，引用来源文档编号。"),
            MessagesPlaceholder(variable_name="messages"),
        ]),
    )

    # ── 模拟多轮对话 ──────────────────────────────────────
    questions = [
        "我入职3年了，能请几天年假？审批流程是什么？",
        "出差坐飞机怎么报销？",
        "VPN 连不上怎么办？",
    ]

    message_history = []

    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 60}")
        print(f"👤 问题 {i}: {question}")
        print(f"{'─' * 60}")

        message_history.append(HumanMessage(content=question))

        result = agent.invoke(
            {"messages": message_history},
            config={"callbacks": [RAGCallbackHandler()]},
        )

        answer = result["messages"][-1].content
        message_history = result["messages"]
        print(f"\n🤖 回答:\n{answer}\n")

    # ── 架构观察 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 LangChain RAG 架构观察:")
    print(f"  累计消息数: {len(message_history)}")
    print()
    print("  ✅ 最佳场景: RAG — 200+ 文档加载器 + 50+ 向量库集成")
    print("  ✅ Retriever 抽象统一所有向量库接口")
    print("  ✅ Chain 组合模式灵活（Retriever | Prompt | LLM）")
    print("  ✅ 多轮对话记忆开箱即用")
    print("  ⚠️  概念多（Chain/Agent/Tool/Memory/Retriever/Splitter）")
    print("  ⚠️  CallbackHandler 粒度粗（~6 类事件 vs Code Puppy 51 Hook）")
    print("  ❌ 无内置取消/重试/安全模型")


if __name__ == "__main__":
    main()
