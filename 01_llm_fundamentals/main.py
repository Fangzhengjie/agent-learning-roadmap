"""大模型（LLM）核心原理 — Agent 的底层基石

核心概念：Agent = LLM + 工具 + 记忆 + 规划。理解 LLM 原理才能用好 Agent。

本示例用纯 Python 展示 LLM 的核心知识体系：
  1. Transformer 架构 — 大模型的统一架构
  2. Self-Attention 机制 — Transformer 的核心创新
  3. Tokenizer — 文本如何变成模型输入
  4. 推理参数 — temperature / top_p / top_k 等
  5. 上下文窗口 — 模型能处理多长的文本
  6. 模型家族 — GPT / Claude / Gemini / Llama / Qwen 对比
  7. 从预训练到 Agent — LLM 能力演进路线
"""

import json
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# 1. Transformer 架构
# ═══════════════════════════════════════════════════════════

def show_transformer():
    """展示 Transformer 架构。"""
    print("▶ 1. Transformer 架构 — 大模型的统一架构")
    print("─" * 60)

    print("""
  Transformer (Vaswani et al., 2017, "Attention Is All You Need")
  ─────────────────────────────────────────────────────────

  三种 Transformer 变体:
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  Encoder-Only          Decoder-Only       Enc-Dec   │
  │  (BERT, RoBERTa)       (GPT, Llama)      (T5, BART)│
  │                                                      │
  │  [理解型]               [生成型]           [转换型]   │
  │  分类/NER/相似度        文本生成/对话       翻译/摘要  │
  │                                                      │
  │  ⭐ Agent 用的是 Decoder-Only（自回归生成）           │
  └──────────────────────────────────────────────────────┘

  Decoder-Only Transformer（GPT 架构）:
  ─────────────────────────────────────────────────────────

  输入: "今天天气"
         ↓ Tokenize
  Token: [今, 天, 天, 气]
         ↓ Token Embedding + Position Embedding
  向量:  [v₁, v₂, v₃, v₄]  (每个 token → d 维向量)
         ↓
  ┌─────────────────────────────────┐
  │  Transformer Block × N 层       │ ← GPT-4: 120层, Llama3: 80层
  │  ┌───────────────────────────┐  │
  │  │ Masked Self-Attention     │  │ ← 核心：每个 token 关注前面的 token
  │  │ + Residual + LayerNorm    │  │
  │  ├───────────────────────────┤  │
  │  │ Feed-Forward Network      │  │ ← MLP: d → 4d → d
  │  │ + Residual + LayerNorm    │  │
  │  └───────────────────────────┘  │
  └─────────────────────────────────┘
         ↓
  输出: 词汇表概率分布 → 采样 → "很" → 继续生成...

  关键概念:
  ─────────────────────────────────────────────────────────
  参数量    │ 权重矩阵中可训练数值的总数（7B = 70亿个参数）
  层数      │ Transformer Block 堆叠几层（越深越强）
  隐藏维度  │ 每个 token 的向量维度 d（GPT-4: ~12288）
  注意力头  │ Multi-Head Attention 的头数（并行关注不同模式）
  词汇表    │ 模型认识的所有 token 集合（~32K~200K）

  主流模型参数:
  ──────────────┬────────┬────────┬────────┬────────
  模型           │ 参数量  │ 层数   │ 隐藏维度│ 注意力头
  ──────────────┼────────┼────────┼────────┼────────
  GPT-4o        │ ~200B? │ ~120   │ ~12288 │ ~96
  Claude 3.5    │ ~175B? │ 未公开  │ 未公开  │ 未公开
  Llama 3.1 70B │ 70B    │ 80     │ 8192   │ 64
  Qwen 2.5 72B  │ 72B    │ 80     │ 8192   │ 64
  Llama 3.1 8B  │ 8B     │ 32     │ 4096   │ 32
  Qwen 2.5 7B   │ 7.6B   │ 28     │ 3584   │ 28""")


# ═══════════════════════════════════════════════════════════
# 2. Self-Attention 机制
# ═══════════════════════════════════════════════════════════

def show_attention():
    """展示 Self-Attention 机制（含数值演示）。"""
    print(f"\n\n▶ 2. Self-Attention 机制 — Transformer 的核心")
    print("─" * 60)

    print("""
  核心公式:
  ─────────────────────────────────────────────────────────
  Attention(Q, K, V) = softmax(Q × Kᵀ / √d_k) × V

  Q (Query):  "我想查找什么"    ← 当前 token 的查询
  K (Key):    "我有什么信息"    ← 所有 token 的键
  V (Value):  "我的具体内容"    ← 所有 token 的值

  过程:
  1. 每个 token 生成 Q, K, V 三个向量
  2. Q × Kᵀ → 注意力分数（谁和谁相关）
  3. / √d_k  → 缩放（防止梯度消失）
  4. softmax → 归一化为概率
  5. × V    → 加权求和得到输出""")

    # 数值演示
    print("\n  数值演示（简化为 2 维）:")
    print("  ─────────────────────────────────────────────")

    # 模拟 3 个 token 的 Q, K, V
    tokens = ["工单", "T-001", "状态"]
    d_k = 2

    # 简化的 Q, K, V 矩阵
    Q = [[1.0, 0.5], [0.8, 1.2], [0.3, 0.9]]
    K = [[1.0, 0.3], [0.7, 1.0], [0.5, 0.8]]
    V = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]

    print(f"  Token:  {tokens}")
    print(f"  Q =     {Q}")
    print(f"  K =     {K}")
    print(f"  V =     {V}")

    # Q × Kᵀ
    scores = []
    for i in range(3):
        row = []
        for j in range(3):
            score = sum(Q[i][k] * K[j][k] for k in range(d_k))
            row.append(round(score, 2))
        scores.append(row)

    print(f"\n  Q × Kᵀ (注意力分数):")
    for i, token in enumerate(tokens):
        print(f"    {token:6s} → {scores[i]}")

    # 缩放 + softmax
    scale = math.sqrt(d_k)
    attention = []
    for row in scores:
        scaled = [s / scale for s in row]
        exp_vals = [math.exp(s) for s in scaled]
        total = sum(exp_vals)
        softmax_row = [round(e / total, 3) for e in exp_vals]
        attention.append(softmax_row)

    print(f"\n  softmax(Q×Kᵀ/√d_k) (注意力权重):")
    for i, token in enumerate(tokens):
        print(f"    {token:6s} → {attention[i]}  (关注: {tokens[attention[i].index(max(attention[i]))]})")

    print("""
  Masked Self-Attention（因果注意力）:
  ─────────────────────────────────────────────────────────
  Decoder 中，每个 token 只能看到自己和前面的 token:

           工单  T-001  状态  是
  工单     ✅    ✖      ✖    ✖
  T-001    ✅    ✅     ✖    ✖
  状态     ✅    ✅     ✅   ✖
  是       ✅    ✅     ✅   ✅

  → 保证自回归生成：预测下一个 token 时看不到未来

  Multi-Head Attention:
  ─────────────────────────────────────────────────────────
  将 d 维向量拆成 h 个头，每个头独立做 Attention:

  head_1: 关注语法关系（"工单" → "状态"）
  head_2: 关注实体关系（"T-001" → "工单"）
  head_3: 关注位置关系（相邻 token）
  ...
  最后 concat 所有头的输出

  → h 个头并行，各自学习不同的注意力模式""")


# ═══════════════════════════════════════════════════════════
# 3. Tokenizer
# ═══════════════════════════════════════════════════════════

class SimpleTokenizer:
    """简易 Tokenizer（模拟 BPE 行为）。

    真实的 BPE (Byte Pair Encoding):
    1. 从字节级别开始
    2. 统计最频繁的相邻 token 对
    3. 合并最频繁的对为新 token
    4. 重复直到达到词汇表大小
    """

    def __init__(self):
        # 模拟词汇表
        self.vocab = {
            "工": 1001, "单": 1002, "T": 1003, "-": 1004,
            "001": 1005, "状": 1006, "态": 1007, "是": 1008,
            "什": 1009, "么": 1010, "查": 1011, "询": 1012,
            " ": 1013, "hello": 2001, "world": 2002,
            "工单": 3001,  # 合并后的 token
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text: str) -> list[str]:
        """分词（模拟 BPE）。"""
        tokens = []
        i = 0
        while i < len(text):
            # 尝试最长匹配
            best = None
            for length in range(min(4, len(text) - i), 0, -1):
                candidate = text[i:i + length]
                if candidate in self.vocab:
                    best = candidate
                    break
            if best:
                tokens.append(best)
                i += len(best)
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def encode(self, text: str) -> list[int]:
        """编码为 token ID。"""
        return [self.vocab.get(t, 0) for t in self.tokenize(text)]


def show_tokenizer():
    """展示 Tokenizer 原理。"""
    print(f"\n\n▶ 3. Tokenizer — 文本如何变成模型输入")
    print("─" * 60)

    tokenizer = SimpleTokenizer()

    examples = [
        "工单T-001状态是什么",
        "hello world",
    ]

    for text in examples:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        print(f"\n  原文: \"{text}\"")
        print(f"  分词: {tokens}")
        print(f"  ID:   {ids}")
        print(f"  Token 数: {len(tokens)}")

    print(f"""
  主流 Tokenizer:
  ─────────────────────────────────────────────────────────
  BPE            │ GPT 系列, Llama, Qwen  │ 字节级合并
  SentencePiece  │ T5, Gemma              │ 无需预分词
  tiktoken       │ OpenAI 专用             │ BPE 优化实现

  Token 与成本:
  ─────────────────────────────────────────────────────────
  1 个英文单词   ≈ 1~2 个 token
  1 个中文字     ≈ 1~2 个 token
  1 行代码      ≈ 10~20 个 token
  1 页 A4 文档  ≈ 500~800 个 token

  GPT-4o 定价: $2.5 / 1M input tokens → 约 ¥0.018/千字
  → Agent 单次对话（含工具调用）约 2K~10K tokens

  为什么 Tokenizer 对 Agent 重要:
  ─────────────────────────────────────────────────────────
  1. Token 数决定成本 — System Prompt 越长越贵
  2. Token 数决定速度 — 输出越多延迟越高（自回归逐个生成）
  3. 上下文窗口限制 — prompt + 回复不能超过模型上限
  4. 微调数据计费 — 按 token 数量收费""")


# ═══════════════════════════════════════════════════════════
# 4. 推理参数 — SamplingSimulator
# ═══════════════════════════════════════════════════════════

class SamplingSimulator:
    """LLM 采样模拟器 — 演示 temperature / top_k / top_p 的真实效果。

    用真实的随机采样 + 频率统计展示参数对输出分布的影响。
    """

    def __init__(self, vocab: list[str], logits: list[float]):
        self.vocab = vocab
        self.logits = logits

    def _softmax(self, values: list[float]) -> list[float]:
        max_v = max(values)
        exp_vals = [math.exp(v - max_v) for v in values]  # 减最大值防溢出
        total = sum(exp_vals)
        return [e / total for e in exp_vals]

    def sample_once(self, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> str:
        """按参数采样一个 token。"""
        # 1. Temperature 缩放
        if temperature <= 0:
            return self.vocab[self.logits.index(max(self.logits))]  # 贪婪
        scaled = [l / temperature for l in self.logits]

        # 2. Softmax
        probs = self._softmax(scaled)

        # 3. Top-K 过滤
        indices = list(range(len(probs)))
        if top_k > 0 and top_k < len(probs):
            ranked = sorted(indices, key=lambda i: probs[i], reverse=True)
            allowed = set(ranked[:top_k])
            probs = [p if i in allowed else 0.0 for i, p in enumerate(probs)]

        # 4. Top-P (nucleus) 过滤
        if top_p < 1.0:
            ranked = sorted(indices, key=lambda i: probs[i], reverse=True)
            cumsum = 0.0
            allowed = set()
            for i in ranked:
                cumsum += probs[i]
                allowed.add(i)
                if cumsum >= top_p:
                    break
            probs = [p if i in allowed else 0.0 for i, p in enumerate(probs)]

        # 重新归一化
        total = sum(probs)
        if total == 0:
            return self.vocab[0]
        probs = [p / total for p in probs]

        # 5. 随机采样
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                return self.vocab[i]
        return self.vocab[-1]

    def sample_n(self, n: int = 100, **kwargs) -> dict:
        """采样 n 次并统计频率分布。"""
        counts = Counter(self.sample_once(**kwargs) for _ in range(n))
        total = sum(counts.values())
        return {token: {"count": counts.get(token, 0), "freq": counts.get(token, 0) / total}
                for token in self.vocab}

    def get_probs(self, temperature: float = 1.0) -> list[float]:
        """获取 softmax 后的概率分布。"""
        if temperature <= 0:
            probs = [0.0] * len(self.logits)
            probs[self.logits.index(max(self.logits))] = 1.0
            return probs
        return self._softmax([l / temperature for l in self.logits])


def show_inference_params():
    """用 SamplingSimulator 演示真实采样。"""
    print(f"\n\n▶ 4. 推理参数 — SamplingSimulator 采样实验")
    print("─" * 60)

    vocab = ["是", "为", "有", "在", "的"]
    logits = [2.0, 1.5, 1.0, 0.5, 0.3]
    sim = SamplingSimulator(vocab, logits)
    random.seed(42)  # 可复现

    # Temperature 对比
    print("\n  Temperature 采样实验（各 200 次）:")
    for temp in [0.0, 0.3, 0.7, 1.5]:
        label = {0.0: "贪婪  ", 0.3: "低温  ", 0.7: "默认  ", 1.5: "高温  "}[temp]
        result = sim.sample_n(200, temperature=temp)
        bar_parts = []
        for token in vocab:
            freq = result[token]["freq"]
            bar = "█" * int(freq * 30)
            bar_parts.append(f"{token}:{freq:4.0%}{'│' + bar if bar else ''}")
        print(f"    T={temp:.1f} ({label}): {' '.join(f'{t}:{result[t]["freq"]:4.0%}' for t in vocab)}")

    # Top-K 对比
    print("\n  Top-K 过滤（T=0.7, 各 200 次）:")
    for k in [2, 3, 5]:
        result = sim.sample_n(200, temperature=0.7, top_k=k)
        active = [t for t in vocab if result[t]["count"] > 0]
        print(f"    top_k={k}: 活跃 {len(active)} 个 token  {' '.join(f'{t}:{result[t]["freq"]:4.0%}' for t in vocab)}")

    # Top-P 对比
    print("\n  Top-P 核采样（T=0.7, 各 200 次）:")
    for p in [0.5, 0.8, 1.0]:
        result = sim.sample_n(200, temperature=0.7, top_p=p)
        active = [t for t in vocab if result[t]["count"] > 0]
        print(f"    top_p={p}: 活跃 {len(active)} 个 token  {' '.join(f'{t}:{result[t]["freq"]:4.0%}' for t in vocab)}")

    print(f"""
  Agent 场景推荐配置:
  ──────────────┬──────────────────────────────────
  工具调用       │ temperature=0, top_p=1
  客服回复       │ temperature=0.3, max_tokens=500
  代码生成       │ temperature=0.2, top_p=0.95
  创意文案       │ temperature=0.8, top_p=0.9
  数据提取       │ temperature=0, response_format=json""")


# ═══════════════════════════════════════════════════════════
# 5. 上下文窗口
# ═══════════════════════════════════════════════════════════

def show_context_window():
    """展示上下文窗口。"""
    print(f"\n\n▶ 5. 上下文窗口 — 模型能处理多长的文本")
    print("─" * 60)

    print(f"""
  上下文窗口 = 输入 + 输出的 token 总数上限

  ┌──────────────────────────────────────────────────┐
  │              Context Window (128K)               │
  │                                                  │
  │  ┌─────────────────────┬──────────────────────┐  │
  │  │  Input Tokens       │  Output Tokens       │  │
  │  │  System Prompt      │  模型生成的回复       │  │
  │  │  + 对话历史         │  + 工具调用           │  │
  │  │  + RAG 上下文       │                      │  │
  │  │  + 用户消息         │                      │  │
  │  └─────────────────────┴──────────────────────┘  │
  │                                                  │
  │  ⚠️ 输入越长 → 留给输出的空间越少               │
  └──────────────────────────────────────────────────┘

  各模型上下文窗口:
  ──────────────────┬────────────┬─────────────────
  模型               │ 上下文窗口  │ 最大输出
  ──────────────────┼────────────┼─────────────────
  GPT-4o            │ 128K       │ 16K
  GPT-4o-mini       │ 128K       │ 16K
  Claude 3.5 Sonnet │ 200K       │ 8K
  Claude 3.5 Haiku  │ 200K       │ 8K
  Gemini 1.5 Pro    │ 2M         │ 8K
  Gemini 2.0 Flash  │ 1M         │ 8K
  Llama 3.1 (8/70B) │ 128K       │ 128K
  Qwen 2.5 (7/72B)  │ 128K       │ 8K
  DeepSeek V3       │ 128K       │ 8K

  Agent 场景的上下文分配:
  ─────────────────────────────────────────────────────
  System Prompt     │ 500~2000 tokens  │ 角色+规则+工具说明
  对话历史          │ 2000~8000 tokens │ 最近 N 轮
  RAG 上下文        │ 1000~4000 tokens │ 检索到的文档片段
  用户消息          │ 100~500 tokens   │ 当前输入
  ─────────────────┼──────────────────┤
  输入合计          │ ~4K~15K tokens   │
  留给输出          │ ~2K~4K tokens    │ 回复 + 工具调用

  上下文过长时的策略:
  ─────────────────────────────────────────────────────
  1. 滑动窗口  │ 只保留最近 N 轮对话
  2. 摘要压缩  │ LLM 摘要旧对话（Compaction）
  3. RAG 替代  │ 长文档存入向量库，按需检索
  4. 分段处理  │ Map-Reduce 模式分段处理""")


# ═══════════════════════════════════════════════════════════
# 6. 模型家族
# ═══════════════════════════════════════════════════════════

def show_model_families():
    """展示主流模型家族。"""
    print(f"\n\n▶ 6. 模型家族 — 主流大模型对比")
    print("─" * 60)

    print(f"""
  闭源模型（API 调用）:
  ──────────────┬──────────────┬──────────────────────────
  模型           │ 公司          │ 特点
  ──────────────┼──────────────┼──────────────────────────
  GPT-4o        │ OpenAI       │ 综合最强，Agent 生态最好
  GPT-4o-mini   │ OpenAI       │ 性价比之王，Agent 首选
  o3/o4-mini    │ OpenAI       │ 推理模型，复杂逻辑推理
  Claude 3.5    │ Anthropic    │ 代码最强，200K 上下文
  Claude 3.5 H  │ Anthropic    │ 快速版，MCP 原生支持
  Gemini 2.0    │ Google       │ 2M 超长上下文，多模态强
  DeepSeek V3   │ DeepSeek     │ 开源最强之一，价格极低
  文心 4.0      │ 百度          │ 国内中文最好之一

  开源模型（自部署）:
  ──────────────┬──────────────┬──────────────────────────
  模型           │ 公司          │ 特点
  ──────────────┼──────────────┼──────────────────────────
  Llama 3.1     │ Meta         │ 开源标杆，8B/70B/405B
  Qwen 2.5      │ 阿里          │ 中文最强开源，0.5B~72B
  DeepSeek V2/3 │ DeepSeek     │ MoE 架构，推理性价比极高
  Mistral/Mixtral│ Mistral AI   │ 高效 MoE，欧洲开源
  GLM-4         │ 智谱          │ 中英双语，工具调用强
  Yi            │ 零一万物      │ 长上下文，34B 开源

  模型能力层级:
  ─────────────────────────────────────────────────────────

  ┌────────────────────────────────────────────────┐
  │  Level 4: 推理模型                              │
  │  o3, o4-mini, DeepSeek-R1                      │
  │  → 复杂数学/逻辑/代码推理                       │
  ├────────────────────────────────────────────────┤
  │  Level 3: 顶级通用模型                          │
  │  GPT-4o, Claude 3.5, Gemini 2.0 Pro           │
  │  → Agent 主力（工具调用 + 复杂规划）            │
  ├────────────────────────────────────────────────┤
  │  Level 2: 高性价比模型                          │
  │  GPT-4o-mini, Claude Haiku, Gemini Flash       │
  │  → Agent 工作马（日常任务，成本低 10x）         │
  ├────────────────────────────────────────────────┤
  │  Level 1: 开源小模型（7B~14B）                  │
  │  Llama 3.1 8B, Qwen 2.5 7B                    │
  │  → 本地部署 / 微调 / 边缘推理                  │
  └────────────────────────────────────────────────┘

  Agent 选模型:
  ─────────────────────────────────────────────────────────
  复杂工具调用 + 规划  │ GPT-4o / Claude 3.5 Sonnet
  日常 Agent 任务      │ GPT-4o-mini（成本低 10x）
  数学/代码推理        │ o3 / o4-mini
  本地/私有化部署      │ Qwen 2.5 7B / Llama 3.1 8B
  超长上下文           │ Gemini 2.0 Flash（1M 窗口）""")


# ═══════════════════════════════════════════════════════════
# 7. 从预训练到 Agent
# ═══════════════════════════════════════════════════════════

def show_llm_to_agent():
    """展示 LLM 到 Agent 的演进。"""
    print(f"\n\n▶ 7. 从预训练到 Agent — LLM 能力演进路线")
    print("─" * 60)

    print(f"""
  LLM 训练四阶段:
  ─────────────────────────────────────────────────────────

  Stage 1: 预训练 (Pre-training)
  ─────────────────────────────────────────────
  目标: 学习语言的统计规律（下一个 token 预测）
  数据: 互联网文本（TB 级别）
  成本: 数千万美元 + 数千张 GPU
  产出: Base Model（只会续写，不会对话）
           ↓
  Stage 2: 指令微调 (Instruction Tuning / SFT)
  ─────────────────────────────────────────────
  目标: 学会遵循指令（问什么答什么）
  数据: (指令, 回复) 对，人工标注（10K~100K 条）
  成本: 数十万美元
  产出: Instruct Model（会对话了）
           ↓
  Stage 3: 对齐 (Alignment / RLHF / DPO)
  ─────────────────────────────────────────────
  目标: 让模型安全、有用、无害
  方法: RLHF（人类反馈强化学习）/ DPO（直接偏好优化）
  数据: 人类偏好标注（哪个回答更好）
  产出: Chat Model（安全有用的对话模型）
           ↓
  Stage 4: Agent 能力 (Tool Use / Function Calling)
  ─────────────────────────────────────────────
  目标: 学会调用工具、结构化输出、多步推理
  方法: 工具调用数据微调 + 特殊 token 训练
  数据: (用户输入, 工具调用 JSON) 标注数据
  产出: Agent-ready Model（能驱动 Agent 框架）

  从 LLM 到 Agent 的完整技术栈:
  ─────────────────────────────────────────────────────────

  ┌─────────────────────────────────────────────────┐
  │                  Agent 应用层                     │
  │  LangChain / LangGraph / CrewAI / Spring AI     │
  ├─────────────────────────────────────────────────┤
  │                  Agent 能力层                     │
  │  Function Calling + Structured Output + Memory  │
  ├─────────────────────────────────────────────────┤
  │                  模型对齐层                       │
  │  RLHF / DPO / Constitutional AI                │
  ├─────────────────────────────────────────────────┤
  │                  指令微调层                       │
  │  SFT + LoRA + 指令数据集                        │
  ├─────────────────────────────────────────────────┤
  │                  预训练模型层                     │
  │  Transformer + Self-Attention + 万亿 Token      │
  ├─────────────────────────────────────────────────┤
  │                  基础设施层                       │
  │  GPU 集群 + 分布式训练 + 数据管道               │
  └─────────────────────────────────────────────────┘

  Agent 对 LLM 的核心依赖:
  ─────────────────────────────────────────────────────────
  Function Calling   │ LLM 输出结构化工具调用（非纯文本）
  指令遵循            │ 严格按 System Prompt 行为
  长上下文            │ 处理多轮对话 + RAG 上下文
  推理能力            │ 多步规划 + 错误修正
  结构化输出          │ 稳定输出 JSON/XML 格式""")


# ═══════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════

def main():
    print("=== 大模型（LLM）核心原理 ===\n")

    # 1. Transformer 架构
    show_transformer()

    # 2. Self-Attention
    show_attention()

    # 3. Tokenizer
    show_tokenizer()

    # 4. 推理参数
    show_inference_params()

    # 5. 上下文窗口
    show_context_window()

    # 6. 模型家族
    show_model_families()

    # 7. 从预训练到 Agent
    show_llm_to_agent()

    # ── 架构总结 ──────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 LLM 核心原理总结:")
    print()
    print("  Transformer → Attention → Token → 生成")
    print("  ────────────────────────────────────────────")
    print("  Transformer │ Decoder-Only 架构，自回归生成")
    print("  Attention   │ Q×Kᵀ/√d → softmax → ×V，信息聚焦")
    print("  Tokenizer   │ 文本→token→ID，决定成本和上下文")
    print("  推理参数     │ temperature 控制确定性/创造性")
    print("  上下文窗口   │ 输入+输出总量限制，合理分配")
    print()
    print("  Agent 开发者需要理解:")
    print("  ────────────────────────────────────────────")
    print("  □ Token 计费原理 — 优化 System Prompt 长度")
    print("  □ temperature=0 — Agent 工具调用的标配")
    print("  □ 上下文分配 — prompt + 历史 + RAG + 输出")
    print("  □ 模型选型 — 复杂用 4o，日常用 4o-mini")
    print("  □ Function Calling — Agent 能力的基石")
    print("  □ 结构化输出 — response_format=json_object")


if __name__ == "__main__":
    main()
