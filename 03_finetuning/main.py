"""大模型微调（Fine-tuning）完整指南

核心概念：让通用大模型学会你的专属任务 — 不是从零训练，而是在预训练基础上适配。

微调 vs RAG vs Prompt Engineering：
  ┌──────────────────┬──────────────────┬──────────────────┐
  │ Prompt Engineering│      RAG         │   Fine-tuning    │
  ├──────────────────┼──────────────────┼──────────────────┤
  │ 改指令            │ 改知识            │ 改模型            │
  │ 零成本            │ 中等成本          │ 高成本            │
  │ 即时生效          │ 数据更新即生效     │ 训练后才生效       │
  │ 通用能力          │ 私有数据问答       │ 专属风格/格式/能力  │
  └──────────────────┴──────────────────┴──────────────────┘
  先试 Prompt → 不够用加 RAG → 还不够再微调

本示例不依赖 GPU，用纯 Python 展示微调的完整知识体系：
  1. 微调方法分类：全量 / LoRA / QLoRA / Prefix Tuning
  2. 数据准备：指令数据集格式 + 数据质量
  3. OpenAI 微调 API 全流程
  4. 开源模型微调流程（HuggingFace + PEFT）
  5. 微调评估与上线
  6. Agent 场景的微调策略
"""

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# 1. 微调方法分类
# ═══════════════════════════════════════════════════════════

def show_finetuning_methods():
    """展示各种微调方法。"""
    print("▶ 1. 微调方法分类")
    print("─" * 60)

    print(f"""
  什么时候需要微调？
  ─────────────────────────────────────────────────────────
  ✅ 需要微调:
  - 模型需要学会特定的输出格式/风格（如 JSON Schema、法律文书）
  - 需要大幅降低推理成本（小模型微调替代大模型）
  - 需要领域专属能力（医疗/金融/代码）
  - Prompt 太长导致 Token 浪费

  ❌ 不需要微调:
  - 只是需要私有数据 → 用 RAG
  - 改改 prompt 就能解决 → 用 Prompt Engineering
  - 数据量不够（< 100 条）→ 用 Few-shot

  微调方法对比:
  ──────────────┬──────────┬──────────┬──────────┬──────────
  方法           │ 可训参数  │ 显存需求  │ 效果     │ 适用场景
  ──────────────┼──────────┼──────────┼──────────┼──────────
  全量微调 (FFT) │ 100%     │ 极高     │ ⭐⭐⭐   │ 资源充足
  LoRA           │ ~1%      │ 低       │ ⭐⭐⭐   │ 主流首选
  QLoRA          │ ~1%      │ 极低     │ ⭐⭐    │ 消费级GPU
  Prefix Tuning  │ <1%      │ 极低     │ ⭐⭐    │ 多任务共享
  Adapter        │ ~2%      │ 低       │ ⭐⭐    │ 早期方法
  P-Tuning v2    │ <1%      │ 极低     │ ⭐⭐    │ 中文模型

  显存估算（7B 模型）:
  ──────────────┬───────────────────────────────────
  全量微调       │ ~60 GB（需要 A100 80G）
  LoRA           │ ~16 GB（单张 V100/A10 可跑）
  QLoRA (4bit)   │ ~6 GB（RTX 4090 / T4 可跑）
  推理           │ ~14 GB（FP16）/ ~4 GB（4bit 量化）""")


def show_lora_details():
    """展示 LoRA 核心原理。"""
    print(f"\n\n▶ 2. LoRA 核心原理 — 微调的主流方案")
    print("─" * 60)

    print(f"""
  LoRA (Low-Rank Adaptation) 核心思想:
  ─────────────────────────────────────────────────────────
  不动原始权重 W，只学一个低秩的增量 ΔW = A × B

  原始:  h = W · x              (W 是 d×d 的大矩阵)
  LoRA:  h = W · x + A · B · x  (A 是 d×r, B 是 r×d, r≪d)

  ┌───────────────────────────────────────────────┐
  │                                               │
  │  输入 x ──┬──→ [W] (冻结) ──────┐             │
  │           │                     ├──→ 输出 h   │
  │           └──→ [A]×[B] (训练) ──┘             │
  │                 r=8~64                        │
  │                                               │
  │  参数量: d²=100M  vs  2×d×r=128K (节省99.9%)  │
  └───────────────────────────────────────────────┘

  LoRA 关键超参数:
  ─────────────────────────────────────────────────────────
  r (秩)          │ 8~64    │ 越大越强，但参数越多
  lora_alpha      │ 16~32   │ 缩放因子，通常 = 2×r
  lora_dropout    │ 0.05    │ 防过拟合
  target_modules  │ q,v,k,o │ 对哪些层加 LoRA

  QLoRA 在 LoRA 基础上加了量化:
  ─────────────────────────────────────────────────────────
  1. 将原始权重 W 量化为 4-bit NF4 格式（显存降 75%）
  2. 计算时反量化回 FP16
  3. LoRA 的 A、B 仍以 FP16 训练
  4. 效果接近 LoRA，显存仅需 1/3""")


# ═══════════════════════════════════════════════════════════
# 3. 数据准备 — DatasetBuilder + JSONL 生成/验证
# ═══════════════════════════════════════════════════════════

@dataclass
class FinetuneExample:
    """一条微调训练样本。"""
    system: str
    user: str
    assistant: str


# 模拟微调数据集（客服 Agent 场景）
FINETUNE_DATASET = [
    FinetuneExample(
        system="你是 SmartFlow 的技术支持助手。根据用户描述判断问题类型，给出诊断步骤。",
        user="登录后页面白屏，控制台报 502 错误",
        assistant=json.dumps({
            "type": "technical",
            "severity": "high",
            "diagnosis": [
                "1. 检查后端服务是否正常: curl http://localhost:8080/health",
                "2. 检查 Nginx 反向代理日志: tail -f /var/log/nginx/error.log",
                "3. 检查 Java 进程: ps aux | grep smartflow",
            ],
            "route_to": "engineering",
        }, ensure_ascii=False, indent=2),
    ),
    FinetuneExample(
        system="你是 SmartFlow 的技术支持助手。根据用户描述判断问题类型，给出诊断步骤。",
        user="流程审批卡在第三步不动了，已经两天了",
        assistant=json.dumps({
            "type": "workflow_stuck",
            "severity": "medium",
            "diagnosis": [
                "1. 登录管理后台 → 实例管理 → 搜索该流程实例",
                "2. 检查第三步的审批人是否已设置",
                "3. 检查是否有外部服务调用超时: grep TIMEOUT /var/log/smartflow/engine.log",
            ],
            "route_to": "support",
        }, ensure_ascii=False, indent=2),
    ),
    FinetuneExample(
        system="你是 SmartFlow 的技术支持助手。根据用户描述判断问题类型，给出诊断步骤。",
        user="API 调用返回 429 Too Many Requests",
        assistant=json.dumps({
            "type": "rate_limit",
            "severity": "low",
            "diagnosis": [
                "1. 确认当前版本: 免费版 100 次/分钟，企业版 10000 次/分钟",
                "2. 建议: 添加请求重试 + 指数退避",
                "3. 如需更高限额: 升级到企业版或联系 sales@smartflow.com",
            ],
            "route_to": "self_service",
        }, ensure_ascii=False, indent=2),
    ),
]


class DatasetBuilder:
    """微调数据集构建器 — 支持多格式导出和数据增强。"""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.examples: list[FinetuneExample] = []

    def add(self, user: str, assistant: str) -> "DatasetBuilder":
        self.examples.append(FinetuneExample(self.system_prompt, user, assistant))
        return self

    def add_batch(self, examples: list[FinetuneExample]) -> "DatasetBuilder":
        self.examples.extend(examples)
        return self

    def to_openai(self) -> list[dict]:
        """转换为 OpenAI 微调 JSONL 格式。"""
        return [{
            "messages": [
                {"role": "system", "content": ex.system},
                {"role": "user", "content": ex.user},
                {"role": "assistant", "content": ex.assistant},
            ]
        } for ex in self.examples]

    def to_alpaca(self) -> list[dict]:
        """转换为 Alpaca 格式。"""
        return [{"instruction": ex.system, "input": ex.user, "output": ex.assistant}
                for ex in self.examples]

    def to_sharegpt(self) -> list[dict]:
        """转换为 ShareGPT 多轮对话格式。"""
        return [{"conversations": [
            {"from": "system", "value": ex.system},
            {"from": "human", "value": ex.user},
            {"from": "gpt", "value": ex.assistant},
        ]} for ex in self.examples]

    def write_jsonl(self, path: str, fmt: str = "openai") -> int:
        """写入 JSONL 文件，返回写入行数。"""
        converter = {"openai": self.to_openai, "alpaca": self.to_alpaca, "sharegpt": self.to_sharegpt}
        data = converter[fmt]()
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return len(data)

    def split(self, train_ratio: float = 0.8) -> tuple["DatasetBuilder", "DatasetBuilder"]:
        """划分训练集/验证集。"""
        split_idx = int(len(self.examples) * train_ratio)
        train = DatasetBuilder(self.system_prompt)
        val = DatasetBuilder(self.system_prompt)
        train.examples = self.examples[:split_idx]
        val.examples = self.examples[split_idx:]
        return train, val

    def stats(self) -> dict:
        """统计数据集信息。"""
        total_chars = sum(len(ex.user) + len(ex.assistant) for ex in self.examples)
        avg_chars = total_chars // len(self.examples) if self.examples else 0
        return {
            "total_examples": len(self.examples),
            "total_chars": total_chars,
            "avg_chars_per_example": avg_chars,
            "estimated_tokens": total_chars * 2,  # 中文粗估
        }


class JSONLValidator:
    """JSONL 文件验证器 — 检查格式、字段完整性、数据质量。"""

    @staticmethod
    def validate(path: str, fmt: str = "openai") -> dict:
        """验证 JSONL 文件，返回验证结果。"""
        errors: list[str] = []
        warnings: list[str] = []
        line_count = 0
        empty_assistant = 0

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line_count += 1
                line = line.strip()
                if not line:
                    warnings.append(f"行 {i}: 空行")
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"行 {i}: JSON 解析失败 - {e}")
                    continue

                if fmt == "openai":
                    if "messages" not in data:
                        errors.append(f"行 {i}: 缺少 'messages' 字段")
                        continue
                    roles = [m.get("role") for m in data["messages"]]
                    if "assistant" not in roles:
                        errors.append(f"行 {i}: 缺少 assistant 回复")
                    for m in data["messages"]:
                        if not m.get("content", "").strip():
                            warnings.append(f"行 {i}: {m.get('role', '?')} 内容为空")
                        if m.get("role") == "assistant" and len(m.get("content", "")) < 5:
                            empty_assistant += 1
                elif fmt == "alpaca":
                    for key in ["instruction", "input", "output"]:
                        if key not in data:
                            errors.append(f"行 {i}: 缺少 '{key}' 字段")

        return {
            "valid": len(errors) == 0,
            "lines": line_count,
            "errors": errors,
            "warnings": warnings,
            "short_assistant_responses": empty_assistant,
        }


def demo_data_preparation():
    """演示 DatasetBuilder + JSONL 验证。"""
    print(f"\n\n▶ 3. 数据准备 — DatasetBuilder + JSONL 验证")
    print("─" * 60)

    # 构建数据集
    builder = DatasetBuilder(
        system_prompt="你是 SmartFlow 的技术支持助手。根据用户描述判断问题类型，给出诊断步骤。"
    )
    builder.add_batch(FINETUNE_DATASET)

    # 统计
    stats = builder.stats()
    print(f"\n  数据集统计:")
    print(f"    样本数: {stats['total_examples']}")
    print(f"    总字符: {stats['total_chars']}")
    print(f"    平均字符/样本: {stats['avg_chars_per_example']}")
    print(f"    预估 Token: ~{stats['estimated_tokens']}")

    # 写入 JSONL（三种格式）
    with tempfile.TemporaryDirectory() as tmpdir:
        for fmt in ["openai", "alpaca", "sharegpt"]:
            path = os.path.join(tmpdir, f"train_{fmt}.jsonl")
            count = builder.write_jsonl(path, fmt=fmt)
            # 读取第一行展示
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            print(f"\n  {fmt:8s} 格式 ({count} 行): {first_line[:100]}...")

        # 验证 JSONL
        openai_path = os.path.join(tmpdir, "train_openai.jsonl")
        result = JSONLValidator.validate(openai_path, fmt="openai")
        print(f"\n  JSONL 验证结果:")
        print(f"    有效: {'✅' if result['valid'] else '❌'}")
        print(f"    行数: {result['lines']}")
        print(f"    错误: {len(result['errors'])}")
        print(f"    警告: {len(result['warnings'])}")

        # 写入一个有错误的文件测试验证
        bad_path = os.path.join(tmpdir, "bad.jsonl")
        with open(bad_path, "w") as f:
            f.write('{"messages": [{"role": "user", "content": "hello"}]}\n')  # 缺 assistant
            f.write('not json at all\n')  # 非法 JSON
            f.write('{"wrong_key": 123}\n')  # 缺 messages
        bad_result = JSONLValidator.validate(bad_path, fmt="openai")
        print(f"\n  错误文件验证:")
        print(f"    有效: {'✅' if bad_result['valid'] else '❌'}")
        for err in bad_result['errors']:
            print(f"    ❌ {err}")

    print(f"""
  数据质量 Checklist:
  ─────────────────────────────────────────────────────────
  □ 数量: 最少 50 条，建议 500~5000 条
  □ 质量: 每条样本人工审核，确保回答准确
  □ 多样性: 覆盖各种边界情况和表述方式
  □ 一致性: 同类问题的回答风格统一
  □ 格式: 严格遵循目标输出格式
  □ 去重: 删除重复或高度相似的样本""")


# ═══════════════════════════════════════════════════════════
# 4. TrainingConfig 生成器
# ═══════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """微调训练配置生成器 — 根据场景自动推荐参数。"""
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    method: str = "lora"  # lora | qlora | full
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    bf16: bool = True
    max_seq_length: int = 2048
    output_dir: str = "./output"

    @classmethod
    def auto(cls, dataset_size: int, gpu_memory_gb: int = 24, task: str = "general") -> "TrainingConfig":
        """根据数据集大小和 GPU 显存自动推荐配置。"""
        config = cls()
        # GPU 显存决定方法
        if gpu_memory_gb <= 8:
            config.method = "qlora"
            config.batch_size = 1
            config.gradient_accumulation = 8
        elif gpu_memory_gb <= 24:
            config.method = "lora"
            config.batch_size = 4
        else:
            config.method = "full" if gpu_memory_gb >= 80 else "lora"
            config.batch_size = 8

        # 数据集大小决定 epochs
        if dataset_size < 100:
            config.num_epochs = 5
        elif dataset_size < 1000:
            config.num_epochs = 3
        else:
            config.num_epochs = 1

        # 任务类型决定 LoRA rank
        if task == "format":
            config.r = 8
        elif task == "tool_calling":
            config.r = 32
            config.max_seq_length = 4096
        elif task == "domain":
            config.r = 16

        config.lora_alpha = config.r * 2
        return config

    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "method": self.method,
            "lora_config": {
                "r": self.r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "target_modules": self.target_modules,
            } if self.method in ("lora", "qlora") else None,
            "training": {
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation,
                "learning_rate": self.learning_rate,
                "warmup_ratio": self.warmup_ratio,
                "bf16": self.bf16,
                "max_seq_length": self.max_seq_length,
            },
            "output_dir": self.output_dir,
        }

    def estimate_resources(self, dataset_size: int) -> dict:
        """估算训练资源需求。"""
        vram = {"full": 60, "lora": 16, "qlora": 6}
        time_per_sample_sec = {"full": 0.5, "lora": 0.3, "qlora": 0.4}
        est_vram = vram.get(self.method, 16)
        est_time_sec = dataset_size * self.num_epochs * time_per_sample_sec.get(self.method, 0.3)
        return {
            "estimated_vram_gb": est_vram,
            "estimated_time_minutes": round(est_time_sec / 60, 1),
            "estimated_cost_a100_usd": round(est_time_sec / 3600 * 3.0, 2),  # ~$3/h A100
        }


def demo_training_config():
    """演示 TrainingConfig 自动生成。"""
    print(f"\n\n▶ 4. TrainingConfig — 自动生成训练配置")
    print("─" * 60)

    scenarios = [
        ("消费级 GPU (8GB)", 200, 8, "format"),
        ("工作站 (24GB)", 500, 24, "tool_calling"),
        ("A100 (80GB)", 5000, 80, "domain"),
    ]

    for label, dataset_size, gpu_gb, task in scenarios:
        config = TrainingConfig.auto(dataset_size, gpu_gb, task)
        resources = config.estimate_resources(dataset_size)
        print(f"\n  {label} | {dataset_size} 条 | {task}:")
        print(f"    方法: {config.method} | r={config.r} | epochs={config.num_epochs} | bs={config.batch_size}")
        print(f"    显存: ~{resources['estimated_vram_gb']}GB | 时间: ~{resources['estimated_time_minutes']}min | 成本: ~${resources['estimated_cost_a100_usd']}")

    # 导出完整配置
    config = TrainingConfig.auto(500, 24, "tool_calling")
    print(f"\n  完整配置 JSON:")
    config_json = json.dumps(config.to_dict(), ensure_ascii=False, indent=4)
    for line in config_json.split("\n")[:12]:
        print(f"    {line}")
    print(f"    ...")

    print(f"""
  OpenAI 微调定价 (2025):
  ──────────┬──────────────┬──────────────────
  模型       │ 训练费用      │ 推理费用
  ──────────┼──────────────┼──────────────────
  gpt-4o-mini│ $0.3/M tokens│ 与原模型相同
  gpt-4o     │ $2.5/M tokens│ 与原模型相同""")


# ═══════════════════════════════════════════════════════════
# 5. 开源模型微调（HuggingFace + PEFT）
# ═══════════════════════════════════════════════════════════

def show_opensource_finetuning():
    """展示开源模型微调流程。"""
    print(f"\n\n▶ 5. 开源模型微调（HuggingFace + PEFT + LoRA）")
    print("─" * 60)

    print("""
  技术栈:
  ─────────────────────────────────────────────────────────
  transformers    │ HuggingFace 模型加载和训练
  peft            │ LoRA / QLoRA 等参数高效微调
  trl             │ SFTTrainer（指令微调专用 Trainer）
  datasets        │ 数据集加载和预处理
  bitsandbytes    │ 4-bit 量化（QLoRA 需要）
  accelerate      │ 多卡训练 + DeepSpeed 集成
  wandb           │ 训练监控可视化

  LoRA 微调代码（完整可运行）:
  ─────────────────────────────────────────────────────────""")

    print("""  from transformers import AutoModelForCausalLM, AutoTokenizer
  from peft import LoraConfig, get_peft_model, TaskType
  from trl import SFTTrainer, SFTConfig
  from datasets import load_dataset

  # 1. 加载基础模型
  model_name = "Qwen/Qwen2.5-7B-Instruct"
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      torch_dtype="auto",
      device_map="auto",
  )
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  # 2. 配置 LoRA
  lora_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      r=16,                          # 秩（8~64）
      lora_alpha=32,                 # 缩放（通常 2×r）
      lora_dropout=0.05,
      target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
  )
  model = get_peft_model(model, lora_config)
  model.print_trainable_parameters()
  # → trainable params: 13.1M || all params: 7.6B || 0.17%

  # 3. 加载数据集
  dataset = load_dataset("json", data_files="train.jsonl")

  # 4. 训练
  trainer = SFTTrainer(
      model=model,
      train_dataset=dataset["train"],
      args=SFTConfig(
          output_dir="./output",
          num_train_epochs=3,
          per_device_train_batch_size=4,
          gradient_accumulation_steps=4,
          learning_rate=2e-4,
          warmup_ratio=0.1,
          logging_steps=10,
          save_strategy="epoch",
          bf16=True,
      ),
      tokenizer=tokenizer,
  )
  trainer.train()

  # 5. 保存 LoRA 权重（仅几十 MB）
  model.save_pretrained("./lora_weights")

  # 6. 合并权重（部署用）
  merged = model.merge_and_unload()
  merged.save_pretrained("./merged_model")""")

    print(f"""
  QLoRA 只需额外加量化配置:
  ─────────────────────────────────────────────────────────
  from transformers import BitsAndBytesConfig

  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True,
  )
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=bnb_config,  # 加这一行
      device_map="auto",
  )

  主流开源基座模型:
  ──────────┬──────────┬──────────────────────────
  模型       │ 参数量    │ 特点
  ──────────┼──────────┼──────────────────────────
  Qwen 2.5  │ 0.5~72B  │ 中文最强，阿里开源
  Llama 3.1 │ 8~405B   │ 英文最强，Meta 开源
  DeepSeek  │ 7~236B   │ 代码 + 推理，性价比高
  Mistral   │ 7~8x22B  │ 高效 MoE 架构
  GLM-4     │ 9B       │ 智谱开源，中英双语
  Yi        │ 6~34B    │ 零一万物，长上下文""")


# ═══════════════════════════════════════════════════════════
# 6. 微调评估
# ═══════════════════════════════════════════════════════════

@dataclass
class FinetuneEvalResult:
    """微调评估结果。"""
    metric: str
    before: float
    after: float

    @property
    def improvement(self) -> str:
        diff = self.after - self.before
        return f"+{diff:.1%}" if diff > 0 else f"{diff:.1%}"


def show_evaluation():
    """展示微调评估方法。"""
    print(f"\n\n▶ 6. 微调评估 — 怎么判断微调有效")
    print("─" * 60)

    # 模拟评估对比
    eval_results = [
        FinetuneEvalResult("输出格式准确率", 0.45, 0.95),
        FinetuneEvalResult("问题分类准确率", 0.70, 0.92),
        FinetuneEvalResult("路由正确率", 0.60, 0.88),
        FinetuneEvalResult("Prompt Token 消耗", 0.0, -0.60),
    ]

    print(f"\n  微调前后对比（模拟）:")
    print(f"  {'指标':18s} {'微调前':>8s} {'微调后':>8s} {'提升':>8s}")
    print(f"  {'─' * 44}")
    for r in eval_results:
        if r.metric == "Prompt Token 消耗":
            print(f"  {r.metric:18s} {'1200':>8s} {'480':>8s} {'-60%':>8s}")
        else:
            print(f"  {r.metric:18s} {r.before:>7.0%} {r.after:>8.0%} {r.improvement:>8s}")

    print(f"""
  评估方法:
  ─────────────────────────────────────────────────────────
  1. 准备测试集（与训练集不重叠，50~200 条）
  2. 分别用原模型和微调模型跑测试集
  3. 对比关键指标

  评估维度:
  ──────────────┬──────────────────────────────────
  格式准确率     │ 输出是否严格符合目标格式
  任务准确率     │ 核心任务（分类/路由/生成）的正确率
  忠实度         │ 是否产生幻觉（微调可能加剧幻觉）
  泛化能力       │ 对训练集之外的输入是否仍有效
  推理效率       │ Prompt Token 是否减少、延迟是否降低
  过拟合检查     │ 验证集 loss 是否上升

  训练曲线监控:
  ──────────────────────────────────────────────────────────
  training loss ↓    │ 正常
  validation loss ↓  │ 正常
  training loss ↓↓   │ ⚠️ 可能过拟合
  validation loss ↑  │ ❌ 已经过拟合，需要:
                     │    - 减少 epoch
                     │    - 增加 dropout
                     │    - 增加训练数据""")


# ═══════════════════════════════════════════════════════════
# 7. Agent 场景的微调策略
# ═══════════════════════════════════════════════════════════

def show_agent_finetuning():
    """展示 Agent 场景的微调策略。"""
    print(f"\n\n▶ 7. Agent 场景的微调策略")
    print("─" * 60)

    print(f"""
  Agent 微调的三大方向:
  ─────────────────────────────────────────────────────────

  ┌─ 1. 工具调用微调 ──────────────────────────────────────┐
  │                                                        │
  │  目标: 让小模型学会 Function Calling                    │
  │  数据: 用户输入 → tool_calls JSON                      │
  │  效果: 7B 模型 ≈ GPT-4o-mini 的工具调用能力            │
  │                                                        │
  │  样本示例:                                             │
  │  User: 帮我查一下工单 T-001                             │
  │  Assistant: {{"tool_calls": [{{"name": "lookup_ticket",  │
  │    "arguments": {{"ticket_id": "T-001"}}}}]}}            │
  └────────────────────────────────────────────────────────┘

  ┌─ 2. 输出格式微调 ──────────────────────────────────────┐
  │                                                        │
  │  目标: 让模型严格输出指定 JSON/XML/Markdown 格式        │
  │  数据: 各种输入 → 标准格式输出                          │
  │  效果: 格式准确率从 ~50% 提升到 ~95%                   │
  │  优势: 省掉 prompt 里的格式说明 → 节省 Token            │
  └────────────────────────────────────────────────────────┘

  ┌─ 3. 领域知识微调 ──────────────────────────────────────┐
  │                                                        │
  │  目标: 让模型掌握领域术语和判断逻辑                      │
  │  数据: 领域 QA 对、专家标注的决策样本                    │
  │  效果: 领域准确率显著提升                                │
  │  注意: 通常与 RAG 配合使用（微调学判断，RAG 补知识）     │
  └────────────────────────────────────────────────────────┘

  微调 + RAG 组合拳:
  ─────────────────────────────────────────────────────────
  ┌─────────┐   ┌──────────┐   ┌──────────────┐
  │ 用户输入 │ → │ RAG 检索  │ → │ 微调模型生成  │
  │         │   │ (补知识)  │   │ (学格式+判断) │
  └─────────┘   └──────────┘   └──────────────┘

  微调学到的:               RAG 提供的:
  - 输出 JSON 格式           - 最新的产品文档
  - 问题分类能力             - 具体的工单数据
  - 领域术语理解             - 实时系统状态

  微调方案选型决策树:
  ─────────────────────────────────────────────────────────
  你的场景？
  │
  ├─ 快速验证 / 预算有限 ──────→ OpenAI 微调（最简单）
  ├─ 数据不能出外网 ───────────→ 开源模型 + LoRA（本地）
  ├─ 消费级 GPU (≤24G) ───────→ QLoRA + 7B 模型
  ├─ 多个任务共享一个模型 ────→ Prefix Tuning / 多 LoRA
  ├─ 追求极致效果 ─────────────→ 全量微调 + 大模型 (70B+)
  └─ 不想微调 ─────────────────→ Prompt Engineering + RAG""")


# ═══════════════════════════════════════════════════════════
# 8. 微调工具链与平台
# ═══════════════════════════════════════════════════════════

def show_tooling():
    """展示微调工具链。"""
    print(f"\n\n▶ 8. 微调工具链与平台")
    print("─" * 60)

    print(f"""
  开源工具:
  ──────────────┬───────────────────────────────────────
  LLaMA-Factory │ 一站式微调框架，Web UI + 100+ 模型
  Axolotl       │ 配置驱动的微调框架，YAML 一键训练
  Unsloth       │ 2x 加速 LoRA 训练，显存降 60%
  TRL (HF)      │ HuggingFace 官方，SFT + RLHF + DPO
  swift (魔搭)  │ 阿里开源，国产模型适配最好

  云平台:
  ──────────────┬───────────────────────────────────────
  OpenAI API    │ 最简单，上传 JSONL → 等待 → 调用
  AutoTrain     │ HuggingFace 零代码微调
  Fireworks AI  │ 按需微调 + 部署，pay-as-you-go
  Together AI   │ 开源模型微调 + 推理 API
  阿里云 PAI    │ 国内首选，Qwen 适配最好
  百度千帆      │ 文心模型微调 + 评测

  训练监控:
  ──────────────┬───────────────────────────────────────
  wandb         │ 最主流，loss 曲线 + 超参搜索
  TensorBoard   │ 本地可视化，HF Trainer 内置
  MLflow        │ 实验管理 + 模型注册

  部署方案:
  ──────────────┬───────────────────────────────────────
  vLLM          │ 最快推理引擎，PagedAttention
  Ollama        │ 本地一键部署，支持 GGUF 量化
  TGI           │ HuggingFace 官方推理服务
  FastChat      │ OpenAI 兼容 API 服务
  Xinference    │ 分布式推理，国产模型支持好""")


# ═══════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════

def main():
    print("=== 大模型微调（Fine-tuning）完整指南 ===\n")

    # 1. 微调方法分类
    show_finetuning_methods()

    # 2. LoRA 核心原理
    show_lora_details()

    # 3. 数据准备
    demo_data_preparation()

    # 4. 训练配置生成
    demo_training_config()

    # 5. 开源模型微调
    show_opensource_finetuning()

    # 6. 微调评估
    show_evaluation()

    # 7. Agent 微调策略
    show_agent_finetuning()

    # 8. 工具链
    show_tooling()

    # ── 架构总结 ──────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 大模型微调总结:")
    print()
    print("  微调方法选择:")
    print("  ────────────────────────────────────────────")
    print("  LoRA          │ 主流首选，平衡效果与成本")
    print("  QLoRA         │ 资源受限时，消费级 GPU 可跑")
    print("  全量微调       │ 追求极致效果，资源充足时")
    print("  OpenAI API    │ 最简单，不想管基础设施时")
    print()
    print("  关键成功因素:")
    print("  ────────────────────────────────────────────")
    print("  1. 数据质量 > 数据数量（100 条好数据 > 10000 条烂数据）")
    print("  2. 先 Prompt Engineering → 再 RAG → 最后微调")
    print("  3. 准备好测试集，量化微调前后差异")
    print("  4. 监控过拟合（验证集 loss）")
    print("  5. 微调 + RAG 组合效果最佳")
    print()
    print("  微调 Checklist:")
    print("  ────────────────────────────────────────────")
    print("  □ 确认 Prompt/RAG 无法满足需求")
    print("  □ 准备高质量标注数据（≥ 500 条）")
    print("  □ 划分训练集/验证集/测试集")
    print("  □ 选择基座模型和微调方法")
    print("  □ 训练并监控 loss 曲线")
    print("  □ 对比微调前后指标")
    print("  □ 检查过拟合和幻觉")
    print("  □ 部署并持续监控线上效果")


if __name__ == "__main__":
    main()
