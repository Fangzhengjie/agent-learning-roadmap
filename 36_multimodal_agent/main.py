"""多模态 Agent — 视觉 / 语音 / 视频 Agent

核心概念：Agent 从纯文本扩展到看图、听声、看视频 — 感知能力决定 Agent 的边界。

多模态 Agent 能力矩阵:
  ┌──────────────────────────────────────────────────────────┐
  │  文本  │ 所有 Agent 的基础（对话 / 工具调用）             │
  │  视觉  │ 看截图 → 理解 UI → 点击操作（Browser Agent）   │
  │  语音  │ 听指令 → 理解意图 → 语音回复（语音助手）       │
  │  视频  │ 看视频 → 理解内容 → 生成摘要（监控 / 教学）    │
  │  文件  │ 读 PDF / PPT / Excel → 提取信息（文档 Agent）  │
  └──────────────────────────────────────────────────────────┘

本示例展示多模态 Agent 的核心知识：
  1. 视觉理解 — GPT-4V / Claude Vision 图片分析
  2. Browser Agent — 截图→LLM→操作 循环
  3. 语音 Agent — STT + LLM + TTS 管道
  4. 视频理解 — 关键帧提取 + 多帧分析
  5. 文档解析 — PDF/PPT/表格的多模态处理
  6. 多模态模型对比
"""

import base64
import json
import os
import struct
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# 1. 视觉理解
# ═══════════════════════════════════════════════════════════

def show_vision():
    """展示视觉理解能力。"""
    print("▶ 1. 视觉理解 — LLM 看懂图片")
    print("─" * 60)

    print("""
  Vision API 调用方式（OpenAI 格式）:
  ─────────────────────────────────────────────────────────

  response = client.chat.completions.create(
      model="gpt-4o",
      messages=[{
          "role": "user",
          "content": [
              {"type": "text", "text": "这个 UI 截图中有什么问题？"},
              {"type": "image_url", "image_url": {
                  "url": "data:image/png;base64,{base64_image}",
                  "detail": "high"  # low / high / auto
              }},
          ],
      }],
  )

  图片输入方式:
  ──────────────┬──────────────────────────────────────
  Base64        │ 图片编码后直接传入（最常用）
  URL           │ 传入图片 URL（需要可公开访问）
  本地文件       │ 读取后转 Base64

  detail 参数:
  ──────────────┬──────────────────────────────────────
  low           │ 512×512 缩放，65 tokens（便宜快速）
  high          │ 原图分块处理，最多 ~1K tokens（精确）
  auto          │ 模型自动判断

  视觉理解能力:
  ─────────────────────────────────────────────────────────
  ✅ UI 截图分析    │ 识别按钮、表单、布局、错误提示
  ✅ 图表理解       │ 读取柱状图/折线图/饼图的数据
  ✅ 文档 OCR       │ 提取手写文字、扫描件内容
  ✅ 代码截图       │ 从截图中读出代码（IDE 截图）
  ✅ 设计稿理解     │ Figma/Sketch 设计稿 → 代码
  ⚠️ 精确计数       │ 图中有多少个点？（不稳定）
  ⚠️ 空间推理       │ A 在 B 的哪个方向？（不稳定）""")


# ═══════════════════════════════════════════════════════════
# 2. Browser Agent（视觉 Agent 典型应用）
# ═══════════════════════════════════════════════════════════

def show_browser_agent():
    """展示 Browser Agent 的视觉循环。"""
    print(f"\n\n▶ 2. Browser Agent — 截图→LLM→操作 循环")
    print("─" * 60)

    print("""
  Browser Agent 核心循环:
  ─────────────────────────────────────────────────────────

  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
  │  │ 1. 截图   │ →  │ 2. 理解   │ →  │ 3. 操作   │     │
  │  │Screenshot│    │ Vision   │    │ Action   │     │
  │  └──────────┘    │ LLM      │    └──────────┘     │
  │       ↑          └──────────┘         │           │
  │       └───────────────────────────────┘           │
  │                                                      │
  └──────────────────────────────────────────────────────┘

  操作类型:
  ──────────────┬──────────────────────────────────────
  click(x, y)   │ 点击页面坐标
  type(text)     │ 在焦点元素输入文本
  scroll(dir)    │ 上下滚动页面
  navigate(url)  │ 跳转到新 URL
  wait(seconds)  │ 等待页面加载
  screenshot()   │ 获取当前截图""")

    # 运行 Browser Agent 模拟
    print("\n  代码模拟 Browser Agent 搜索商品:")
    browser = MockBrowser()
    agent = BrowserAgent(browser)
    agent.run("在商城搜索机械键盘，找到最便宜的")

    print(f"""
  Browser Agent 方案:
  ──────────────┬──────────┬──────────────────────────
  Browser Use   │ 开源      │ Playwright+截图，可本地运行
  Playwright+LLM│ 自建      │ 灵活度最高
  Anthropic CUA │ Claude    │ 全屏幕截图+操控，通用性最强
  LaVague       │ 开源      │ HTML 语义理解
  Skyvern      │ 开源      │ 专注表单填写和数据提取

  适用场景: 网页数据采集 / E2E 测试 / RPA / 竞品监控
  ⚠️ 局限: 每步截图+LLM，速度慢、Token 消耗大""")


# ═══════════════════════════════════════════════════════════
# 3. 语音 Agent
# ═══════════════════════════════════════════════════════════

def show_voice_agent():
    """展示语音 Agent 管道。"""
    print(f"\n\n▶ 3. 语音 Agent — STT + LLM + TTS 管道")
    print("─" * 60)

    print(f"""
  语音 Agent 架构:
  ─────────────────────────────────────────────────────────

  ┌──────┐   ┌───────┐   ┌──────┐   ┌───────┐   ┌──────┐
  │ 用户  │→ │  STT  │→ │ LLM  │→ │  TTS  │→ │ 用户  │
  │ 说话  │   │语音转文│   │ 理解  │   │文字转音│   │ 听到  │
  └──────┘   └───────┘   └──────┘   └───────┘   └──────┘
   麦克风     Whisper     GPT-4o     TTS-1       扬声器
              ~1s         ~1s        ~0.5s       

  传统管道 vs 原生多模态:
  ─────────────────────────────────────────────────────────

  传统管道（STT→LLM→TTS）:
  ┌────────────────────────────────────────────────┐
  │ 语音 → Whisper → 文本 → GPT → 文本 → TTS → 语音│
  │ 延迟: ~3s（不自然，丢失语气信息）                │
  └────────────────────────────────────────────────┘

  原生多模态（GPT-4o Realtime）:
  ┌────────────────────────────────────────────────┐
  │ 语音 → GPT-4o → 语音                           │
  │ 延迟: ~0.3s（自然，保留语气和情感）              │
  └────────────────────────────────────────────────┘

  语音技术栈:
  ──────────────┬──────────┬──────────────────────
  组件           │ 方案      │ 特点
  ──────────────┼──────────┼──────────────────────
  STT           │ Whisper  │ OpenAI 开源，100+ 语言
                │ Deepgram │ 实时 STT，延迟最低
                │ Azure STT│ 微软，企业级
  ──────────────┼──────────┼──────────────────────
  TTS           │ OpenAI   │ TTS-1 / TTS-1-HD
                │ ElevenLabs│ 最自然的 AI 语音
                │ Azure TTS│ 微软，SSML 支持
                │ Edge TTS │ 免费，微软边缘
  ──────────────┼──────────┼──────────────────────
  Realtime      │ GPT-4o   │ Realtime API (WebSocket)
                │ Gemini   │ Live API (原生多模态)

  OpenAI Realtime API:
  ─────────────────────────────────────────────────────────

  import asyncio, websockets, json

  async def voice_agent():
      url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime"
      headers = {{"Authorization": "Bearer sk-...",
                  "OpenAI-Beta": "realtime=v1"}}

      async with websockets.connect(url, extra_headers=headers) as ws:
          # 配置会话
          await ws.send(json.dumps({{
              "type": "session.update",
              "session": {{
                  "modalities": ["text", "audio"],
                  "instructions": "你是 SmartFlow 语音助手",
                  "voice": "alloy",
                  "tools": [...],  # Agent 工具
              }}
          }}))

          # 发送音频 → 接收音频回复
          # ...流式音频处理...

  语音 Agent 应用场景:
  ──────────────┬──────────────────────────────────────
  电话客服       │ 接听来电 → 语音对话 → 工单处理
  语音助手       │ 类 Siri/Alexa，支持工具调用
  会议助手       │ 实时转录 + 总结 + 行动项提取
  无障碍交互     │ 视障用户的语音界面""")


# ═══════════════════════════════════════════════════════════
# 4. 视频理解
# ═══════════════════════════════════════════════════════════

def show_video_understanding():
    """展示视频理解方案。"""
    print(f"\n\n▶ 4. 视频理解 — 关键帧提取 + 多帧分析")
    print("─" * 60)

    print(f"""
  视频理解的两种方式:
  ─────────────────────────────────────────────────────────

  方式 1: 关键帧提取（通用方式）
  ┌──────────────────────────────────────────────────────┐
  │ 视频 → 每 N 秒截取一帧 → 多张图片发给 Vision LLM     │
  │                                                      │
  │ 实现:                                                 │
  │ import cv2                                            │
  │ cap = cv2.VideoCapture("video.mp4")                   │
  │ frames = []                                           │
  │ while cap.isOpened():                                  │
  │     ret, frame = cap.read()                            │
  │     if frame_count % (fps * interval) == 0:            │
  │         frames.append(encode_base64(frame))            │
  │                                                      │
  │ # 将多帧发给 Vision LLM                               │
  │ response = client.chat.completions.create(             │
  │     model="gpt-4o",                                    │
  │     messages=[{{"role": "user", "content": [           │
  │         {{"type": "text", "text": "分析这个视频"}},     │
  │         *[{{"type": "image_url", ...}} for f in frames]│
  │     ]}}],                                              │
  │ )                                                      │
  └──────────────────────────────────────────────────────┘

  方式 2: 原生视频理解（Gemini）
  ┌──────────────────────────────────────────────────────┐
  │ Gemini 1.5/2.0 支持直接上传视频文件:                   │
  │                                                      │
  │ import google.generativeai as genai                    │
  │ video = genai.upload_file("video.mp4")                 │
  │ response = model.generate_content(                     │
  │     ["总结这个视频的关键内容", video]                   │
  │ )                                                      │
  │                                                      │
  │ 支持: 最长 1 小时视频, 原生时间戳理解                  │
  └──────────────────────────────────────────────────────┘

  视频理解应用:
  ──────────────┬──────────────────────────────────────
  安防监控       │ 异常行为检测 → 告警
  教学分析       │ 课程视频 → 知识点提取 → 字幕生成
  质量检测       │ 生产线视频 → 缺陷检测
  会议录像       │ 录像 → 转录 → 摘要 → 行动项
  内容审核       │ UGC 视频 → 违规检测""")


# ═══════════════════════════════════════════════════════════
# 5. 文档解析
# ═══════════════════════════════════════════════════════════

def show_document_parsing():
    """展示多模态文档解析。"""
    print(f"\n\n▶ 5. 文档解析 — PDF/PPT/表格的多模态处理")
    print("─" * 60)

    print(f"""
  文档解析方案:
  ─────────────────────────────────────────────────────────

  ┌─ 纯文本提取（传统方式）──────────────────────────────┐
  │  PDF → PyPDF2/pdfplumber → 纯文本                    │
  │  ✅ 快速、便宜                                        │
  │  ❌ 丢失表格/图表/布局信息                             │
  └──────────────────────────────────────────────────────┘

  ┌─ 视觉理解（多模态方式）──────────────────────────────┐
  │  PDF → 渲染为图片 → Vision LLM                        │
  │  ✅ 保留表格/图表/布局                                 │
  │  ❌ 成本高、速度慢                                     │
  └──────────────────────────────────────────────────────┘

  ┌─ 专用文档模型（最佳实践）────────────────────────────┐
  │  PDF → 专用解析 → 结构化输出                          │
  │  工具: Unstructured / Docling / MinerU / Marker       │
  │  ✅ 表格/图表/公式都能处理                             │
  │  ✅ 成本介于两者之间                                   │
  └──────────────────────────────────────────────────────┘

  文档解析工具:
  ──────────────┬──────────────────────────────────────
  Unstructured  │ 最全面，支持 20+ 格式，开源
  Docling (IBM) │ PDF 专用，表格提取最强
  MinerU        │ 上海 AI Lab，PDF 转 Markdown
  Marker        │ PDF → Markdown，开源轻量
  LlamaParse    │ LlamaIndex 出品，RAG 优化

  各格式处理方式:
  ──────────────┬──────────────────────────────────────
  PDF (文本型)   │ pdfplumber 提取 + LLM 理解
  PDF (扫描件)   │ 渲染图片 → Vision LLM / OCR
  PPT/PPTX      │ python-pptx 提取 + 图片 Vision
  Excel/CSV     │ pandas 结构化 + LLM 分析
  Word/DOCX     │ python-docx 提取
  图片文档       │ 直接 Vision LLM""")


# ═══════════════════════════════════════════════════════════
# 6. 多模态模型对比
# ═══════════════════════════════════════════════════════════

def show_model_comparison():
    """展示多模态模型对比。"""
    print(f"\n\n▶ 6. 多模态模型对比")
    print("─" * 60)

    print(f"""
  多模态能力矩阵:
  ──────────────┬──────┬──────┬──────┬──────┬──────
  模型           │ 图片  │ 视频  │ 音频  │ 文件  │ 生成
  ──────────────┼──────┼──────┼──────┼──────┼──────
  GPT-4o        │ ✅    │ ⚠️帧  │ ✅    │ ✅   │ ✅ 图
  GPT-4o-mini   │ ✅    │ ⚠️帧  │ ✅    │ ✅   │ ❌
  Claude 3.5    │ ✅    │ ❌    │ ❌    │ ✅   │ ❌
  Gemini 2.0    │ ✅    │ ✅原生 │ ✅    │ ✅   │ ✅ 图
  Llama 3.2     │ ✅    │ ❌    │ ❌    │ ❌   │ ❌
  Qwen2-VL      │ ✅    │ ✅    │ ❌    │ ❌   │ ❌

  各模态的 Agent 应用:
  ─────────────────────────────────────────────────────────

  ┌──────────────────────────────────────────────────────┐
  │  输入模态          Agent 应用                         │
  ├──────────────────────────────────────────────────────┤
  │  📸 截图/照片  →   Browser Agent / 设计稿→代码       │
  │  📄 PDF/PPT   →   文档问答 / RAG 数据源              │
  │  📊 图表      →   数据分析 Agent / BI 报表           │
  │  🎤 语音      →   电话客服 / 语音助手                │
  │  🎥 视频      →   监控告警 / 会议摘要                │
  │  🖥️ 屏幕     →   Computer Use / 桌面自动化          │
  └──────────────────────────────────────────────────────┘

  多模态 Agent 架构:
  ─────────────────────────────────────────────────────────

  ┌───────────────────────────────────────────┐
  │              多模态 Agent                  │
  │                                           │
  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
  │  │视觉  │ │语音  │ │文档  │ │文本  │       │
  │  │感知  │ │感知  │ │解析  │ │理解  │       │
  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘       │
  │     └───────┴───────┴───────┘            │
  │              ↓ 统一为文本/embedding        │
  │         ┌────────────┐                    │
  │         │ LLM 推理    │                    │
  │         └─────┬──────┘                    │
  │               ↓                           │
  │         ┌────────────┐                    │
  │         │ 工具调用     │                    │
  │         └────────────┘                    │
  └───────────────────────────────────────────┘""")


# ═══════════════════════════════════════════════════════════
# Browser Agent 模拟组件（合并自 16_browser_use）
# ═══════════════════════════════════════════════════════════

class ActionType(Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    EXTRACT = "extract"
    NAVIGATE = "navigate"
    DONE = "done"


@dataclass
class BrowserAction:
    """一个浏览器操作。"""
    action: ActionType
    selector: str = ""
    value: str = ""
    reasoning: str = ""


@dataclass
class PageState:
    """页面状态。"""
    url: str
    title: str
    elements: list[dict]
    text_content: str


class MockBrowser:
    """模拟浏览器。"""
    PAGES = {
        "https://shop.example.com": PageState(
            url="https://shop.example.com", title="示例商城 - 首页",
            elements=[
                {"tag": "input", "text": "搜索商品...", "input": True, "id": "search-box"},
                {"tag": "button", "text": "搜索", "clickable": True, "id": "search-btn"},
            ],
            text_content="欢迎来到示例商城。热门: 机械键盘、4K 显示器。"),
        "https://shop.example.com/search?q=机械键盘": PageState(
            url="https://shop.example.com/search?q=机械键盘", title="搜索: 机械键盘",
            elements=[
                {"tag": "div", "text": "K8 Pro ¥899", "clickable": True, "id": "product-1"},
                {"tag": "div", "text": "G915 ¥1,499", "clickable": True, "id": "product-2"},
                {"tag": "div", "text": "Anne Pro 2 ¥599", "clickable": True, "id": "product-3"},
            ],
            text_content="找到 3 件商品。"),
        "https://shop.example.com/product/k8-pro": PageState(
            url="https://shop.example.com/product/k8-pro", title="K8 Pro 机械键盘",
            elements=[
                {"tag": "span", "text": "¥899", "id": "price"},
                {"tag": "span", "text": "有货 (42件)", "id": "stock"},
                {"tag": "button", "text": "加入购物车", "clickable": True, "id": "add-to-cart"},
            ],
            text_content="K8 Pro ¥899，Gateron Pro 轴，PBT 键帽，库存 42。"),
    }

    def __init__(self):
        self.current_url = ""

    def execute(self, action: BrowserAction) -> PageState:
        if action.action == ActionType.NAVIGATE:
            self.current_url = action.value
        elif action.action == ActionType.TYPE and action.selector == "search-box":
            self.current_url = f"https://shop.example.com/search?q={action.value}"
        elif action.action == ActionType.CLICK and action.selector == "product-1":
            self.current_url = "https://shop.example.com/product/k8-pro"
        return self.PAGES.get(self.current_url,
            PageState(url=self.current_url, title="404", elements=[], text_content=""))


class BrowserAgent:
    """模拟 Browser Agent 的决策过程。"""

    def __init__(self, browser: MockBrowser):
        self.browser = browser

    def run(self, task: str):
        actions = [
            BrowserAction(ActionType.NAVIGATE, value="https://shop.example.com",
                          reasoning="打开商城首页"),
            BrowserAction(ActionType.TYPE, selector="search-box", value="机械键盘",
                          reasoning="输入搜索关键词"),
            BrowserAction(ActionType.CLICK, selector="product-1",
                          reasoning="点击第一个商品查看详情"),
            BrowserAction(ActionType.EXTRACT, selector="price,stock",
                          reasoning="提取价格和库存信息"),
            BrowserAction(ActionType.DONE, value="已获取商品信息",
                          reasoning="任务完成"),
        ]
        icons = {ActionType.NAVIGATE: "🌐", ActionType.CLICK: "👆", ActionType.TYPE: "⌨️",
                 ActionType.SCREENSHOT: "📸", ActionType.EXTRACT: "📋", ActionType.DONE: "✅"}
        for i, action in enumerate(actions, 1):
            icon = icons.get(action.action, "▶")
            info = f" [{action.selector}]" if action.selector else ""
            info += f" → '{action.value}'" if action.value else ""
            print(f"    Step {i}: {icon} {action.action.value}{info}")
            print(f"           💭 {action.reasoning}")
            if action.action == ActionType.DONE:
                break
            page = self.browser.execute(action)
            if action.action in (ActionType.NAVIGATE, ActionType.TYPE, ActionType.CLICK):
                print(f"           📄 {page.title} ({len(page.elements)} 个元素)")
            if action.action == ActionType.EXTRACT:
                print(f"           📋 {page.text_content[:80]}")


# ═══════════════════════════════════════════════════════════
# 7. 可运行组件 — VoicePipeline / ImageEncoder / DocumentParser
# ═══════════════════════════════════════════════════════════

class ImageEncoder:
    """图片 → Base64 编码器（Vision API 所需格式）。"""

    @staticmethod
    def encode_file(path: str) -> str:
        """读取图片文件并编码为 base64。"""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def make_vision_message(text: str, image_b64: str, detail: str = "auto") -> dict:
        """构建 OpenAI Vision API 消息体。"""
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                    "detail": detail,
                }},
            ],
        }

    @staticmethod
    def estimate_tokens(width: int, height: int, detail: str = "high") -> int:
        """估算图片 token 消耗。"""
        if detail == "low":
            return 85
        # high: 缩放到 2048 最长边，然后按 512x512 分块
        scale = min(2048 / max(width, height), 1.0)
        w, h = int(width * scale), int(height * scale)
        tiles_w = max(1, (w + 511) // 512)
        tiles_h = max(1, (h + 511) // 512)
        return 85 + 170 * tiles_w * tiles_h


class VoicePipelineSimulator:
    """语音管道模拟器 — 生成真实 WAV + 模拟 STT→LLM→TTS。"""

    @staticmethod
    def generate_wav(path: str, duration_s: float = 1.0, freq: int = 440, sample_rate: int = 16000):
        """生成一个真实的 WAV 文件（正弦波）。"""
        import math
        n_samples = int(sample_rate * duration_s)
        samples = []
        for i in range(n_samples):
            t = i / sample_rate
            value = int(32767 * 0.5 * math.sin(2 * math.pi * freq * t))
            samples.append(struct.pack("<h", value))

        with open(path, "wb") as f:
            data = b"".join(samples)
            # WAV header
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + len(data)))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))       # chunk size
            f.write(struct.pack("<H", 1))        # PCM
            f.write(struct.pack("<H", 1))        # mono
            f.write(struct.pack("<I", sample_rate))
            f.write(struct.pack("<I", sample_rate * 2))  # byte rate
            f.write(struct.pack("<H", 2))        # block align
            f.write(struct.pack("<H", 16))       # bits per sample
            f.write(b"data")
            f.write(struct.pack("<I", len(data)))
            f.write(data)

    @staticmethod
    def mock_stt(wav_path: str) -> str:
        """模拟 STT（Whisper）。"""
        size = os.path.getsize(wav_path)
        return f"帮我查一下工单 T-001 的状态"  # 模拟识别结果

    @staticmethod
    def mock_llm(text: str) -> str:
        """模拟 LLM 处理。"""
        if "工单" in text:
            return "工单 T-001 是技术类问题，状态为处理中，已分配给工程团队。"
        return f"收到您的消息：{text}"

    @staticmethod
    def mock_tts(text: str, output_path: str):
        """模拟 TTS — 生成静音 WAV 作为输出。"""
        VoicePipelineSimulator.generate_wav(output_path, duration_s=0.5, freq=0)

    def run_pipeline(self, input_wav: str, output_wav: str) -> dict:
        """运行完整 STT → LLM → TTS 管道。"""
        transcript = self.mock_stt(input_wav)
        response = self.mock_llm(transcript)
        self.mock_tts(response, output_wav)
        return {
            "input_size": os.path.getsize(input_wav),
            "transcript": transcript,
            "response": response,
            "output_size": os.path.getsize(output_wav),
        }


class DocumentParser:
    """简易文档解析器 — 模拟不同格式的文本提取。"""

    MOCK_DOCS = {
        ".pdf": {
            "title": "Q4 财务报告",
            "pages": 12,
            "text": "2024年第四季度，公司实现营收 ¥2.5 亿，同比增长 18%。净利润 ¥3,200 万。",
            "tables": [{"name": "营收摘要", "rows": 4, "cols": 3}],
            "images": 2,
        },
        ".pptx": {
            "title": "产品路线图 2025",
            "pages": 24,
            "text": "Q1: AI 客服上线。Q2: 多模态 Agent。Q3: 浏览器自动化。Q4: 语音 Agent。",
            "tables": [],
            "images": 15,
        },
        ".xlsx": {
            "title": "销售数据",
            "pages": 3,
            "text": "Sheet1: 华东区 ¥1.2亿, 华南区 ¥8,500万, 华北区 ¥9,800万",
            "tables": [{"name": "Sheet1", "rows": 150, "cols": 8}],
            "images": 0,
        },
    }

    def parse(self, filename: str) -> dict:
        """解析文档（模拟）。"""
        ext = os.path.splitext(filename)[1].lower()
        doc = self.MOCK_DOCS.get(ext)
        if not doc:
            return {"error": f"不支持的格式: {ext}"}

        # 选择解析策略
        if doc["images"] > 0:
            strategy = "视觉理解 (Vision LLM)"
            token_cost = doc["pages"] * 800
        else:
            strategy = "纯文本提取"
            token_cost = len(doc["text"]) * 2

        return {
            "filename": filename,
            "strategy": strategy,
            "title": doc["title"],
            "pages": doc["pages"],
            "text_preview": doc["text"][:80],
            "tables": len(doc["tables"]),
            "images": doc["images"],
            "estimated_tokens": token_cost,
        }


# ═══════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════

def demo_voice_pipeline():
    """演示真实 WAV 生成 + 语音管道。"""
    print(f"\n\n▶ 7. 语音管道实战（生成 WAV + STT→LLM→TTS）")
    print("─" * 60)

    sim = VoicePipelineSimulator()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_wav = os.path.join(tmpdir, "input.wav")
        output_wav = os.path.join(tmpdir, "output.wav")

        # 生成真实 WAV
        sim.generate_wav(input_wav, duration_s=2.0, freq=440)
        print(f"  生成输入 WAV: {os.path.getsize(input_wav)} bytes (2s, 440Hz)")

        # 运行管道
        result = sim.run_pipeline(input_wav, output_wav)
        print(f"  STT 识别: \"{result['transcript']}\"")
        print(f"  LLM 回复: \"{result['response']}\"")
        print(f"  TTS 输出: {result['output_size']} bytes")


def demo_image_encoder():
    """演示图片编码和 token 估算。"""
    print(f"\n\n▶ 8. 图片编码 + Token 估算")
    print("─" * 60)

    enc = ImageEncoder()

    # 创建一个小 PNG（1x1 像素红点）
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        # 最小合法 PNG
        png_data = (
            b"\x89PNG\r\n\x1a\n"  # signature
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
            b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
            b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        f.write(png_data)
        tmp_path = f.name

    b64 = enc.encode_file(tmp_path)
    msg = enc.make_vision_message("这是什么图片？", b64, detail="high")
    os.unlink(tmp_path)

    print(f"  Base64 长度: {len(b64)} chars")
    print(f"  消息结构: role={msg['role']}, content types={[c['type'] for c in msg['content']]}")

    # Token 估算
    for w, h in [(512, 512), (1920, 1080), (4096, 3072)]:
        tokens_low = enc.estimate_tokens(w, h, "low")
        tokens_high = enc.estimate_tokens(w, h, "high")
        print(f"  {w}x{h}: low={tokens_low} tokens, high={tokens_high} tokens")


def demo_document_parser():
    """演示文档解析。"""
    print(f"\n\n▶ 9. 文档解析器")
    print("─" * 60)

    parser = DocumentParser()
    for filename in ["report.pdf", "roadmap.pptx", "sales.xlsx", "notes.md"]:
        result = parser.parse(filename)
        if "error" in result:
            print(f"  ❌ {filename}: {result['error']}")
        else:
            print(f"  📄 {filename} → {result['strategy']}")
            print(f"     标题: {result['title']}, {result['pages']} 页, "
                  f"{result['tables']} 表格, {result['images']} 图片")
            print(f"     预览: {result['text_preview']}")
            print(f"     Token 估算: {result['estimated_tokens']}")


def main():
    print("=== 多模态 Agent ===\n")

    show_vision()
    show_browser_agent()
    show_voice_agent()
    show_video_understanding()
    show_document_parsing()
    show_model_comparison()

    demo_voice_pipeline()
    demo_image_encoder()
    demo_document_parser()

    # ── 架构总结 ──────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 多模态 Agent 总结:")
    print()
    print("  模态         │ 技术方案       │ 主要模型")
    print("  ─────────────┼───────────────┼──────────────")
    print("  图片理解      │ Vision API     │ GPT-4o / Gemini")
    print("  浏览器操控    │ 截图→LLM→点击  │ GPT-4o + Playwright")
    print("  语音对话      │ STT+LLM+TTS   │ Whisper + GPT + TTS")
    print("  实时语音      │ Realtime API   │ GPT-4o / Gemini Live")
    print("  视频理解      │ 关键帧 / 原生  │ GPT-4o / Gemini")
    print("  文档解析      │ 专用工具 + LLM │ Unstructured / Docling")
    print()
    print("  关键决策:")
    print("  ────────────────────────────────────────────")
    print("  □ 需要视频原生理解 → Gemini（唯一支持）")
    print("  □ 需要实时语音 → GPT-4o Realtime API")
    print("  □ 需要 Browser Agent → GPT-4o + Browser Use")
    print("  □ 需要文档解析 → Unstructured + RAG")
    print("  □ 需要屏幕操控 → Claude Computer Use")


if __name__ == "__main__":
    main()
