"""流式处理模式 — 演示入口"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from stream_parser import (StreamSimulator, StreamParser, StreamGuardrail,
                               BackpressureController, StreamEvent)
except ImportError:
    from .stream_parser import (StreamSimulator, StreamParser, StreamGuardrail,
                                BackpressureController, StreamEvent)


def demo_stream_parsing():
    print("▶ 1. 流式 Token 解析 + JSON 检测")
    print("─" * 60)

    text = '分析结果如下：{"category": "技术问题", "priority": "high", "action": "转接技术支持"}。请稍等，正在处理。'
    sim = StreamSimulator(text)
    parser = StreamParser()

    print(f"  原文: {text[:60]}...")
    print(f"  流式解析:")

    collected_text = ""
    for token in sim.stream():
        events = parser.feed(token)
        for e in events:
            if e.type == "json_start":
                print(f"    📌 检测到 JSON 开始")
            elif e.type == "json_complete":
                print(f"    ✅ JSON 完成: {e.data}")
            elif e.type == "tool_call":
                print(f"    🔧 工具调用: {e.data}")
        collected_text += token

    print(f"  总事件: {len(parser.events)} (text: {sum(1 for e in parser.events if e.type == 'text')}, "
          f"json: {sum(1 for e in parser.events if e.type in ('json_start', 'json_complete'))})")


def demo_stream_tool_detection():
    print(f"\n\n▶ 2. 流式工具调用检测")
    print("─" * 60)

    # 模拟 LLM 输出含工具调用
    text = '让我查一下天气。{"name": "get_weather", "arguments": {"city": "上海"}}。上海今天晴天，32度。'
    sim = StreamSimulator(text)
    parser = StreamParser()

    print(f"  LLM 输出: {text}")
    print(f"  实时检测:")

    for token in sim.stream():
        events = parser.feed(token)
        for e in events:
            if e.type == "tool_call":
                print(f"    🔧 检测到工具调用! → {e.data['name']}({e.data.get('arguments', {})})")
                print(f"       → 暂停流式输出 → 执行工具 → 继续")


def demo_stream_guardrail():
    print(f"\n\n▶ 3. 流式安全护栏")
    print("─" * 60)

    guard = StreamGuardrail()

    # 安全输出
    safe_text = "您的订单已处理完成，预计明天送达。如有问题请联系客服。"
    print(f"  安全输出:")
    sim = StreamSimulator(safe_text)
    for token in sim.stream():
        violation = guard.check(token)
        if violation:
            print(f"    🚨 中断! {violation}")
            break
    if not guard.interrupted:
        print(f"    ✅ 安全通过")
    guard.reset()

    # 危险输出 — 泄露 API Key
    danger_text = "这是配置信息：api_key=sk-abc123def456ghi789jkl012mno345pqr 请保存好。"
    print(f"\n  危险输出:")
    sim = StreamSimulator(danger_text)
    output_so_far = ""
    for token in sim.stream():
        violation = guard.check(token)
        if violation:
            print(f"    🚨 流式中断! 位置={len(output_so_far)} 类型={violation['category']}")
            print(f"    用户看到的: '{output_so_far[:40]}...[已中断]'")
            break
        output_so_far += token
    guard.reset()

    # 危险输出 — 危险命令
    cmd_text = "要清理缓存，执行 rm -rf /tmp/cache 即可。"
    print(f"\n  命令输出:")
    sim = StreamSimulator(cmd_text)
    output_so_far = ""
    for token in sim.stream():
        violation = guard.check(token)
        if violation:
            print(f"    🚨 流式中断! 类型={violation['category']}")
            break
        output_so_far += token
    if not guard.interrupted:
        print(f"    ✅ 安全通过")


def demo_backpressure():
    print(f"\n\n▶ 4. Backpressure 背压控制")
    print("─" * 60)

    bp = BackpressureController(max_buffer=10, high_water=0.8, low_water=0.3)

    text = "这是一段很长的流式输出文本，用来测试背压控制机制。当缓冲区满了就要降速。"
    sim = StreamSimulator(text)

    print(f"  缓冲区大小: {bp.max_buffer}, 高水位: {bp.high_water}, 低水位: {bp.low_water}")
    print(f"  模拟: 快速生产 + 慢速消费\n")

    produce_count = 0
    consume_count = 0
    for i, token in enumerate(sim.stream()):
        accepted = bp.produce(token)
        produce_count += 1

        status = "⏸️ PAUSED" if bp.is_paused else "▶️ FLOWING"
        icon = "✅" if accepted else "❌"

        # 每 3 个 token 消费 1 个（模拟慢消费者）
        consumed = []
        if i % 3 == 0:
            consumed = bp.consume(2)
            consume_count += len(consumed)

        if accepted and (bp.is_paused or not accepted or consumed):
            print(f"    [{status}] 缓冲={bp.buffer_usage:.0%} "
                  f"产:{icon} 消:{''.join(consumed) if consumed else '-'}")

    # 消费剩余
    while bp.buffer_usage > 0:
        consumed = bp.consume(3)
        consume_count += len(consumed)

    print(f"\n  统计: 产出={bp.stats['total_tokens']} 暂停={bp.stats['paused_count']}次 "
          f"丢弃={bp.stats['dropped_count']}")


def main():
    print("=== 流式处理模式 ===\n")
    demo_stream_parsing()
    demo_stream_tool_detection()
    demo_stream_guardrail()
    demo_backpressure()

    print(f"\n\n{'=' * 60}")
    print("📊 流式处理模式总结:")
    print(f"""
  模式             │ 解决的问题               │ 框架支持
  ────────────────┼────────────────────────┼──────────────────
  流式 Token 解析   │ 实时显示 + 结构检测       │ Vercel AI streamText
  流式 JSON 检测    │ 流中提取工具调用          │ OpenAI stream tool_calls
  流式 Guardrail   │ 输出安全实时拦截           │ NeMo Guardrails
  Backpressure    │ 消费者慢时防溢出           │ Node.js Streams / Reactive
  流式取消          │ 用户中途取消              │ AbortController

  SSE vs WebSocket 流式对比:
  - SSE:  单向推送, 自动重连, HTTP 协议 → 适合 LLM 流式输出
  - WS:   双向通信, 手动重连, 独立协议 → 适合交互式 Agent""")


if __name__ == "__main__":
    main()
