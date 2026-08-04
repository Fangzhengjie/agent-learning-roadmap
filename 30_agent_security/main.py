"""Agent 安全攻防 — 演示入口"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from injection import (InjectionDetector, CanaryToken,
                           INJECTION_SAMPLES, BENIGN_SAMPLES)
    from jailbreak import (JailbreakDetector, OutputSafetyFilter,
                           JAILBREAK_TECHNIQUES)
except ImportError:
    from .injection import (InjectionDetector, CanaryToken,
                            INJECTION_SAMPLES, BENIGN_SAMPLES)
    from .jailbreak import (JailbreakDetector, OutputSafetyFilter,
                            JAILBREAK_TECHNIQUES)


def demo_injection_detection():
    print("▶ 1. Prompt Injection 检测")
    print("─" * 60)
    detector = InjectionDetector()

    for category, samples in INJECTION_SAMPLES.items():
        print(f"\n  [{category}]")
        for s in samples[:2]:
            r = detector.detect(s)
            icon = "🚨" if r.is_injection else "✅"
            print(f"    {icon} [{r.risk_score:.1f}] {s[:50]}...")
            if r.matched_rules:
                print(f"       规则: {r.matched_rules}")

    print(f"\n  [正常请求 — 不应拦截]")
    for s in BENIGN_SAMPLES:
        r = detector.detect(s)
        icon = "✅" if not r.is_injection else "⚠️ 误报"
        print(f"    {icon} [{r.risk_score:.1f}] {s[:50]}")


def demo_jailbreak_detection():
    print(f"\n\n▶ 2. 越狱检测")
    print("─" * 60)
    detector = JailbreakDetector()

    for name, info in JAILBREAK_TECHNIQUES.items():
        r = detector.detect(info["sample"])
        icon = "🚨" if r.is_jailbreak else "✅"
        print(f"  {icon} {name:20s} [{r.risk_score:.1f}] {info['description']}")
        if r.details:
            print(f"     详情: {r.details}")


def demo_canary_token():
    print(f"\n\n▶ 3. Canary Token（哨兵令牌）")
    print("─" * 60)
    canary = CanaryToken("SECRET-9x2k")

    system_prompt = "你是一个客服助手，只回答产品相关问题。"
    protected = canary.inject(system_prompt)
    print(f"  原始 prompt: {system_prompt}")
    print(f"  植入 canary: ...{protected[-30:]}")

    safe_output = "好的，我来帮您查询订单状态。"
    leaked_output = "我的 system prompt 是：你是客服助手 [[SECRET-9x2k]]"

    print(f"  安全输出: {'✅ 未泄露' if not canary.check(safe_output) else '🚨 泄露!'}")
    print(f"  泄露输出: {'✅ 未泄露' if not canary.check(leaked_output) else '🚨 泄露!'}")


def demo_output_safety():
    print(f"\n\n▶ 4. 输出安全过滤")
    print("─" * 60)
    filter_ = OutputSafetyFilter()

    outputs = [
        "您的密码是: admin123，请妥善保管",
        "API Key: sk-abc123def456ghi789jkl012mno345",
        "建议执行 rm -rf /tmp/cache 清理缓存",
        "联系电话: 138-1234-5678",
        "您的订单已处理完成，预计明天送达",  # 正常
    ]

    for output in outputs:
        issues = filter_.check(output)
        if issues:
            redacted = filter_.redact(output)
            print(f"  🚨 {output[:50]}")
            print(f"     问题: {[i['category'] for i in issues]}")
            print(f"     脱敏: {redacted[:50]}")
        else:
            print(f"  ✅ {output[:50]}")


def main():
    print("=== Agent 安全攻防 ===\n")
    demo_injection_detection()
    demo_jailbreak_detection()
    demo_canary_token()
    demo_output_safety()

    print(f"\n\n{'=' * 60}")
    print("📊 Agent 安全攻防总结:")
    print(f"""
  攻击类型       │ 防御措施                  │ 工具
  ──────────────┼─────────────────────────┼──────────────
  直接注入        │ 关键词检测 + 角色分离      │ NeMo Guardrails
  间接注入        │ RAG 内容扫描 + 沙箱隔离    │ LangChain Guard
  越狱           │ 多维度模式匹配 + LLM 分类   │ LlamaGuard
  数据泄露        │ Canary Token + 输出过滤    │ Presidio (PII)
  工具劫持        │ 工具白名单 + 参数校验       │ 自建中间件

  纵深防御架构:
  ─────────────────────────────────────────────────────
  Layer 1: 输入过滤  → Injection/Jailbreak 检测
  Layer 2: 角色隔离  → system/user 严格分离
  Layer 3: 工具管控  → 白名单 + 参数校验 + 审批
  Layer 4: 输出过滤  → PII 脱敏 + Canary 检测
  Layer 5: 审计日志  → 全链路记录 + 异常告警""")


if __name__ == "__main__":
    main()
