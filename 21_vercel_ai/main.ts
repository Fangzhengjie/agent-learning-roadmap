/**
 * Vercel AI SDK — 流式对话 + 工具调用
 *
 * 最佳场景：全栈 AI 应用（Next.js / Node.js）— 流式渲染 + 服务端工具调用。
 *
 * 核心模式：
 *   - generateText / streamText：同步/流式文本生成
 *   - tool()：Zod schema 定义工具参数（类型安全）
 *   - maxSteps：自动工具调用循环（调用工具 → 喂回结果 → 再调用）
 *   - AI RSC（React Server Components）：在 Next.js 中流式渲染 AI 组件
 *
 * 为什么全栈 AI 选 Vercel AI SDK：
 *   - Next.js 原生集成（useChat / useCompletion hooks）
 *   - 流式渲染性能最优（Edge Runtime + streaming）
 *   - 统一的 provider 接口（OpenAI / Anthropic / Google 等）
 *   - Zod schema = 工具定义 + 运行时验证 + TypeScript 类型，三合一
 */

import { generateText, tool } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";

// ── 模拟业务数据 ──────────────────────────────────────────

const PRODUCTS: Record<string, { name: string; price: number; stock: number }> = {
  "SKU-001": { name: "机械键盘 K8 Pro", price: 899, stock: 42 },
  "SKU-002": { name: '4K 显示器 27"', price: 2499, stock: 15 },
  "SKU-003": { name: "降噪耳机 WH-1000", price: 1999, stock: 0 },
  "SKU-004": { name: "Type-C 扩展坞", price: 399, stock: 128 },
};

const ORDERS: Record<string, { user: string; items: string[]; status: string; total: number }> = {
  "ORD-101": { user: "alice", items: ["SKU-001", "SKU-004"], status: "shipped", total: 1298 },
  "ORD-102": { user: "bob", items: ["SKU-002"], status: "processing", total: 2499 },
  "ORD-103": { user: "charlie", items: ["SKU-003"], status: "cancelled", total: 1999 },
};

// ── 工具定义（Zod schema = 参数类型 + 验证 + 文档） ─────────

const searchProducts = tool({
  description: "搜索商品信息，支持按名称或SKU查询",
  parameters: z.object({
    query: z.string().describe("搜索关键词或SKU编号"),
  }),
  execute: async ({ query }) => {
    const results = Object.entries(PRODUCTS)
      .filter(
        ([sku, p]) =>
          sku.toLowerCase().includes(query.toLowerCase()) ||
          p.name.toLowerCase().includes(query.toLowerCase())
      )
      .map(([sku, p]) => ({
        sku,
        ...p,
        availability: p.stock > 0 ? `有货 (${p.stock}件)` : "缺货",
      }));
    return results.length > 0 ? results : [{ message: `未找到匹配 "${query}" 的商品` }];
  },
});

const lookupOrder = tool({
  description: "查询订单状态",
  parameters: z.object({
    orderId: z.string().describe("订单编号，如 ORD-101"),
  }),
  execute: async ({ orderId }) => {
    const order = ORDERS[orderId];
    if (!order) return { error: `订单 ${orderId} 不存在` };
    const statusMap: Record<string, string> = {
      processing: "处理中",
      shipped: "已发货",
      cancelled: "已取消",
    };
    return { orderId, ...order, statusText: statusMap[order.status] || order.status };
  },
});

const checkStock = tool({
  description: "批量检查商品库存",
  parameters: z.object({
    skus: z.array(z.string()).describe("SKU编号列表"),
  }),
  execute: async ({ skus }) => {
    return skus.map((sku) => {
      const p = PRODUCTS[sku];
      if (!p) return { sku, error: "商品不存在" };
      return { sku, name: p.name, stock: p.stock, available: p.stock > 0 };
    });
  },
});

// ── 主流程 ────────────────────────────────────────────────

async function main() {
  const modelName = process.env.MODEL_NAME || "gpt-4o-mini";
  console.log("=== Vercel AI SDK 电商助手 ===");
  console.log(`模型: ${modelName}\n`);

  const queries = [
    "帮我查一下有没有键盘卖？价格多少？",
    "我的订单 ORD-101 到哪了？",
    "SKU-003 还有货吗？如果没有，推荐类似的产品",
  ];

  for (const query of queries) {
    console.log("─".repeat(60));
    console.log(`👤 用户: ${query}`);
    console.log("─".repeat(60));

    // generateText: 同步生成（生产中 Next.js 场景用 streamText 流式渲染）
    // maxSteps: 允许最多 5 轮工具调用循环
    const result = await generateText({
      model: openai(modelName),
      system:
        "你是电商客服助手。帮用户查询商品、订单和库存。用中文回复，简洁专业。",
      prompt: query,
      tools: { searchProducts, lookupOrder, checkStock },
      maxSteps: 5,
    });

    // 展示工具调用过程
    for (const step of result.steps) {
      for (const tc of step.toolCalls) {
        console.log(`  🔧 工具: ${tc.toolName}(${JSON.stringify(tc.args)})`);
      }
      for (const tr of step.toolResults) {
        const preview = JSON.stringify(tr.result).slice(0, 80);
        console.log(`  ✅ 结果: ${preview}...`);
      }
    }

    console.log(`\n🤖 回复: ${result.text}\n`);
  }

  // ── 架构观察 ──────────────────────────────────────────
  console.log("=".repeat(60));
  console.log("📊 Vercel AI SDK 架构观察:");
  console.log();
  console.log("  ✅ 最佳场景: Next.js 全栈 AI 应用（流式渲染 + RSC）");
  console.log("  ✅ Zod schema = 参数定义 + 运行时验证 + TS 类型，三合一");
  console.log("  ✅ maxSteps 自动工具调用循环（无需手动循环）");
  console.log("  ✅ 统一 provider 接口（openai/anthropic/google 一行切换）");
  console.log("  ✅ streamText + useChat 实现打字机效果");
  console.log("  ⚠️  纯库，无 Agent 编排能力（无 Handoff / GroupChat）");
  console.log("  ⚠️  无内置记忆/历史管理（需自己维护）");
  console.log("  ❌ 无取消/重试/安全模型");
}

main().catch(console.error);
