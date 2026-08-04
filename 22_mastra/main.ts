/**
 * Mastra — GitHub Issue → PR 自动化工作流
 *
 * 最佳场景：TypeScript 原生 Agent 工作流 — 多步骤编排 + 工具集成。
 *
 * 核心模式：
 *   - Agent：带 instructions + tools 的 AI 实体
 *   - Workflow：多步骤编排引擎（Step → 条件分支 → 并行）
 *   - Step：工作流中的原子执行单元
 *   - createTool：Zod schema 工具定义
 *
 * 为什么自动化工作流选 Mastra：
 *   - TypeScript 原生（vs Python 框架的 TS binding）
 *   - Workflow engine 内置（vs Vercel AI SDK 只有工具循环）
 *   - 步骤间类型安全的数据传递
 *   - 集成 Vercel AI SDK 作为 LLM 层
 */

import { Agent, Mastra, Step, Workflow, createTool } from "@mastra/core";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";

// ── 模拟 GitHub API ──────────────────────────────────────

const ISSUES: Record<string, { title: string; body: string; labels: string[]; author: string }> = {
  "42": {
    title: "登录页面在 Safari 上白屏",
    body: "复现步骤：1. 打开 Safari 16.x 2. 访问 /login 3. 页面空白\n控制台报错：TypeError: CSS.supports is not a function",
    labels: ["bug", "frontend"],
    author: "alice",
  },
  "43": {
    title: "添加用户导出 CSV 功能",
    body: "产品需求：管理员可以在用户管理页面导出用户列表为 CSV 文件，包含 name/email/created_at 字段。",
    labels: ["feature", "backend"],
    author: "bob",
  },
};

// ── Mastra 工具定义 ──────────────────────────────────────

const fetchIssue = createTool({
  id: "fetch-issue",
  description: "获取 GitHub Issue 详情",
  inputSchema: z.object({
    issueNumber: z.string().describe("Issue 编号"),
  }),
  outputSchema: z.object({
    issueNumber: z.string(),
    title: z.string(),
    body: z.string(),
    labels: z.array(z.string()),
    author: z.string(),
  }),
  execute: async ({ context }) => {
    const issue = ISSUES[context.issueNumber];
    if (!issue) throw new Error(`Issue #${context.issueNumber} not found`);
    console.log(`  📋 获取 Issue #${context.issueNumber}: ${issue.title}`);
    return { issueNumber: context.issueNumber, ...issue };
  },
});

const analyzeIssue = createTool({
  id: "analyze-issue",
  description: "分析 Issue 类型和影响范围",
  inputSchema: z.object({
    title: z.string(),
    body: z.string(),
    labels: z.array(z.string()),
  }),
  outputSchema: z.object({
    type: z.enum(["bug", "feature", "refactor"]),
    priority: z.enum(["low", "medium", "high", "critical"]),
    affectedFiles: z.array(z.string()),
    estimatedHours: z.number(),
  }),
  execute: async ({ context }) => {
    const isBug = context.labels.includes("bug");
    const isFrontend = context.labels.includes("frontend");

    console.log(`  🔍 分析 Issue: type=${isBug ? "bug" : "feature"}`);

    return {
      type: isBug ? "bug" as const : "feature" as const,
      priority: isBug ? "high" as const : "medium" as const,
      affectedFiles: isFrontend
        ? ["src/pages/login.tsx", "src/styles/auth.css"]
        : ["src/api/users.ts", "src/utils/csv.ts"],
      estimatedHours: isBug ? 2 : 8,
    };
  },
});

const generatePRDescription = createTool({
  id: "generate-pr-description",
  description: "生成 PR 描述",
  inputSchema: z.object({
    issueNumber: z.string(),
    title: z.string(),
    type: z.string(),
    affectedFiles: z.array(z.string()),
  }),
  outputSchema: z.object({
    prTitle: z.string(),
    prBody: z.string(),
    branch: z.string(),
  }),
  execute: async ({ context }) => {
    const prefix = context.type === "bug" ? "fix" : "feat";
    const slug = context.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 30);
    const branch = `${prefix}/issue-${context.issueNumber}-${slug}`;

    console.log(`  🌿 生成分支: ${branch}`);

    return {
      prTitle: `${prefix}: ${context.title} (closes #${context.issueNumber})`,
      prBody: [
        `## 关联 Issue`,
        `Closes #${context.issueNumber}`,
        ``,
        `## 变更说明`,
        `- 类型: ${context.type}`,
        `- 影响文件:`,
        ...context.affectedFiles.map((f: string) => `  - \`${f}\``),
        ``,
        `## 测试清单`,
        `- [ ] 单元测试通过`,
        `- [ ] E2E 测试通过`,
        `- [ ] Code Review`,
      ].join("\n"),
      branch,
    };
  },
});

// ── Mastra Agent ─────────────────────────────────────────

const modelName = process.env.MODEL_NAME || "gpt-4o-mini";

const issueAgent = new Agent({
  name: "Issue 分析助手",
  instructions:
    "你是 GitHub Issue 分析助手。分析 Issue 内容，判断类型和优先级，" +
    "识别可能受影响的文件，然后生成规范的 PR 描述。用中文回复。",
  model: openai(modelName),
  tools: { fetchIssue, analyzeIssue, generatePRDescription },
});

// ── Mastra Workflow（多步骤编排）──────────────────────────

const issueWorkflow = new Workflow({
  name: "issue-to-pr",
  triggerSchema: z.object({
    issueNumber: z.string(),
  }),
});

// Step 1: 获取 Issue
const fetchStep = new Step({
  id: "fetch",
  execute: async ({ context }) => {
    const issueNumber = context.triggerData.issueNumber;
    const issue = ISSUES[issueNumber];
    if (!issue) throw new Error(`Issue #${issueNumber} not found`);
    console.log(`\n  📋 Step 1 [fetch]: Issue #${issueNumber} — ${issue.title}`);
    return { issueNumber, ...issue };
  },
});

// Step 2: 分析 Issue
const analyzeStep = new Step({
  id: "analyze",
  execute: async ({ context }) => {
    const prev = context.getStepResult<{
      labels: string[];
      title: string;
      body: string;
    }>("fetch");
    if (!prev) throw new Error("Missing fetch result");

    const isBug = prev.labels.includes("bug");
    const priority = isBug ? "high" : "medium";
    const affectedFiles = prev.labels.includes("frontend")
      ? ["src/pages/login.tsx", "src/styles/auth.css"]
      : ["src/api/users.ts", "src/utils/csv.ts"];

    console.log(`  🔍 Step 2 [analyze]: type=${isBug ? "bug" : "feature"}, priority=${priority}`);
    return { type: isBug ? "bug" : "feature", priority, affectedFiles, estimatedHours: isBug ? 2 : 8 };
  },
});

// Step 3: 生成 PR
const generateStep = new Step({
  id: "generate-pr",
  execute: async ({ context }) => {
    const fetchResult = context.getStepResult<{ issueNumber: string; title: string }>("fetch");
    const analyzeResult = context.getStepResult<{ type: string; affectedFiles: string[] }>("analyze");
    if (!fetchResult || !analyzeResult) throw new Error("Missing previous step results");

    const prefix = analyzeResult.type === "bug" ? "fix" : "feat";
    const slug = fetchResult.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 30);
    const branch = `${prefix}/issue-${fetchResult.issueNumber}-${slug}`;

    console.log(`  🌿 Step 3 [generate-pr]: branch=${branch}`);
    return {
      prTitle: `${prefix}: ${fetchResult.title} (closes #${fetchResult.issueNumber})`,
      branch,
      filesChanged: analyzeResult.affectedFiles,
    };
  },
});

// 编排步骤
issueWorkflow.step(fetchStep).then(analyzeStep).then(generateStep).commit();

// ── 执行 ──────────────────────────────────────────────────

const mastra = new Mastra({
  agents: { issueAgent },
  workflows: { "issue-to-pr": issueWorkflow },
});

async function main() {
  console.log("=== Mastra GitHub Issue → PR 自动化 ===");
  console.log(`模型: ${modelName}\n`);

  // ── 方案 1: Workflow（确定性多步编排）──────────────────
  console.log("▶ 方案 1: Workflow（确定性步骤编排）");
  console.log("  流程: fetch → analyze → generate-pr\n");

  for (const issueNum of ["42", "43"]) {
    console.log("─".repeat(60));
    const wf = mastra.getWorkflow("issue-to-pr");
    const run = wf.createRun();
    const result = await run.start({ triggerData: { issueNumber: issueNum } });

    const prResult = result.results?.["generate-pr"];
    if (prResult && "output" in prResult) {
      const output = prResult.output as Record<string, unknown>;
      console.log(`\n  📝 生成 PR:`);
      console.log(`     标题: ${output.prTitle}`);
      console.log(`     分支: ${output.branch}`);
    }
    console.log();
  }

  // ── 方案 2: Agent（LLM 自主决策）──────────────────────
  console.log("─".repeat(60));
  console.log("▶ 方案 2: Agent（LLM 自主决策工具调用顺序）\n");

  const agent = mastra.getAgent("issueAgent");
  const response = await agent.generate(
    "分析 Issue #42，判断优先级，然后生成 PR 描述"
  );

  console.log(`\n🤖 Agent 回复:\n${response.text?.slice(0, 400)}`);

  // ── 架构观察 ──────────────────────────────────────────
  console.log("\n" + "=".repeat(60));
  console.log("📊 Mastra 架构观察:");
  console.log();
  console.log("  ✅ 最佳场景: TypeScript Agent 工作流（步骤编排 + 工具集成）");
  console.log("  ✅ Workflow 引擎内置（step → then → commit）");
  console.log("  ✅ 步骤间类型安全数据传递（getStepResult<T>）");
  console.log("  ✅ Agent 和 Workflow 可混合使用");
  console.log("  ✅ TypeScript 原生（非 Python 框架的 TS binding）");
  console.log("  ⚠️  生态尚早期（vs LangChain 的 200+ 集成）");
  console.log("  ⚠️  多 Agent 协作能力有限");
  console.log("  ❌ 无取消/重试/安全模型");
}

main().catch(console.error);
