/**
 * Semantic Kernel — 企业 Copilot 示例
 *
 * 最佳场景：.NET 企业 Copilot — 插件编排 + 规划器 + Azure OpenAI 原生。
 *
 * 核心模式：
 *   - Kernel：核心容器（管理插件、AI 服务、内存）
 *   - KernelPlugin：一组相关函数（[KernelFunction] 注解）
 *   - Planner：自动规划工具调用顺序（Handlebars / Stepwise）
 *   - ChatCompletionAgent：对话式 Agent
 *   - AgentGroupChat：多 Agent 协作
 *
 * 为什么 .NET 企业选 Semantic Kernel：
 *   - 微软官方出品，Azure OpenAI 原生支持
 *   - 与 Microsoft 365 Copilot 同一架构
 *   - .NET / Java 双语言支持
 *   - 插件系统可复用 OpenAI Plugin 标准
 */

using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using System.ComponentModel;

// ── 插件定义 ──────────────────────────────────────────────
// Semantic Kernel 的 Plugin = 一组 [KernelFunction] 方法
// 类似 Spring AI 的 @Tool，但支持 OpenAI Plugin 标准

public class TicketPlugin
{
    private static readonly Dictionary<string, Dictionary<string, object>> Tickets = new()
    {
        ["T-001"] = new() { ["user"] = "alice", ["type"] = "technical", ["issue"] = "登录后页面白屏", ["priority"] = "high" },
        ["T-002"] = new() { ["user"] = "bob", ["type"] = "billing", ["issue"] = "订阅扣费但功能无法使用", ["amount"] = 99.0 },
        ["T-003"] = new() { ["user"] = "charlie", ["type"] = "general", ["issue"] = "如何升级到企业版", ["priority"] = "low" },
    };

    [KernelFunction, Description("查询工单详情")]
    public string LookupTicket(
        [Description("工单编号，如 T-001")] string ticketId)
    {
        if (!Tickets.TryGetValue(ticketId, out var ticket))
            return $"工单 {ticketId} 不存在";

        var info = string.Join(", ", ticket.Select(kv => $"{kv.Key}={kv.Value}"));
        return $"工单 {ticketId}: {info}";
    }

    [KernelFunction, Description("将工单路由到指定团队")]
    public string RouteTicket(
        [Description("工单编号")] string ticketId,
        [Description("目标团队: engineering / billing / support")] string team,
        [Description("路由原因")] string reason)
    {
        return $"工单 {ticketId} 已路由到 {team} 团队。原因: {reason}";
    }

    [KernelFunction, Description("处理退款")]
    public string ProcessRefund(
        [Description("工单编号")] string ticketId,
        [Description("退款金额")] double amount,
        [Description("退款原因")] string reason)
    {
        return $"退款 ¥{amount} 已处理 (工单 {ticketId})。原因: {reason}。预计 3-5 个工作日到账。";
    }
}

// ── 主程序 ────────────────────────────────────────────────

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("=== Semantic Kernel 企业 Copilot ===\n");

        var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")
            ?? throw new Exception("请设置 OPENAI_API_KEY 环境变量");
        var modelName = Environment.GetEnvironmentVariable("MODEL_NAME") ?? "gpt-4o-mini";

        // ── 构建 Kernel ──────────────────────────────────
        // Kernel = 核心容器，管理 AI 服务 + 插件
        var builder = Kernel.CreateBuilder();
        builder.AddOpenAIChatCompletion(modelName, apiKey);
        builder.Plugins.AddFromType<TicketPlugin>();  // 注册插件

        var kernel = builder.Build();

        // ── 自动工具调用（FunctionCallingBehavior）────────
        // 类似 OpenAI Agents SDK 的 Runner.run()
        var settings = new OpenAIPromptExecutionSettings
        {
            FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
        };

        var chatService = kernel.GetRequiredService<IChatCompletionService>();

        // ── 模拟客服对话 ─────────────────────────────────
        var conversations = new[]
        {
            "工单 T-001，用户反映登录后白屏，帮我分析一下并路由到合适的团队",
            "工单 T-002，用户要求退款，99 元订阅费，功能无法使用",
            "工单 T-003，用户想了解企业版，请路由到合适团队",
        };

        foreach (var message in conversations)
        {
            Console.WriteLine(new string('─', 60));
            Console.WriteLine($"👤 用户: {message}");
            Console.WriteLine(new string('─', 60));

            var history = new ChatHistory();
            history.AddSystemMessage(
                "你是企业 Copilot 客服助手。分析工单内容，使用工具查询详情、路由工单或处理退款。用中文回复。");
            history.AddUserMessage(message);

            var result = await chatService.GetChatMessageContentAsync(
                history, settings, kernel);

            Console.WriteLine($"\n🤖 Copilot: {result.Content}\n");
        }

        // ── 架构观察 ─────────────────────────────────────
        Console.WriteLine(new string('=', 60));
        Console.WriteLine("📊 Semantic Kernel 架构观察:");
        Console.WriteLine();
        Console.WriteLine("  ✅ 最佳场景: .NET 企业 Copilot / Azure OpenAI 集成");
        Console.WriteLine("  ✅ 微软官方出品，与 M365 Copilot 同架构");
        Console.WriteLine("  ✅ [KernelFunction] 注解 = 方法即工具");
        Console.WriteLine("  ✅ 插件系统兼容 OpenAI Plugin 标准");
        Console.WriteLine("  ✅ AgentGroupChat 支持多 Agent 协作");
        Console.WriteLine("  ✅ Planner 可自动规划工具调用顺序");
        Console.WriteLine("  ⚠️  .NET 生态的 AI 社区不如 Python 活跃");
        Console.WriteLine("  ⚠️  Planner 在复杂场景下不够稳定");
        Console.WriteLine("  ❌ 无内置 human-in-the-loop / checkpoint");
    }
}
