package com.example;

import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * 工单相关工具 — 用 @Tool 注解将普通 Spring Bean 方法暴露给 LLM。
 *
 * Spring AI 的 @Tool 注解等价于 Python 框架的 @function_tool，
 * 但天然支持 Spring 的依赖注入（可以注入 Repository、Service 等）。
 */
@Component
public class TicketTools {

    // 模拟工单数据
    private static final Map<String, Map<String, Object>> TICKETS = Map.of(
        "T-001", Map.of("user", "alice", "type", "technical",
                        "issue", "登录后页面白屏", "status", "open", "priority", "high"),
        "T-002", Map.of("user", "bob", "type", "billing",
                        "issue", "订阅扣费但功能无法使用", "status", "open",
                        "priority", "high", "amount", 99.0),
        "T-003", Map.of("user", "charlie", "type", "general",
                        "issue", "如何升级到企业版", "status", "open", "priority", "low")
    );

    private static final Map<String, String> SYSTEM_STATUS = Map.of(
        "auth", "healthy",
        "payment", "degraded",
        "api", "healthy",
        "web", "healthy"
    );

    @Tool(description = "查询工单详情，返回工单类型、状态、优先级等信息")
    public Map<String, Object> lookupTicket(
            @ToolParam(description = "工单编号，如 T-001") String ticketId) {
        var ticket = TICKETS.get(ticketId);
        if (ticket == null) {
            return Map.of("error", "工单 " + ticketId + " 不存在");
        }
        return Map.of("ticketId", ticketId, "data", ticket);
    }

    @Tool(description = "检查后端服务状态（auth/payment/api/web）")
    public Map<String, String> checkSystemStatus(
            @ToolParam(description = "服务名称") String service) {
        String status = SYSTEM_STATUS.getOrDefault(service, "unknown");
        return Map.of("service", service, "status", status);
    }

    @Tool(description = "将工单路由到指定团队处理")
    public Map<String, String> routeTicket(
            @ToolParam(description = "工单编号") String ticketId,
            @ToolParam(description = "目标团队: engineering / billing / support") String targetTeam,
            @ToolParam(description = "路由原因") String reason) {
        return Map.of(
            "ticketId", ticketId,
            "routedTo", targetTeam,
            "reason", reason,
            "status", "routed"
        );
    }

    @Tool(description = "处理退款申请")
    public Map<String, Object> processRefund(
            @ToolParam(description = "工单编号") String ticketId,
            @ToolParam(description = "退款金额") double amount,
            @ToolParam(description = "退款原因") String reason) {
        return Map.of(
            "ticketId", ticketId,
            "refundAmount", amount,
            "reason", reason,
            "status", "processed",
            "eta", "3-5 个工作日"
        );
    }
}
