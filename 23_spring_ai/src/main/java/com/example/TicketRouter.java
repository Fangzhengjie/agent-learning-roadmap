package com.example;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.web.bind.annotation.*;

/**
 * 智能工单路由 Controller
 *
 * 展示 Spring AI 的核心 Agent 模式：
 *   1. ChatClient.builder() — 链式构建 AI 调用（类似 WebClient）
 *   2. .defaultTools(ticketTools) — 注入 @Tool Bean 作为工具
 *   3. .defaultSystem(...) — 设置系统 prompt
 *   4. .defaultAdvisors(...) — Advisor 链（请求/响应拦截）
 *
 * 调用示例：
 *   POST http://localhost:8080/ticket/route
 *   Body: {"message": "工单 T-001，用户反映登录白屏"}
 */
@RestController
@RequestMapping("/ticket")
public class TicketRouter {

    private final ChatClient chatClient;

    /**
     * 通过构造器注入 ChatClient.Builder（Spring AI 自动配置提供）
     * 和 TicketTools（Spring 自动扫描的 @Tool Bean）
     */
    public TicketRouter(ChatClient.Builder chatClientBuilder, TicketTools ticketTools) {
        this.chatClient = chatClientBuilder
            .defaultSystem("""
                你是智能工单路由系统。根据用户描述的问题：
                1. 用 lookupTicket 查看工单详情
                2. 根据工单类型判断应该路由到哪个团队：
                   - 技术问题（白屏、报错、连接）→ engineering
                   - 计费/退款问题 → billing
                   - 一般咨询 → support
                3. 如果是技术问题，先用 checkSystemStatus 检查相关服务状态
                4. 如果是退款问题，用 processRefund 处理退款
                5. 最后用 routeTicket 将工单路由到目标团队
                用中文回复，给出分析过程和最终路由结果。
                """)
            .defaultTools(ticketTools)
            .build();
    }

    /**
     * 工单路由接口
     *
     * Spring AI 的 ChatClient.prompt().user().call() 链路：
     * - 自动调用 LLM
     * - LLM 决定调用哪些 @Tool 方法
     * - Spring AI 自动执行工具、回传结果
     * - 循环直到 LLM 给出最终回复
     */
    @PostMapping("/route")
    public TicketResponse routeTicket(@RequestBody TicketRequest request) {
        String response = chatClient.prompt()
            .user(request.message())
            .call()
            .content();

        return new TicketResponse(response);
    }

    /**
     * 流式响应版本（SSE）
     * 适合前端实时显示处理进度
     */
    @PostMapping("/route/stream")
    public org.reactor.core.publisher.Flux<String> routeTicketStream(
            @RequestBody TicketRequest request) {
        return chatClient.prompt()
            .user(request.message())
            .stream()
            .content();
    }

    record TicketRequest(String message) {}
    record TicketResponse(String result) {}
}
