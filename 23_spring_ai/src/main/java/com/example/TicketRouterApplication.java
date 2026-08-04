package com.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Spring AI 智能工单路由服务
 *
 * 最佳场景：Java 企业级 AI 集成 — 与 Spring Boot 生态无缝融合。
 *
 * 核心模式：
 *   - ChatClient：流式/同步调用 LLM（类似 RestClient 的链式 API）
 *   - @Tool 注解：方法即工具（Spring 自动扫描、注入）
 *   - Advisor 链：请求/响应拦截器（日志、RAG、安全过滤）
 *   - 结构化输出：BeanOutputConverter 自动映射到 Java Record
 *
 * 为什么 Java 企业选 Spring AI：
 *   - Spring Boot 原生（自动配置、Actuator 监控、Profile 切环境）
 *   - @Tool 注解让现有 Service Bean 直接成为 Agent 工具
 *   - Advisor 链 = AOP 思想在 AI 领域的延伸
 *   - 与 Spring Security / Spring Data / Spring Batch 无缝集成
 */
@SpringBootApplication
public class TicketRouterApplication {
    public static void main(String[] args) {
        SpringApplication.run(TicketRouterApplication.class, args);
    }
}
