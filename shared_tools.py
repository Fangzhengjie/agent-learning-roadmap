"""Shared mock data and utilities used across demos.

These simulate real business backends so demos run without external services.
Each demo picks the subset it needs.
"""

import json
import os
from datetime import datetime
from typing import Optional


# ── 知识库 (01_langchain RAG) ─────────────────────────────

KNOWLEDGE_BASE = [
    {
        "id": "KB-001",
        "title": "员工请假政策",
        "content": (
            "年假：入职满1年享5天，满10年享10天，满20年享15天。"
            "病假：持医院证明可请病假，前3天全薪，超出按80%发放。"
            "事假：需提前3天申请，无薪，每月上限5天。"
            "审批流程：3天以内直属主管审批，3天以上需部门总监审批。"
        ),
        "category": "HR",
    },
    {
        "id": "KB-002",
        "title": "报销制度",
        "content": (
            "交通：市内出租车/网约车单程不超过200元可报销，需提供电子发票。"
            "差旅：经济舱机票、四星以下酒店，需提前在OA系统申请出差单。"
            "餐饮：商务宴请每人不超过300元，需注明宴请对象和事由。"
            "审批流程：500元以下部门主管审批，500-5000元总监审批，5000以上VP审批。"
        ),
        "category": "Finance",
    },
    {
        "id": "KB-003",
        "title": "VPN 连接指南",
        "content": (
            "1. 下载 GlobalProtect 客户端。"
            "2. 服务器地址：vpn.company.com。"
            "3. 使用域账号登录（格式：DOMAIN\\username）。"
            "4. 首次连接需安装根证书（IT部门邮件中的附件）。"
            "常见问题：连接超时请检查防火墙设置，认证失败请重置AD密码。"
        ),
        "category": "IT",
    },
]


def search_knowledge(query: str) -> str:
    """在知识库中搜索相关文档。"""
    query_lower = query.lower()
    results = []
    for doc in KNOWLEDGE_BASE:
        score = sum(1 for kw in query_lower.split() if kw in doc["content"].lower() or kw in doc["title"].lower())
        if score > 0:
            results.append({**doc, "relevance": score})
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return json.dumps(results[:3] if results else [{"message": "未找到相关文档"}], ensure_ascii=False)


# ── 订单系统 (02_langgraph 审批流) ────────────────────────

ORDERS = {
    "ORD-2024-001": {"customer": "张三", "amount": 45000, "items": ["服务器x2", "交换机x1"], "status": "pending_review", "risk_score": 0.3},
    "ORD-2024-002": {"customer": "李四", "amount": 128000, "items": ["数据库许可证x10"], "status": "pending_review", "risk_score": 0.8},
    "ORD-2024-003": {"customer": "王五", "amount": 3500, "items": ["键盘x5", "鼠标x5"], "status": "pending_review", "risk_score": 0.1},
}


def get_order(order_id: str) -> str:
    """查询订单详情。"""
    order = ORDERS.get(order_id)
    if not order:
        return json.dumps({"error": f"订单 {order_id} 不存在"}, ensure_ascii=False)
    return json.dumps({"order_id": order_id, **order}, ensure_ascii=False)


def check_inventory(items: list) -> str:
    """检查库存。"""
    result = {item: {"available": True, "lead_time_days": 3} for item in items}
    if any("许可证" in i for i in items):
        for k in result:
            if "许可证" in k:
                result[k] = {"available": True, "lead_time_days": 0, "note": "电子交付"}
    return json.dumps(result, ensure_ascii=False)


def check_credit(customer: str) -> str:
    """检查客户信用。"""
    credits = {"张三": {"level": "A", "limit": 100000, "used": 20000}, "李四": {"level": "B", "limit": 200000, "used": 150000}, "王五": {"level": "A", "limit": 50000, "used": 5000}}
    return json.dumps(credits.get(customer, {"level": "C", "limit": 10000, "used": 0}), ensure_ascii=False)


def approve_order(order_id: str, decision: str, reason: str) -> str:
    """审批订单。"""
    if order_id in ORDERS:
        ORDERS[order_id]["status"] = decision
        return json.dumps({"order_id": order_id, "decision": decision, "reason": reason, "timestamp": datetime.now().isoformat()}, ensure_ascii=False)
    return json.dumps({"error": f"订单 {order_id} 不存在"}, ensure_ascii=False)


# ── 代码审查 (03_autogen) ─────────────────────────────────

CODE_SNIPPET = '''
def process_payment(user_id, amount, currency="USD"):
    conn = sqlite3.connect("payments.db")
    cursor = conn.cursor()
    cursor.execute(f"INSERT INTO payments VALUES ('{user_id}', {amount}, '{currency}')")
    conn.commit()
    conn.close()
    print(f"Payment processed: {amount}")
    return True
'''


def analyze_code(code: str, aspect: str) -> str:
    """分析代码的特定方面（security/performance/style）。"""
    issues = {
        "security": [
            {"severity": "CRITICAL", "line": 4, "issue": "SQL 注入漏洞", "detail": "使用 f-string 拼接 SQL，应使用参数化查询"},
            {"severity": "HIGH", "line": 1, "issue": "无输入验证", "detail": "amount 未校验类型和范围，可能导致负数支付"},
        ],
        "performance": [
            {"severity": "MEDIUM", "line": 2, "issue": "每次调用创建连接", "detail": "应使用连接池（如 sqlalchemy）"},
            {"severity": "LOW", "line": 5, "issue": "缺少事务管理", "detail": "建议使用 context manager 管理事务"},
        ],
        "style": [
            {"severity": "LOW", "line": 6, "issue": "使用 print 而非 logging", "detail": "生产代码应使用标准 logging 模块"},
            {"severity": "LOW", "line": 1, "issue": "缺少类型注解", "detail": "建议添加 -> bool 返回类型注解"},
        ],
    }
    return json.dumps(issues.get(aspect, []), ensure_ascii=False)


def suggest_fix(issue_description: str) -> str:
    """为代码问题生成修复建议。"""
    fixes = {
        "SQL 注入": 'cursor.execute("INSERT INTO payments VALUES (?, ?, ?)", (user_id, amount, currency))',
        "连接池": "engine = create_engine('sqlite:///payments.db', pool_size=5)\nwith engine.connect() as conn: ...",
        "输入验证": "if not isinstance(amount, (int, float)) or amount <= 0:\n    raise ValueError(f'Invalid amount: {amount}')",
    }
    for keyword, fix in fixes.items():
        if keyword in issue_description:
            return json.dumps({"fix": fix, "explanation": f"修复 {keyword} 问题"}, ensure_ascii=False)
    return json.dumps({"fix": "# TODO: 需要人工审查", "explanation": "未找到自动修复方案"}, ensure_ascii=False)


# ── 客服系统 (05_openai_agents) ───────────────────────────

TICKETS = {
    "T-001": {"user": "alice", "type": "technical", "issue": "登录后页面白屏", "status": "open"},
    "T-002": {"user": "bob", "type": "refund", "issue": "订阅扣费但功能无法使用", "status": "open", "amount": 99.0},
    "T-003": {"user": "charlie", "type": "general", "issue": "如何升级到企业版", "status": "open"},
}


def lookup_ticket(ticket_id: str) -> str:
    """查询工单详情。"""
    ticket = TICKETS.get(ticket_id)
    if not ticket:
        return json.dumps({"error": f"工单 {ticket_id} 不存在"}, ensure_ascii=False)
    return json.dumps({"ticket_id": ticket_id, **ticket}, ensure_ascii=False)


def check_system_status(service: str) -> str:
    """检查系统服务状态。"""
    statuses = {"auth": "healthy", "payment": "degraded", "api": "healthy", "web": "healthy"}
    s = statuses.get(service, "unknown")
    return json.dumps({"service": service, "status": s, "uptime": "99.2%"}, ensure_ascii=False)


def process_refund(ticket_id: str, amount: float, reason: str) -> str:
    """处理退款。"""
    return json.dumps({"ticket_id": ticket_id, "refund_amount": amount, "status": "processed", "reason": reason, "eta": "3-5 business days"}, ensure_ascii=False)


def escalate_ticket(ticket_id: str, target_team: str, notes: str) -> str:
    """升级工单到指定团队。"""
    return json.dumps({"ticket_id": ticket_id, "escalated_to": target_team, "notes": notes, "status": "escalated"}, ensure_ascii=False)


# ── 数据库运维 (06_pydantic_ai) ───────────────────────────

DB_TABLES = {
    "users": {"columns": ["id", "name", "email", "created_at", "status"], "rows": 15420, "size_mb": 2.3},
    "orders": {"columns": ["id", "user_id", "amount", "status", "created_at"], "rows": 89200, "size_mb": 12.7},
    "products": {"columns": ["id", "name", "price", "category", "stock"], "rows": 3500, "size_mb": 0.8},
    "logs": {"columns": ["id", "level", "message", "timestamp", "service"], "rows": 2450000, "size_mb": 890.5},
}


def list_tables() -> str:
    """列出所有数据库表。"""
    result = [{"table": name, "rows": info["rows"], "size_mb": info["size_mb"]} for name, info in DB_TABLES.items()]
    return json.dumps(result, ensure_ascii=False)


def describe_table(table_name: str) -> str:
    """查看表结构。"""
    info = DB_TABLES.get(table_name)
    if not info:
        return json.dumps({"error": f"表 '{table_name}' 不存在"}, ensure_ascii=False)
    return json.dumps({"table": table_name, **info}, ensure_ascii=False)


def run_query(sql: str) -> str:
    """执行 SQL 查询（模拟）。"""
    sql_lower = sql.lower().strip()
    if any(kw in sql_lower for kw in ["drop", "delete", "truncate", "alter"]):
        return json.dumps({"error": "危险操作被拦截！只读查询模式。", "blocked_sql": sql}, ensure_ascii=False)
    if "count" in sql_lower and "logs" in sql_lower:
        return json.dumps({"result": [{"count": 2450000}], "execution_time_ms": 120}, ensure_ascii=False)
    if "count" in sql_lower and "users" in sql_lower:
        return json.dumps({"result": [{"count": 15420}], "execution_time_ms": 5}, ensure_ascii=False)
    if "size" in sql_lower or "pg_total_relation_size" in sql_lower:
        return json.dumps({"result": [{"table": "logs", "size_mb": 890.5}, {"table": "orders", "size_mb": 12.7}], "execution_time_ms": 45}, ensure_ascii=False)
    return json.dumps({"result": [{"message": "查询已执行（模拟结果）"}], "execution_time_ms": 10}, ensure_ascii=False)


# ── 通用文件工具 ──────────────────────────────────────────

def write_file(filename: str, content: str) -> str:
    """将内容写入文件。"""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已保存到 {filepath}（{len(content)} 字符）"


def read_file(filename: str) -> str:
    """读取文件内容。"""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    filepath = os.path.join(output_dir, filename)
    if not os.path.exists(filepath):
        return f"文件 '{filepath}' 不存在"
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
