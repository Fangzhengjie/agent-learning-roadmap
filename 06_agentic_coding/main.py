"""Agentic Coding — 代码 Agent 工程原理

核心概念：让 AI 像程序员一样读、写、调试代码 — 从辅助补全到自主编程。

代码 Agent 产品矩阵:
  ┌──────────────────────────────────────────────────────────┐
  │  Level 1: 补全         Copilot / Codeium / Tabnine      │
  │  Level 2: 对话         ChatGPT / Claude Chat             │
  │  Level 3: 编辑器内 Agent  Cursor / Windsurf / Cline      │
  │  Level 4: 自主 Agent     Devin / OpenAI Codex / SWE-agent│
  │  Level 5: 终端 Agent     Claude Code / Warp AI           │
  └──────────────────────────────────────────────────────────┘

本示例展示代码 Agent 的核心工程原理：
  1. 代码 Agent 架构 — 感知→规划→编辑→验证循环
  2. 沙箱与安全 — 代码执行的隔离环境
  3. 文件编辑策略 — Search-Replace / AST Diff / 全文重写
  4. Test-Driven Agent Loop — 测试驱动的自动修复
  5. 上下文工程 — 如何给 Agent 足够的代码上下文
  6. 产品对比与选型
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# 1. 代码 Agent 架构
# ═══════════════════════════════════════════════════════════

def show_architecture():
    """展示代码 Agent 核心架构。"""
    print("▶ 1. 代码 Agent 架构 — 感知→规划→编辑→验证")
    print("─" * 60)

    print("""
  代码 Agent 核心循环:
  ─────────────────────────────────────────────────────────

  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │   ┌──────────┐    ┌──────────┐    ┌──────────┐     │
  │   │ 1. 感知   │ →  │ 2. 规划   │ →  │ 3. 编辑   │    │
  │   │ Perceive │    │ Plan     │    │ Edit     │    │
  │   └──────────┘    └──────────┘    └──────────┘     │
  │       ↑                                ↓           │
  │   ┌──────────┐                   ┌──────────┐     │
  │   │ 5. 反馈   │ ←──────────────  │ 4. 验证   │    │
  │   │ Feedback │                   │ Verify   │     │
  │   └──────────┘                   └──────────┘     │
  │                                                     │
  └─────────────────────────────────────────────────────┘

  每一步的具体工作:
  ─────────────────────────────────────────────────────────
  1. 感知 (Perceive):
     - 读取项目结构 (file tree)
     - 读取相关文件内容
     - 搜索代码 (grep / AST)
     - 读取错误日志 / 测试结果
     - 获取 git diff / blame

  2. 规划 (Plan):
     - 分析用户需求
     - 确定需要修改哪些文件
     - 制定修改步骤（先改什么后改什么）
     - 预估影响范围

  3. 编辑 (Edit):
     - 创建新文件
     - 修改现有文件（Search-Replace / Diff）
     - 删除文件
     - 添加依赖

  4. 验证 (Verify):
     - 运行测试 (pytest / jest / go test)
     - 运行 linter (eslint / ruff / clippy)
     - 类型检查 (tsc / mypy)
     - 构建检查 (build / compile)

  5. 反馈 (Feedback):
     - 测试通过 → 完成
     - 测试失败 → 分析错误 → 回到规划
     - Lint 错误 → 自动修复 → 重新验证

  Agent 工具集:
  ──────────────┬──────────────────────────────────────
  read_file     │ 读取文件内容
  write_file    │ 创建新文件
  edit_file     │ 修改文件（search-replace）
  list_dir      │ 列出目录结构
  grep_search   │ 搜索代码
  run_command   │ 执行终端命令（在沙箱中）
  git_diff      │ 查看变更
  ask_user      │ 向用户确认关键决策""")


# ═══════════════════════════════════════════════════════════
# 2. 沙箱与安全
# ═══════════════════════════════════════════════════════════

def show_sandbox():
    """展示代码执行沙箱。"""
    print(f"\n\n▶ 2. 沙箱与安全 — 代码执行的隔离环境")
    print("─" * 60)

    print(f"""
  为什么需要沙箱？
  ─────────────────────────────────────────────────────────
  代码 Agent 需要执行 rm / pip install / git push 等命令。
  如果不隔离 → Agent 可能删库、安装恶意包、推送坏代码。

  沙箱方案对比:
  ──────────────┬──────────────┬──────────────┬──────────
  方案           │ 隔离级别      │ 启动速度      │ 适用场景
  ──────────────┼──────────────┼──────────────┼──────────
  Docker 容器   │ ⭐⭐⭐ 强     │ 秒级          │ 生产首选
  microVM       │ ⭐⭐⭐ 最强   │ 毫秒级        │ Firecracker
  gVisor        │ ⭐⭐⭐ 强     │ 毫秒级        │ Google 方案
  nsjail        │ ⭐⭐ 中       │ 毫秒级        │ 轻量沙箱
  chroot        │ ⭐ 弱         │ 即时          │ 简单隔离
  WebAssembly   │ ⭐⭐ 中       │ 即时          │ 浏览器端

  各产品的沙箱方案:
  ──────────────┬──────────────────────────────────────
  OpenAI Codex  │ microVM（每个任务独立 VM，云端异步）
  Devin         │ Docker 容器（完整开发环境）
  Cursor        │ 本地执行（用户确认机制，无沙箱）
  Claude Code   │ 本地 + 权限控制（用户审批命令）
  SWE-agent     │ Docker 容器（标准化环境）
  E2B           │ 云端微沙箱（SDK 调用，按秒计费）

  安全层级:
  ─────────────────────────────────────────────────────────

  Layer 1: 命令白名单
  ┌──────────────────────────────────────────────────────┐
  │ 允许: ls, cat, grep, python, pytest, npm test        │
  │ 禁止: rm -rf, curl (外网), sudo, chmod 777           │
  └──────────────────────────────────────────────────────┘

  Layer 2: 用户确认
  ┌──────────────────────────────────────────────────────┐
  │ 安全命令: 自动执行（ls, cat, grep）                   │
  │ 风险命令: 需要用户确认（pip install, git push）       │
  │ 危险命令: 直接拒绝（rm -rf /, DROP TABLE）            │
  └──────────────────────────────────────────────────────┘

  Layer 3: 环境隔离
  ┌──────────────────────────────────────────────────────┐
  │ 文件系统: 只允许访问项目目录                          │
  │ 网络:     禁止外网（或仅允许 npm/pip registry）       │
  │ 进程:     超时自动杀死（30s~5min）                    │
  │ 资源:     CPU/内存限制                                │
  └──────────────────────────────────────────────────────┘""")


# ═══════════════════════════════════════════════════════════
# 3. 文件编辑策略 — FileEditor 引擎
# ═══════════════════════════════════════════════════════════

@dataclass
class EditResult:
    """编辑结果。"""
    success: bool
    message: str
    file_path: str = ""
    lines_changed: int = 0


class FileEditor:
    """代码 Agent 文件编辑器 — 支持 Search-Replace / 全文重写。

    这是 Cursor / Windsurf / Cline 等 Agent IDE 的核心组件。
    """

    def __init__(self, workspace: str):
        self.workspace = workspace
        self.history: list[dict] = []  # 编辑历史（可回退）

    def _resolve(self, path: str) -> str:
        return os.path.join(self.workspace, path)

    def read(self, path: str) -> str | None:
        """读取文件。"""
        full = self._resolve(path)
        if not os.path.exists(full):
            return None
        with open(full, "r", encoding="utf-8") as f:
            return f.read()

    def write(self, path: str, content: str) -> EditResult:
        """全文重写。"""
        full = self._resolve(path)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        old = self.read(path)
        self.history.append({"type": "write", "path": path, "old": old, "new": content})
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return EditResult(True, f"写入 {len(content)} 字符", path, content.count("\n") + 1)

    def search_replace(self, path: str, old_string: str, new_string: str) -> EditResult:
        """Search-Replace 精确替换（Cursor/Windsurf 方式）。"""
        content = self.read(path)
        if content is None:
            return EditResult(False, f"文件不存在: {path}", path)
        count = content.count(old_string)
        if count == 0:
            return EditResult(False, f"未找到匹配: '{old_string[:40]}...'", path)
        if count > 1:
            return EditResult(False, f"匹配不唯一: 找到 {count} 处", path)
        self.history.append({"type": "search_replace", "path": path, "old": content})
        new_content = content.replace(old_string, new_string, 1)
        with open(self._resolve(path), "w", encoding="utf-8") as f:
            f.write(new_content)
        changed = abs(new_string.count("\n") - old_string.count("\n")) + 1
        return EditResult(True, f"替换成功", path, changed)

    def undo(self) -> EditResult:
        """撤销上一次编辑。"""
        if not self.history:
            return EditResult(False, "无编辑历史")
        last = self.history.pop()
        if last["old"] is not None:
            with open(self._resolve(last["path"]), "w", encoding="utf-8") as f:
                f.write(last["old"])
        return EditResult(True, f"已撤销 {last['type']} on {last['path']}")

    def list_files(self, pattern: str = "*.py") -> list[str]:
        """列出工作区文件。"""
        result = []
        for root, _, files in os.walk(self.workspace):
            for f in files:
                if pattern == "*" or f.endswith(pattern.lstrip("*")):
                    rel = os.path.relpath(os.path.join(root, f), self.workspace)
                    result.append(rel)
        return sorted(result)


def demo_edit_strategies():
    """用 FileEditor 演示真实文件编辑。"""
    print(f"\n\n▶ 3. 文件编辑策略 — FileEditor 引擎")
    print("─" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        editor = FileEditor(tmpdir)

        # 创建源文件
        original_code = '''def process_order(order_id):
    order = db.get(order_id)
    if order.status == "pending":
        order.status = "processing"
        db.save(order)
        return True
    return False
'''
        editor.write("order.py", original_code)

        print(f"\n  原始代码 (order.py):")
        for i, line in enumerate(original_code.strip().split("\n"), 1):
            print(f"    {i:2d} │ {line}")

        # Search-Replace
        print(f"\n  Search-Replace 编辑:")
        result = editor.search_replace(
            "order.py",
            '        order.status = "processing"',
            '        order.status = "processing"\n        order.updated_at = now()'
        )
        print(f"    {('✅' if result.success else '❌')} {result.message}")

        # 验证
        updated = editor.read("order.py")
        print(f"\n  编辑后:")
        for i, line in enumerate(updated.strip().split("\n"), 1):
            marker = "  + " if "updated_at" in line else "    "
            print(f"  {marker}{i:2d} │ {line}")

        # 测试错误情况
        print(f"\n  错误处理:")
        r1 = editor.search_replace("order.py", "nonexistent code", "new")
        print(f"    ❌ 匹配失败: {r1.message}")
        r2 = editor.search_replace("missing.py", "x", "y")
        print(f"    ❌ 文件不存在: {r2.message}")

        # Undo
        editor.undo()
        restored = editor.read("order.py")
        print(f"    ↩️  Undo: {'成功' if restored == original_code else '失败'}")

    print(f"""
  策略选型:
  ──────────────┬──────────────────────────────────────
  小文件 (<100行)│ 全文重写（简单可靠）
  中文件 (100-1K)│ Search-Replace（精确最小化）
  大文件 (>1K行) │ Search-Replace / Diff（必须精确）
  新文件          │ 全文重写（write_file）
  多处修改        │ 多次 Search-Replace 或 Multi-Edit""")


# ═══════════════════════════════════════════════════════════
# 4. Test-Driven Agent Loop — TestRunner
# ═══════════════════════════════════════════════════════════

@dataclass
class TestResult:
    """测试运行结果。"""
    passed: bool
    output: str
    failures: list[str] = field(default_factory=list)
    return_code: int = 0


class TestRunner:
    """测试运行器 — 执行代码并收集结果。"""

    @staticmethod
    def run_python(file_path: str, timeout: int = 10) -> TestResult:
        """运行 Python 文件并捕获输出。"""
        try:
            result = subprocess.run(
                [sys.executable, file_path],
                capture_output=True, text=True, timeout=timeout,
                cwd=os.path.dirname(file_path)
            )
            failures = []
            if result.returncode != 0:
                for line in result.stderr.split("\n"):
                    if "Error" in line or "assert" in line.lower():
                        failures.append(line.strip())
            return TestResult(
                passed=result.returncode == 0,
                output=result.stdout + result.stderr,
                failures=failures,
                return_code=result.returncode
            )
        except subprocess.TimeoutExpired:
            return TestResult(False, "超时", ["执行超时"], -1)
        except Exception as e:
            return TestResult(False, str(e), [str(e)], -1)


def demo_test_driven_loop():
    """用 FileEditor + TestRunner 运行真实 Test-Driven Loop。"""
    print(f"\n\n▶ 4. Test-Driven Agent Loop — 真实修复循环")
    print("─" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        editor = FileEditor(tmpdir)
        runner = TestRunner()

        # Step 1: 创建有 bug 的源码和测试
        buggy_code = '''import datetime

def add_numbers(a, b):
    return a - b  # BUG: should be a + b

def get_greeting(name):
    return f"Hello {name}"
'''
        test_code = '''import sys
sys.path.insert(0, ".")
from code import add_numbers, get_greeting

# Test 1
result = add_numbers(2, 3)
assert result == 5, f"Expected 5 but got {result}"

# Test 2
greeting = get_greeting("World")
assert greeting == "Hello World", f"Expected 'Hello World' but got '{greeting}'"

print("All tests passed!")
'''
        editor.write("code.py", buggy_code)
        editor.write("test_code.py", test_code)

        print("\n  Step 1: 运行测试 → 发现失败")
        result = runner.run_python(os.path.join(tmpdir, "test_code.py"))
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        print(f"    {status}")
        if result.failures:
            print(f"    错误: {result.failures[0][:80]}")

        # Step 2: 分析 + 修复
        print("\n  Step 2: 分析错误 → 编辑修复")
        fix_result = editor.search_replace("code.py", "return a - b  # BUG: should be a + b", "return a + b")
        print(f"    ✏️  {fix_result.message}")

        # Step 3: 重新运行测试
        print("\n  Step 3: 重新运行测试 → 验证修复")
        result2 = runner.run_python(os.path.join(tmpdir, "test_code.py"))
        status2 = "✅ PASSED" if result2.passed else "❌ FAILED"
        print(f"    {status2}")
        if result2.passed:
            print(f"    输出: {result2.output.strip()}")

    print(f"""
  关键工程细节:
  ─────────────────────────────────────────────────────────
  最大重试次数   │ 通常 3~5 次（防无限循环）
  错误分类       │ 编译错误 vs 运行时错误 vs 测试失败
  回退策略       │ 多次失败后 git checkout 回退
  增量验证       │ 每次改动后只运行相关测试（快速反馈）
  上下文注入     │ 把测试输出 + 错误栈追踪喂给 LLM""")


# ═══════════════════════════════════════════════════════════
# 5. 上下文工程 — ContextRanker
# ═══════════════════════════════════════════════════════════

class ContextRanker:
    """代码上下文排序器 — 找到与任务最相关的文件。

    生产中用向量相似度 + AST 分析，这里用关键词 + 路径匹配演示。
    """

    def __init__(self):
        self.files: list[dict] = []  # {path, content, size}

    def index_workspace(self, workspace: str, pattern: str = ".py"):
        """索引工作区文件。"""
        self.files = []
        for root, _, filenames in os.walk(workspace):
            for fn in filenames:
                if fn.endswith(pattern):
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, workspace)
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    self.files.append({"path": rel, "content": content, "size": len(content)})

    def rank(self, query: str, top_k: int = 5) -> list[dict]:
        """根据查询排序文件相关性。"""
        scored = []
        query_terms = set(query.lower().split())
        for f in self.files:
            score = 0
            content_lower = f["content"].lower()
            path_lower = f["path"].lower()
            # 路径匹配
            for term in query_terms:
                if term in path_lower:
                    score += 10
                if term in content_lower:
                    score += content_lower.count(term)
            # test 文件加权
            if "test" in path_lower and any(t in query.lower() for t in ["test", "bug", "fix"]):
                score += 5
            scored.append({**f, "relevance": score})
        scored.sort(key=lambda x: x["relevance"], reverse=True)
        return [{"path": f["path"], "size": f["size"], "relevance": f["relevance"]}
                for f in scored[:top_k] if f["relevance"] > 0]


def demo_context_engineering():
    """用 ContextRanker 演示上下文排序。"""
    print(f"\n\n▶ 5. 上下文工程 — ContextRanker")
    print("─" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        editor = FileEditor(tmpdir)

        # 创建模拟项目
        editor.write("src/order.py", "class Order:\n    def process(self): pass\n    def cancel(self): pass\n")
        editor.write("src/payment.py", "def charge(amount): pass\ndef refund(amount): pass\n")
        editor.write("src/auth.py", "def login(user, pwd): pass\ndef logout(): pass\n")
        editor.write("src/models/order_model.py", "class OrderModel:\n    status: str\n")
        editor.write("tests/test_order.py", "def test_process_order(): assert True\ndef test_cancel_order(): assert True\n")
        editor.write("tests/test_payment.py", "def test_charge(): assert True\n")
        editor.write("config.py", "DB_URL = 'sqlite:///app.db'\n")

        # 索引
        ranker = ContextRanker()
        ranker.index_workspace(tmpdir)
        print(f"\n  工作区文件: {len(ranker.files)} 个")
        for f in ranker.files:
            print(f"    {f['path']:30s} ({f['size']:4d} bytes)")

        # 排序
        queries = [
            "修复 order process 的 bug",
            "payment charge refund",
            "auth login 认证问题",
        ]
        for query in queries:
            results = ranker.rank(query, top_k=3)
            print(f"\n  查询: '{query}'")
            for r in results:
                print(f"    📄 {r['path']:30s} (相关度: {r['relevance']})")

    print(f"""
  上下文窗口分配（代码 Agent）:
  ─────────────────────────────────────────────────────────
  System Prompt    │  2K~5K tokens  │ 角色 + 工具说明 + 编码规范
  用户指令          │  100~500 tokens│ "修复这个 bug"
  相关代码          │  10K~50K tokens│ 源码 + 测试 + 依赖
  对话历史          │  5K~20K tokens │ 之前的修改和反馈
  工具结果          │  2K~10K tokens │ 测试输出 + 错误日志
  合计              │ ~20K~80K tokens│ 在 128K 窗口内""")


# ═══════════════════════════════════════════════════════════
# 6. 产品对比
# ═══════════════════════════════════════════════════════════

def show_product_comparison():
    """展示代码 Agent 产品对比。"""
    print(f"\n\n▶ 6. 代码 Agent 产品对比")
    print("─" * 60)

    print(f"""
  编辑器内 Agent:
  ──────────────┬──────────┬──────────┬──────────────────
  产品           │ 编辑方式  │ 沙箱     │ 特点
  ──────────────┼──────────┼──────────┼──────────────────
  Cursor        │ S&R+Diff │ 本地     │ VSCode fork，最主流
  Windsurf      │ S&R      │ 本地     │ Cascade 上下文引擎
  Cline         │ S&R+Diff │ 本地     │ VSCode 插件，开源
  Continue      │ S&R      │ 本地     │ 开源，可接多模型
  Aider         │ Diff/S&R │ 本地     │ 终端工具，git 集成

  自主 Agent:
  ──────────────┬──────────┬──────────┬──────────────────
  产品           │ 编辑方式  │ 沙箱     │ 特点
  ──────────────┼──────────┼──────────┼──────────────────
  OpenAI Codex  │ 全文重写  │ microVM  │ 异步执行，云端
  Devin         │ 全文     │ Docker   │ 端到端自主开发
  Claude Code   │ S&R      │ 本地     │ 终端 Agent，无 IDE
  SWE-agent     │ Diff     │ Docker   │ 开源研究项目

  代码 Agent 的关键指标:
  ──────────────┬──────────────────────────────────────
  SWE-bench     │ GitHub Issue 修复准确率（标准评测）
  HumanEval     │ 代码生成准确率
  首次修复率     │ 一次就改对的比例
  平均轮次       │ 改几次才能通过测试
  上下文精度     │ 找到正确文件的比例

  SWE-bench 排行 (2025):
  ──────────────┬──────────────────────────────────────
  Claude Code   │ ~72% (Claude 3.5 Sonnet)
  OpenAI Codex  │ ~70% (o3)
  Devin         │ ~55%
  SWE-agent     │ ~30% (开源基线)""")


# ═══════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════

def main():
    print("=== Agentic Coding 代码 Agent 工程原理 ===\n")

    show_architecture()
    show_sandbox()
    demo_edit_strategies()
    demo_test_driven_loop()
    demo_context_engineering()
    show_product_comparison()

    # ── 架构总结 ──────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 代码 Agent 工程总结:")
    print()
    print("  核心循环: 感知 → 规划 → 编辑 → 验证 → 反馈")
    print("  ────────────────────────────────────────────")
    print("  感知  │ 读文件 + 搜索代码 + Git 历史")
    print("  规划  │ 分析需求 → 确定改哪些文件")
    print("  编辑  │ Search-Replace / Diff / 全文重写")
    print("  验证  │ 运行测试 + Lint + 类型检查")
    print("  反馈  │ 失败 → 分析错误 → 重新编辑")
    print()
    print("  关键工程要素:")
    print("  ────────────────────────────────────────────")
    print("  □ 沙箱隔离 — Docker/microVM 执行用户代码")
    print("  □ 文件编辑 — Search-Replace 最精确最小化")
    print("  □ 测试驱动 — 每次编辑后运行测试验证")
    print("  □ 上下文工程 — 文件树 + RAG + AST + Git")
    print("  □ 安全层级 — 白名单 + 用户确认 + 隔离")
    print("  □ 重试限制 — max 3~5 次防无限循环")


if __name__ == "__main__":
    main()
