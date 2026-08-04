"""Agent 部署与生产化 — 从代码到生产的最后一公里

核心概念：Agent 写好了只是第一步，部署到生产需要一整套基建。

Agent 生产化清单:
  ┌──────────────────────────────────────────────────────────┐
  │  ✅ 配置管理     │ 环境变量 / 密钥注入 / 多环境          │
  │  ✅ 健康检查     │ Liveness + Readiness + Startup        │
  │  ✅ 优雅关闭     │ SIGTERM → 完成在途请求 → 退出         │
  │  ✅ 容器化       │ Dockerfile + 多阶段构建               │
  │  ✅ 密钥管理     │ 加密存储 / Vault / 环境变量            │
  │  ✅ 并发控制     │ 请求队列 / Worker 池                   │
  │  ✅ 版本管理     │ 模型版本 + Prompt 版本 + 配置版本      │
  └──────────────────────────────────────────────────────────┘

本示例展示:
  1. ConfigManager — 多环境配置 + 密钥管理
  2. HealthCheck — Liveness / Readiness / Startup 探针
  3. AgentServer — HTTP 服务器（含优雅关闭、并发控制）
  4. Dockerfile 生成器 — 多阶段构建模板
  5. VersionManager — Agent/模型/Prompt 版本追踪
"""

import hashlib
import json
import os
import signal
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# 1. ConfigManager — 多环境配置 + 密钥
# ═══════════════════════════════════════════════════════════

class ConfigManager:
    """配置管理器 — 多环境 + 密钥加密 + 热加载。

    加载优先级: 环境变量 > .env 文件 > 默认值
    """

    def __init__(self, env: str = "development"):
        self.env = env
        self._config: dict[str, Any] = {}
        self._secrets: dict[str, str] = {}
        self._defaults = {
            "development": {
                "LLM_MODEL": "gpt-4o-mini",
                "LLM_TEMPERATURE": 0.0,
                "LLM_MAX_TOKENS": 2000,
                "MAX_CONCURRENT": 5,
                "LOG_LEVEL": "DEBUG",
                "TIMEOUT_S": 30,
            },
            "staging": {
                "LLM_MODEL": "gpt-4o-mini",
                "LLM_TEMPERATURE": 0.0,
                "LLM_MAX_TOKENS": 4000,
                "MAX_CONCURRENT": 20,
                "LOG_LEVEL": "INFO",
                "TIMEOUT_S": 15,
            },
            "production": {
                "LLM_MODEL": "gpt-4o",
                "LLM_TEMPERATURE": 0.0,
                "LLM_MAX_TOKENS": 4000,
                "MAX_CONCURRENT": 50,
                "LOG_LEVEL": "WARNING",
                "TIMEOUT_S": 10,
            },
        }
        self._load_defaults()

    def _load_defaults(self):
        self._config = dict(self._defaults.get(self.env, self._defaults["development"]))

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置（环境变量优先）。"""
        env_val = os.environ.get(key)
        if env_val is not None:
            return env_val
        return self._config.get(key, default)

    def get_secret(self, key: str) -> str | None:
        """获取密钥（从环境变量或加密存储）。"""
        # 优先环境变量
        val = os.environ.get(key)
        if val:
            return val
        return self._secrets.get(key)

    def set_secret(self, key: str, value: str):
        """设置密钥（模拟加密存储）。"""
        self._secrets[key] = value

    def set(self, key: str, value: Any):
        self._config[key] = value

    def to_dict(self, mask_secrets: bool = True) -> dict:
        """导出配置（可选隐藏密钥）。"""
        result = dict(self._config)
        result["__env__"] = self.env
        for k, v in self._secrets.items():
            result[k] = "***" if mask_secrets else v
        return result

    def save_env_file(self, path: str):
        """导出为 .env 文件。"""
        with open(path, "w") as f:
            f.write(f"# Auto-generated for {self.env}\n")
            for k, v in self._config.items():
                f.write(f"{k}={v}\n")
            for k in self._secrets:
                f.write(f"# {k}=*** (use Vault/Akeyless)\n")


# ═══════════════════════════════════════════════════════════
# 2. HealthCheck — 探针系统
# ═══════════════════════════════════════════════════════════

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthResult:
    status: HealthStatus
    checks: dict[str, dict] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "checks": self.checks,
            "timestamp": self.timestamp,
        }


class HealthCheck:
    """健康检查 — Kubernetes Liveness/Readiness/Startup 探针。"""

    def __init__(self):
        self._checks: dict[str, Callable[[], dict]] = {}
        self._started = False
        self._ready = False

    def register(self, name: str, check_fn: Callable[[], dict]):
        """注册一个健康检查项。check_fn 返回 {"healthy": bool, "detail": str}"""
        self._checks[name] = check_fn

    def mark_started(self):
        self._started = True

    def mark_ready(self):
        self._ready = True

    def mark_not_ready(self):
        self._ready = False

    def liveness(self) -> HealthResult:
        """存活检查 — 进程是否还活着。"""
        return HealthResult(
            status=HealthStatus.HEALTHY if self._started else HealthStatus.UNHEALTHY,
            checks={"started": {"healthy": self._started}},
        )

    def readiness(self) -> HealthResult:
        """就绪检查 — 是否可以接收流量。"""
        if not self._ready:
            return HealthResult(
                status=HealthStatus.UNHEALTHY,
                checks={"ready": {"healthy": False, "detail": "Not ready"}},
            )
        all_checks = {}
        overall = HealthStatus.HEALTHY
        for name, fn in self._checks.items():
            result = fn()
            all_checks[name] = result
            if not result.get("healthy", False):
                overall = HealthStatus.UNHEALTHY
        return HealthResult(status=overall, checks=all_checks)

    def startup(self) -> HealthResult:
        """启动检查 — 初始化是否完成。"""
        return self.liveness()


# ═══════════════════════════════════════════════════════════
# 3. AgentServer — HTTP 服务器模拟
# ═══════════════════════════════════════════════════════════

class AgentServer:
    """Agent 服务器 — 含优雅关闭、并发控制、请求处理。

    模拟生产级 Agent HTTP 服务（不依赖 Flask/FastAPI）。
    """

    def __init__(self, config: ConfigManager, health: HealthCheck):
        self.config = config
        self.health = health
        self._running = False
        self._active_requests = 0
        self._total_requests = 0
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

    def start(self):
        """启动服务器。"""
        self.health.mark_started()
        self._running = True
        # 模拟初始化（加载模型配置等）
        time.sleep(0.05)
        self.health.mark_ready()

    def handle_request(self, request: dict) -> dict:
        """处理一个请求。"""
        max_concurrent = int(self.config.get("MAX_CONCURRENT", 5))

        with self._lock:
            if not self._running:
                return {"error": "Server shutting down", "status": 503}
            if self._active_requests >= max_concurrent:
                return {"error": "Too many requests", "status": 429}
            self._active_requests += 1
            self._total_requests += 1

        try:
            # 模拟 Agent 处理
            time.sleep(0.02)
            return {
                "answer": f"处理请求: {request.get('query', '')}",
                "model": self.config.get("LLM_MODEL"),
                "status": 200,
            }
        finally:
            with self._lock:
                self._active_requests -= 1

    def graceful_shutdown(self, timeout_s: float = 5.0):
        """优雅关闭 — 等待在途请求完成。"""
        self.health.mark_not_ready()  # 停止接收新请求
        self._running = False

        # 等待在途请求完成
        deadline = time.time() + timeout_s
        while self._active_requests > 0 and time.time() < deadline:
            time.sleep(0.1)

        return {
            "drained": self._active_requests == 0,
            "active_at_shutdown": self._active_requests,
            "total_served": self._total_requests,
        }

    def stats(self) -> dict:
        return {
            "running": self._running,
            "active_requests": self._active_requests,
            "total_requests": self._total_requests,
            "health": self.health.readiness().to_dict(),
        }


# ═══════════════════════════════════════════════════════════
# 4. Dockerfile 生成器
# ═══════════════════════════════════════════════════════════

class DockerfileGenerator:
    """生成 Agent 服务的 Dockerfile（多阶段构建）。"""

    @staticmethod
    def generate(app_name: str = "agent-service",
                 python_version: str = "3.12",
                 port: int = 8080,
                 use_uv: bool = True) -> str:
        if use_uv:
            return f"""# === {app_name} Dockerfile (多阶段构建 + uv) ===

# Stage 1: 安装依赖
FROM python:{python_version}-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Stage 2: 运行
FROM python:{python_version}-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE {port}
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\
  CMD curl -f http://localhost:{port}/health || exit 1

# 非 root 用户运行
RUN useradd -m appuser
USER appuser

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{port}"]
"""
        else:
            return f"""# === {app_name} Dockerfile (多阶段构建 + pip) ===

FROM python:{python_version}-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/deps -r requirements.txt

FROM python:{python_version}-slim AS runtime
WORKDIR /app
COPY --from=builder /deps /deps
COPY . .
ENV PYTHONPATH="/deps:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
EXPOSE {port}
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\
  CMD curl -f http://localhost:{port}/health || exit 1
RUN useradd -m appuser
USER appuser
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{port}"]
"""


# ═══════════════════════════════════════════════════════════
# 5. VersionManager — 版本追踪
# ═══════════════════════════════════════════════════════════

@dataclass
class AgentVersion:
    """Agent 版本信息。"""
    agent_version: str
    model: str
    prompt_hash: str
    config_hash: str
    deployed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "agent_version": self.agent_version,
            "model": self.model,
            "prompt_hash": self.prompt_hash[:12],
            "config_hash": self.config_hash[:12],
            "deployed_at": self.deployed_at,
        }


class VersionManager:
    """版本管理器 — 追踪 Agent、模型、Prompt 的变更。"""

    def __init__(self):
        self.versions: list[AgentVersion] = []

    @staticmethod
    def hash_content(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def register(self, agent_version: str, model: str,
                 system_prompt: str, config: dict) -> AgentVersion:
        v = AgentVersion(
            agent_version=agent_version,
            model=model,
            prompt_hash=self.hash_content(system_prompt),
            config_hash=self.hash_content(json.dumps(config, sort_keys=True)),
        )
        self.versions.append(v)
        return v

    def current(self) -> AgentVersion | None:
        return self.versions[-1] if self.versions else None

    def diff(self, v1: AgentVersion, v2: AgentVersion) -> dict:
        """对比两个版本的差异。"""
        changes = {}
        if v1.model != v2.model:
            changes["model"] = {"old": v1.model, "new": v2.model}
        if v1.prompt_hash != v2.prompt_hash:
            changes["prompt"] = "changed"
        if v1.config_hash != v2.config_hash:
            changes["config"] = "changed"
        return changes


# ═══════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════

def demo_config():
    """演示多环境配置管理。"""
    print("▶ 1. ConfigManager — 多环境配置")
    print("─" * 60)

    for env in ["development", "staging", "production"]:
        cfg = ConfigManager(env=env)
        cfg.set_secret("OPENAI_API_KEY", "sk-fake-key-12345")
        print(f"  [{env}] model={cfg.get('LLM_MODEL')}, "
              f"concurrent={cfg.get('MAX_CONCURRENT')}, "
              f"log={cfg.get('LOG_LEVEL')}")

    # .env 文件生成
    with tempfile.NamedTemporaryFile(suffix=".env", delete=False, mode="w") as f:
        cfg = ConfigManager("production")
        cfg.set_secret("OPENAI_API_KEY", "sk-real-key")
        cfg.save_env_file(f.name)
        env_path = f.name

    with open(env_path) as f:
        content = f.read()
    os.unlink(env_path)
    print(f"\n  生成 .env 文件:\n    {content.replace(chr(10), chr(10) + '    ')}")


def demo_health_check():
    """演示健康检查探针。"""
    print(f"\n▶ 2. HealthCheck — 探针系统")
    print("─" * 60)

    hc = HealthCheck()

    # 注册检查项
    hc.register("llm_api", lambda: {"healthy": True, "detail": "GPT-4o reachable"})
    hc.register("vector_db", lambda: {"healthy": True, "detail": "Chroma connected"})

    # 启动前
    print(f"  启动前: liveness={hc.liveness().status.value}, "
          f"readiness={hc.readiness().status.value}")

    hc.mark_started()
    hc.mark_ready()

    # 启动后
    r = hc.readiness()
    print(f"  启动后: liveness={hc.liveness().status.value}, "
          f"readiness={r.status.value}")
    for name, check in r.checks.items():
        print(f"    {name}: {check}")


def demo_server():
    """演示 Agent 服务器。"""
    print(f"\n\n▶ 3. AgentServer — 请求处理 + 优雅关闭")
    print("─" * 60)

    cfg = ConfigManager("production")
    hc = HealthCheck()
    hc.register("llm", lambda: {"healthy": True})
    server = AgentServer(cfg, hc)
    server.start()

    # 处理请求
    for i in range(3):
        resp = server.handle_request({"query": f"查询工单 T-{i+1:03d}"})
        print(f"  请求 #{i+1}: [{resp['status']}] {resp.get('answer', resp.get('error'))}")

    print(f"  服务状态: {server.stats()['total_requests']} 请求已处理")

    # 优雅关闭
    result = server.graceful_shutdown(timeout_s=2.0)
    print(f"  优雅关闭: drained={result['drained']}, total_served={result['total_served']}")

    # 关闭后的请求
    resp = server.handle_request({"query": "test"})
    print(f"  关闭后请求: [{resp['status']}] {resp.get('error')}")


def demo_dockerfile():
    """演示 Dockerfile 生成。"""
    print(f"\n\n▶ 4. Dockerfile 生成器")
    print("─" * 60)

    df = DockerfileGenerator.generate("smartflow-agent", port=8080, use_uv=True)
    lines = df.strip().split("\n")
    print(f"  生成 Dockerfile ({len(lines)} 行):")
    for line in lines[:12]:
        print(f"    {line}")
    print(f"    ... ({len(lines) - 12} more lines)")


def demo_version():
    """演示版本管理。"""
    print(f"\n\n▶ 5. VersionManager — 版本追踪")
    print("─" * 60)

    vm = VersionManager()

    v1 = vm.register("1.0.0", "gpt-4o-mini",
                      "你是 SmartFlow 客服助手", {"temperature": 0, "max_tokens": 2000})
    print(f"  v1: {v1.to_dict()}")

    v2 = vm.register("1.1.0", "gpt-4o",
                      "你是 SmartFlow 智能客服助手，请用中文回复", {"temperature": 0, "max_tokens": 4000})
    print(f"  v2: {v2.to_dict()}")

    diff = vm.diff(v1, v2)
    print(f"  版本差异: {diff}")


def main():
    print("=== Agent 部署与生产化 ===\n")

    demo_config()
    demo_health_check()
    demo_server()
    demo_dockerfile()
    demo_version()

    # ── 架构总结 ──────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 Agent 生产化清单:")
    print()
    print("  阶段        │ 组件                │ 生产工具")
    print("  ────────────┼────────────────────┼────────────────────")
    print("  配置管理     │ ConfigManager       │ dotenv / Consul / NACOS")
    print("  密钥管理     │ 加密存储            │ Vault / Akeyless / AWS SM")
    print("  健康检查     │ HealthCheck 探针    │ K8s probes / Spring Actuator")
    print("  优雅关闭     │ SIGTERM handler     │ K8s preStop / uvicorn")
    print("  容器化       │ Dockerfile          │ Docker / Podman / Buildah")
    print("  版本管理     │ VersionManager      │ Git tag / MLflow / W&B")
    print("  并发控制     │ Worker Pool         │ uvicorn workers / Gunicorn")
    print()
    print("  部署平台:")
    print("  - Kubernetes: 企业首选，HPA 自动扩缩")
    print("  - AWS ECS / Azure Container Apps: 托管容器")
    print("  - Modal / Fly.io: Serverless 容器")
    print("  - LangServe: LangChain 官方部署方案")


if __name__ == "__main__":
    main()
