"""Small, dependency-free security controls shared by the web boundary and LLM client."""
import ipaddress
import os
import socket
import threading
import time
from urllib.parse import urlsplit


PUBLIC_MODE = os.environ.get("SCENIC_PUBLIC_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name, default, minimum=1):
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


class RateLimitExceeded(Exception):
    def __init__(self, retry_after):
        self.retry_after = max(1, int(retry_after))
        super().__init__("rate limit exceeded")


class FixedWindowLimiter:
    """Bounded fixed-window limiter suitable for the app's single worker."""

    def __init__(self, limit, window_seconds, max_keys=10000):
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._entries = {}
        self._lock = threading.Lock()

    def check(self, key):
        with self._lock:
            now = time.monotonic()
            start, count = self._entries.get(key, (now, 0))
            if now - start >= self.window_seconds:
                start, count = now, 0
            if count >= self.limit:
                raise RateLimitExceeded(self.window_seconds - (now - start))
            self._entries[key] = (start, count + 1)
            if len(self._entries) > self.max_keys:
                cutoff = now - self.window_seconds
                self._entries = {k: v for k, v in self._entries.items() if v[0] > cutoff}


LOGIN_LIMITER = FixedWindowLimiter(_int_env("SCENIC_LOGIN_LIMIT", 10), _int_env("SCENIC_LOGIN_WINDOW", 900))
REGISTER_LIMITER = FixedWindowLimiter(_int_env("SCENIC_REGISTER_LIMIT", 5), _int_env("SCENIC_REGISTER_WINDOW", 3600))
RECOVERY_LIMITER = FixedWindowLimiter(_int_env("SCENIC_RECOVERY_LIMIT", 5), _int_env("SCENIC_RECOVERY_WINDOW", 900))
CHAT_LIMITER = FixedWindowLimiter(_int_env("SCENIC_CHAT_LIMIT", 30), _int_env("SCENIC_CHAT_WINDOW", 60))


def allowed_model_hosts():
    configured = os.environ.get("SCENIC_ALLOWED_MODEL_HOSTS", "")
    if configured.strip():
        return {item.strip().lower().rstrip(".") for item in configured.split(",") if item.strip()}
    return {"api.deepseek.com", "api.openai.com"}


def validate_model_endpoint(base_url, *, public=None):
    """Reject malformed and, in public mode, user-controlled SSRF endpoints."""
    value = str(base_url or "").strip().rstrip("/")
    parsed = urlsplit(value)
    is_public = PUBLIC_MODE if public is None else public
    if parsed.scheme not in (("https",) if is_public else ("http", "https")):
        raise ValueError("公开部署只允许使用 HTTPS 模型服务地址。" if is_public else "请填写有效的模型服务地址（http 或 https）。")
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("请填写有效的模型服务地址（不得包含账号、密码、查询参数或片段）。")
    host = parsed.hostname.lower().rstrip(".")
    if is_public:
        if parsed.port not in (None, 443):
            raise ValueError("公开部署的模型服务地址只允许 HTTPS 默认端口。")
        if host not in allowed_model_hosts():
            raise ValueError("该模型服务地址不在允许列表中，请联系管理员配置。")
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
        except OSError:
            raise ValueError("无法解析模型服务地址，请检查域名。") from None
        for address in addresses:
            try:
                if not ipaddress.ip_address(address).is_global:
                    raise ValueError("模型服务地址解析到了非公网地址，已拒绝连接。")
            except ValueError as exc:
                if "模型服务地址" in str(exc):
                    raise
                raise ValueError("模型服务地址解析结果无效。") from None
    return value
