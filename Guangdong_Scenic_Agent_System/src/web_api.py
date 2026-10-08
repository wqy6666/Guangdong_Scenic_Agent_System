"""Same-origin HTTP API and frontend. Run with python src/web_api.py."""
import os
import secrets
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

# Optional project-local web dependencies; leaves the existing model environment intact.
ROOT = Path(__file__).resolve().parent.parent
if (ROOT / ".web-deps").is_dir():
    sys.path.insert(0, str(ROOT / ".web-deps"))

from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from protocol1 import build_error_response
from service_errors import LLMServiceError
from web_service import WebService, ServiceError
from security_controls import (
    CHAT_LIMITER, LOGIN_LIMITER, RECOVERY_LIMITER, REGISTER_LIMITER,
    PUBLIC_MODE, RateLimitExceeded,
)
from operations_monitor import record_error, record_request_result

COOKIE_NAME = "scenic_session"
AUTH_TTL = 8 * 60 * 60


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class PasswordReset(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    old_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=8, max_length=256)


class IdentityAnswers(BaseModel):
    full_name: str = Field(max_length=200)
    birth_date: str = Field(max_length=40)
    verification_phrase: str = Field(max_length=500)


class PasswordResetByAnswers(IdentityAnswers):
    username: str = Field(min_length=1, max_length=64)
    new_password: str = Field(min_length=8, max_length=256)


class FeedbackRequest(BaseModel):
    content: str = Field(min_length=1, max_length=1000)


class ModelSettings(BaseModel):
    base_url: str = Field(min_length=1, max_length=2048)
    model_name: str = Field(min_length=1, max_length=200)
    api_key: str | None = Field(default=None, max_length=4096)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex, pattern=r"^[A-Za-z0-9_-]{1,64}$")


def create_app(service=None):
    service = service or WebService()
    auth = {}
    auth_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        await run_in_threadpool(service.start)
        yield
        await run_in_threadpool(service.stop)
        auth.clear()

    app = FastAPI(title="粤游智答", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.service = service

    def error(status, message, request_id=None):
        return JSONResponse(status_code=status, content=build_error_response(
            request_id=request_id or uuid.uuid4().hex, code=status, message=message,
            error_type="service_error", retryable=status >= 500 or status == 429,
        ))

    @app.middleware("http")
    async def browser_boundary(request, call_next):
        if request.url.path.startswith("/api/") and request.method not in ("GET", "HEAD", "OPTIONS"):
            # A custom header + no CORS allowance prevents cross-site form requests.
            if request.headers.get("x-requested-with") != "scenic-web":
                return error(403, "请求来源无效，请从本网站操作。")
            origin = request.headers.get("origin")
            if origin:
                configured = {item.strip().rstrip("/") for item in os.environ.get("SCENIC_ALLOWED_ORIGINS", "").split(",") if item.strip()}
                expected = f"{request.url.scheme}://{request.url.netloc}"
                if (configured and origin.rstrip("/") not in configured) or (not configured and origin != expected):
                    return error(403, "不允许跨站请求。")
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return error(exc.status, exc.message)

    @app.exception_handler(LLMServiceError)
    async def model_error(request, exc):
        return error(502, str(exc))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Pydantic errors can include submitted passwords/keys; never echo them.
        return error(422, "提交内容格式不正确，请检查必填项和长度。")

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        import logging
        logging.getLogger(__name__).error("Web request failed: %s", type(exc).__name__)
        return error(500, "服务处理失败，请稍后重试。")

    def current_user(request: Request):
        token = request.cookies.get(COOKIE_NAME, "")
        with auth_lock:
            entry = auth.get(token)
            if not entry or entry[1] <= time.time():
                auth.pop(token, None)
                raise ServiceError(401, "登录已失效，请重新登录。")
            return entry[0]

    def request_ip(request):
        return request.client.host if request.client else "unknown"

    def enforce_limit(limiter, key):
        try:
            limiter.check(key)
        except RateLimitExceeded as exc:
            raise ServiceError(429, f"请求过于频繁，请在约 {exc.retry_after} 秒后重试。") from None

    def record_chat_metrics(outcome, duration_ms, *, user_id, request_id, error_info=None):
        import logging
        try:
            if error_info:
                record_error(error_info[0], error_info[1], error_info[2], stage=error_info[3],
                             user_id=user_id, request_id=request_id)
            record_request_result(outcome, duration_ms)
        except OSError:
            logging.getLogger(__name__).exception("Unable to persist chat monitoring metrics")

    def require_admin(user_id=Depends(current_user)):
        if not service.is_admin(user_id):
            raise ServiceError(403, "需要管理员权限。")
        return user_id

    def revoke_user_tokens(target):
        # Drop every session token previously issued to this account.
        with auth_lock:
            for key in [token for token, value in auth.items() if value[0].casefold() == target.casefold()]:
                auth.pop(key, None)

    def issue_login(request, response, user_id):
        token = secrets.token_urlsafe(32)
        with auth_lock:
            expired = [key for key, value in auth.items() if value[1] <= time.time()]
            for key in expired:
                auth.pop(key, None)
            auth.pop(request.cookies.get(COOKIE_NAME, ""), None)
            auth[token] = (user_id, time.time() + AUTH_TTL)
        response.set_cookie(COOKIE_NAME, token, max_age=AUTH_TTL, httponly=True,
                            secure=PUBLIC_MODE or request.url.scheme == "https", samesite="strict", path="/")
        return {"username": user_id, "role": service.role(user_id), "settings": service.config(user_id)}

    @app.get("/api/health")
    def health():
        return {"ready": service.engine is not None,
                "message": "问答服务就绪" if service.engine is not None else "问答服务未就绪，请检查启动日志"}

    @app.post("/api/auth/register")
    def register(data: Credentials, request: Request, response: Response):
        enforce_limit(REGISTER_LIMITER, request_ip(request))
        try:
            user_id = service.authenticate(data.username, data.password, register=True)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from None
        return issue_login(request, response, user_id)

    @app.post("/api/auth/login")
    def login(data: Credentials, request: Request, response: Response):
        enforce_limit(LOGIN_LIMITER, request_ip(request))
        try:
            user_id = service.authenticate(data.username, data.password)
        except ValueError:
            raise ServiceError(401, "用户名或密码错误。") from None
        return issue_login(request, response, user_id)

    @app.post("/api/auth/verify-identity")
    def verify_identity(data: Credentials, request: Request):
        enforce_limit(RECOVERY_LIMITER, f"{request_ip(request)}:{data.username.casefold()}")
        # Step 1 of the forgot-password flow: check the account and its
        # current password without issuing a session cookie.
        try:
            service.authenticate(data.username, data.password)
        except ValueError:
            raise ServiceError(401, "用户名或密码错误。") from None
        return {"ok": True}

    @app.post("/api/auth/reset-password")
    def reset_password(data: PasswordReset, request: Request):
        enforce_limit(RECOVERY_LIMITER, f"{request_ip(request)}:{data.username.casefold()}")
        # Step 2 re-submits username and old password with the new one, so the
        # server keeps no reset state between the two steps.
        try:
            user_id = service.reset_password(data.username, data.old_password, data.new_password)
        except ValueError:
            raise ServiceError(400, "用户名格式不正确。") from None
        # A new password revokes every session previously issued to the account.
        revoke_user_tokens(user_id)
        return {"ok": True}

    @app.post("/api/auth/reset-password-by-answers")
    def reset_password_by_answers(data: PasswordResetByAnswers, request: Request):
        enforce_limit(RECOVERY_LIMITER, f"{request_ip(request)}:{data.username.casefold()}")
        try:
            user_id = service.reset_password_by_answers(
                data.username, data.full_name, data.birth_date,
                data.verification_phrase, data.new_password,
            )
        except ValueError:
            raise ServiceError(400, "用户名格式不正确。") from None
        revoke_user_tokens(user_id)
        return {"ok": True}

    @app.post("/api/feedback")
    def submit_feedback(data: FeedbackRequest, user_id=Depends(current_user)):
        content = data.content.strip()
        if not content:
            raise ServiceError(400, "请填写反馈内容。")
        service.add_feedback(user_id, content)
        return {"ok": True}

    @app.get("/api/admin/summary")
    def admin_summary(user_id=Depends(require_admin)):
        return {**service.stats(), "users": len(service.users()), "feedback": len(service.feedback())}

    @app.get("/api/admin/users")
    def admin_users(user_id=Depends(require_admin)):
        return {"users": service.users()}

    @app.post("/api/admin/users/{username}/disable")
    def admin_disable(username: str, user_id=Depends(require_admin)):
        try:
            name = service.set_user_disabled(username, True)
        except ValueError:
            raise ServiceError(400, "用户名格式不正确。") from None
        revoke_user_tokens(name)
        return {"ok": True}

    @app.post("/api/admin/users/{username}/activate")
    def admin_activate(username: str, user_id=Depends(require_admin)):
        try:
            service.set_user_disabled(username, False)
        except ValueError:
            raise ServiceError(400, "用户名格式不正确。") from None
        return {"ok": True}

    @app.get("/api/admin/feedback")
    def admin_feedback(user_id=Depends(require_admin)):
        return {"feedback": service.feedback()}

    @app.get("/api/admin/system")
    def admin_system(user_id=Depends(require_admin)):
        return service.system_status()

    @app.get("/api/admin/errors")
    def admin_errors(days: int = Query(default=7, ge=1, le=90), limit: int = Query(default=100, ge=1, le=200),
                     user_id=Depends(require_admin)):
        return service.errors(days, limit)

    @app.get("/api/admin/usage")
    def admin_usage(period: str = Query(default="today", pattern="^(today|month)$"),
                    user_id=Depends(require_admin)):
        return service.usage(period)

    @app.get("/api/auth/me")
    def me(user_id=Depends(current_user)):
        return {"username": user_id, "role": service.role(user_id), "settings": service.config(user_id)}

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response):
        with auth_lock:
            auth.pop(request.cookies.get(COOKIE_NAME, ""), None)
        response.delete_cookie(COOKIE_NAME, path="/")
        return {"ok": True}

    @app.get("/api/settings/llm")
    def settings(user_id=Depends(current_user)):
        return service.config(user_id)

    @app.put("/api/settings/llm")
    def save_settings(data: ModelSettings, user_id=Depends(current_user)):
        try:
            return service.save_config(user_id, data.base_url, data.model_name, data.api_key)
        except ValueError:
            raise ServiceError(400, "模型服务地址格式不正确。") from None

    @app.get("/api/settings/security-questions")
    def security_questions(user_id=Depends(current_user)):
        return {"configured": service.has_identity_answers(user_id)}

    @app.put("/api/settings/security-questions")
    def save_security_questions(data: IdentityAnswers, user_id=Depends(current_user)):
        return service.save_identity_answers(
            user_id, data.full_name, data.birth_date, data.verification_phrase,
        )

    @app.get("/api/sessions")
    def sessions(user_id=Depends(current_user)):
        return {"sessions": service.sessions(user_id)}

    @app.post("/api/sessions")
    def new_session(user_id=Depends(current_user)):
        return {"session_id": service.create_session(user_id)}

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str, user_id=Depends(current_user)):
        service.delete_session(user_id, session_id)
        return {"ok": True}

    @app.get("/api/sessions/{session_id}/messages")
    def messages(session_id: str, before: int | None = Query(default=None, ge=0), user_id=Depends(current_user)):
        return service.history(user_id, session_id, before)

    @app.post("/api/chat")
    def chat(data: ChatRequest, user_id=Depends(current_user)):
        enforce_limit(CHAT_LIMITER, user_id.casefold())
        started = time.perf_counter()
        try:
            result = service.chat(user_id, data.session_id, data.message, data.request_id)
        except ServiceError as exc:
            record_chat_metrics("failed", (time.perf_counter() - started) * 1000, user_id=user_id,
                                request_id=data.request_id, error_info=("ServiceError", exc.status, exc.message, "chat"))
            raise
        except LLMServiceError as exc:
            record_chat_metrics("failed", (time.perf_counter() - started) * 1000, user_id=user_id,
                                request_id=data.request_id, error_info=("LLMServiceError", 502, str(exc), "model"))
            raise
        record_chat_metrics("succeeded", (time.perf_counter() - started) * 1000, user_id=user_id,
                            request_id=data.request_id)
        return result

    @app.post("/api/chat/jobs")
    def submit_chat_job(data: ChatRequest, user_id=Depends(current_user)):
        enforce_limit(CHAT_LIMITER, user_id.casefold())
        return service.submit_chat(user_id, data.session_id, data.message, data.request_id)

    @app.get("/api/chat/jobs/{session_id}/{request_id}")
    def chat_job(session_id: str, request_id: str, user_id=Depends(current_user)):
        return service.chat_job(user_id, session_id, request_id)

    @app.get("/")
    def index():
        service.record_open()
        return FileResponse(ROOT / "frontend" / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/admin")
    def admin_page(user_id=Depends(require_admin)):
        return FileResponse(ROOT / "frontend" / "admin.html", headers={"Cache-Control": "no-cache"})

    @app.get("/styles.css")
    def styles():
        return FileResponse(ROOT / "frontend" / "styles.css", media_type="text/css")

    @app.get("/app.js")
    def javascript():
        return FileResponse(ROOT / "frontend" / "app.js", media_type="text/javascript")

    @app.get("/admin.js")
    def admin_javascript():
        return FileResponse(ROOT / "frontend" / "admin.js", media_type="text/javascript")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    # One worker owns the in-memory logins, contexts and inference lock.
    uvicorn.run(app, host=os.environ.get("SCENIC_HOST", "0.0.0.0"),
                port=int(os.environ.get("SCENIC_PORT", "8000")))
