"""Adapt existing console services for a single-process web server."""
import json
import hashlib
import logging
import queue
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import DEFAULT_LLM_TEMPLATE, ADMIN_USERNAME
from file_store import write_json
from protocol1 import build_success_response
from service_errors import strict_llm_errors, LLMServiceError
from session_memory_manager import SessionMemoryManager
from user_account import UserAccount, validate_user_id, validate_password, USER_ROOT_DIR
from security_controls import validate_model_endpoint
from operations_monitor import (
    error_report, record_error, record_request_result, request_report, system_snapshot, usage_report,
)


class ServiceError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message
        super().__init__(message)


class AgentEngine:
    def __init__(self):
        # Import once at application startup; loads the local models once.
        import main_agent
        self.agent = main_agent

    def query(self, key, question, config, manager):
        try:
            if key not in self.agent.SESSION_STORE:
                session, _ = self.agent.get_or_create_session(key)
                self.agent.restore_session_context(key, session["session_ctx"], manager.load_recent_paras(5))
                session["session_ctx"].update(manager.load_context())
            answer, docs, _, meta = self.agent.agent_chat(
                question, config, key, curr_mem_mgr=manager, return_metadata=True, raise_errors=True,
            )
            manager.save_context(self.agent.SESSION_STORE[key]["session_ctx"])
            return answer, docs, meta
        except Exception:
            # A failed query must not leave partial routing state for the next turn.
            self.agent.SESSION_STORE.pop(key, None)
            self.agent.memory_manager.clear_session(key)
            raise

    def forget_session(self, key):
        # The conversation was deleted on disk; drop its in-memory agent state too.
        self.agent.SESSION_STORE.pop(key, None)
        self.agent.memory_manager.clear_session(key)


class WebService:
    def __init__(self, engine_factory=AgentEngine):
        self.engine_factory = engine_factory
        self.engine = None
        self.engine_error = None
        self._guard = threading.Lock()
        self._locks = {}
        # Local embedding/reranker instances are shared. Serialize inference while
        # HTTP and account/history operations continue on separate worker threads.
        self._inference_lock = threading.Lock()
        # Site-wide statistics and feedback live next to the user directories.
        self._stats_lock = threading.Lock()
        self._feedback_lock = threading.Lock()
        self._stats_path = Path(USER_ROOT_DIR) / "site_stats.json"
        self._feedback_path = Path(USER_ROOT_DIR) / "feedback.json"
        self._job_queue = queue.Queue(maxsize=16)
        self._job_stop = threading.Event()
        self._job_worker = None

    def start(self):
        try:
            self.engine = self.engine_factory()
        except Exception as exc:
            self.engine_error = type(exc).__name__
            import logging
            logging.getLogger(__name__).exception("问答模型加载失败")
        self._recover_jobs()
        self._job_stop.clear()
        self._job_worker = threading.Thread(target=self._job_loop, name="scenic-chat-worker", daemon=True)
        self._job_worker.start()

    def stop(self):
        self._job_stop.set()
        if self._job_worker and self._job_worker.is_alive():
            self._job_worker.join(timeout=5)
        self._job_worker = None

    def lock(self, user_id):
        # Windows paths are case insensitive: Alice and alice share the same lock.
        with self._guard:
            return self._locks.setdefault(user_id.casefold(), threading.RLock())

    def account(self, user_id):
        if not UserAccount.check_user_exist(user_id):
            raise ServiceError(401, "账户不存在，请重新登录。")
        return UserAccount(user_id)

    def authenticate(self, user_id, password, register=False):
        user_id = validate_user_id(user_id)
        with self.lock(user_id):
            if register:
                validate_password(password)
                if self.is_admin(user_id):
                    raise ServiceError(403, "管理员账户只能由后台初始化命令创建或修改。")
                if UserAccount.find_existing_user(user_id):
                    raise ServiceError(409, "该用户名已存在，请登录或更换用户名。")
                user = UserAccount.create_user(user_id, password)
            else:
                user = UserAccount.login_user(user_id, password)
                if user is None:
                    raise ServiceError(401, "用户名或密码错误。")
                if user.is_disabled():
                    raise ServiceError(403, "该账户已被关停，请联系管理员。")
            return user.meta.get("user_id", user.user_id)

    def is_admin(self, user_id):
        return user_id.casefold() == ADMIN_USERNAME.casefold()

    def role(self, user_id):
        return "admin" if self.is_admin(user_id) else "user"

    @staticmethod
    def _read_json(path, default):
        try:
            with open(path, encoding="utf-8") as stream:
                return json.load(stream)
        except (OSError, ValueError):
            return default

    def _bump_stat(self, field):
        try:
            with self._stats_lock:
                data = self._read_json(self._stats_path, {})
                data[field] = int(data.get(field, 0)) + 1
                write_json(self._stats_path, data)
        except OSError:
            pass  # Statistics are best-effort; never block a page load or answer.

    def record_open(self):
        self._bump_stat("opens")

    def record_question(self):
        self._bump_stat("questions")

    def stats(self):
        data = self._read_json(self._stats_path, {})
        return {"opens": int(data.get("opens", 0)), "questions": int(data.get("questions", 0))}

    def add_feedback(self, user_id, content):
        entry = {"id": uuid.uuid4().hex, "username": user_id, "content": content,
                 "create_time": datetime.now(timezone.utc).isoformat()}
        with self._feedback_lock:
            items = self._read_json(self._feedback_path, [])
            items.append(entry)
            write_json(self._feedback_path, items)
        return entry

    def feedback(self):
        return list(reversed(self._read_json(self._feedback_path, [])))

    def users(self):
        root = Path(USER_ROOT_DIR)
        items = []
        usage_today = usage_report("today")["users"]
        usage_month = usage_report("month")["users"]
        for child in sorted(root.iterdir()) if root.is_dir() else []:
            meta_file = child / "user_meta.json"
            if not child.is_dir() or not meta_file.is_file():
                continue
            meta = self._read_json(meta_file, {})
            items.append({"username": meta.get("user_id") or child.name,
                          "create_time": meta.get("create_time", ""),
                          "disabled": bool(meta.get("disabled", False)),
                          "sessions": len(meta.get("session_list", [])),
                          "usage_today": usage_today.get(meta.get("user_id") or child.name, {}),
                          "usage_month": usage_month.get(meta.get("user_id") or child.name, {})})
        return items

    def system_status(self):
        return {**system_snapshot(self._job_queue.qsize(), self.engine is not None, self.engine_error),
                "requests_7d": request_report(7)}

    @staticmethod
    def errors(days=7, limit=100):
        return error_report(days, limit)

    @staticmethod
    def usage(period="today"):
        if period not in {"today", "month"}:
            raise ServiceError(400, "统计周期仅支持 today 或 month。")
        return usage_report(period)

    def set_user_disabled(self, target, disabled):
        target = validate_user_id(target)
        if self.is_admin(target):
            raise ServiceError(403, "管理员账户不能被关停。")
        with self.lock(target):
            if not UserAccount.check_user_exist(target):
                raise ServiceError(404, "用户不存在。")
            user = UserAccount(target)
            user.set_disabled(disabled)
        return user.meta.get("user_id", target)

    def reset_password(self, user_id, old_password, new_password):
        user_id = validate_user_id(user_id)
        try:
            validate_password(new_password)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from None
        with self.lock(user_id):
            # login_user returns None for both a missing account and a wrong
            # password, so the unified 401 never reveals which one failed.
            user = UserAccount.login_user(user_id, old_password)
            if user is None:
                raise ServiceError(401, "用户名或原密码不正确。")
            user.set_password(new_password)
        return user.meta.get("user_id", user.user_id)

    def save_identity_answers(self, user_id, full_name, birth_date, verification_phrase):
        values = [full_name.strip(), birth_date.strip(), verification_phrase.strip()]
        if any(values) and not all(values):
            raise ServiceError(400, "请填写全部三项验证信息，或全部留空以关闭密码找回。")
        with self.lock(user_id):
            user = self.account(user_id)
            user.set_identity_answers(*values)
        return {"configured": all(values)}

    def has_identity_answers(self, user_id):
        return self.account(user_id).has_identity_answers()

    def reset_password_by_answers(self, user_id, full_name, birth_date, verification_phrase, new_password):
        user_id = validate_user_id(user_id)
        if self.is_admin(user_id):
            raise ServiceError(403, "管理员账户不能通过公开找回流程重置，请在服务器运行 admin_init.py。")
        try:
            validate_password(new_password)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from None
        supplied = [full_name.strip(), birth_date.strip(), verification_phrase.strip()]
        with self.lock(user_id):
            try:
                user = self.account(user_id)
            except ServiceError:
                raise ServiceError(401, "验证信息不正确，或账户未设置找回答案。") from None
            if not user.has_identity_answers():
                raise ServiceError(401, "验证信息不正确，或账户未设置找回答案。")
            if not user.verify_identity_answers(*supplied):
                raise ServiceError(401, "验证信息不正确，或账户未设置找回答案。")
            user.set_password(new_password)
        return user.meta.get("user_id", user.user_id)

    @staticmethod
    def public_config(config):
        return {"base_url": config.get("base_url", ""), "model_name": config.get("model_name", ""),
                "configured": bool(config.get("api_key") and config.get("base_url") and config.get("model_name"))}

    def config(self, user_id):
        with self.lock(user_id):
            value, _ = self.account(user_id).load_user_llm_config()
            return self.public_config(value or DEFAULT_LLM_TEMPLATE)

    def save_config(self, user_id, base_url, model_name, api_key):
        base_url, model_name = base_url.strip().rstrip("/"), model_name.strip()
        try:
            base_url = validate_model_endpoint(base_url)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from None
        if not model_name:
            raise ServiceError(400, "请填写模型名称。")
        with self.lock(user_id):
            user = self.account(user_id)
            old, _ = user.load_user_llm_config()
            key = api_key.strip() if api_key else old.get("api_key", "")
            if not key:
                raise ServiceError(400, "首次配置需要填写 API 密钥。")
            config = {"base_url": base_url, "model_name": model_name, "api_key": key}
            ok, _ = user.save_user_llm_config(config)
            if not ok:
                raise ServiceError(500, "配置保存失败，请重试。")
            return self.public_config(config)

    def session_path(self, user, session_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", session_id) or session_id not in user.list_all_sessions():
            raise ServiceError(404, "会话不存在或不属于当前账户。")
        path = (Path(user.user_folder) / session_id).resolve()
        if path.parent != Path(user.user_folder).resolve() or not path.is_dir():
            raise ServiceError(404, "会话不存在。")
        return path

    def sessions(self, user_id):
        with self.lock(user_id):
            user = self.account(user_id)
            items = []
            for sid in user.list_all_sessions():
                try:
                    path = self.session_path(user, sid)
                except ServiceError:
                    continue
                history = path / "global_memory.txt"
                title = "新对话"
                if history.exists():
                    with history.open(encoding="utf-8") as stream:
                        title = stream.readline().removeprefix("用户：").strip()[:40] or title
                stamp = (history if history.exists() else path).stat().st_mtime
                items.append({"session_id": sid, "title": title,
                              "updated_at": datetime.fromtimestamp(stamp, timezone.utc).isoformat()})
            return sorted(items, key=lambda item: item["updated_at"], reverse=True)

    def create_session(self, user_id):
        with self.lock(user_id):
            return self.account(user_id).create_new_session()

    def delete_session(self, user_id, session_id):
        with self.lock(user_id):
            user = self.account(user_id)
            path = self.session_path(user, session_id)
            shutil.rmtree(path)
            user.remove_session(session_id)
        if self.engine is not None:
            self.engine.forget_session(f"{user_id.casefold()}:{session_id}")

    @staticmethod
    def _job_hash(request_id):
        return hashlib.sha256(request_id.encode()).hexdigest() + ".json"

    def _job_path(self, user, session_id, request_id):
        return Path(user.user_folder) / session_id / "web_jobs" / self._job_hash(request_id)

    @staticmethod
    def _job_public(job):
        result = {"request_id": job["request_id"], "status": job["status"]}
        if job.get("status") == "succeeded" and job.get("result") is not None:
            result["result"] = job["result"]
        if job.get("error"):
            result["error"] = job["error"]
        return result

    def _recover_jobs(self):
        root = Path(USER_ROOT_DIR)
        if not root.is_dir():
            return
        for path in root.glob("*/**/web_jobs/*.json"):
            job = self._read_json(path, {})
            if job.get("status") in {"queued", "running"}:
                job["status"] = "interrupted"
                job["error"] = {"status": 503, "message": "服务重启中断了本次问答，请重新发送。"}
                job["updated_at"] = datetime.now(timezone.utc).isoformat()
                try:
                    write_json(path, job)
                except OSError:
                    logging.getLogger(__name__).warning("无法更新中断任务: %s", path.name)

    def _job_loop(self):
        while not self._job_stop.is_set():
            try:
                job = self._job_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._run_job(job)
            finally:
                self._job_queue.task_done()

    def _run_job(self, job):
        started = time.perf_counter()
        path = Path(job["job_path"])
        # A queued job may race with session deletion. Never recreate a deleted
        # session directory just to persist a late worker state.
        if not path.parent.parent.is_dir():
            return
        job["status"] = "running"
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        job["updated_at"] = job["started_at"]
        write_json(path, {key: value for key, value in job.items() if key != "job_path"})
        try:
            wait_ms = (time.time() - job["queued_epoch"]) * 1000
            logging.getLogger(__name__).info("chat_job_start request_id=%s queue_wait_ms=%.0f", job["request_id"], wait_ms)
            result = self.chat(job["user_id"], job["session_id"], job["question"], job["request_id"])
            job["status"], job["result"], job["error"] = "succeeded", result, None
        except ServiceError as exc:
            job["status"], job["error"] = "failed", {"status": exc.status, "message": exc.message}
            try:
                record_error("ServiceError", exc.status, exc.message, stage="chat", user_id=job["user_id"], request_id=job["request_id"])
            except OSError:
                logging.getLogger(__name__).exception("failed to persist service error event")
        except LLMServiceError as exc:
            job["status"], job["error"] = "failed", {"status": 502, "message": str(exc)}
            try:
                record_error("LLMServiceError", 502, str(exc), stage="model", user_id=job["user_id"], request_id=job["request_id"])
            except OSError:
                logging.getLogger(__name__).exception("failed to persist model error event")
        except Exception as exc:
            logging.getLogger(__name__).exception("chat_job_failed request_id=%s", job["request_id"])
            job["status"], job["error"] = "failed", {"status": 500, "message": "服务处理失败，请稍后重试。"}
            try:
                record_error(type(exc).__name__, 500, "服务处理失败，请稍后重试。", stage="chat", user_id=job["user_id"], request_id=job["request_id"])
            except OSError:
                logging.getLogger(__name__).exception("failed to persist unhandled error event")
        try:
            record_request_result(job["status"], (time.perf_counter() - started) * 1000)
        except OSError:
            logging.getLogger(__name__).exception("failed to persist request metrics")
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        job.pop("job_path", None)
        job.pop("queued_epoch", None)
        try:
            if path.parent.parent.is_dir():
                write_json(path, job)
        except OSError:
            logging.getLogger(__name__).warning("无法保存任务结果: %s", path.name)

    def submit_chat(self, user_id, session_id, question, request_id):
        if self.engine is None:
            raise ServiceError(503, "问答服务未就绪，请检查后端启动日志后重启服务。")
        question = question.strip()
        if not question:
            raise ServiceError(400, "请输入问题。")
        with self.lock(user_id):
            user = self.account(user_id)
            session_path = self.session_path(user, session_id)
            response_path = session_path / "web_requests" / self._job_hash(request_id)
            if response_path.exists():
                previous = self._read_json(response_path, {})
                if previous.get("request_id") == request_id:
                    if previous.get("data", {}).get("query") != question:
                        raise ServiceError(409, "请求编号已用于其他问题。")
                    return {"request_id": request_id, "status": "succeeded", "result": previous}
            config, _ = user.load_user_llm_config()
            if not self.public_config(config)["configured"]:
                raise ServiceError(409, "请先配置模型服务。")
            path = self._job_path(user, session_id, request_id)
            existing = self._read_json(path, {}) if path.exists() else None
            if existing:
                if existing.get("question") != question:
                    raise ServiceError(409, "请求编号已用于其他问题。")
                if existing.get("status") in {"queued", "running"}:
                    return self._job_public(existing)
            job = {
                "request_id": request_id, "user_id": user.user_id, "session_id": session_id,
                "question": question, "status": "queued",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(), "error": None,
                "queued_epoch": time.time(), "job_path": str(path),
            }
            write_json(path, {key: value for key, value in job.items() if key not in {"job_path", "queued_epoch"}})
            try:
                self._job_queue.put_nowait(job)
            except queue.Full:
                job["status"] = "failed"
                job["error"] = {"status": 429, "message": "当前问答任务较多，请稍后重试。"}
                job.pop("job_path", None)
                job.pop("queued_epoch", None)
                write_json(path, job)
                raise ServiceError(429, job["error"]["message"]) from None
            return self._job_public(job)

    def chat_job(self, user_id, session_id, request_id):
        user = self.account(user_id)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", session_id):
            raise ServiceError(404, "会话不存在或不属于当前账户。")
        user_root = Path(user.user_folder).resolve()
        session_path = (user_root / session_id).resolve()
        if session_path.parent != user_root or not session_path.is_dir():
            raise ServiceError(404, "会话不存在或不属于当前账户。")
        # Job files are atomically replaced; polling must not wait for the user
        # lock held across a potentially minutes-long model inference.
        job = self._read_json(session_path / "web_jobs" / self._job_hash(request_id), {})
        if not job or job.get("user_id", "").casefold() != user.user_id.casefold():
            raise ServiceError(404, "问答任务不存在。")
        return self._job_public(job)

    def history(self, user_id, session_id, before=None):
        with self.lock(user_id):
            path = self.session_path(self.account(user_id), session_id)
            manager = SessionMemoryManager(str(path))
            if before is None:
                paras = manager.load_recent_paras(6)
                end = manager.get_max_paragraph_id() + 1
                ids = list(range(end - len(paras), end))
            else:
                paras, ids = manager.load_prev_paras(before)
            messages = []
            for pid, text in zip(ids, paras):
                question, _, answer = text.partition("\n助手：")
                messages.extend([
                    {"id": f"{pid}-user", "role": "user", "text": question.removeprefix("用户：")},
                    {"id": f"{pid}-assistant", "role": "assistant", "text": answer},
                ])
            oldest = ids[0] if ids else None
            return {"messages": messages, "before": oldest, "has_more": oldest is not None and oldest > 0}

    def chat(self, user_id, session_id, question, request_id):
        if self.engine is None:
            raise ServiceError(503, "问答服务未就绪，请检查后端启动日志后重启服务。")
        question = question.strip()
        if not question:
            raise ServiceError(400, "请输入问题。")
        started = time.perf_counter()
        with self.lock(user_id):
            user = self.account(user_id)
            path = self.session_path(user, session_id)
            response_path = path / "web_requests" / (hashlib.sha256(request_id.encode()).hexdigest() + ".json")
            if response_path.exists():
                with response_path.open(encoding="utf-8") as stream:
                    previous = json.load(stream)
                if previous.get("request_id") == request_id:
                    if previous["data"]["query"] != question:
                        raise ServiceError(409, "请求编号已用于其他问题。")
                    return previous
            config, _ = user.load_user_llm_config()
            if not self.public_config(config)["configured"]:
                raise ServiceError(409, "请先配置模型服务。")
            manager = SessionMemoryManager(str(path))
            # Same legacy session IDs in different accounts must never share memory.
            memory_key = f"{user_id.casefold()}:{session_id}"
            token = strict_llm_errors.set(True)
            from operations_monitor import reset_request_context, set_request_context
            request_token = set_request_context(user.meta.get("user_id", user.user_id), request_id)
            try:
                # Count the question once it is actually handed to the engine;
                # idempotent replays above never reach this line.
                self.record_question()
                with self._inference_lock:
                    answer, docs, meta = self.engine.query(memory_key, question, config, manager)
            finally:
                reset_request_context(request_token)
                strict_llm_errors.reset(token)
            result = build_success_response(request_id=request_id, user_query=question,
                                            session_id=session_id, answer=answer, docs=docs, query_meta=meta)
            write_json(response_path, result)
            logging.getLogger(__name__).info("chat_total_ms=%.0f request_id=%s", (time.perf_counter() - started) * 1000, request_id)
            return result
