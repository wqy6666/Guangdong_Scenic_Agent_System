"""HTTP integration tests using real account/history storage and a deterministic engine.

Run: python -B src/test_suite/test_web_api.py
Temporary accounts never touch the project's users directory or call an external model.
"""
import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".web-deps"))
from fastapi.testclient import TestClient
from web_api import create_app
from web_service import WebService
from user_account import UserAccount
from session_memory_manager import SessionMemoryManager
from service_errors import LLMServiceError, strict_llm_errors
from protocol1 import normalize_spots
from file_store import write_json
from security_controls import CHAT_LIMITER, LOGIN_LIMITER, RECOVERY_LIMITER, REGISTER_LIMITER


class DeterministicEngine:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.max_active = 0

    def query(self, key, question, config, manager):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append((key, question))
            assert strict_llm_errors.get()
            if question == "模型失败":
                raise LLMServiceError("模型服务连接失败或超时，请检查配置后重试。")
            time.sleep(.01)
            previous = manager.load_recent_paras(1)
            answer = f"回答：{question}；上一轮：{previous[0] if previous else '无'}"
            manager.append_dialogue_paragraph(question, answer, [question])
            return answer, [], {"route": "rag", "spots": ["丹霞山", {"spot_name": "丹霞山", "city": "韶关"}]}
        finally:
            self.active -= 1


class WebIntegrationTests(unittest.TestCase):
    def setUp(self):
        for limiter in (CHAT_LIMITER, LOGIN_LIMITER, RECOVERY_LIMITER, REGISTER_LIMITER):
            with limiter._lock:
                limiter._entries.clear()
        self.temp = tempfile.TemporaryDirectory(prefix="scenic-web-test-")
        self.root_patch = patch("user_account.USER_ROOT_DIR", self.temp.name)
        self.root_patch.start()
        self.engine = DeterministicEngine()
        self.service = WebService(lambda: self.engine)
        self.client = TestClient(create_app(self.service), raise_server_exceptions=False)
        self.client.__enter__()
        self.client.headers["X-Requested-With"] = "scenic-web"

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.root_patch.stop()
        self.temp.cleanup()

    def register(self, name="alice"):
        result = self.client.post("/api/auth/register", json={"username": name, "password": "password123"})
        self.assertEqual(result.status_code, 200, result.text)
        return result

    def configure(self):
        response = self.client.put("/api/settings/llm", json={"base_url": "https://example.com/v1", "model_name": "test-model", "api_key": "test-secret-never-return"})
        self.assertEqual(response.status_code, 200, response.text)

    def new_session(self):
        return self.client.post("/api/sessions").json()["session_id"]

    def chat(self, sid, text="你好", request_id="req-1"):
        return self.client.post("/api/chat", json={"session_id": sid, "message": text, "request_id": request_id})

    def test_registration_login_logout_and_no_secret_exposure(self):
        response = self.register()
        self.assertIn("httponly", response.headers["set-cookie"].lower())
        self.assertIn("samesite=strict", response.headers["set-cookie"].lower())
        self.configure()
        for endpoint in ("/api/auth/me", "/api/settings/llm"):
            response = self.client.get(endpoint)
            self.assertNotIn("test-secret", response.text)
            self.assertNotIn("password_hash", response.text)
        self.client.post("/api/auth/logout")
        self.assertEqual(self.client.get("/api/sessions").status_code, 401)
        bad = self.client.post("/api/auth/login", json={"username": "alice", "password": "wrong"})
        self.assertEqual(bad.status_code, 401)
        good = self.client.post("/api/auth/login", json={"username": "alice", "password": "password123"})
        self.assertTrue(good.json()["settings"]["configured"])

    def test_settings_blank_key_preserves_existing_and_bad_url_is_rejected(self):
        self.register(); self.configure()
        saved = self.client.put("/api/settings/llm", json={"base_url": "https://example.com/v2", "model_name": "new-model"})
        self.assertTrue(saved.json()["configured"])
        self.assertEqual(UserAccount("alice").load_user_llm_config()[0]["api_key"], "test-secret-never-return")
        rejected = self.client.put("/api/settings/llm", json={"base_url": "file:///secret", "model_name": "x"})
        self.assertEqual(rejected.status_code, 400)

    def test_unauthenticated_and_cross_site_requests_are_blocked(self):
        self.assertEqual(self.client.post("/api/sessions").status_code, 401)
        data = {"username": "alice", "password": "password123"}
        self.assertEqual(self.client.post("/api/auth/register", json=data, headers={"Origin": "https://evil.example"}).status_code, 403)
        self.assertEqual(self.client.post("/api/auth/register", json=data, headers={"X-Requested-With": ""}).status_code, 403)

    def test_username_paths_duplicate_registration_and_short_password(self):
        for name in ("../escape", "admin/foo", "CON", "..", "admin:other"):
            response = self.client.post("/api/auth/register", json={"username": name, "password": "password123"})
            self.assertEqual(response.status_code, 400, response.text)
        with REGISTER_LIMITER._lock:
            REGISTER_LIMITER._entries.clear()
        self.register()
        self.assertEqual(self.client.post("/api/auth/register", json={"username": "alice", "password": "password123"}).status_code, 409)
        self.assertEqual(self.client.post("/api/auth/register", json={"username": "bob", "password": "123"}).status_code, 400)

    def test_chat_history_pagination_retry_and_secret_free_protocol(self):
        self.register(); self.configure(); sid = self.new_session()
        for index in range(8):
            response = self.chat(sid, f"问题{index}", f"req-{index}")
            self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["data"]["session_id"], sid)
        self.assertEqual(result["data"]["spots"], [{"spot_name": "丹霞山", "city": "韶关", "level": None, "address": None, "ticket": None, "open_time": None, "telephone": None}])
        self.assertNotIn("test-secret", response.text)
        count = len(self.engine.calls)
        self.assertEqual(self.chat(sid, "问题7", "req-7").json(), result)
        self.assertEqual(len(self.engine.calls), count)
        self.assertEqual(self.chat(sid, "不同问题", "req-7").status_code, 409)
        self.assertEqual(self.chat(sid, "问题0", "req-0").json()["data"]["query"], "问题0")
        self.assertEqual(len(self.engine.calls), count)
        history = self.client.get(f"/api/sessions/{sid}/messages").json()
        self.assertEqual(len(history["messages"]), 12)
        self.assertEqual(history["messages"][0]["text"], "问题2")
        older = self.client.get(f"/api/sessions/{sid}/messages?before={history['before']}").json()
        self.assertEqual(len(older["messages"]), 4)
        self.assertFalse(older["has_more"])
        self.assertEqual(self.client.get("/api/sessions").json()["sessions"][0]["title"], "问题0")

    def test_session_ownership_and_identical_legacy_ids_are_isolated(self):
        self.register(); self.configure()
        alice = UserAccount("alice"); alice.create_new_session("session_shared")
        self.assertEqual(self.chat("session_shared", "alice问题").status_code, 200)
        alice_only = self.new_session()
        self.register("bob"); self.configure()
        bob = UserAccount("bob"); bob.create_new_session("session_shared")
        self.assertEqual(self.client.get(f"/api/sessions/{alice_only}/messages").status_code, 404)
        self.assertEqual(self.chat(alice_only).status_code, 404)
        response = self.chat("session_shared", "bob问题")
        self.assertNotIn("alice问题", response.text)
        self.assertEqual(self.engine.calls[0][0], "alice:session_shared")
        self.assertEqual(self.engine.calls[-1][0], "bob:session_shared")

    def test_empty_unconfigured_and_model_failure_do_not_append_history(self):
        self.register(); sid = self.new_session()
        self.assertEqual(self.chat(sid).status_code, 409)
        self.configure()
        self.assertEqual(self.chat(sid, "   ").status_code, 400)
        result = self.chat(sid, "模型失败")
        self.assertEqual(result.status_code, 502)
        self.assertTrue(result.json()["error"]["retryable"])
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/messages").json()["messages"], [])

    def test_unique_session_ids_and_serial_inference(self):
        self.register(); self.configure()
        sids = [self.new_session() for _ in range(3)]
        self.assertEqual(len(set(sids)), 3)
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda item: self.service.chat("alice", item[1], f"q{item[0]}", f"r{item[0]}"), enumerate(sids)))
        self.assertEqual(len(results), 3)
        self.assertEqual(self.engine.max_active, 1)

    def test_chat_job_poll_does_not_wait_for_user_inference_lock(self):
        self.register(); self.configure(); sid = self.new_session()
        request_id = "poll-while-running"
        user = UserAccount("alice")
        write_json(self.service._job_path(user, sid, request_id), {
            "request_id": request_id, "user_id": "alice", "session_id": sid,
            "status": "running", "error": None,
        })
        lock = self.service.lock("alice")
        lock.acquire()
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self.client.get, f"/api/chat/jobs/{sid}/{request_id}")
        response = None
        try:
            try:
                response = future.result(timeout=.5)
            except FutureTimeoutError:
                pass
        finally:
            lock.release()
        if response is None:
            response = future.result(timeout=2)
        pool.shutdown(wait=True)
        self.assertIsNotNone(response, "状态轮询被问答执行期间的用户锁阻塞")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "running")

    def test_persistence_survives_service_restart(self):
        self.register(); self.configure(); sid = self.new_session(); self.chat(sid)
        restarted = WebService(DeterministicEngine); restarted.start()
        self.assertTrue(restarted.config("alice")["configured"])
        self.assertEqual(restarted.history("alice", sid)["messages"][0]["text"], "你好")
        result = restarted.chat("alice", sid, "继续", "restart-1")
        self.assertIn("你好", result["data"]["answer"])

    def test_context_persists_only_conversation_fields(self):
        manager = SessionMemoryManager(self.temp.name)
        manager.save_context({"city": "广州", "candidate_spot_list": ["白云山", "长隆"],
                              "list_pages": {"city": "广州", "offset": 15}, "llm_cfg": {"api_key": "secret"}})
        result = SessionMemoryManager(self.temp.name).load_context()
        self.assertEqual(result["candidate_spot_list"], ["白云山", "长隆"])
        self.assertEqual(result["list_pages"]["offset"], 15)
        self.assertNotIn("secret", json.dumps(result))

    def test_existing_sha256_password_is_accepted_and_upgraded(self):
        self.register()
        user = UserAccount("alice")
        user.meta["password_hash"] = hashlib.sha256(b"legacy-password").hexdigest(); user._save_meta()
        self.assertIsNotNone(UserAccount.login_user("alice", "legacy-password"))
        self.assertTrue(UserAccount("alice").meta["password_hash"].startswith("pbkdf2_sha256$"))

    def test_forgot_password_verifies_old_password_resets_and_revokes_sessions(self):
        self.register(); self.configure()
        verified = self.client.post("/api/auth/verify-identity", json={"username": "alice", "password": "password123"})
        self.assertEqual(verified.status_code, 200, verified.text)
        self.assertTrue(verified.json()["ok"])
        self.assertNotIn("set-cookie", verified.headers)  # verification must not start a session
        wrong = self.client.post("/api/auth/verify-identity", json={"username": "alice", "password": "wrong-pass"})
        unknown = self.client.post("/api/auth/verify-identity", json={"username": "ghost", "password": "whatever1"})
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(unknown.status_code, 401)
        # A wrong password and an unknown account share one message.
        self.assertEqual(wrong.json()["message"], unknown.json()["message"])
        blocked = self.client.post("/api/auth/reset-password",
                                   json={"username": "alice", "old_password": "password123", "new_password": "newpassword1"},
                                   headers={"X-Requested-With": ""})
        self.assertEqual(blocked.status_code, 403)
        short = self.client.post("/api/auth/reset-password",
                                 json={"username": "alice", "old_password": "password123", "new_password": "short7"})
        self.assertEqual(short.status_code, 422)
        self.assertNotIn("short7", short.text)
        bad_old = self.client.post("/api/auth/reset-password",
                                   json={"username": "alice", "old_password": "wrong-pass", "new_password": "newpassword1"})
        self.assertEqual(bad_old.status_code, 401)
        reset = self.client.post("/api/auth/reset-password",
                                 json={"username": "alice", "old_password": "password123", "new_password": "newpassword1"})
        self.assertEqual(reset.status_code, 200, reset.text)
        self.assertTrue(reset.json()["ok"])
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)  # sessions issued before the reset are revoked
        self.assertEqual(self.client.post("/api/auth/login", json={"username": "alice", "password": "password123"}).status_code, 401)
        good = self.client.post("/api/auth/login", json={"username": "alice", "password": "newpassword1"})
        self.assertEqual(good.status_code, 200, good.text)
        self.assertTrue(good.json()["settings"]["configured"])

    def test_static_files_are_scoped_and_validation_does_not_echo_inputs(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/app.js").status_code, 200)
        self.assertEqual(self.client.get("/users/alice/llm_config.json").status_code, 404)
        self.assertEqual(self.client.get("/src/main_agent.py").status_code, 404)
        result = self.client.post("/api/auth/login", json={"username": [], "password": "sensitive-input"})
        self.assertEqual(result.status_code, 422)
        self.assertNotIn("sensitive-input", result.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
