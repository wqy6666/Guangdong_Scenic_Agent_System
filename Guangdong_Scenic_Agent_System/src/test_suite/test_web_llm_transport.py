"""Exercise the real OpenAI SDK with an in-process HTTP transport; no network."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".web-deps"))
import httpx
from openai import OpenAI
from llm_answer import generate_answer_by_api
from service_errors import LLMServiceError, LLMOutputError, strict_llm_errors


class ModelTransportTests(unittest.TestCase):
    def call_model(self, handler, base_url="https://model.test/v1", **options):
        self.http = httpx.Client(transport=httpx.MockTransport(handler))
        client = OpenAI(api_key="test-secret", base_url=base_url, http_client=self.http, max_retries=0)
        token = strict_llm_errors.set(True)
        try:
            with patch("llm_answer.OpenAI", return_value=client) as create_client:
                result = generate_answer_by_api(
                    [{"role": "user", "content": "介绍白云山"}],
                    {"api_key": "test-secret", "base_url": base_url, "model_name": "model-test"}, **options,
                )
                if options.get("purpose") in ("verification", "rewrite"):
                    self.assertEqual(create_client.call_args.kwargs["max_retries"], 0)
                return result
        finally:
            strict_llm_errors.reset(token)
            self.assertTrue(self.http.is_closed)

    def test_streamed_answer_and_config_reach_real_sdk(self):
        def reply(request):
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-secret")
            body = json.loads(request.content)
            self.assertEqual(body["model"], "model-test")
            self.assertTrue(body["stream"])
            self.assertEqual(body["messages"][0]["content"], "介绍白云山")
            chunks = [{"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "model-test",
                       "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                      for text in ("白云山", "位于广州。")]
            stream = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=stream)
        self.assertEqual(self.call_model(reply), "白云山位于广州。")

    def test_provider_errors_are_actionable_and_do_not_echo_secrets(self):
        for status, message in ((401, "密钥"), (403, "权限"), (402, "余额不足"), (429, "限流"), (404, "模型名称"), (400, "服务地址"), (500, "暂时不可用")):
            with self.subTest(status=status):
                with self.assertRaises(LLMServiceError) as caught:
                    self.call_model(lambda request: httpx.Response(status, json={"error": {"message": "sensitive-provider-error test-secret"}}))
                self.assertIn(message, str(caught.exception))
                self.assertNotIn("test-secret", str(caught.exception))

    def test_windows_network_permission_failure_has_a_specific_recovery(self):
        def blocked(request):
            cause = OSError(10013, "sensitive-path test-secret")
            cause.winerror = 10013
            raise httpx.ConnectError("sensitive-provider-error", request=request) from cause
        with self.assertLogs("travel_agent", level="WARNING") as logs:
            with self.assertRaisesRegex(LLMServiceError, "被系统禁止联网") as caught:
                self.call_model(blocked)
        self.assertIn("start_web.ps1", str(caught.exception))
        self.assertIn("10013", "".join(logs.output))
        self.assertNotIn("test-secret", "".join(logs.output))
        self.assertNotIn("sensitive-path", "".join(logs.output))

    def test_timeout_and_empty_stream_fail_without_a_fake_answer(self):
        def timeout(request):
            raise httpx.ReadTimeout("sensitive-provider-error", request=request)
        with self.assertRaisesRegex(LLMServiceError, "超时"):
            self.call_model(timeout)
        with self.assertRaisesRegex(LLMServiceError, "空回答"):
            self.call_model(lambda request: httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text="data: [DONE]\n\n"))

    @staticmethod
    def streamed_response(content="", reasoning="", finish="stop"):
        chunk = {"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "model-test",
                 "choices": [{"index": 0, "delta": {"content": content, "reasoning_content": reasoning}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 500, "total_tokens": 600,
                           "completion_tokens_details": {"reasoning_tokens": 480}}}
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
                              text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n")

    def test_thinking_is_disabled_only_for_deepseek_verification_and_rewrite(self):
        for base_url, purpose, disabled in (("https://api.deepseek.com", "verification", True),
                                            ("https://api.deepseek.com", "rewrite", True),
                                            ("https://api.deepseek.com/v1", "rewrite", True),
                                            ("https://model.test/v1", "rewrite", False),
                                            ("https://model.test/v1", "verification", False),
                                            ("https://api.deepseek.com", "answer", False)):
            with self.subTest(base_url=base_url, purpose=purpose):
                def reply(request):
                    body = json.loads(request.content)
                    self.assertEqual(body.get("thinking"), {"type": "disabled"} if disabled else None)
                    self.assertEqual(body["max_tokens"], 1500)
                    return self.streamed_response(content="正常正文", reasoning="private-reasoning")
                self.assertEqual(self.call_model(reply, base_url, purpose=purpose, max_tokens=1500), "正常正文")

    def test_truncated_and_reasoning_only_streams_are_not_accepted(self):
        for content, finish, expected_reason in (("", "length", "length"), ("半截答案", "length", "length"), ("", "stop", "empty")):
            with self.subTest(content=content, finish=finish):
                with self.assertLogs("travel_agent", level="INFO") as logs:
                    with self.assertRaises(LLMOutputError) as caught:
                        self.call_model(lambda request: self.streamed_response(content, "private-reasoning", finish))
                self.assertEqual(caught.exception.reason, expected_reason)
                logged = "".join(logs.output)
                self.assertIn("completion_tokens=500", logged)
                self.assertIn("reasoning_tokens=480", logged)
                self.assertNotIn("private-reasoning", logged)
                self.assertNotIn("test-secret", logged)


if __name__ == "__main__":
    unittest.main(verbosity=2)
