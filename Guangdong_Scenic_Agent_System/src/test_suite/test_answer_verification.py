"""Verification recovery tests: load local retrieval models, never call paid APIs."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".web-deps"))
import agent_reflection_loop as agent
import rag_hybrid_query as hybrid
from service_errors import LLMServiceError, LLMOutputError
from protocol1 import build_success_response


VERDICT = {"coverage": "fully", "missing_points": [], "unsupported_claims": [], "action": "done", "reason": "资料支持"}
CONFIG = {"base_url": "https://api.deepseek.com", "model_name": "deepseek-flash", "api_key": "test-only"}


def context():
    return {"session_alias_map": {}, "latest_subject": None, "candidate_spot_list": [], "city": None,
            "level": None, "last_intent_type": "", "user_preference": {}, "focus_mode": False,
            "focus_keep_round": 0, "allow_cross_city": False}


class VerificationTests(unittest.TestCase):
    def verify(self):
        return agent.verify_answer_quality("介绍丹霞山", ["介绍丹霞山"], "丹霞山以丹霞地貌著称。",
                                           "片段1：丹霞山以丹霞地貌著称。", context(), CONFIG)

    def test_empty_or_truncated_verification_retries_once(self):
        for first in ("", LLMOutputError("empty", "empty"), LLMOutputError("length", "length")):
            with self.subTest(first=str(first)):
                with patch.object(agent, "generate_answer_by_api", side_effect=[first, json.dumps(VERDICT)]) as generate:
                    self.assertEqual(self.verify(), VERDICT)
                self.assertEqual([call.kwargs["max_tokens"] for call in generate.call_args_list], [1500, 3000])
                self.assertTrue(all(call.kwargs["purpose"] == "verification" for call in generate.call_args_list))

    def test_exhausted_empty_retry_returns_unavailable(self):
        with patch.object(agent, "generate_answer_by_api", side_effect=LLMOutputError("empty", "empty")) as generate:
            self.assertIsNone(self.verify())
            self.assertEqual(generate.call_count, 2)

    def test_service_errors_do_not_retry_or_escape_verifier(self):
        for error in (LLMServiceError("无权限"), LLMOutputError("未正常结束", "incomplete")):
            with self.subTest(error=type(error).__name__):
                with patch.object(agent, "generate_answer_by_api", side_effect=error) as generate:
                    self.assertIsNone(self.verify())
                    self.assertEqual(generate.call_count, 1)

    def test_missing_or_invalid_verdict_is_never_treated_as_passed(self):
        invalid = ("{}", '{"coverage":"fully"}', "not-json", {**VERDICT, "missing_points": [None]},
                   {**VERDICT, "missing_points": [""]}, {**VERDICT, "unsupported_claims": [None]},
                   {**VERDICT, "coverage": "unknown"}, {**VERDICT, "action": "unknown"})
        for data in invalid:
            raw = json.dumps(data) if isinstance(data, dict) else data
            with self.subTest(raw=raw), patch.object(agent, "generate_answer_by_api", return_value=raw):
                self.assertIsNone(self.verify())

    def test_introduction_and_ticket_drafts_survive_failed_verification(self):
        for question, draft, spot in (("介绍一下丹霞山景区，有什么值得看的？", "丹霞山以丹霞地貌著称。", "韶关市丹霞山景区"),
                                      ("白云山的门票是多少", "门票信息：5元/人", "广州市白云山风景区")):
            with self.subTest(question=question):
                docs = [{"spot_name": spot, "chunk_text": draft, "source_url": "https://example.com/source"}]
                with patch.object(hybrid, "hybrid_query", return_value=(draft, docs, {"route": "rag", "spots": [spot]})), \
                        patch.object(agent, "_needs_query_planning", return_value=False), \
                        patch.object(agent, "generate_answer_by_api", side_effect=LLMOutputError("empty", "empty")) as generate:
                    answer, returned_docs, meta = agent.agent_reflective_query(question, [], context(), "verification-test", 100,
                                                                              CONFIG, return_metadata=True)
                self.assertIn(draft, answer)
                self.assertIn("未完成自动核验", answer)
                self.assertEqual(meta["answer_type"], "agent_unverified_answer")
                self.assertEqual(returned_docs, docs)
                self.assertEqual(generate.call_count, 2)  # no merge call after verification fails
                payload = build_success_response(request_id="test", user_query=question, session_id="test",
                                                 answer=answer, docs=docs, query_meta=meta)
                self.assertEqual(payload["data"]["answer_type"], "agent_unverified_answer")

    def test_successful_verification_keeps_normal_answer(self):
        docs = [{"spot_name": "韶关市丹霞山景区", "chunk_text": "丹霞山以丹霞地貌著称。"}]
        with patch.object(hybrid, "hybrid_query", return_value=(docs[0]["chunk_text"], docs, {"route": "rag", "spots": []})), \
                patch.object(agent, "_needs_query_planning", return_value=False), \
                patch.object(agent, "generate_answer_by_api", return_value=json.dumps(VERDICT)) as generate:
            answer, _, meta = agent.agent_reflective_query("介绍丹霞山", [], context(), "verified-test", 100,
                                                         CONFIG, return_metadata=True)
        self.assertEqual(answer, docs[0]["chunk_text"])
        self.assertEqual(meta["answer_type"], "agent_reflective_answer")
        self.assertEqual(generate.call_count, 1)

    def test_post_verification_failure_is_also_labelled(self):
        partial = {**VERDICT, "coverage": "partially", "missing_points": [{"point": "看点", "source": None}]}
        docs = [{"spot_name": "韶关市丹霞山景区", "chunk_text": "丹霞山资料"}]
        with patch.object(hybrid, "hybrid_query", return_value=("介绍草稿", docs, {"route": "rag", "spots": []})), \
                patch.object(agent, "_needs_query_planning", return_value=False), \
                patch.object(agent, "generate_answer_by_api", side_effect=[json.dumps(partial), "整理后的介绍", "", ""]) as generate:
            answer, _, meta = agent.agent_reflective_query("介绍丹霞山", [], context(), "post-test", 100,
                                                         CONFIG, return_metadata=True)
        self.assertIn("整理后的介绍", answer)
        self.assertIn("未完成自动核验", answer)
        self.assertEqual(meta["answer_type"], "agent_unverified_answer")
        self.assertEqual(generate.call_count, 4)

    def test_rewrite_retries_empty_or_truncated_output_then_verifies(self):
        question = "我带一个7岁的孩子去深圳玩"
        partial = {**VERDICT, "coverage": "partially", "missing_points": [{"point": "亲子看点", "source": None}]}
        docs = [{"spot_name": "深圳市仙湖植物园", "chunk_text": "植物园资料"}]
        for failure in (LLMOutputError("length", "length"), LLMOutputError("empty", "empty"), ""):
            with self.subTest(failure=str(failure)), \
                    patch.object(hybrid, "hybrid_query", return_value=("深圳亲子推荐草稿", docs, {"route": "recommendation"})), \
                    patch.object(agent, "_needs_query_planning", return_value=False), \
                    patch.object(agent, "generate_answer_by_api", side_effect=[json.dumps(partial), failure,
                                 "整理后的深圳亲子推荐", json.dumps(VERDICT)]) as generate:
                answer, returned_docs, meta = agent.agent_reflective_query(question, [], context(), "rewrite-test", 100,
                                                                         CONFIG, return_metadata=True)
            self.assertEqual(answer, "整理后的深圳亲子推荐")
            self.assertEqual(returned_docs, docs)
            self.assertEqual(meta["answer_type"], "agent_reflective_answer")
            calls = [c for c in generate.call_args_list if c.kwargs.get("purpose") == "rewrite"]
            self.assertEqual([c.kwargs["max_tokens"] for c in calls], [2000, 4000])
            self.assertEqual(generate.call_count, 4)

    def test_exhausted_rewrite_keeps_draft_and_known_verification_notes(self):
        partial = {**VERDICT, "coverage": "partially", "missing_points": [{"point": "儿童优惠", "source": None}],
                   "unsupported_claims": ["7岁儿童全部免费"]}
        docs = [{"spot_name": "深圳市仙湖植物园", "chunk_text": "植物园资料"}]
        for reason in ("length", "empty"):
            with self.subTest(reason=reason), \
                    patch.object(hybrid, "hybrid_query", return_value=("深圳亲子推荐草稿", docs, {"route": "recommendation"})), \
                    patch.object(agent, "_needs_query_planning", return_value=False), \
                    patch.object(agent, "generate_answer_by_api", side_effect=[json.dumps(partial),
                                 LLMOutputError(reason, reason), LLMOutputError(reason, reason)]) as generate:
                answer, returned_docs, meta = agent.agent_reflective_query("我带一个7岁的孩子去深圳玩", [], context(),
                                                                         "rewrite-failed", 100, CONFIG, return_metadata=True)
            self.assertIn("深圳亲子推荐草稿", answer)
            self.assertIn("回答整理未完成", answer)
            self.assertIn("【补充】关于儿童优惠", answer)
            self.assertIn("【更正】回答中关于", answer)
            self.assertIn("7岁儿童全部免费", answer)
            self.assertEqual(meta["answer_type"], "agent_unverified_answer")
            self.assertEqual(returned_docs, docs)
            self.assertEqual(generate.call_count, 3)

    def test_rewrite_does_not_retry_unrelated_failures(self):
        partial = {**VERDICT, "coverage": "partially", "missing_points": [{"point": "看点", "source": None}]}
        for error in (LLMServiceError("密钥失效"), LLMOutputError("未正常完成", "incomplete")):
            with self.subTest(error=str(error)), \
                    patch.object(hybrid, "hybrid_query", return_value=("推荐草稿", [], {"route": "recommendation"})), \
                    patch.object(agent, "_needs_query_planning", return_value=False), \
                    patch.object(agent, "generate_answer_by_api", side_effect=[json.dumps(partial), error]) as generate:
                with self.assertRaises(LLMServiceError):
                    agent.agent_reflective_query("我带一个7岁的孩子去深圳玩", [], context(), "rewrite-error", 100,
                                                CONFIG, return_metadata=True)
            self.assertEqual(generate.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
