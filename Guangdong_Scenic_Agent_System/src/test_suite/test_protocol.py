import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Importing the real retrieval module loads local embedding and reranker models.
# Protocol tests replace that dependency so they remain fast and deterministic.
fake_rag_module = types.ModuleType("rag_hybrid_query")
fake_rag_module.hybrid_query = None
sys.modules.setdefault("rag_hybrid_query", fake_rag_module)

from main_agent import agent_chat_json, agent_chat_response
import main_agent


class ProtocolTests(unittest.TestCase):
    @staticmethod
    def fake_hybrid_query(query, history, cache, return_metadata=False):
        if not return_metadata:
            raise AssertionError("protocol entry must request metadata")
        return (
            "丹霞山位于韶关市。",
            [
                {
                    "spot_name": "丹霞山",
                    "city": "韶关",
                    "level": "5A",
                    "chunk_index": 7,
                    "source_url": "https://example.com/danxia",
                    "chunk_text": "internal retrieval text",
                }
            ],
            {"route": "rag", "answer_type": "spot_detail"},
        )

    def test_protocol_serializes_normalized_success_response(self):
        with patch.object(main_agent, "hybrid_query", self.fake_hybrid_query):
            response = agent_chat_response("丹霞山在哪里", request_id="req-test")

            self.assertEqual(response["protocol"], "scenic_customer_service")
            self.assertEqual(response["version"], "1.0")
            self.assertEqual(response["request_id"], "req-test")
            self.assertEqual(response["code"], 0)
            self.assertEqual(response["data"]["route"], "rag")
            self.assertEqual(response["data"]["spots"], [
                {
                    "spot_name": "丹霞山",
                    "city": "韶关",
                    "level": "5A",
                    "address": None,
                    "ticket": None,
                    "open_time": None,
                    "telephone": None,
                }
            ])
            self.assertEqual(response["data"]["sources"][0]["source_url"], "https://example.com/danxia")
            self.assertNotIn("chunk_text", json.dumps(response, ensure_ascii=False))

            serialized = agent_chat_json("丹霞山在哪里", request_id="req-json")
            self.assertEqual(json.loads(serialized)["request_id"], "req-json")


    def test_protocol_returns_validation_error_without_running_agent(self):
        def fail_if_called(*args, **kwargs):
            raise AssertionError("agent should not run for invalid input")

        with patch.object(main_agent, "hybrid_query", fail_if_called):
            response = agent_chat_response("  ", request_id="req-invalid")

        self.assertEqual(response["code"], 1001)
        self.assertEqual(response["error"]["type"], "validation_error")
        self.assertIsNone(response["data"])


    def test_protocol_returns_query_error_with_stable_shape(self):
        def failing_hybrid_query(query, history, cache, return_metadata=False):
            raise RuntimeError("model unavailable")

        with patch.object(main_agent, "hybrid_query", failing_hybrid_query):
            response = agent_chat_response("丹霞山介绍", request_id="req-error")

        self.assertEqual(response["code"], 2001)
        self.assertEqual(response["message"], "query failed")
        self.assertTrue(response["error"]["retryable"])
        self.assertIsNone(response["data"])


if __name__ == "__main__":
    unittest.main()
