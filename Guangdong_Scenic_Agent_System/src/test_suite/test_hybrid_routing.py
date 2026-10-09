import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# The routing integration tests do not need to load the real ML dependencies.
torch_module = types.ModuleType("torch")
torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)
torch_module.device = lambda value: value
sys.modules.setdefault("torch", torch_module)
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules["numpy"].load = lambda path: []
sys.modules["numpy"].dot = lambda *args, **kwargs: None
sys.modules["numpy"].argsort = lambda *args, **kwargs: []

transformers_module = types.ModuleType("transformers")
transformers_module.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: object())
transformers_module.AutoModel = types.SimpleNamespace(
    from_pretrained=lambda path: types.SimpleNamespace(to=lambda device: object())
)
sys.modules.setdefault("transformers", transformers_module)

sentence_module = types.ModuleType("sentence_transformers")
sentence_module.CrossEncoder = lambda *args, **kwargs: object()
sys.modules.setdefault("sentence_transformers", sentence_module)

llm_module = types.ModuleType("llm_answer")
llm_module.build_chat_messages = lambda *args, **kwargs: []
llm_module.generate_answer_by_api = lambda *args, **kwargs: "模拟回答"
llm_module.OUTPUT_FORMAT_RULE = ""
sys.modules.setdefault("llm_answer", llm_module)

with patch("pickle.load", return_value=[]), patch("sys.stdout", new=io.StringIO()):
    import rag_hybrid_query


class HybridRoutingTests(unittest.TestCase):
    def test_greeting_skips_database_and_rag(self):
        with patch.object(
            rag_hybrid_query, "get_query_embedding", side_effect=AssertionError("RAG should not run")
        ), patch.object(
            rag_hybrid_query, "get_single_spot_detail", side_effect=AssertionError("SQL should not run")
        ):
            answer, docs, meta = rag_hybrid_query.hybrid_query(
                "你好", [], [None], return_metadata=True
            )

        self.assertIn("你好", answer)
        self.assertEqual(docs, [])
        self.assertEqual(meta["route"], "greeting")
        self.assertEqual(meta["answer_type"], "greeting")

    def test_agent_profile_skips_database_and_rag(self):
        with patch.object(
            rag_hybrid_query, "get_query_embedding", side_effect=AssertionError("RAG should not run")
        ), patch.object(
            rag_hybrid_query, "get_single_spot_detail", side_effect=AssertionError("SQL should not run")
        ):
            answer, docs, meta = rag_hybrid_query.hybrid_query(
                "你是谁", [], [None], return_metadata=True
            )

        self.assertIn("广东省A级旅游景区智能客服助手", answer)
        self.assertEqual(docs, [])
        self.assertEqual(meta["route"], "agent_profile")
        self.assertEqual(meta["answer_type"], "agent_intro")

    def test_structured_detail_skips_rag(self):
        detail = {
            "spot_name": "韶关市丹霞山景区",
            "address": "韶关市仁化县",
            "ticket": "100元",
            "open_time": "08:00-18:00",
            "city": "韶关",
            "level": "5A",
            "telephone": "0751-0000000",
        }
        with patch.object(rag_hybrid_query, "get_single_spot_detail", return_value=detail), patch.object(
            rag_hybrid_query, "get_query_embedding", side_effect=AssertionError("RAG should not run")
        ):
            answer, docs, meta = rag_hybrid_query.hybrid_query(
                "丹霞山门票多少钱", [], [None], return_metadata=True
            )

        self.assertIn("100元", answer)
        self.assertNotIn("开放时间", answer)
        self.assertEqual(docs, [])
        self.assertEqual(meta["route"], "structured_detail")
        self.assertGreater(meta["confidence"], 0.9)

    def test_structured_detail_updates_pronoun_cache(self):
        detail = {
            "spot_name": "韶关市丹霞山景区",
            "address": "韶关市仁化县",
            "ticket": "100元",
            "open_time": "08:00-18:00",
            "city": "韶关",
            "level": "5A",
            "telephone": "0751-0000000",
        }
        cache = [None]
        with patch.object(rag_hybrid_query, "get_single_spot_detail", return_value=detail):
            rag_hybrid_query.hybrid_query("丹霞山门票多少钱", [], cache)

        self.assertEqual(cache[0], "韶关市丹霞山景区")

    def test_count_route_skips_llm(self):
        with patch.object(rag_hybrid_query, "query_scenic_count", return_value=16), patch.object(
            rag_hybrid_query, "generate_answer_by_api", side_effect=AssertionError("LLM should not run")
        ):
            answer, docs, meta = rag_hybrid_query.hybrid_query(
                "广州一共有多少5A景区", [], [None], return_metadata=True
            )

        self.assertEqual(answer, "广州共有16个5A级旅游景区。")
        self.assertEqual(docs, [])
        self.assertEqual(meta["route"], "scenic_count")

    def test_list_route_uses_sql_result_without_llm(self):
        rows = [
            {"spot_name": "景区甲", "city": "茂名", "level": "4A", "chunk_text": "甲"},
            {"spot_name": "景区乙", "city": "茂名", "level": "4A", "chunk_text": "乙"},
        ]
        with patch.object(rag_hybrid_query, "query_city_scenic_list", return_value=rows), patch.object(
            rag_hybrid_query, "generate_answer_by_api", side_effect=AssertionError("LLM should not run")
        ):
            answer, docs, meta = rag_hybrid_query.hybrid_query(
                "茂名有哪些4A景区", [], [None], return_metadata=True
            )

        self.assertIn("共有2个", answer)
        self.assertEqual(docs, rows)
        self.assertEqual(meta["route"], "scenic_list")

    def test_list_continuation_returns_next_page(self):
        rows = [
            {"spot_name": f"景区{index}", "city": "广州", "level": "4A", "ticket": "暂无", "chunk_text": "资料"}
            for index in range(1, 38)
        ]
        context = {
            "list_pages": {"city": "广州", "level": "4A", "offset": 15, "total": 37},
            "candidate_spot_list": [], "latest_subject": None, "city": "广州", "level": "4A",
            "allow_cross_city": False, "focus_mode": False, "focus_keep_round": 0,
            "user_preference": {}, "last_intent_type": "scenic_list",
        }
        with patch.object(rag_hybrid_query, "query_city_scenic_list", return_value=rows), patch.object(
            rag_hybrid_query, "generate_answer_by_api", side_effect=AssertionError("pagination must not call LLM")
        ):
            answer, docs, meta = rag_hybrid_query.hybrid_query(
                "广州还有吗", [], context, "test-session", 0, {}, return_metadata=True
            )

        self.assertIn("第2批，第16-30家", answer)
        self.assertIn("景区16", answer)
        self.assertNotIn("景区1（", answer)
        self.assertEqual([item["spot_name"] for item in docs], [f"景区{index}" for index in range(16, 31)])
        self.assertEqual(context["list_pages"]["offset"], 30)

    def test_completed_list_answers_no_more_without_restarting_at_first_page(self):
        rows = [
            {"spot_name": f"景区{index}", "city": "广州", "level": "4A", "ticket": "暂无", "chunk_text": "资料"}
            for index in range(1, 38)
        ]
        context = {
            "candidate_spot_list": [], "latest_subject": None, "city": "广州", "level": "4A",
            "allow_cross_city": False, "focus_mode": False, "focus_keep_round": 0,
            "user_preference": {},
        }

        with patch.object(rag_hybrid_query, "query_city_scenic_list", return_value=rows), patch.object(
            rag_hybrid_query, "generate_answer_by_api", side_effect=AssertionError("list answers must not call LLM")
        ):
            pages = []
            for query in ("广州有哪些4A景区", "广州继续", "广州继续"):
                answer, docs, _ = rag_hybrid_query.hybrid_query(
                    query, [], context, "test-session", 0, {}, return_metadata=True
                )
                pages.append((answer, docs))

            self.assertIn("一共有37个", pages[0][0])
            self.assertIn("第16-30家", pages[1][0])
            self.assertIn("第31-37家", pages[2][0])
            self.assertTrue(context["list_pages"]["complete"])
            self.assertEqual(context["list_pages"]["offset"], 37)

            answer, docs, meta = rag_hybrid_query.hybrid_query(
                "还有吗？还是已经没有了", [], context, "test-session", 0, {}, return_metadata=True
            )

        self.assertIn("已经全部展示完毕", answer)
        self.assertIn("目前没有更多未列出的景区", answer)
        self.assertNotIn("景区1（", answer)
        self.assertEqual(docs, [])
        self.assertEqual(meta["route"], "scenic_list")
        self.assertTrue(context["list_pages"]["complete"])

    def test_short_list_keeps_completed_cursor_for_followup(self):
        rows = [
            {"spot_name": f"景区{index}", "city": "茂名", "level": "4A", "ticket": "暂无", "chunk_text": "资料"}
            for index in range(1, 4)
        ]
        context = {"city": "茂名", "level": "4A", "candidate_spot_list": [], "user_preference": {}}
        with patch.object(rag_hybrid_query, "query_city_scenic_list", return_value=rows), patch.object(
            rag_hybrid_query, "generate_answer_by_api", side_effect=AssertionError("list answers must not call LLM")
        ):
            rag_hybrid_query.hybrid_query("茂名有哪些4A景区", [], context, "test-session", 0, {})
            self.assertEqual(context["list_pages"]["offset"], 3)
            self.assertTrue(context["list_pages"]["complete"])

            answer, docs, _ = rag_hybrid_query.hybrid_query(
                "还有吗", [], context, "test-session", 0, {}, return_metadata=True
            )

        self.assertIn("共3家", answer)
        self.assertIn("目前没有更多未列出的景区", answer)
        self.assertEqual(docs, [])

    def test_missing_pronoun_context_returns_clarification(self):
        with patch.object(
            rag_hybrid_query, "get_query_embedding", side_effect=AssertionError("RAG should not run")
        ):
            answer, docs, meta = rag_hybrid_query.hybrid_query(
                "它几点开放", [], [None], return_metadata=True
            )

        self.assertIn("具体的城市或景区名称", answer)
        self.assertEqual(docs, [])
        self.assertEqual(meta["route"], "clarify")

    def test_pronoun_rag_query_uses_resolved_spot_name(self):
        captured = {}

        def fake_embedding(query, tokenizer, model):
            captured["embedding_query"] = query
            return []

        with patch.object(rag_hybrid_query, "get_query_embedding", side_effect=fake_embedding), patch.object(
            rag_hybrid_query, "vector_search_numpy", return_value=[]
        ), patch.object(rag_hybrid_query, "keyword_search_db", return_value=[]), patch.object(
            rag_hybrid_query, "rerank", return_value=[]
        ), patch.object(rag_hybrid_query, "generate_answer_by_api", return_value="景区介绍"):
            answer, docs, meta = rag_hybrid_query.hybrid_query(
                "它有什么特色", [], ["韶关市丹霞山景区"], return_metadata=True
            )

        self.assertEqual(answer, "景区介绍")
        self.assertEqual(docs, [])
        self.assertEqual(meta["route"], "rag")
        self.assertIn("韶关市丹霞山景区", captured["embedding_query"])


if __name__ == "__main__":
    unittest.main()
