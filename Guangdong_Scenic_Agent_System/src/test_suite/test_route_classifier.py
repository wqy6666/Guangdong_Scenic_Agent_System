import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import route_classifier


class RouteClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        route_classifier.clear_entity_cache()

    def test_detail_query_has_structured_route(self):
        result = route_classifier.classify_query("丹霞山门票多少钱")
        self.assertEqual(result["route"], route_classifier.ROUTE_STRUCTURED_DETAIL)
        self.assertIn("丹霞山", result["entities"]["spot_name"])
        self.assertIn("ticket", result["entities"]["fields"])

    def test_agent_profile_does_not_require_database_entities(self):
        with patch.object(
            route_classifier, "load_entities", side_effect=AssertionError("database should not run")
        ):
            result = route_classifier.classify_query("你是谁？你能做什么？")
        self.assertEqual(result["route"], route_classifier.ROUTE_AGENT_PROFILE)
        self.assertEqual(result["answer_type"], "agent_intro")
        self.assertEqual(result["entities"]["spot_name"], None)

    def test_standalone_greeting_does_not_require_database_entities(self):
        with patch.object(
            route_classifier, "load_entities", side_effect=AssertionError("database should not run")
        ):
            result = route_classifier.classify_query("你好！")
        self.assertEqual(result["route"], route_classifier.ROUTE_GREETING)
        self.assertEqual(result["answer_type"], "greeting")

    def test_greeting_with_a_scenic_question_keeps_business_route(self):
        result = route_classifier.classify_query("你好，丹霞山门票多少钱？")
        self.assertEqual(result["route"], route_classifier.ROUTE_STRUCTURED_DETAIL)

    def test_city_list_uses_database_city_entities(self):
        result = route_classifier.classify_query("茂名有哪些4A景区")
        self.assertEqual(result["route"], route_classifier.ROUTE_SCENIC_LIST)
        self.assertEqual(result["entities"]["level"], "4A")
        self.assertIsNotNone(result["entities"]["city"])

    def test_city_count_has_dedicated_route(self):
        result = route_classifier.classify_query("广州一共有多少景区")
        self.assertEqual(result["route"], route_classifier.ROUTE_SCENIC_COUNT)

    def test_list_continuation_with_city_does_not_restart_first_page(self):
        context = {
            "list_pages": {"city": "广州", "level": "4A", "offset": 15, "total": 37},
            "candidate_spot_list": [], "latest_subject": None, "city": "广州",
            "level": "4A", "last_intent_type": route_classifier.ROUTE_SCENIC_LIST,
        }
        for query in ("继续", "还有吗", "广州还有吗", "广州4A还有哪些"):
            result = route_classifier.classify_query(query, context)
            self.assertEqual(result["route"], route_classifier.ROUTE_SCENIC_LIST, query)
            self.assertTrue(result["entities"].get("list_continue"), query)
            self.assertEqual(result["entities"]["city"], "广州")
            self.assertEqual(result["entities"]["level"], "4A")

    def test_recommendation_does_not_use_sql_list_route(self):
        result = route_classifier.classify_query("广州适合亲子游的景区推荐")
        self.assertEqual(result["route"], route_classifier.ROUTE_RECOMMENDATION)

    def test_pronoun_without_context_asks_for_clarification(self):
        result = route_classifier.classify_query("它的开放时间")
        self.assertEqual(result["route"], route_classifier.ROUTE_CLARIFY)

    def test_pronoun_with_context_uses_context_spot(self):
        result = route_classifier.classify_query("它几点开放", last_spot="丹霞山")
        self.assertEqual(result["route"], route_classifier.ROUTE_STRUCTURED_DETAIL)
        self.assertIn("丹霞山", result["entities"]["spot_name"])

    def test_explicit_spot_with_pronoun_does_not_require_history(self):
        result = route_classifier.classify_query("丹霞山这个景区门票多少钱")
        self.assertEqual(result["route"], route_classifier.ROUTE_STRUCTURED_DETAIL)
        self.assertIn("丹霞山", result["entities"]["spot_name"])

    def test_unsupported_level_is_clarified(self):
        result = route_classifier.classify_query("广州有哪些8A景区")
        self.assertEqual(result["route"], route_classifier.ROUTE_CLARIFY)
        self.assertEqual(result["entities"]["invalid_level"], "8A")


if __name__ == "__main__":
    unittest.main()
