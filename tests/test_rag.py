import unittest
from src.state import AgentGraphState
from src.nodes.router import run_heuristic_router, router_node
from src.nodes.grader import run_heuristic_grader, grader_node
from src.nodes.rewriter import run_heuristic_rewriter, rewriter_node
from src.nodes.generator import run_heuristic_generator, generator_node, direct_response_node
from src.nodes.retriever import get_chroma_collection, retriever_node
from src.tools.web_search import execute_web_search, web_search_node

class TestAsyncEnterpriseRAG(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize chroma collection
        get_chroma_collection()

    def test_router_heuristic_greetings(self):
        """Verify that greetings route to direct_response."""
        res = run_heuristic_router("Hi, my name is Tejas")
        self.assertEqual(res.route, "direct_response")

    def test_router_heuristic_specs(self):
        """Verify that specification terms route to vectorstore."""
        res = run_heuristic_router("Tell me about Stellarex Nova-9 specifications")
        self.assertEqual(res.route, "vectorstore")

    def test_grader_heuristic_relevant(self):
        """Verify that matching terms grade as relevant."""
        query = "Aetheris quantum specs"
        docs = ["Project Aetheris is a next-generation quantum computing initiative with 128 qubits."]
        res = run_heuristic_grader(query, docs)
        self.assertEqual(res.is_relevant, "yes")

    def test_grader_heuristic_irrelevant(self):
        """Verify that mismatched categories grade as irrelevant."""
        query = "stock prices stellarex"
        docs = ["The Nova-9 propulsion system is designed for deep-space transit to Mars."]
        res = run_heuristic_grader(query, docs)
        self.assertEqual(res.is_relevant, "no")

    async def test_router_node_async(self):
        """Verify async router node execution and telemetry metrics."""
        state = {
            "input_query": "Hello there, how can I help you?",
            "optimized_query": "",
            "retrieved_documents": [],
            "verified_context": [],
            "routing_target": "",
            "current_loop_count": 0,
            "system_logs": [],
            "final_generation": ""
        }
        res = await router_node(state)
        self.assertEqual(res["routing_target"], "direct_response")
        self.assertEqual(len(res["system_logs"]), 1)
        self.assertEqual(res["system_logs"][0]["node"], "router")
        self.assertIn("elapsed_time_ms", res["system_logs"][0])

    async def test_retriever_node_async(self):
        """Verify async retriever node execution and telemetry metrics."""
        state = {
            "input_query": "quantum CPU specs",
            "optimized_query": "",
            "retrieved_documents": [],
            "verified_context": [],
            "routing_target": "",
            "current_loop_count": 0,
            "system_logs": [],
            "final_generation": ""
        }
        res = await retriever_node(state)
        self.assertGreaterEqual(len(res["retrieved_documents"]), 0)
        self.assertEqual(len(res["system_logs"]), 1)
        self.assertEqual(res["system_logs"][0]["node"], "retriever")
        self.assertIn("elapsed_time_ms", res["system_logs"][0])

    async def test_grader_node_async_irrelevant(self):
        """Verify async grader node relevance grading and telemetry logs."""
        state = {
            "input_query": "stock price Stellarex",
            "optimized_query": "",
            "retrieved_documents": ["The Nova-9 engine test lasted 500 hours."],
            "verified_context": [],
            "routing_target": "",
            "current_loop_count": 0,
            "system_logs": [],
            "final_generation": ""
        }
        res = await grader_node(state)
        self.assertEqual(len(res["verified_context"]), 0) # Should be empty due to category mismatch
        self.assertEqual(len(res["system_logs"]), 1)
        self.assertEqual(res["system_logs"][0]["node"], "grader")
        self.assertEqual(res["system_logs"][0]["graded_relevant"], "no")

    async def test_rewriter_node_async(self):
        """Verify async rewriter node increments count and updates optimized query."""
        state = {
            "input_query": "What is the stock price of Stellarex?",
            "optimized_query": "",
            "retrieved_documents": [],
            "verified_context": [],
            "routing_target": "",
            "current_loop_count": 0,
            "system_logs": [],
            "final_generation": ""
        }
        res = await rewriter_node(state)
        self.assertEqual(res["current_loop_count"], 1)
        self.assertEqual(res["system_logs"][0]["node"], "rewriter")
        self.assertTrue(len(res["optimized_query"]) > 0)

if __name__ == "__main__":
    unittest.main()
