import logging
from langgraph.graph import StateGraph, START, END
from config.settings import settings
from config.logger import setup_logger
from src.state import AgentGraphState
from src.nodes.router import router_node
from src.nodes.retriever import retriever_node
from src.nodes.grader import grader_node
from src.nodes.generator import generator_node, direct_response_node
from src.nodes.rewriter import rewriter_node
from src.tools.web_search import web_search_node

logger = setup_logger("graph")

# ==========================================
# Conditional Routing Helper Functions
# ==========================================

def route_after_router(state: AgentGraphState) -> str:
    """
    Evaluates the router node's target selection.
    """
    target = state.get("routing_target", "direct_response")
    logger.info(f"Conditional Router Edge -> Moving to node: '{target}'")
    return target

def route_after_grader(state: AgentGraphState) -> str:
    """
    Decides whether to generate the answer or rewrite the query.
    If context is verified (not empty), we generate. Otherwise, we rewrite.
    """
    context = state.get("verified_context", [])
    if context:
        logger.info("Conditional Grader Edge -> Context is relevant. Proceeding to 'generator'.")
        return "generator"
    else:
        logger.info("Conditional Grader Edge -> Context is irrelevant. Redirecting to 'rewriter'.")
        return "rewriter"

def route_after_rewriter(state: AgentGraphState) -> str:
    """
    Enforces loop boundaries. If current loops exceed the configuration,
    routes to web_search; otherwise, retries retrieving.
    """
    loop_count = state.get("current_loop_count", 0) or 0
    max_loops = settings.MAX_REWRITE_LOOPS
    
    if loop_count >= max_loops:
        logger.info(f"Conditional Rewriter Edge -> Loop limit reached ({loop_count}/{max_loops}). Routing to 'web_search'.")
        return "web_search"
    else:
        logger.info(f"Conditional Rewriter Edge -> Retry loop {loop_count}/{max_loops}. Re-routing to 'retriever'.")
        return "retriever"

# ==========================================
# Graph Compilation Wiring
# ==========================================

def create_graph():
    """
    Assembles the RAG workflow graph and compiles it.
    """
    # Initialize state graph
    workflow = StateGraph(AgentGraphState)
    
    # Register all nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("rewriter", rewriter_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("direct_response", direct_response_node)
    workflow.add_node("generator", generator_node)
    
    # Establish entry point
    workflow.add_edge(START, "router")
    
    # Router conditional edges
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "vectorstore": "retriever",
            "web_search": "web_search",
            "direct_response": "direct_response"
        }
    )
    
    # Connect retriever directly to grader
    workflow.add_edge("retriever", "grader")
    
    # Grader conditional edges
    workflow.add_conditional_edges(
        "grader",
        route_after_grader,
        {
            "generator": "generator",
            "rewriter": "rewriter"
        }
    )
    
    # Rewriter conditional edges
    workflow.add_conditional_edges(
        "rewriter",
        route_after_rewriter,
        {
            "retriever": "retriever",
            "web_search": "web_search"
        }
    )
    
    # Connect web search back to final generation
    workflow.add_edge("web_search", "generator")
    
    # Terminal edges
    workflow.add_edge("generator", END)
    workflow.add_edge("direct_response", END)
    
    # Compile
    return workflow.compile()
