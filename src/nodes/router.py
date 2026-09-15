import time
import logging
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from config.settings import settings
from config.logger import setup_logger
from src.state import AgentGraphState
from src.prompts import ROUTER_PROMPT
from src.agents import (
    RouterOutput,
    get_structured_chain,
    run_heuristic_router,
    normalize_route
)

logger = setup_logger("nodes.router")

async def router_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Asynchronously classifies user intent and sets routing_target using Google GenAI (ADK/Gemini)
    or local open-source LLM.
    Appends structured telemetry transaction logs containing execution latency.
    """
    start_time = time.time()
    query = state["input_query"]
    logger.info(f"Routing query asynchronously: '{query}'")
    
    status = "success"
    try:
        router_chain = get_structured_chain(RouterOutput)
        if router_chain is not None:
            prompt = ChatPromptTemplate.from_messages([
                ("system", ROUTER_PROMPT),
                ("human", "Query to route: {query}")
            ])
            router_res = await router_chain.ainvoke(prompt.format_messages(query=query))
            route = normalize_route(router_res.route)
            reasoning = router_res.reasoning
        else:
            heuristic_res = run_heuristic_router(query)
            route = heuristic_res.route
            reasoning = f"Heuristic | {heuristic_res.reasoning}"
    except Exception as e:
        logger.warning(f"Router LLM invocation failed ({e}). Falling back to heuristics.")
        status = "fallback"
        heuristic_res = run_heuristic_router(query)
        route = heuristic_res.route
        reasoning = f"Fallback (Error: {e}) | {heuristic_res.reasoning}"

    elapsed_ms = int((time.time() - start_time) * 1000)
    
    telemetry = {
        "node": "router",
        "action": "intent_classification",
        "status": status,
        "elapsed_time_ms": elapsed_ms,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "route_selected": route,
        "reasoning": reasoning
    }
    
    logger.info(f"Router Node completed in {elapsed_ms}ms. Selected route: '{route}'")
    
    return {
        "routing_target": route,
        "system_logs": [telemetry]
    }

