import time
import logging
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config.settings import settings
from config.logger import setup_logger
from src.state import AgentGraphState
from src.prompts import ROUTER_PROMPT

logger = setup_logger("nodes.router")

class RouterOutput(BaseModel):
    """Pydantic model representing the router classification output."""
    route: str = Field(
        description="The target route. Must be strictly one of: 'vectorstore', 'web_search', or 'direct_response'."
    )
    reasoning: str = Field(
        description="Explanation behind the routing choice."
    )

# Initialize Ollama LLM
is_ollama_active = True
router_chain = None

try:
    llm = ChatOllama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0
    )
    router_chain = llm.with_structured_output(RouterOutput)
    logger.info(f"Open-source Ollama Router initialized successfully with model '{settings.LLM_MODEL}'.")
except Exception as e:
    logger.error(f"Error initializing Ollama Router Agent: {e}. Using fallback heuristics.")
    is_ollama_active = False

def run_heuristic_router(query: str) -> RouterOutput:
    query_lower = query.lower()
    
    if any(greet in query_lower for greet in ["hi", "hello", "hey", "how are you", "who are you"]):
        return RouterOutput(
            route="direct_response",
            reasoning="Query is a basic greeting or conversation starter, requiring no external knowledge."
        )
    elif any(term in query_lower for term in ["aetheris", "quantum", "xyz", "stellarex", "nova-9", "nova 9", "propulsion"]):
        return RouterOutput(
            route="vectorstore",
            reasoning="Query targets specific internal/proprietary knowledge terms present in documents."
        )
    else:
        return RouterOutput(
            route="web_search",
            reasoning="Query appears to seek real-time, current events, or general knowledge suited for web search."
        )

def normalize_route(route: str) -> str:
    r = (route or "").lower().strip()
    if r in ["vectorstore", "web_search", "direct_response"]:
        return r
    if any(k in r for k in ["vector", "store", "doc", "spec", "internal"]):
        return "vectorstore"
    if any(k in r for k in ["web", "search", "news", "real-time", "internet"]):
        return "web_search"
    return "direct_response"

async def router_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Asynchronously classifies user intent and sets routing_target using local open-source LLM.
    Appends structured telemetry transaction logs containing execution latency.
    """
    start_time = time.time()
    query = state["input_query"]
    logger.info(f"Routing query asynchronously: '{query}'")
    
    status = "success"
    try:
        if is_ollama_active and router_chain is not None:
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
        logger.warning(f"Ollama router invocation failed ({e}). Falling back to heuristics.")
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

