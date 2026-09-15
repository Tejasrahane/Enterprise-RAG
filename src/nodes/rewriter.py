import time
import logging
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from config.settings import settings
from config.logger import setup_logger
from src.state import AgentGraphState
from src.prompts import REWRITER_PROMPT
from src.agents import (
    RewriterOutput,
    get_structured_chain,
    run_heuristic_rewriter
)

logger = setup_logger("nodes.rewriter")

async def rewriter_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Asynchronously optimizes user search queries to improve retriever hits using Google GenAI (ADK/Gemini)
    or local open-source LLM.
    Appends structured telemetry transaction logs.
    """
    start_time = time.time()
    query = state.get("optimized_query") or state["input_query"]
    loop_count = state.get("current_loop_count", 0) or 0
    
    logger.info(f"Rewriting query '{query}' asynchronously (Loop: {loop_count})")
    
    status = "success"
    try:
        rewriter_chain = get_structured_chain(RewriterOutput)
        if rewriter_chain is not None:
            prompt = ChatPromptTemplate.from_messages([
                ("system", REWRITER_PROMPT),
                ("human", "Original Query: {query}")
            ])
            rewriter_res = await rewriter_chain.ainvoke(prompt.format_messages(query=query))
            rewritten_query = rewriter_res.rewritten_query or query
            reasoning = rewriter_res.reasoning
        else:
            heuristic_res = run_heuristic_rewriter(query)
            rewritten_query = heuristic_res.rewritten_query
            reasoning = f"Heuristic | {heuristic_res.reasoning}"
    except Exception as e:
        logger.warning(f"Rewriter LLM failed ({e}). Falling back to heuristics.")
        status = "fallback"
        heuristic_res = run_heuristic_rewriter(query)
        rewritten_query = heuristic_res.rewritten_query
        reasoning = f"Fallback (Error: {e}) | {heuristic_res.reasoning}"
        
    new_loop_count = loop_count + 1
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    telemetry = {
        "node": "rewriter",
        "action": "query_rewriting",
        "status": status,
        "elapsed_time_ms": elapsed_ms,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_query": query,
        "rewritten_query": rewritten_query,
        "new_loop_count": new_loop_count
    }
    
    logger.info(f"Rewriter Node completed in {elapsed_ms}ms. New query: '{rewritten_query}' (Loop: {new_loop_count})")
    
    return {
        "optimized_query": rewritten_query,
        "current_loop_count": new_loop_count,
        "system_logs": [telemetry]
    }

