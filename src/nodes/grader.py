import time
import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from config.settings import settings
from config.logger import setup_logger
from src.state import AgentGraphState
from src.prompts import GRADER_PROMPT
from src.agents import (
    GraderOutput,
    get_structured_chain,
    run_heuristic_grader,
    normalize_relevance
)

logger = setup_logger("nodes.grader")

async def grader_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Asynchronously grades the relevance of retrieved documents using Google GenAI (ADK/Gemini)
    or local open-source LLM with structured output.
    Appends structured telemetry transaction logs.
    """
    start_time = time.time()
    query = state.get("optimized_query") or state["input_query"]
    docs = state.get("retrieved_documents", []) or []
    
    logger.info(f"Grading {len(docs)} documents asynchronously for query: '{query}'")
    
    status = "success"
    try:
        grader_chain = get_structured_chain(GraderOutput)
        if grader_chain is not None:
            prompt = ChatPromptTemplate.from_messages([
                ("system", GRADER_PROMPT),
                ("human", "Query: {query}\n\nDocuments:\n{context}")
            ])
            context = "\n\n".join(docs)
            grader_res = await grader_chain.ainvoke(prompt.format_messages(query=query, context=context))
            is_relevant = normalize_relevance(grader_res.is_relevant)
            reasoning = grader_res.reasoning
        else:
            heuristic_res = run_heuristic_grader(query, docs)
            is_relevant = heuristic_res.is_relevant
            reasoning = f"Heuristic | {heuristic_res.reasoning}"
    except Exception as e:
        logger.warning(f"Grader LLM failed ({e}). Falling back to heuristics.")
        status = "fallback"
        heuristic_res = run_heuristic_grader(query, docs)
        is_relevant = heuristic_res.is_relevant
        reasoning = f"Fallback (Error: {e}) | {heuristic_res.reasoning}"
        
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    telemetry = {
        "node": "grader",
        "action": "relevance_grading",
        "status": status,
        "elapsed_time_ms": elapsed_ms,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "graded_relevant": is_relevant,
        "reasoning": reasoning
    }
    
    logger.info(f"Grader Node completed in {elapsed_ms}ms. Relevance score: '{is_relevant}'")
    
    # If relevant, store in verified_context; if not, leave empty
    verified = docs if is_relevant == "yes" else []
    
    return {
        "verified_context": verified,
        "system_logs": [telemetry]
    }

