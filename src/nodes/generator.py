import time
import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from config.settings import settings
from config.logger import setup_logger
from src.state import AgentGraphState
from src.prompts import GENERATOR_PROMPT
from src.agents import get_llm, run_heuristic_generator

logger = setup_logger("nodes.generator")

async def generator_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Asynchronously synthesizes a final response grounded strictly in the gathered context documents
    using Google GenAI (ADK/Gemini) or local open-source LLM.
    Appends structured telemetry transaction logs.
    """
    start_time = time.time()
    query = state["input_query"]
    docs = state.get("verified_context", []) or state.get("retrieved_documents", []) or []
    
    logger.info(f"Generating grounded answer asynchronously for query: '{query}' using {len(docs)} documents.")
    
    status = "success"
    try:
        llm = get_llm()
        if llm is not None:
            prompt = ChatPromptTemplate.from_messages([
                ("system", GENERATOR_PROMPT),
                ("human", "User Query: {query}\n\nRetrieved Context:\n{context}")
            ])
            context = "\n\n".join(docs)
            res = await llm.ainvoke(prompt.format_messages(query=query, context=context))
            answer = str(res.content)
        else:
            answer = run_heuristic_generator(query, docs)
    except Exception as e:
        logger.warning(f"Generator LLM failed ({e}). Falling back to heuristics.")
        status = "fallback"
        answer = run_heuristic_generator(query, docs)
        
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    telemetry = {
        "node": "generator",
        "action": "grounded_generation",
        "status": status,
        "elapsed_time_ms": elapsed_ms,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "context_docs_count": len(docs)
    }
    
    logger.info(f"Generator Node completed in {elapsed_ms}ms.")
    
    return {
        "final_generation": answer,
        "system_logs": [telemetry]
    }

async def direct_response_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Asynchronously handles simple greetings or general tasks directly without context lookups
    using Google GenAI (ADK/Gemini) or local open-source LLM.
    Appends structured telemetry transaction logs.
    """
    start_time = time.time()
    query = state["input_query"]
    logger.info(f"Generating direct conversational response asynchronously for query: '{query}'")
    
    status = "success"
    try:
        llm = get_llm()
        if llm is not None:
            res = await llm.ainvoke([HumanMessage(content=query)])
            answer = str(res.content)
        else:
            answer = "Hello! I am your AI assistant. How can I help you today?"
    except Exception as e:
        logger.warning(f"Direct response LLM failed ({e}). Running fallback.")
        status = "fallback"
        answer = f"Hello! How can I help you today? (Fallback notice: {e})"
        
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    telemetry = {
        "node": "direct_response",
        "action": "direct_conversation",
        "status": status,
        "elapsed_time_ms": elapsed_ms,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    logger.info(f"Direct Response Node completed in {elapsed_ms}ms.")
    
    return {
        "final_generation": answer,
        "system_logs": [telemetry]
    }

