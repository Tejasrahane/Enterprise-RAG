import logging
from typing import Dict, Any
from src.state import AgentState
from src.agents import call_router, call_grader, call_rewriter, call_generator
from src.vector_store import retrieve_from_store
from src.tools import web_search_tool

logger = logging.getLogger(__name__)

def route_query_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the query and routes it to the correct path.
    """
    query = state["query"]
    logger.info(f"Node [route_query]: Routing query '{query}'")
    
    router_res = call_router(query)
    
    steps = state.get("steps", []) or []
    new_steps = list(steps) + [f"Routed query to '{router_res.route}' | Reason: {router_res.reasoning}"]
    
    return {
        "route": router_res.route,
        "steps": new_steps
    }

def retrieve_docs_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves documents from the vector database using the current search query.
    """
    query = state["query"]
    logger.info(f"Node [retrieve_docs]: Searching vector store for query '{query}'")
    
    docs = retrieve_from_store(query)
    doc_contents = [d.page_content for d in docs]
    
    steps = state.get("steps", []) or []
    new_steps = list(steps) + [f"Retrieved {len(doc_contents)} documents from vector store."]
    
    return {
        "documents": doc_contents,
        "steps": new_steps
    }

def grade_documents_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates retrieved documents for relevance to the user's query.
    """
    query = state["query"]
    docs = state.get("documents", []) or []
    logger.info(f"Node [grade_documents]: Grading relevance of {len(docs)} docs for query '{query}'")
    
    grader_res = call_grader(query, docs)
    
    steps = state.get("steps", []) or []
    new_steps = list(steps) + [f"Graded document relevance: '{grader_res.is_relevant}' | Reason: {grader_res.reasoning}"]
    
    return {
        "is_relevant": grader_res.is_relevant,
        "steps": new_steps
    }

def rewrite_query_node(state: AgentState) -> Dict[str, Any]:
    """
    Optimizes the user query to yield better vector search results.
    """
    query = state["query"]
    loop_count = state.get("loop_count", 0) or 0
    logger.info(f"Node [rewrite_query]: Rewriting query '{query}' (Loop: {loop_count})")
    
    rewriter_res = call_rewriter(query)
    new_loop_count = loop_count + 1
    
    steps = state.get("steps", []) or []
    new_steps = list(steps) + [
        f"Rewrote query from '{query}' to '{rewriter_res.rewritten_query}' (Loop: {new_loop_count}) | Reason: {rewriter_res.reasoning}"
    ]
    
    return {
        "query": rewriter_res.rewritten_query,
        "loop_count": new_loop_count,
        "steps": new_steps
    }

def web_search_node(state: AgentState) -> Dict[str, Any]:
    """
    Queries web search and stores results as documents.
    """
    query = state["query"]
    logger.info(f"Node [web_search]: Searching web for query '{query}'")
    
    search_results = web_search_tool(query)
    
    steps = state.get("steps", []) or []
    new_steps = list(steps) + [f"Web search retrieved {len(search_results)} results."]
    
    return {
        "documents": search_results,
        "steps": new_steps
    }

def direct_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates a conversational answer directly, skipping vector search and web lookup.
    """
    query = state["original_query"]
    logger.info(f"Node [direct_response]: Direct conversational response for '{query}'")
    
    from langchain_core.messages import HumanMessage
    from src.agents import llm, is_llm_active
    
    if is_llm_active:
        try:
            res = llm.invoke([HumanMessage(content=query)])
            generation = str(res.content)
        except Exception as e:
            generation = f"Hello! How can I help you today? I detected you wanted to chat directly. (LLM error: {e})"
    else:
        generation = (
            f"Hello! This is a direct response mode. "
            f"You asked a conversational query: '{query}'. How can I assist you further?"
        )
        
    steps = state.get("steps", []) or []
    new_steps = list(steps) + ["Generated direct conversational response."]
    
    return {
        "generation": generation,
        "steps": new_steps
    }

def generate_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates a final response grounded strictly in the gathered documents.
    """
    original_query = state["original_query"]
    docs = state.get("documents", []) or []
    logger.info(f"Node [generate_answer]: Generating grounded answer for query '{original_query}'")
    
    generation = call_generator(original_query, docs)
    
    steps = state.get("steps", []) or []
    new_steps = list(steps) + ["Synthesized final response grounded in gathered docs."]
    
    return {
        "generation": generation,
        "steps": new_steps
    }
