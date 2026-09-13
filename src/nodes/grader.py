import time
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config.settings import settings
from config.logger import setup_logger
from src.state import AgentGraphState
from src.prompts import GRADER_PROMPT

logger = setup_logger("nodes.grader")

class GraderOutput(BaseModel):
    """Pydantic model representing the relevance grading output."""
    is_relevant: str = Field(
        description="Relevance decision. Must be 'yes' or 'no'."
    )
    reasoning: str = Field(
        description="Detailed explanation justifying the decision."
    )

# Initialize Ollama LLM
is_ollama_active = True
grader_chain = None

try:
    llm = ChatOllama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0
    )
    grader_chain = llm.with_structured_output(GraderOutput)
    logger.info(f"Open-source Ollama Grader initialized successfully with model '{settings.LLM_MODEL}'.")
except Exception as e:
    logger.error(f"Error initializing Ollama Grader Agent: {e}. Using fallback heuristics.")
    is_ollama_active = False

def run_heuristic_grader(query: str, docs: List[str]) -> GraderOutput:
    if not docs:
        return GraderOutput(is_relevant="no", reasoning="No documents were retrieved.")
    
    query_lower = query.lower()
    combined_docs = "\n".join(docs).lower()
    
    financial_terms = {"stock", "price", "revenue", "profit", "earnings", "cost", "financial", "$"}
    has_financial_query = any(term in query_lower for term in financial_terms)
    has_financial_docs = any(term in combined_docs for term in financial_terms)
    
    if has_financial_query and not has_financial_docs:
        return GraderOutput(
            is_relevant="no",
            reasoning="Query asks for financial metrics (e.g. stock price), but retrieved documents contain only technical specs without financial values."
        )
    
    query_words = set(query_lower.replace("?", "").split())
    stop_words = {"what", "how", "why", "where", "who", "which", "when", "does", "after", "before", "about", "with", "from", "their", "under", "system"}
    important_query_words = {w for w in query_words if len(w) > 3 and w not in stop_words}
    
    max_overlap = 0
    for doc in docs:
        doc_words = set(doc.lower().split())
        overlap = len(important_query_words & doc_words)
        if overlap > max_overlap:
            max_overlap = overlap
            
    if max_overlap >= 2:
        return GraderOutput(
            is_relevant="yes",
            reasoning=f"Found significant overlap of key query concepts in retrieved docs (overlap: {max_overlap} terms)."
        )
            
    return GraderOutput(
        is_relevant="no",
        reasoning=f"Insufficient overlap of key semantic query terms (max overlap: {max_overlap} terms)."
    )

def normalize_relevance(decision: str) -> str:
    d = (decision or "").lower().strip()
    return "yes" if "yes" in d else "no"

async def grader_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Asynchronously grades the relevance of retrieved documents using local open-source LLM.
    Appends structured telemetry transaction logs.
    """
    start_time = time.time()
    query = state.get("optimized_query") or state["input_query"]
    docs = state.get("retrieved_documents", []) or []
    
    logger.info(f"Grading {len(docs)} documents asynchronously for query: '{query}'")
    
    status = "success"
    try:
        if is_ollama_active and grader_chain is not None:
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
        logger.warning(f"Ollama grader failed ({e}). Falling back to heuristics.")
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

