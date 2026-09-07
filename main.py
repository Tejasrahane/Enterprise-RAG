import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from config.settings import settings
from config.logger import setup_logger
from src.graph import create_graph
from src.tools.ingest import ingest_all_documents

logger = setup_logger("main_gateway")

# ==========================================
# FastAPI Lifespan (Startup Data Ingestion)
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI Enterprise RAG gateway...")
    try:
        logger.info("Initializing vector database index...")
        ingest_all_documents()
    except Exception as e:
        logger.error(f"Failed to initialize vector database index: {e}")
    yield
    logger.info("Shutting down FastAPI Enterprise RAG gateway...")

app = FastAPI(
    title="Async Enterprise RAG Gateway API",
    description="FastAPI asynchronous LangGraph serving gateway with telemetry tracing logs.",
    version="2.0.0",
    lifespan=lifespan
)

# Instantiate the compiled LangGraph workflow
graph = create_graph()

# ==========================================
# API Request/Response Schemas
# ==========================================

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    input_query: str
    final_generation: str
    routing_target: str
    current_loop_count: int
    system_logs: list[dict] # Pydantic V2 list of telemetry log dicts

# ==========================================
# API Routes
# ==========================================

@app.get("/health")
async def health_check():
    """Simple API health check endpoint."""
    return {"status": "healthy", "service": "async-enterprise-rag"}

@app.post("/api/v1/chat/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Synchronously runs the user query through the Async RAG LangGraph workflow.
    """
    logger.info(f"Received synchronous query request: '{request.query}'")
    
    initial_state = {
        "input_query": request.query,
        "optimized_query": "",
        "retrieved_documents": [],
        "verified_context": [],
        "routing_target": "",
        "current_loop_count": 0,
        "system_logs": [],
        "final_generation": ""
    }
    
    try:
        # Run graph invocation (inherently async under the hood for async nodes)
        final_state = await graph.ainvoke(initial_state)
        
        return QueryResponse(
            input_query=final_state.get("input_query", request.query),
            final_generation=final_state.get("final_generation", "No generation produced."),
            routing_target=final_state.get("routing_target", "unknown"),
            current_loop_count=final_state.get("current_loop_count", 0),
            system_logs=final_state.get("system_logs", [])
        )
    except Exception as e:
        logger.error(f"Error during graph execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG workflow execution failed: {str(e)}")

@app.post("/api/v1/chat/stream")
async def query_rag_stream(request: QueryRequest):
    """
    Runs the user query and streams back LangGraph node state transitions and telemetry logs
    as Server-Sent Events (SSE).
    """
    logger.info(f"Received streaming query request: '{request.query}'")
    
    initial_state = {
        "input_query": request.query,
        "optimized_query": "",
        "retrieved_documents": [],
        "verified_context": [],
        "routing_target": "",
        "current_loop_count": 0,
        "system_logs": [],
        "final_generation": ""
    }
    
    async def event_generator():
        try:
            # astream streams updates when nodes complete execution
            async for event in graph.astream(initial_state):
                # Format event details for SSE
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0.05) # Preservation delay
                
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Direct script entry runner
if __name__ == "__main__":
    import uvicorn
    # Boot server on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
