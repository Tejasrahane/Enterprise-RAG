from typing import List, TypedDict, Annotated
from operator import add

class AgentGraphState(TypedDict):
    # Core Queries
    input_query: str
    optimized_query: str
    
    # Document state management
    retrieved_documents: List[str]
    verified_context: List[str]
    
    # Telemetry and loop trackers
    routing_target: str
    current_loop_count: int
    system_logs: Annotated[List[dict], add] # Append-only structured trace logs (latency, status, node metadata)
    
    # Output
    final_generation: str
