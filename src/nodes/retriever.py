import os
import time
import logging
import asyncio
from typing import Dict, Any, List
from config.settings import settings
from config.logger import setup_logger
from src.state import AgentGraphState

logger = setup_logger("nodes.retriever")

# Global reference for Chroma Client, collection, and embedder
_chroma_client = None
_collection = None
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder

    google_api_key = (
        settings.GOOGLE_API_KEY or 
        settings.GEMINI_API_KEY or 
        os.getenv("GOOGLE_API_KEY") or 
        os.getenv("GEMINI_API_KEY") or 
        ""
    ).strip()

    # Cloud Google GenAI Embeddings (zero server RAM/GPU compute needed)
    if settings.EMBEDDING_PROVIDER == "gemini" or (google_api_key and settings.EMBEDDING_PROVIDER != "local"):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            class GoogleEmbedderWrapper:
                def __init__(self, model_name: str, api_key: str):
                    if api_key:
                        self.client = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
                    else:
                        self.client = GoogleGenerativeAIEmbeddings(model=model_name)
                def encode(self, texts: List[str], convert_to_numpy: bool = True):
                    return self.client.embed_documents(texts)
            _embedder = GoogleEmbedderWrapper(settings.GEMINI_EMBEDDING_MODEL, google_api_key)
            logger.info(f"Loaded Google GenAI Cloud Embeddings: '{settings.GEMINI_EMBEDDING_MODEL}'")
            return _embedder
        except Exception as e:
            logger.warning(f"Could not load Google GenAI Embeddings: {e}. Falling back to SentenceTransformers.")

    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        return _embedder
    except Exception as e:
        logger.warning(f"Could not load SentenceTransformer embedder: {e}")
        return None

def get_chroma_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection
        
    import chromadb
    
    # 1. Try to connect to HTTP Server (Docker container)
    try:
        _chroma_client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
        _chroma_client.heartbeat()
        logger.info(f"Connected to ChromaDB HTTP Server at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}")
    except Exception as e:
        logger.warning(f"Could not connect to ChromaDB HTTP server: {e}. Falling back to PersistentClient.")
        # 2. Fallback to Persistent Client
        try:
            _chroma_client = chromadb.PersistentClient(path=settings.DB_PERSIST_DIR)
            logger.info(f"Initialized ChromaDB PersistentClient at path: {settings.DB_PERSIST_DIR}")
        except Exception as ex:
            logger.error(f"Failed to initialize ChromaDB PersistentClient: {ex}")
            return None
            
    # Get or create collection
    try:
        _collection = _chroma_client.get_or_create_collection(name=settings.DB_COLLECTION_NAME)
        logger.info(f"Collection '{settings.DB_COLLECTION_NAME}' is ready.")
    except Exception as e:
        logger.error(f"Error accessing collection: {e}")
        _collection = None
        
    return _collection

def query_chroma(query: str, k: int = 2) -> List[str]:
    collection = get_chroma_collection()
    if collection is None:
        return []
        
    # Check if collection is empty
    try:
        count = collection.count()
        if count == 0:
            logger.warning("Chroma collection is empty.")
            return []
    except Exception as e:
        logger.error(f"Failed to count collection items: {e}")
        return []
        
    embedder = get_embedder()
    
    if embedder is not None:
        try:
            raw_query_vector = embedder.encode([query], convert_to_numpy=True)
            query_vector = raw_query_vector.tolist() if hasattr(raw_query_vector, "tolist") else raw_query_vector
            
            results = collection.query(
                query_embeddings=query_vector,
                n_results=k
            )
            documents = []
            if results and "documents" in results and results["documents"]:
                documents = results["documents"][0]
            logger.info(f"Retrieved {len(documents)} documents using active Embeddings provider.")
            return documents
        except Exception as e:
            logger.error(f"Error querying with Embeddings: {e}. Falling back to text search.")
            
    # Text-matching Fallback (for offline/pure-text run)
    try:
        all_docs = collection.get()
        doc_texts = all_docs.get("documents", [])
        
        if not doc_texts:
            return []
            
        # Simple word-overlap scoring
        query_lower = query.lower()
        query_words = set(query_lower.replace("?", "").split())
        scored = []
        for text in doc_texts:
            text_words = text.lower().split()
            overlap = sum(1 for w in query_words if w in text_words)
            score = overlap / (len(query_words) + 1.0)
            scored.append((text, score))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        retrieved = [text for text, score in scored[:k] if score > 0]
        
        if not retrieved:
            retrieved = doc_texts[:k]
            
        logger.info(f"Retrieved {len(retrieved)} documents using text overlap fallback.")
        return retrieved
    except Exception as e:
        logger.error(f"Error during fallback text search: {e}")
        return []

async def retriever_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Asynchronously retrieves matching documents from ChromaDB using open-source embeddings.
    Runs blocking database I/O inside a separate thread via asyncio.to_thread.
    """
    start_time = time.time()
    query = state.get("optimized_query") or state["input_query"]
    logger.info(f"Retrieving documents asynchronously for query: '{query}'")
    
    status = "success"
    try:
        docs = await asyncio.to_thread(query_chroma, query)
    except Exception as e:
        logger.error(f"Retriever query thread crashed: {e}")
        status = "failed"
        docs = []

    elapsed_ms = int((time.time() - start_time) * 1000)
    
    telemetry = {
        "node": "retriever",
        "action": "document_retrieval",
        "status": status,
        "elapsed_time_ms": elapsed_ms,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "retrieved_count": len(docs)
    }
    
    logger.info(f"Retriever Node completed in {elapsed_ms}ms. Retrieved {len(docs)} documents.")
    
    return {
        "retrieved_documents": docs,
        "system_logs": [telemetry]
    }

