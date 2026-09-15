import os
import logging
from typing import List, Tuple
from langchain_core.documents import Document
from config.settings import settings

logger = logging.getLogger(__name__)

# Try to import sentence_transformers and chromadb
CHROMA_AVAILABLE = False
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    CHROMA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import chromadb or sentence_transformers: {e}. Using fallback vector store.")

# A robust, pure-python fallback vector store
class SimpleInMemoryVectorStore:
    def __init__(self):
        self.documents: List[Document] = []
        logger.info("Initialized SimpleInMemoryVectorStore fallback.")

    def add_documents(self, documents: List[Document]):
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents to Fallback Vector Store.")

    def similarity_search_with_score(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in self.documents:
            doc_words = doc.page_content.lower().split()
            overlap = sum(1 for w in query_words if w in doc_words)
            score = overlap / (len(query_words) + 1.0)
            scored_docs.append((doc, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k)]

# Global vector store reference
_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    sample_docs = [
        Document(
            page_content="Project Aetheris is a next-generation quantum computing initiative. The Project Aetheris quantum CPU utilizes a 128-qubit topological architecture operating at 15 millikelvin. It achieved a quantum volume of 16,777,216 (2^24) in June 2026.",
            metadata={"source": "project_aetheris_specs.txt"}
        ),
        Document(
            page_content="Company XYZ reported a record net profit of $4.2 billion for Q2 2026, driven by a 45% year-over-year increase in cloud AI infrastructure sales and enterprise contracts.",
            metadata={"source": "xyz_q2_2026_report.txt"}
        ),
        Document(
            page_content="The Nova-9 propulsion system is a hybrid ion-chemical drive developed by Stellarex. It is designed specifically for deep-space transit to Mars and has a specific impulse of 3,200 seconds, running on liquid xenon fuel.",
            metadata={"source": "nova9_propulsion_manual.txt"}
        ),
        Document(
            page_content="Under the Stellarex Deep Space program, the Nova-9 engine achieved a continuous burn test of 500 hours, validating its thermal shielding and thrust vectoring capabilities.",
            metadata={"source": "stellarex_newsletter.txt"}
        )
    ]

    if CHROMA_AVAILABLE:
        try:
            from src.nodes.retriever import get_embedder
            embedder = get_embedder()
            client = chromadb.PersistentClient(path=settings.DB_PERSIST_DIR)
            collection = client.get_or_create_collection(name=settings.DB_COLLECTION_NAME)
            
            # Populate if empty
            if collection.count() == 0:
                texts = [d.page_content for d in sample_docs]
                ids = [f"sample_doc_{i}" for i in range(len(sample_docs))]
                metadatas = [d.metadata for d in sample_docs]
                if embedder:
                    raw_emb = embedder.encode(texts, convert_to_numpy=True)
                    embeddings = raw_emb.tolist() if hasattr(raw_emb, "tolist") else raw_emb
                    collection.upsert(
                        documents=texts,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                else:
                    collection.upsert(
                        documents=texts,
                        metadatas=metadatas,
                        ids=ids
                    )
            logger.info("ChromaDB vector store successfully created/connected with active Embeddings provider.")
            
            class ChromaStoreWrapper:
                def __init__(self, coll, emb):
                    self.coll = coll
                    self.emb = emb
                def similarity_search(self, query: str, k: int = 2) -> List[Document]:
                    if self.emb:
                        raw_q_emb = self.emb.encode([query], convert_to_numpy=True)
                        q_emb = raw_q_emb.tolist() if hasattr(raw_q_emb, "tolist") else raw_q_emb
                        res = self.coll.query(query_embeddings=q_emb, n_results=k)
                    else:
                        res = self.coll.query(query_texts=[query], n_results=k)
                    docs = []
                    if res and "documents" in res and res["documents"]:
                        for text, meta in zip(res["documents"][0], res["metadatas"][0] if res.get("metadatas") else [{}] * len(res["documents"][0])):
                            docs.append(Document(page_content=text, metadata=meta or {}))
                    return docs

            _vector_store = ChromaStoreWrapper(collection, embedder)
            return _vector_store
        except Exception as e:
            logger.error(f"Error initializing ChromaDB with Embeddings: {e}. Falling back to In-Memory store.")
    
    # Fallback to SimpleInMemoryStore
    _vector_store = SimpleInMemoryVectorStore()
    _vector_store.add_documents(sample_docs)
    return _vector_store

def retrieve_from_store(query: str, k: int = 2) -> List[Document]:
    store = get_vector_store()
    try:
        return store.similarity_search(query, k=k)
    except Exception as e:
        logger.error(f"Error during similarity search: {e}. Returning empty list.")
        return []

