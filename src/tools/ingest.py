import os
import glob
import uuid
import logging
from typing import List
from config.settings import settings
from config.logger import setup_logger
from src.nodes.retriever import get_chroma_collection, get_embedder

logger = setup_logger("tools.ingest")

def write_sample_data():
    """Writes default sample specifications to data/raw/ if empty."""
    raw_dir = "./data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    sample_files = {
        "project_aetheris_specs.txt": (
            "Project Aetheris is a next-generation quantum computing initiative. "
            "The Project Aetheris quantum CPU utilizes a 128-qubit topological architecture operating at 15 millikelvin. "
            "It achieved a quantum volume of 16,777,216 (2^24) in June 2026."
        ),
        "xyz_q2_2026_report.txt": (
            "Company XYZ reported a record net profit of $4.2 billion for Q2 2026, "
            "driven by a 45% year-over-year increase in cloud AI infrastructure sales and enterprise contracts."
        ),
        "nova9_propulsion_manual.txt": (
            "The Nova-9 propulsion system is a hybrid ion-chemical drive developed by Stellarex. "
            "It is designed specifically for deep-space transit to Mars and has a specific impulse of 3,200 seconds, "
            "running on liquid xenon fuel."
        ),
        "stellarex_newsletter.txt": (
            "Under the Stellarex Deep Space program, the Nova-9 engine achieved a continuous burn test of 500 hours, "
            "validating its thermal shielding and thrust vectoring capabilities."
        )
    }
    
    # Check if raw files exist
    existing_files = glob.glob(os.path.join(raw_dir, "*"))
    if not existing_files:
        logger.info("No raw files found in data/raw/. Generating default sample documents...")
        for filename, content in sample_files.items():
            filepath = os.path.join(raw_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Created sample raw file: '{filepath}'")
    else:
        logger.info(f"Found {len(existing_files)} existing files in data/raw/. Skipping generation.")

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Splits a long document string into overlapping chunks."""
    words = text.split()
    chunks = []
    
    # Simple word-count chunking fallback
    step = chunk_size - chunk_overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        if chunk:
            chunks.append(chunk)
            
    return chunks

def ingest_all_documents():
    """Reads raw files, chunks them, embeds them with open-source SentenceTransformers, and uploads to Chroma DB."""
    # Ensure sample raw files exist
    write_sample_data()
    
    raw_dir = "./data/raw"
    txt_files = glob.glob(os.path.join(raw_dir, "*.txt"))
    md_files = glob.glob(os.path.join(raw_dir, "*.md"))
    all_files = txt_files + md_files
    
    if not all_files:
        logger.warning("No text or markdown files found to ingest.")
        return
        
    logger.info(f"Beginning open-source ingestion for {len(all_files)} files...")
    
    # Get collection
    collection = get_chroma_collection()
    if collection is None:
        logger.error("Could not obtain Chroma collection. Ingestion aborted.")
        return
        
    embedder = get_embedder()
    
    # Process files
    total_chunks = 0
    for filepath in all_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            filename = os.path.basename(filepath)
            chunks = chunk_text(content, chunk_size=150, chunk_overlap=20)
            
            if not chunks:
                continue
                
            ids = [f"{filename}_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
            
            # Embed chunks with active embeddings provider
            if embedder:
                raw_emb = embedder.encode(chunks, convert_to_numpy=True)
                embeddings = raw_emb.tolist() if hasattr(raw_emb, "tolist") else raw_emb
                collection.upsert(
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                # Add text directly (let Chroma use its default embedding model or keyword matching)
                collection.upsert(
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids
                )
                
            total_chunks += len(chunks)
            logger.info(f"Ingested file '{filename}': Created {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Error processing file '{filepath}': {e}", exc_info=True)
            
    logger.info(f"Ingestion completed! Total processed chunks added: {total_chunks}")
    logger.info(f"Current collection count: {collection.count()} items.")

if __name__ == "__main__":
    ingest_all_documents()

