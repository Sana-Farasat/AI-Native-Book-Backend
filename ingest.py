"""
Ingestion script for RAG chatbot with local embeddings (free, no API cost)
Reads all markdown files from project-root `docs/` folder,
chunks content intelligently, generates embeddings, and upserts to Qdrant.
"""

import asyncio
import os
import re
import random
from pathlib import Path
from typing import List, Dict, Any
import logging

from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import PointStruct
import frontmatter
from dotenv import load_dotenv

load_dotenv()

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Paths
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR.parent / "docs"  # ROOT/docs

# -----------------------
# Local embedding model
# -----------------------
logger.info("Loading local embedding model BAAI/bge-large-en-v1.5 (~1GB, first-time download)")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def get_embedding(text: str) -> List[float]:
    """Get embedding from local model (sync)"""
    return model.encode(text, normalize_embeddings=True).tolist()

# -----------------------
# Qdrant client
# -----------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "book_chunks")

# Increase timeout to 120s for larger batches
qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True, timeout=120)

# -----------------------
# Markdown utilities
# -----------------------
def extract_headings(content: str) -> List[Dict[str, Any]]:
    """Extract markdown headings with their line numbers"""
    headings = []
    lines = content.splitlines()
    for i, line in enumerate(lines):
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            headings.append({
                "line": i,
                "level": len(match.group(1)),
                "text": match.group(2).strip()
            })
    return headings

def get_section_context(headings: List[Dict], current_line: int) -> tuple:
    """Get nearest heading above the line"""
    relevant_heading = None
    for heading in reversed(headings):
        if heading['line'] < current_line:
            relevant_heading = heading
            break
    if relevant_heading:
        return relevant_heading['text'], relevant_heading['level']
    return "Introduction", 1

def chunk_content(content: str, max_tokens: int = 500) -> List[Dict[str, Any]]:
    """Chunk content preserving headings"""
    lines = content.splitlines()
    chunks, current_chunk, current_tokens = [], [], 0
    headings = extract_headings(content)

    for line in lines:
        line_tokens = len(line.split())  # rough token estimate
        if current_tokens + line_tokens > max_tokens and current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                first_line_idx = lines.index(current_chunk[0])
                section_title, _ = get_section_context(headings, first_line_idx)
                chunks.append({"content": chunk_text, "section": section_title})
            current_chunk = [line]
            current_tokens = line_tokens
        else:
            current_chunk.append(line)
            current_tokens += line_tokens

    if current_chunk:
        chunk_text = "\n".join(current_chunk).strip()
        if chunk_text:
            first_line_idx = lines.index(current_chunk[0])
            section_title, _ = get_section_context(headings, first_line_idx)
            chunks.append({"content": chunk_text, "section": section_title})

    return chunks

# -----------------------
# Qdrant utilities
# -----------------------
async def ensure_collection_exists():
    """Ensure Qdrant collection exists with correct vector size"""
    try:
        await qdrant_client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted old collection '{COLLECTION_NAME}'")
    except Exception:
        pass  # ignore if it doesn't exist

    await qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1024,  # BGE embedding dim
            distance=models.Distance.COSINE
        ),
    )
    logger.info(f"Created collection '{COLLECTION_NAME}' with 1024-dim vectors")

async def process_file(file_path: Path) -> List[PointStruct]:
    """Process a single markdown file into Qdrant points"""
    logger.info(f"Processing file: {file_path.name}")
    try:
        post = frontmatter.load(file_path)
        content = post.content
        metadata = post.metadata
    except Exception as e:
        logger.error(f"Failed to parse {file_path.name}: {e}")
        return []

    chapter = metadata.get("title", file_path.stem)
    chunks = chunk_content(content)
    points = []

    for chunk in chunks:
        chunk_id = random.randint(1_000_000, 9_999_999)  # Qdrant-safe integer ID
        full_content = f"Chapter: {chapter}\nSection: {chunk['section']}\n\n{chunk['content']}"
        embedding = get_embedding(full_content)
        payload = {
            "chapter": chapter,
            "section": chunk['section'],
            "title": chunk['section'],
            "source_file": str(file_path.name),
            "content_preview": chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
        }
        points.append(PointStruct(id=chunk_id, vector=embedding, payload=payload))
    logger.info(f"Processed {len(points)} chunks from {file_path.name}")
    return points

# -----------------------
# Main ingestion with batching
# -----------------------
BATCH_SIZE = 20  # smaller batch size avoids Deadline Exceeded

async def ingest_docs():
    logger.info(f"Reading markdown files from: {DOCS_DIR.resolve()}")
    if not DOCS_DIR.exists():
        logger.error(f"Docs folder not found: {DOCS_DIR}")
        return

    md_files = list(DOCS_DIR.rglob("*.md")) + list(DOCS_DIR.rglob("*.mdx"))
    if not md_files:
        logger.error("❌ No markdown files found")
        return

    logger.info(f"📄 Found {len(md_files)} markdown files")
    await ensure_collection_exists()
    all_points = []

    for file in md_files:
        points = await process_file(file)
        all_points.extend(points)

    if all_points:
        logger.info(f"Upserting {len(all_points)} points to Qdrant in batches of {BATCH_SIZE}...")
        for i in range(0, len(all_points), BATCH_SIZE):
            batch = all_points[i:i + BATCH_SIZE]
            await qdrant_client.upsert(collection_name=COLLECTION_NAME, points=batch)
            logger.info(f"✅ Upserted batch {i//BATCH_SIZE + 1} ({len(batch)} points)")
        logger.info("✅ Ingestion completed successfully!")
    else:
        logger.warning("No points to upsert")

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    asyncio.run(ingest_docs())
