# """
# Ingestion script for the RAG chatbot system.

# This script parses all MDX/Markdown files from the Docusaurus docs folder,
# chunks the content intelligently with headings/metadata, generates embeddings,
# and upserts the data into Qdrant vector store.
# """

# import asyncio
# import os
# import re
# from pathlib import Path
# from typing import List, Dict, Any
# import logging

# import tiktoken
# from openai import AsyncOpenAI
# from qdrant_client import AsyncQdrantClient
# from qdrant_client.http import models
# from qdrant_client.http.models import PointStruct
# import frontmatter

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize clients
# openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# qdrant_client = AsyncQdrantClient(
#     url=os.getenv("QDRANT_URL"),
#     api_key=os.getenv("QDRANT_API_KEY"),
#     prefer_grpc=True
# )

# # Initialize tokenizer
# tokenizer = tiktoken.get_encoding("cl100k_base")

# # Collection name
# COLLECTION_NAME = os.getenv("COLLECTION_NAME", "book_chunks")

# async def num_tokens_from_string(string: str) -> int:
#     """Returns the number of tokens in a text string."""
#     return len(tokenizer.encode(string))

# def extract_headings(content: str) -> List[Dict[str, Any]]:
#     """Extract headings from content with their positions."""
#     headings = []
#     lines = content.splitlines()
    
#     for i, line in enumerate(lines):
#         # Match markdown headings (h1, h2, h3, etc.)
#         match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
#         if match:
#             heading_level = len(match.group(1))
#             heading_text = match.group(2).strip()
#             headings.append({
#                 'line': i,
#                 'level': heading_level,
#                 'text': heading_text
#             })
    
#     return headings

# def get_section_context(headings: List[Dict], current_line: int) -> tuple:
#     """Get the most relevant section context for a given line position."""
#     relevant_heading = None
#     for heading in reversed(headings):
#         if heading['line'] < current_line:
#             relevant_heading = heading
#             break
    
#     if relevant_heading:
#         return relevant_heading['text'], relevant_heading['level']
#     return "Introduction", 1

# def chunk_content(content: str, max_tokens: int = 500) -> List[Dict[str, Any]]:
#     """Chunk content intelligently, preserving document structure."""
#     lines = content.splitlines()
#     chunks = []
#     current_chunk = []
#     current_tokens = 0
#     headings = extract_headings(content)
    
#     for line in lines:
#         line_tokens = len(tokenizer.encode(line))
        
#         # If adding this line would exceed the token limit
#         if current_tokens + line_tokens > max_tokens and current_chunk:
#             # Save the current chunk
#             chunk_content = "\n".join(current_chunk).strip()
#             if chunk_content:
#                 # Get section context for this chunk
#                 first_line_idx = lines.index(current_chunk[0])
#                 section_title, _ = get_section_context(headings, first_line_idx)
                
#                 chunks.append({
#                     'content': chunk_content,
#                     'section': section_title
#                 })
            
#             # Start a new chunk with the current line
#             current_chunk = [line]
#             current_tokens = line_tokens
#         else:
#             current_chunk.append(line)
#             current_tokens += line_tokens
    
#     # Add the last chunk if it has content
#     if current_chunk:
#         chunk_content = "\n".join(current_chunk).strip()
#         if chunk_content:
#             first_line_idx = lines.index(current_chunk[0])
#             section_title, _ = get_section_context(headings, first_line_idx)
            
#             chunks.append({
#                 'content': chunk_content,
#                 'section': section_title
#             })
    
#     return chunks

# async def get_embedding(text: str) -> List[float]:
#     """Get embedding for text using OpenAI API."""
#     response = await openai_client.embeddings.create(
#         input=text,
#         model="text-embedding-3-small"
#     )
#     return response.data[0].embedding

# async def ensure_collection_exists():
#     """Ensure the Qdrant collection exists with proper configuration."""
#     try:
#         # Try to get collection info
#         await qdrant_client.get_collection(COLLECTION_NAME)
#         logger.info(f"Collection '{COLLECTION_NAME}' already exists")
#     except Exception:
#         # Create collection if it doesn't exist
#         await qdrant_client.create_collection(
#             collection_name=COLLECTION_NAME,
#             vectors_config=models.VectorParams(
#                 size=1536,  # Size of text-embedding-3-small vectors
#                 distance=models.Distance.COSINE
#             ),
#         )
#         logger.info(f"Created collection '{COLLECTION_NAME}'")

# async def process_file(file_path: Path, source_dir: Path) -> List[PointStruct]:
#     """Process a single MDX/Markdown file and return Qdrant points."""
#     logger.info(f"Processing file: {file_path}")
    
#     # Extract frontmatter and content
#     with open(file_path, 'r', encoding='utf-8') as f:
#         post = frontmatter.load(f)
#         content = post.content
#         metadata = post.metadata
    
#     # Extract chapter/section info from file path and frontmatter
#     relative_path = file_path.relative_to(source_dir)
#     file_parts = relative_path.parts
    
#     chapter = metadata.get('title', file_parts[-1].replace('.mdx', '').replace('.md', ''))
    
#     # Add directory name as part of the section info if available
#     if len(file_parts) > 1:
#         section_info = " > ".join(file_parts[:-1])
#         if section_info:
#             chapter = f"{section_info} > {chapter}"
    
#     # Chunk the content
#     chunks = chunk_content(content)
    
#     points = []
#     for i, chunk in enumerate(chunks):
#         # Create a unique ID for this chunk
#         chunk_id = f"{file_path.name}-{i}"
        
#         # Prepare content with context
#         full_content = f"Chapter: {chapter}\nSection: {chunk['section']}\n\n{chunk['content']}"
        
#         # Get embedding
#         embedding = await get_embedding(full_content)
        
#         # Create metadata for the point
#         point_metadata = {
#             "chapter": chapter,
#             "section": chunk['section'],
#             "title": chunk['section'],
#             "source_file": str(relative_path),
#             "file_name": file_path.name,
#             "content_preview": chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
#         }
        
#         # Add any additional metadata from frontmatter
#         for key, value in metadata.items():
#             if key not in point_metadata:
#                 point_metadata[key] = str(value) if not isinstance(value, (str, int, float, bool)) else value
        
#         # Create point
#         point = PointStruct(
#             id=chunk_id,
#             vector=embedding,
#             payload=point_metadata
#         )
        
#         points.append(point)
    
#     logger.info(f"Processed {len(points)} chunks from {file_path}")
#     return points

# async def ingest_docs(docs_dir: str = "C:/Spec Kit Plus/ai-native-book-with-chatbot/docs"):
#     """Main ingestion function to process all MDX/Markdown files."""
#     logger.info("Starting ingestion process...")
    
#     # Ensure the collection exists
#     await ensure_collection_exists()
    
#     # Get all MDX and Markdown files
#     docs_path = Path(docs_dir)
#     if not docs_path.exists():
#         logger.error(f"Docs directory does not exist: {docs_dir}")
#         return
    
#     md_files = list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.mdx"))
    
#     if not md_files:
#         logger.warning(f"No MD or MDX files found in {docs_dir}")
#         return
    
#     logger.info(f"Found {len(md_files)} files to process")
    
#     # Process all files
#     all_points = []
#     for file_path in md_files:
#         try:
#             points = await process_file(file_path, docs_path)
#             all_points.extend(points)
#         except Exception as e:
#             logger.error(f"Error processing file {file_path}: {str(e)}")
#             continue
    
#     # Upsert all points to Qdrant
#     if all_points:
#         logger.info(f"Upserting {len(all_points)} points to Qdrant...")
#         await qdrant_client.upsert(
#             collection_name=COLLECTION_NAME,
#             points=all_points,
#             wait=True
#         )
#         logger.info("Ingestion completed successfully!")
#     else:
#         logger.warning("No points to upsert - check if files were processed correctly")

# async def main():
#     """Main entry point for the ingestion script."""
#     # Check required environment variables
#     required_env_vars = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
#     missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
#     if missing_vars:
#         logger.error(f"Missing required environment variables: {missing_vars}")
#         return
    
#     try:
#         await ingest_docs()
#     except Exception as e:
#         logger.error(f"Error during ingestion: {str(e)}")

# if __name__ == "__main__":
#     asyncio.run(main())