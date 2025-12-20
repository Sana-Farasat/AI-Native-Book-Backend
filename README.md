# RAG Chatbot Backend

This is the backend service for the Physical AI and Humanoid Robotics textbook RAG chatbot. It provides a REST API for conversational interactions with the book content using retrieval-augmented generation.

## Features

- FastAPI-based REST API
- LangChain-powered RAG pipeline
- Qdrant vector store for book content
- OpenAI integration for embeddings and generation
- Anonymous session management
- Streaming responses via Server-Sent Events

## Prerequisites

- Python 3.11+
- An OpenAI API key
- Access to Qdrant Cloud (free tier account)
- Neon Serverless Postgres account (free tier)

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   # Edit .env with your specific configuration
   ```

## Usage

### Running the server

```bash
uvicorn src.api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`, with interactive documentation at `http://localhost:8000/docs`.

### Ingesting book content

Before using the chatbot, you need to ingest the book content into the vector database:

```bash
python -m src.ingestion.ingest
```

This will parse all MDX/Markdown files from the Docusaurus docs folder, chunk the content intelligently with headings/metadata, generate embeddings, and upsert the data into Qdrant.

### API Endpoints

- `POST /v1/chat` - Chat endpoint with RAG
- `GET /v1/history` - Retrieve conversation history
- `GET /health` - Health check

## Environment Variables

See `.env.example` for all required environment variables.

## Architecture

This backend implements a RAG (Retrieval-Augmented Generation) system:

1. Book content is ingested and stored in a Qdrant vector database with metadata
2. When a query comes in, relevant content is retrieved from the vector database
3. The retrieved content is used to augment the prompt for the LLM
4. The LLM generates a response based only on the provided context
5. Response is streamed back to the client if requested

This ensures responses are grounded in the book content and prevents hallucinations.# AI-Native-Book-Backend
