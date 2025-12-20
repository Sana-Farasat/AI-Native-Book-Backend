# from fastapi import FastAPI, HTTPException, Depends, Request
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Optional, List, Dict, Any
# import os
# import uuid
# import asyncio
# import logging
# from contextlib import asynccontextmanager
# from datetime import datetime

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import Qdrant
# from qdrant_client import QdrantClient
# import pytz

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Models
# class ChatRequest(BaseModel):
#     message: str
#     session_id: Optional[str] = None
#     selected_text: Optional[str] = None
#     stream: Optional[bool] = True
#     temperature: Optional[float] = 0.7

# class SourceReference(BaseModel):
#     chunk_id: str
#     chapter: str
#     section: str
#     title: str
#     relevance_score: float
#     text_preview: str

# class ChatResponse(BaseModel):
#     response: str
#     session_id: str
#     sources: List[SourceReference]
#     latency: float
#     timestamp: datetime

# class HistoryRequest(BaseModel):
#     session_id: str
#     limit: Optional[int] = 50

# class Message(BaseModel):
#     id: str
#     session_id: str
#     role: str
#     content: str
#     timestamp: datetime
#     sources: Optional[List[Dict]] = None
#     token_count: Optional[int] = None
#     latency: Optional[float] = None

# class HistoryResponse(BaseModel):
#     session_id: str
#     messages: List[Message]
#     total_count: int


# # Global variables to hold our services
# qdrant_client = None
# llm = None
# retriever = None


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Initialize services on startup"""
#     global qdrant_client, llm, retriever
    
#     # Initialize Qdrant client
#     qdrant_client = QdrantClient(
#         url=os.getenv("QDRANT_URL"),
#         api_key=os.getenv("QDRANT_API_KEY"),
#     )
    
#     # Initialize embeddings and LLM
#     embeddings = OpenAIEmbeddings(
#         model="text-embedding-3-small",
#         openai_api_key=os.getenv("OPENAI_API_KEY")
#     )
    
#     llm = ChatOpenAI(
#         model="gpt-4o-mini",
#         temperature=0.7,
#         openai_api_key=os.getenv("OPENAI_API_KEY")
#     )
    
#     # Initialize Qdrant vector store and retriever
#     vector_store = Qdrant(
#         client=qdrant_client,
#         collection_name=os.getenv("COLLECTION_NAME", "book_chunks"),
#         embeddings=embeddings,
#     )
    
#     retriever = vector_store.as_retriever(
#         search_kwargs={
#             "k": 5,  # Retrieve top 5 most relevant chunks
#             "filter": None  # No specific filters for now
#         }
#     )
    
#     logger.info("Services initialized successfully")
#     yield
    
#     # Cleanup on shutdown
#     logger.info("Shutting down services")


# # Initialize FastAPI app
# app = FastAPI(
#     title="RAG Chatbot API",
#     description="API for the Physical AI and Humanoid Robotics textbook RAG chatbot",
#     version="1.0.0",
#     lifespan=lifespan
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# def format_docs(docs):
#     """Format retrieved documents for the prompt."""
#     return "\n\n".join([doc.page_content for doc in docs])


# # Define the RAG chain
# template = """
# You are an AI assistant for the Physical AI and Humanoid Robotics textbook. Your purpose is to answer questions about the content of this book.

# Only use the following context to answer the question. Do not use any external knowledge.

# Context:
# {context}

# Question: {question}

# Instructions:
# 1. Answer only based on the provided context
# 2. If the context doesn't contain the answer, respond with: "I can only answer questions based on the content of this book. That information is not covered here."
# 3. Reference relevant book sections where possible
# 4. Keep the answer concise but informative
# 5. Use student-friendly language as much as possible

# Answer:
# """
# prompt = ChatPromptTemplate.from_template(template)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )


# @app.post("/v1/chat", response_model=ChatResponse)
# async def chat_endpoint(chat_request: ChatRequest):
#     """Handle chat requests with RAG"""
#     start_time = datetime.now(pytz.utc)
    
#     try:
#         # Create or validate session ID
#         session_id = chat_request.session_id or str(uuid.uuid4())
        
#         # Prepare the query, including selected text if provided
#         query = chat_request.message
#         if chat_request.selected_text:
#             query = f"Explain this selected text: {chat_request.selected_text}\n\nQuestion about this: {chat_request.message}"
        
#         # Get response from RAG chain
#         response_text = await rag_chain.ainvoke(query)
        
#         # For now, we're not capturing sources properly in this simplified version
#         # In a full implementation, we would capture which documents were used
#         sources = []
        
#         # Calculate latency
#         end_time = datetime.now(pytz.utc)
#         latency = (end_time - start_time).total_seconds()
        
#         # Create response
#         response = ChatResponse(
#             response=response_text,
#             session_id=session_id,
#             sources=sources,
#             latency=latency,
#             timestamp=end_time
#         )
        
#         # TODO: Save to conversation history in Postgres
        
#         return response
        
#     except Exception as e:
#         logger.error(f"Error in chat endpoint: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/v1/sessions/clear")
# async def clear_session(session_data: dict):
#     """Clear conversation history for a session"""
#     # TODO: Implement session clearing in Postgres
#     # For now, just return success
#     session_id = session_data.get("session_id")
#     return {"session_id": session_id, "message": "Session history cleared successfully"}


# @app.get("/v1/history", response_model=HistoryResponse)
# async def get_history(history_request: HistoryRequest = Depends()):
#     """Retrieve conversation history for a session"""
#     # TODO: Implement history retrieval from Postgres
#     # For now, return an empty history
#     return HistoryResponse(
#         session_id=history_request.session_id,
#         messages=[],
#         total_count=0
#     )


# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {"status": "healthy"}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)