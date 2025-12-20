from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uuid
import logging
import os

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1-70b-versatile")
MODEL_NAME= os.getenv('MODEL_NAME','llama-3.1-8b-instant')

# ---------------- REQUEST MODELS ----------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class SourceReference(BaseModel):
    chapter: str
    section: str
    preview: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[SourceReference]

# ---------------- LOCAL EMBEDDINGS ----------------
class LocalBGEEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()

# ---------------- INIT ----------------
logger.info("Initializing services...")

embeddings = LocalBGEEmbeddings()

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

vector_store = QdrantVectorStore(
    client=qdrant,
    collection_name=os.getenv("COLLECTION_NAME", "book_chunks"),
    embedding=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 6})

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def format_docs(docs):
    out = []
    for d in docs:
        out.append(
            f"From {d.metadata.get('chapter')} > {d.metadata.get('section')}:\n{d.page_content}"
        )
    return "\n\n".join(out)

prompt = ChatPromptTemplate.from_template("""
You are an expert assistant for the Physical AI textbook.

Context:
{context}

Question: {question}

Rules:
- Answer ONLY from the context
- If not present, say it is not covered
- Clear and student friendly

Answer:
""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---------------- FASTAPI ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        session_id = req.session_id or str(uuid.uuid4())

        answer = await rag_chain.ainvoke(req.message)
        docs = await retriever.ainvoke(req.message)

        sources = [
            SourceReference(
                chapter=d.metadata.get("chapter", ""),
                section=d.metadata.get("section", ""),
                preview=d.page_content[:200]
            )
            for d in docs
        ]

        return ChatResponse(
            response=answer,
            session_id=session_id,
            sources=sources
        )

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))
