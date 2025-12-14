"""
Module: api_server.py
Task 3: REST API with Conversation Context

This module provides a FastAPI application with:
- General chatbot endpoint (Task 1)
- CV-specific RAG endpoint (Task 2)
- Conversation history management (Task 3)
- Automatic API documentation (Swagger UI)
- Health checks
- Error handling

API Endpoints:
- POST /chat/general - General questions (Task 1)
- POST /chat/cv - CV-specific questions (Task 2 with RAG)
- GET /chat/history/{session_id} - Get conversation history
- DELETE /chat/history/{session_id} - Clear conversation history
- GET /health - Health check
- GET /stats - System statistics
"""

import os
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import uuid

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from dotenv import load_dotenv

from modules.llm_handler import LLMHandler
from modules.rag_engine import AdvancedRAGEngine
from modules.conversation_manager import ConversationManager


# Load environment variables
load_dotenv()


# ===== Pydantic Models =====

class ChatRequest(BaseModel):
    """Request model for chat endpoints."""
    query: str = Field(..., description="User's question", min_length=1)
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session ID for conversation tracking"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are my technical skills?",
                "session_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


class Source(BaseModel):
    """Source citation from CV."""
    page: int = Field(..., description="Page number in CV")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    confidence: float = Field(..., description="Confidence score (0-1)")
    preview: str = Field(..., description="Text preview from source")


class ChatResponse(BaseModel):
    """Response model for chat endpoints."""
    answer: str = Field(..., description="Generated answer")
    session_id: str = Field(..., description="Session ID")
    sources: Optional[List[Source]] = Field(None, description="Source citations (for CV queries)")
    conversation_history: List[Dict[str, str]] = Field(
        ...,
        description="Recent conversation turns"
    )
    metadata: Optional[Dict] = Field(None, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "I have 5 years of experience with Python...",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "sources": [
                    {
                        "page": 1,
                        "chunk_id": "page_1_chunk_0",
                        "confidence": 0.95,
                        "preview": "Python developer with 5 years..."
                    }
                ],
                "conversation_history": [
                    {"role": "user", "content": "What are my skills?"},
                    {"role": "assistant", "content": "Your main skills are..."}
                ],
                "metadata": {
                    "num_chunks_used": 3,
                    "query_enhanced": True
                }
            }
        }


class HistoryResponse(BaseModel):
    """Response model for conversation history."""
    session_id: str
    num_turns: int
    history: List[Dict[str, str]]
    session_info: Dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    llm_provider: str
    llm_model: str
    rag_enabled: bool
    vector_db_count: int


class StatsResponse(BaseModel):
    """System statistics response."""
    llm_info: Dict
    rag_stats: Dict
    conversation_stats: Dict
    cv_loaded: bool


# ===== Application Lifecycle =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes components on startup, cleans up on shutdown.
    """
    # Startup
    logger.info("🚀 Starting RAG Chatbot API...")
    
    # Initialize LLM
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    logger.info(f"Initializing LLM with provider: {provider}")
    
    if provider == "groq":
        app.state.llm = LLMHandler(
            provider="groq",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        )
    else:
        app.state.llm = LLMHandler(
            provider="ollama",
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        )
    
    # Initialize RAG Engine
    logger.info("Initializing Advanced RAG Engine...")
    app.state.rag = AdvancedRAGEngine(
        llm_handler=app.state.llm,
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./vector_store"),
        collection_name=os.getenv("COLLECTION_NAME", "cv_documents"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", "10")),
        top_k_rerank=int(os.getenv("TOP_K_RERANK", "3"))
    )
    
    # Load CV if available
    cv_path = os.getenv("CV_PATH", "./data/cv.pdf")
    if os.path.exists(cv_path):
        logger.info(f"Loading CV from: {cv_path}")
        app.state.rag.load_pdf(cv_path)
        app.state.cv_loaded = True
    else:
        logger.warning(f"CV not found at {cv_path}. RAG functionality will be limited.")
        app.state.cv_loaded = False
    
    # Initialize Conversation Manager
    logger.info("Initializing Conversation Manager...")
    app.state.conversation_manager = ConversationManager(
        max_history_per_session=int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))
    )
    
    logger.info("✅ RAG Chatbot API is ready!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down RAG Chatbot API...")


# ===== FastAPI Application =====

app = FastAPI(
    title="Advanced RAG Chatbot API",
    description="A sophisticated chatbot with RAG capabilities, conversation context, and CV-specific Q&A",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== API Endpoints =====

@app.post(
    "/chat/general",
    response_model=ChatResponse,
    summary="General Chat (Task 1)",
    description="Ask any general question. Uses lightweight LLM without RAG."
)
async def chat_general(request: ChatRequest) -> ChatResponse:
    """
    General chatbot endpoint (Task 1).
    
    Answers any general question using the base LLM.
    Maintains conversation context.
    """
    try:
        # Get conversation history
        history = app.state.conversation_manager.get_history_as_messages(
            request.session_id
        )
        
        # Generate response
        answer = await app.state.llm.generate_response(
            query=request.query,
            conversation_history=history[-6:] if len(history) > 6 else history  # Last 3 turns
        )
        
        # Store in conversation history
        app.state.conversation_manager.add_turn(
            session_id=request.session_id,
            question=request.query,
            answer=answer
        )
        
        # Get updated history
        updated_history = app.state.conversation_manager.get_history_as_messages(
            request.session_id
        )
        
        return ChatResponse(
            answer=answer,
            session_id=request.session_id,
            sources=None,
            conversation_history=updated_history,
            metadata={
                "type": "general",
                "model": app.state.llm.model
            }
        )
        
    except Exception as e:
        logger.error(f"Error in general chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )


@app.post(
    "/chat/cv",
    response_model=ChatResponse,
    summary="CV-Specific Chat (Task 2)",
    description="Ask questions about the CV. Uses Advanced RAG with hybrid search, re-ranking, and source citations."
)
async def chat_cv(request: ChatRequest) -> ChatResponse:
    """
    CV-specific chatbot endpoint (Task 2).
    
    Answers questions about the CV using Advanced RAG:
    - Hybrid search (BM25 + Vector)
    - Cross-encoder re-ranking
    - Query enhancement
    - Source citations
    
    Maintains conversation context.
    """
    try:
        if not app.state.cv_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CV has not been loaded. Please ensure cv.pdf is in the data/ directory."
            )
        
        # Query RAG system
        rag_result = await app.state.rag.query(
            question=request.query,
            use_query_enhancement=True,
            use_hybrid_search=True,
            use_reranking=True
        )
        
        # Store in conversation history
        app.state.conversation_manager.add_turn(
            session_id=request.session_id,
            question=request.query,
            answer=rag_result["answer"]
        )
        
        # Get conversation history
        history = app.state.conversation_manager.get_history_as_messages(
            request.session_id
        )
        
        # Convert sources to Pydantic models
        sources = [
            Source(**source) for source in rag_result["sources"]
        ]
        
        return ChatResponse(
            answer=rag_result["answer"],
            session_id=request.session_id,
            sources=sources,
            conversation_history=history,
            metadata={
                "type": "cv_rag",
                "num_chunks_retrieved": rag_result["num_chunks_retrieved"],
                "num_chunks_used": rag_result["num_chunks_used"],
                "query_enhanced": rag_result["query_enhanced"],
                "enhanced_query": rag_result.get("enhanced_query")
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in CV chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )


@app.get(
    "/chat/history/{session_id}",
    response_model=HistoryResponse,
    summary="Get Conversation History",
    description="Retrieve conversation history for a specific session."
)
async def get_history(session_id: str) -> HistoryResponse:
    """Get conversation history for a session."""
    try:
        session = app.state.conversation_manager.get_session(session_id)
        history = session.get_history_as_messages()
        
        return HistoryResponse(
            session_id=session_id,
            num_turns=len(session.history),
            history=history,
            session_info=session.get_info()
        )
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@app.delete(
    "/chat/history/{session_id}",
    summary="Clear Conversation History",
    description="Clear all conversation history for a specific session."
)
async def clear_history(session_id: str) -> Dict[str, str]:
    """Clear conversation history for a session."""
    try:
        success = app.state.conversation_manager.clear_session(session_id)
        
        if success:
            return {
                "status": "success",
                "message": f"History cleared for session {session_id}"
            }
        else:
            return {
                "status": "info",
                "message": f"No history found for session {session_id}"
            }
            
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear history: {str(e)}"
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is running and all components are initialized."
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        llm_info = app.state.llm.get_info()
        rag_stats = app.state.rag.get_stats()
        
        return HealthResponse(
            status="healthy",
            llm_provider=llm_info["provider"],
            llm_model=llm_info["model"],
            rag_enabled=app.state.cv_loaded,
            vector_db_count=rag_stats["vector_db_count"]
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="System Statistics",
    description="Get detailed statistics about the system."
)
async def get_stats() -> StatsResponse:
    """Get system statistics."""
    try:
        return StatsResponse(
            llm_info=app.state.llm.get_info(),
            rag_stats=app.state.rag.get_stats(),
            conversation_stats=app.state.conversation_manager.get_stats(),
            cv_loaded=app.state.cv_loaded
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve stats: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Advanced RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "general_chat": "POST /chat/general",
            "cv_chat": "POST /chat/cv",
            "history": "GET /chat/history/{session_id}",
            "clear_history": "DELETE /chat/history/{session_id}",
            "stats": "GET /stats"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_server:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8002")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true"
    )
