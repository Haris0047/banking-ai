"""
Main FastAPI application for Vanna.AI.
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from app.utils.logger import logger

from config.settings import settings
from app import VannaAI
from app.utils.exceptions import VannaException
from .models.request_models import (
    QuestionRequest, TrainDDLRequest, TrainSQLPairRequest, 
    TrainDocumentationRequest, DatabaseConnectionRequest,
    ExecuteSQLRequest, FeedbackRequest, BatchSQLPairsRequest
)

# Initialize FastAPI app
app = FastAPI(
    title="Vanna.AI API",
    description="Professional text-to-SQL generation API using RAG and LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global VannaAI instances per user
vanna_instances = {}


def get_vanna(user_id: Optional[int] = None) -> VannaAI:
    """Dependency to get VannaAI instance for a specific user."""
    global vanna_instances
    
    # Use 0 as default user_id for anonymous
    effective_user_id = user_id if user_id is not None else 0
    
    if effective_user_id not in vanna_instances:
        try:
            vanna_instances[effective_user_id] = VannaAI(user_id=effective_user_id)
            logger.info(f"VannaAI instance initialized for user: {effective_user_id}")
        except Exception as e:
            logger.error(f"Failed to initialize VannaAI for user {effective_user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize VannaAI for user {effective_user_id}: {str(e)}")
    
    return vanna_instances[effective_user_id]


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Vanna.AI Text-to-SQL API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        vanna = get_vanna()
        stats = vanna.get_training_stats()
        return {
            "status": "healthy",
            "training_data": stats
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


# === Query Endpoints ===

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Generate SQL from natural language question and optionally execute it."""
    try:
        logger.info(f"API request from user {request.user_id}: {request.question[:100]}...")
        
        vanna = get_vanna(request.user_id)
        
        result = vanna.ask(
            question=request.question,
            execute_sql=request.execute_sql,
            generate_summary=request.generate_summary,
            max_context_length=request.max_context_length
        )
        
        return {"success": True, "result": result, "user_id": vanna.get_user_id()}
    
    except VannaException as e:
        logger.error(f"Vanna error in ask: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in ask: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/execute")
async def execute_sql(request: ExecuteSQLRequest):
    """Execute SQL query on connected database."""
    try:
        vanna = get_vanna(request.user_id)
        result = vanna.run_sql(request.sql)
        return {"success": True, "result": result, "user_id": vanna.get_user_id()}
    
    except VannaException as e:
        logger.error(f"Vanna error in execute: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in execute: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/explain")
async def explain_sql(request: ExecuteSQLRequest):
    """Get explanation for SQL query."""
    try:
        vanna = get_vanna(request.user_id)
        explanation = vanna.explain_sql(request.sql)
        return {"success": True, "explanation": explanation, "user_id": vanna.get_user_id()}
    
    except VannaException as e:
        logger.error(f"Vanna error in explain: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in explain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# === Training Endpoints ===

@app.post("/train/ddl")
async def train_ddl(request: TrainDDLRequest):
    """Train with DDL statement."""
    try:
        vanna = get_vanna(request.user_id)
        doc_id = vanna.train_ddl(request.ddl_statement, request.table_name)
        return {"success": True, "document_id": doc_id, "message": "DDL training data added", "user_id": vanna.get_user_id()}
    
    except VannaException as e:
        logger.error(f"Vanna error in train_ddl: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in train_ddl: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/train/sql-pair")
async def train_sql_pair(request: TrainSQLPairRequest):
    """Train with question-SQL pair."""
    try:
        vanna = get_vanna(request.user_id)
        doc_id = vanna.train_sql_pair(request.question, request.sql, request.explanation)
        return {"success": True, "document_id": doc_id, "message": "SQL pair training data added", "user_id": vanna.get_user_id()}
    
    except VannaException as e:
        logger.error(f"Vanna error in train_sql_pair: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in train_sql_pair: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/train/documentation")
async def train_documentation(request: TrainDocumentationRequest):
    """Train with table documentation."""
    try:
        vanna = get_vanna(request.user_id)
        doc_id = vanna.train_documentation(
            request.table_name, 
            request.description, 
            request.column_descriptions
        )
        return {"success": True, "document_id": doc_id, "message": "Documentation training data added", "user_id": vanna.get_user_id()}
    
    except VannaException as e:
        logger.error(f"Vanna error in train_documentation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in train_documentation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/train/batch-sql-pairs")
async def train_batch_sql_pairs(request: BatchSQLPairsRequest, vanna: VannaAI = Depends(get_vanna)):
    """Train with batch of SQL pairs."""
    try:
        summary = vanna.data_ingestion.ingest_sql_pairs_batch(request.sql_pairs)
        return {"success": True, "summary": summary}
    
    except VannaException as e:
        logger.error(f"Vanna error in batch training: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in batch training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/train/from-database")
async def train_from_database(request: DatabaseConnectionRequest, vanna: VannaAI = Depends(get_vanna)):
    """Train from database schema."""
    try:
        summary = vanna.train_from_database(request.db_type, request.connection_params)
        return {"success": True, "summary": summary}
    
    except VannaException as e:
        logger.error(f"Vanna error in database training: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in database training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# === Database Connection Endpoints ===

@app.post("/database/connect")
async def connect_database(request: DatabaseConnectionRequest, vanna: VannaAI = Depends(get_vanna)):
    """Connect to database."""
    try:
        vanna.connect_to_database(request.db_type, request.connection_params)
        return {"success": True, "message": f"Connected to {request.db_type} database"}
    
    except VannaException as e:
        logger.error(f"Vanna error in database connection: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in database connection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/database/disconnect")
async def disconnect_database(vanna: VannaAI = Depends(get_vanna)):
    """Disconnect from database."""
    try:
        vanna.disconnect_database()
        return {"success": True, "message": "Disconnected from database"}
    
    except Exception as e:
        logger.error(f"Error in database disconnection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# === Utility Endpoints ===

@app.post("/feedback")
async def add_feedback(request: FeedbackRequest, vanna: VannaAI = Depends(get_vanna)):
    """Add feedback to improve model."""
    try:
        vanna.add_feedback(
            request.question,
            request.generated_sql,
            request.correct_sql,
            request.explanation
        )
        return {"success": True, "message": "Feedback added successfully"}
    
    except VannaException as e:
        logger.error(f"Vanna error in feedback: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/similar-questions")
async def get_similar_questions(question: str, n_results: int = 5, vanna: VannaAI = Depends(get_vanna)):
    """Get similar questions from training data."""
    try:
        similar = vanna.get_similar_questions(question, n_results)
        return {"success": True, "similar_questions": similar}
    
    except Exception as e:
        logger.error(f"Error in similar questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/training-stats")
async def get_training_stats(vanna: VannaAI = Depends(get_vanna)):
    """Get training data statistics."""
    try:
        stats = vanna.get_training_stats()
        return {"success": True, "stats": stats}
    
    except Exception as e:
        logger.error(f"Error in training stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.delete("/training-data")
async def clear_training_data(vanna: VannaAI = Depends(get_vanna)):
    """Clear all training data."""
    try:
        vanna.clear_training_data()
        return {"success": True, "message": "All training data cleared"}
    
    except VannaException as e:
        logger.error(f"Vanna error in clearing data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in clearing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting Vanna.AI API server...")
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    ) 