"""
Pydantic request models for the Vanna.AI API.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., description="Natural language question")
    user_id: Optional[int] = Field(None, description="User identifier for this request (integer)")
    execute_sql: bool = Field(True, description="Whether to execute the generated SQL (requires database connection)")
    generate_summary: bool = Field(True, description="Whether to generate natural language summary of results")
    max_context_length: int = Field(4000, description="Maximum length of context to retrieve from vector store")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "How many users are there in the database?",
                "user_id": 123,
                "execute_sql": True,
                "generate_summary": True,
                "max_context_length": 4000
            }
        }


class TrainDDLRequest(BaseModel):
    """Request model for training with DDL statements."""
    ddl_statement: str = Field(..., description="DDL statement (CREATE TABLE, etc.)")
    user_id: Optional[int] = Field(None, description="User identifier for this request (integer)")
    table_name: Optional[str] = Field(None, description="Optional table name for metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "ddl_statement": "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100), email VARCHAR(100))",
                "user_id": 123,
                "table_name": "users"
            }
        }


class TrainSQLPairRequest(BaseModel):
    """Request model for training with SQL pairs."""
    question: str = Field(..., description="Natural language question")
    sql: str = Field(..., description="Corresponding SQL query")
    user_id: Optional[int] = Field(None, description="User identifier for this request (integer)")
    explanation: Optional[str] = Field(None, description="Optional explanation of the query")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "How many users are there?",
                "sql": "SELECT COUNT(*) FROM users",
                "user_id": 123,
                "explanation": "This query counts the total number of rows in the users table"
            }
        }


class TrainDocumentationRequest(BaseModel):
    """Request model for training with documentation."""
    table_name: str = Field(..., description="Name of the table")
    description: str = Field(..., description="Description of the table's purpose")
    user_id: Optional[int] = Field(None, description="User identifier for this request (integer)")
    column_descriptions: Optional[Dict[str, str]] = Field(None, description="Descriptions for each column")
    
    class Config:
        schema_extra = {
            "example": {
                "table_name": "users",
                "description": "Contains customer information and account details",
                "user_id": 123,
                "column_descriptions": {
                    "id": "Unique identifier for each user",
                    "name": "Full name of the user",
                    "email": "User's email address"
                }
            }
        }


class DatabaseConnectionRequest(BaseModel):
    """Request model for database connections."""
    db_type: str = Field(..., description="Database type (mysql, sqlite)")
    user_id: Optional[int] = Field(None, description="User identifier for this request (integer)")
    connection_params: Optional[Dict[str, Any]] = Field(None, description="Connection parameters")
    
    class Config:
        schema_extra = {
            "example": {
                "db_type": "mysql",
                "user_id": 123,
                "connection_params": {
                    "host": "localhost",
                    "port": 3306,
                    "database": "mydb",
                    "user": "username",
                    "password": "password"
                }
            }
        }


class ExecuteSQLRequest(BaseModel):
    """Request model for executing SQL queries."""
    sql: str = Field(..., description="SQL query to execute")
    user_id: Optional[int] = Field(None, description="User identifier for this request (integer)")
    
    class Config:
        schema_extra = {
            "example": {
                "sql": "SELECT COUNT(*) FROM users",
                "user_id": 123
            }
        }


class FeedbackRequest(BaseModel):
    """Request model for providing feedback."""
    question: str = Field(..., description="Original question")
    generated_sql: str = Field(..., description="SQL that was generated")
    correct_sql: str = Field(..., description="Correct SQL query")
    user_id: Optional[int] = Field(None, description="User identifier for this request (integer)")
    explanation: Optional[str] = Field(None, description="Optional explanation")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "How many active users are there?",
                "generated_sql": "SELECT COUNT(*) FROM users",
                "correct_sql": "SELECT COUNT(*) FROM users WHERE status = 'active'",
                "user_id": 123,
                "explanation": "Need to filter for active users only"
            }
        }


class BatchSQLPairsRequest(BaseModel):
    """Request model for batch training with SQL pairs."""
    sql_pairs: List[Dict[str, str]] = Field(..., description="List of SQL pairs")
    user_id: Optional[int] = Field(None, description="User identifier for this request (integer)")
    
    class Config:
        schema_extra = {
            "example": {
                "sql_pairs": [
                    {
                        "question": "How many users?",
                        "sql": "SELECT COUNT(*) FROM users"
                    },
                    {
                        "question": "List all active users",
                        "sql": "SELECT * FROM users WHERE status = 'active'"
                    }
                ],
                "user_id": 123
            }
        } 