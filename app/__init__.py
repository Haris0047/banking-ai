"""
Vanna.AI Custom Implementation

A professional text-to-SQL system using three training mechanisms:
- Text-to-SQL pairs
- DDL statements  
- Documentation
"""

from app.core.base import VannaAI
from app.core.sql_generator import SQLGenerator
from app.training.data_ingestion import DataIngestion
from app.utils.exceptions import VannaException, DatabaseConnectionError, LLMError

__version__ = "1.0.0"
__author__ = "Vanna.AI Custom Implementation"

__all__ = [
    "VannaAI",
    "SQLGenerator", 
    "DataIngestion",
    "VannaException",
    "DatabaseConnectionError",
    "LLMError"
] 