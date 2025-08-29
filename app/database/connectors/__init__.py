"""
Database connectors module.
"""

from typing import Dict, Any, Optional
from .base_connector import DatabaseConnector
from .postgres_connector import PostgreSQLConnector
from app.utils.exceptions import ConfigurationError


def get_connector(db_type: str, 
                 connection_params: Optional[Dict[str, Any]] = None) -> DatabaseConnector:
    """
    Factory function to create database connectors.
    
    Args:
        db_type: Database type ('postgresql', 'postgres')
        connection_params: Connection parameters
        
    Returns:
        Database connector instance
    """
    db_type = db_type.lower()
    
    if db_type in ['postgresql', 'postgres']:
        return PostgreSQLConnector(connection_params)
    else:
        raise ConfigurationError(f"Unsupported database type: {db_type}. Only PostgreSQL is supported.")


__all__ = [
    'DatabaseConnector',
    'PostgreSQLConnector',
    'get_connector'
] 