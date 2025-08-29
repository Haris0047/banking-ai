"""
Base database connector abstract class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd


class DatabaseConnector(ABC):
    """Abstract base class for database connectors."""
    
    def __init__(self, connection_params: Optional[Dict[str, Any]] = None):
        """Initialize database connector."""
        self.connection_params = connection_params or {}
        self.connection = None
    
    @abstractmethod
    def connect(self):
        """Establish database connection."""
        pass
    
    @abstractmethod
    def close(self):
        """Close database connection."""
        pass
    
    @abstractmethod
    def execute_query(self, sql: str) -> Dict[str, Any]:
        """
        Execute SQL query and return results.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Dictionary containing results, columns, and metadata
        """
        pass
    
    @abstractmethod
    def get_table_names(self) -> List[str]:
        """Get list of table names in the database."""
        pass
    
    @abstractmethod
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Get schema information for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary containing column information
        """
        pass
    
    @abstractmethod
    def get_ddl_statement(self, table_name: str) -> str:
        """
        Get DDL (CREATE TABLE) statement for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            DDL statement as string
        """
        pass
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """
        Get sample data from a table.
        
        Args:
            table_name: Name of the table
            limit: Number of rows to sample
            
        Returns:
            DataFrame with sample data
        """
        sql = f"SELECT * FROM {table_name} LIMIT {limit}"
        result = self.execute_query(sql)
        return pd.DataFrame(result["data"], columns=result["columns"])
    
    def test_connection(self) -> bool:
        """Test if database connection is working."""
        try:
            self.connect()
            # Try a simple query
            self.execute_query("SELECT 1")
            return True
        except Exception:
            return False
        finally:
            self.close()
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 