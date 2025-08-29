"""
SQLite database connector implementation.
"""

import sqlite3
from typing import Dict, Any, List, Optional
from app.utils.logger import logger
from config.settings import settings, get_database_url
from .base_connector import DatabaseConnector
from app.utils.exceptions import DatabaseConnectionError


class SQLiteConnector(DatabaseConnector):
    """SQLite database connector."""
    
    def __init__(self, connection_params: Optional[Dict[str, Any]] = None):
        """Initialize SQLite connector."""
        super().__init__(connection_params)
        
        # Get database path from params or settings
        if connection_params and 'database' in connection_params:
            self.db_path = connection_params['database']
        elif connection_params and 'path' in connection_params:
            self.db_path = connection_params['path']
        else:
            self.db_path = settings.sqlite_path if hasattr(settings, 'sqlite_path') else "./data/sample.db"
        
        logger.info(f"Initialized SQLite connector for: {self.db_path}")
    
    def connect(self):
        """Establish SQLite connection."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            logger.info("Connected to SQLite database")
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {str(e)}")
            raise DatabaseConnectionError(f"SQLite connection failed: {str(e)}")
    
    def close(self):
        """Close SQLite connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Closed SQLite connection")
    
    def execute_query(self, sql: str) -> Dict[str, Any]:
        """Execute SQL query and return results."""
        if not self.connection:
            self.connect()
        
        try:
            logger.debug(f"Executing SQLite query: {sql[:100]}...")
            cursor = self.connection.cursor()
            cursor.execute(sql)
            
            # Handle different query types
            if sql.strip().upper().startswith(('SELECT', 'WITH')):
                # Query returns data
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description] if cursor.description else []
                data = [list(row) for row in rows]
                
                result = {
                    "data": data,
                    "columns": columns,
                    "row_count": len(data),
                    "query_type": "SELECT"
                }
            else:
                # Query modifies data
                self.connection.commit()
                result = {
                    "data": [],
                    "columns": [],
                    "row_count": cursor.rowcount,
                    "query_type": "MODIFY"
                }
            
            cursor.close()
            logger.info(f"SQLite query executed successfully, returned {result['row_count']} rows")
            return result
            
        except Exception as e:
            logger.error(f"SQLite query execution failed: {str(e)}")
            raise DatabaseConnectionError(f"Query execution failed: {str(e)}")
    
    def get_table_names(self) -> List[str]:
        """Get list of table names in the SQLite database."""
        sql = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
        
        result = self.execute_query(sql)
        table_names = [row[0] for row in result["data"]]
        logger.info(f"Found {len(table_names)} tables in SQLite database")
        return table_names
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a SQLite table."""
        sql = f"PRAGMA table_info({table_name})"
        
        result = self.execute_query(sql)
        
        columns = []
        for row in result["data"]:
            # SQLite PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
            columns.append({
                "column_name": row[1],
                "data_type": row[2],
                "is_nullable": not bool(row[3]),
                "default_value": row[4],
                "is_primary_key": bool(row[5])
            })
        
        schema_info = {
            "table_name": table_name,
            "columns": columns,
            "column_count": len(columns)
        }
        
        logger.debug(f"Retrieved schema for table {table_name}: {len(columns)} columns")
        return schema_info
    
    def get_ddl_statement(self, table_name: str) -> str:
        """Get DDL statement for a SQLite table."""
        sql = f"""
        SELECT sql FROM sqlite_master 
        WHERE type='table' AND name='{table_name}'
        """
        
        result = self.execute_query(sql)
        
        if result["data"]:
            ddl = result["data"][0][0]
            logger.debug(f"Retrieved DDL for table {table_name}")
            return ddl
        else:
            logger.error(f"Table '{table_name}' not found")
            raise DatabaseConnectionError(f"Table '{table_name}' not found")
    
    def get_foreign_keys(self, table_name: str) -> List[Dict[str, Any]]:
        """Get foreign key information for a SQLite table."""
        sql = f"PRAGMA foreign_key_list({table_name})"
        
        result = self.execute_query(sql)
        
        foreign_keys = []
        for row in result["data"]:
            # SQLite PRAGMA foreign_key_list returns: id, seq, table, from, to, on_update, on_delete, match
            foreign_keys.append({
                "column_name": row[3],
                "referenced_table": row[2],
                "referenced_column": row[4],
                "on_update": row[5],
                "on_delete": row[6]
            })
        
        logger.debug(f"Found {len(foreign_keys)} foreign keys for table {table_name}")
        return foreign_keys
    
    def get_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Get index information for a SQLite table."""
        sql = f"PRAGMA index_list({table_name})"
        
        result = self.execute_query(sql)
        
        indexes = []
        for row in result["data"]:
            # Get index details
            index_name = row[1]
            index_sql = f"PRAGMA index_info({index_name})"
            index_result = self.execute_query(index_sql)
            
            columns = [col_row[2] for col_row in index_result["data"]]
            
            indexes.append({
                "index_name": index_name,
                "is_unique": bool(row[2]),
                "columns": columns
            })
        
        logger.debug(f"Found {len(indexes)} indexes for table {table_name}")
        return indexes 