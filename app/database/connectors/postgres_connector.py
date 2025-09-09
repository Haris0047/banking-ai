"""
PostgreSQL database connector implementation.
"""

import psycopg2
import psycopg2.extras
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
from app.database.connectors.base_connector import DatabaseConnector
from app.utils.logger import logger
from app.utils.exceptions import DatabaseConnectionError


class PostgreSQLConnector(DatabaseConnector):
    """PostgreSQL database connector."""
    
    def __init__(self, connection_params: Optional[Dict[str, Any]] = None):
        """
        Initialize PostgreSQL connector.
        
        Args:
            connection_params: Connection parameters including:
                - connection_url: PostgreSQL connection URL
                - host: Database host
                - port: Database port
                - database: Database name
                - user: Username
                - password: Password
        """
        super().__init__(connection_params)
        self.connection = None
        self.cursor = None
        
        # Parse connection URL if provided
        if 'connection_url' in self.connection_params:
            self._parse_connection_url(self.connection_params['connection_url'])
    
    def _parse_connection_url(self, connection_url: str):
        """Parse PostgreSQL connection URL and extract parameters for display purposes."""
        try:
            parsed = urlparse(connection_url)
            
            # Store parsed info for display/logging purposes only
            # The actual connection will use the raw URL
            self.connection_params.update({
                'host': parsed.hostname,
                'port': parsed.port or 5432,
                'database': parsed.path.lstrip('/'),
                'user': parsed.username,
                'password': parsed.password
            })
            
            logger.info(f"Parsed PostgreSQL connection URL for database: {self.connection_params.get('database', 'unknown')}")
            
        except Exception as e:
            logger.warning(f"Could not parse connection URL for display: {str(e)}")
            # Don't raise error - we'll still try to connect with the raw URL
    
    def connect(self):
        """Establish PostgreSQL connection."""
        try:
            if self.connection and not self.connection.closed:
                return
            
            # If we have a connection_url, use it directly
            if 'connection_url' in self.connection_params:
                connection_url = self.connection_params['connection_url']
                logger.info(f"Connecting to PostgreSQL using direct URL: {connection_url[:50]}...")
                
                # Connect directly using the URL string
                self.connection = psycopg2.connect(connection_url)
                self.connection.autocommit = True
                self.cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                logger.info("PostgreSQL connection established successfully using direct URL")
                
            else:
                # Fallback to individual parameters
                conn_params = {
                    'host': self.connection_params.get('host', 'localhost'),
                    'port': self.connection_params.get('port', 5432),
                    'database': self.connection_params.get('database'),
                    'user': self.connection_params.get('user'),
                    'password': self.connection_params.get('password')
                }
                
                # Remove None values
                conn_params = {k: v for k, v in conn_params.items() if v is not None}
                
                logger.info(f"Connecting to PostgreSQL database: {conn_params.get('database')} at {conn_params.get('host')}")
                
                self.connection = psycopg2.connect(**conn_params)
                self.connection.autocommit = True
                self.cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                logger.info("PostgreSQL connection established successfully using parameters")
            
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL connection failed: {str(e)}")
            raise DatabaseConnectionError(f"Failed to connect to PostgreSQL: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to PostgreSQL: {str(e)}")
            raise DatabaseConnectionError(f"Unexpected connection error: {str(e)}")
    
    def close(self):
        """Close PostgreSQL connection."""
        try:
            if self.cursor:
                self.cursor.close()
                self.cursor = None
            
            if self.connection and not self.connection.closed:
                self.connection.close()
                self.connection = None
                logger.info("PostgreSQL connection closed")
                
        except Exception as e:
            logger.warning(f"Error closing PostgreSQL connection: {str(e)}")
    
    def execute_query(self, sql: str, params: Optional[tuple] = None) -> Dict[str, Any]:
            """
            Execute SQL query and return results.
            
            Args:
                sql: SQL query to execute
                params: Optional parameters for the SQL query
                
            Returns:
                Dictionary containing results, columns, and metadata
            """
            try:
                if not self.connection or self.connection.closed:
                    self.connect()
                
                logger.debug(f"Executing PostgreSQL query: {sql[:100]}... with params: {params}")
                
                self.cursor.execute(sql, params) # Pass parameters to execute
                
                # Handle different query types
                if self.cursor.description:
                    # SELECT query - fetch results
                    rows = self.cursor.fetchall()
                    columns = [desc[0] for desc in self.cursor.description]
                    
                    # Convert RealDictRow to regular dict/list
                    data = []
                    for row in rows:
                        if isinstance(row, psycopg2.extras.RealDictRow):
                            data.append(list(row.values()))
                        else:
                            data.append(list(row))
                    
                    result = {
                        "data": data,
                        "columns": columns,
                        "row_count": len(data),
                        "query_type": "SELECT"
                    }
                else:
                    # Non-SELECT query (INSERT, UPDATE, DELETE, etc.)
                    result = {
                        "data": [],
                        "columns": [],
                        "row_count": self.cursor.rowcount,
                        "query_type": "MODIFY"
                    }
                
                logger.debug(f"Query executed successfully, returned {result['row_count']} rows")
                return result
                
            except psycopg2.Error as e:
                logger.error(f"PostgreSQL query execution failed: {str(e)}")
                raise DatabaseConnectionError(f"Query execution failed: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error executing query: {str(e)}")
                raise DatabaseConnectionError(f"Unexpected query error: {str(e)}")
    def get_table_names(self) -> List[str]:
        """Get list of table names in the database."""
        try:
            sql = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """
            
            result = self.execute_query(sql)
            table_names = [row[0] for row in result["data"]]
            
            logger.info(f"Found {len(table_names)} tables in PostgreSQL database")
            return table_names
            
        except Exception as e:
            logger.error(f"Failed to get table names: {str(e)}")
            raise DatabaseConnectionError(f"Failed to get table names: {str(e)}")
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Get schema information for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary containing column information
        """
        try:
            sql = """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = %s
                ORDER BY ordinal_position
            """
            
            result = self.execute_query(sql, (table_name,)) # Pass parameters as a tuple
            rows = result["data"]
            
            columns = []
            for row_values in rows: # Iterate through row_values (which are lists from execute_query)
                column_info = {
                    "name": row_values[0],
                    "type": row_values[1],
                    "nullable": row_values[2] == 'YES',
                    "default": row_values[3],
                    "max_length": row_values[4],
                    "precision": row_values[5],
                    "scale": row_values[6]
                }
                columns.append(column_info)
            
            schema = {
                "table_name": table_name,
                "columns": columns,
                "column_count": len(columns)
            }
            
            logger.debug(f"Retrieved schema for table {table_name}: {len(columns)} columns")
            return schema
            
        except Exception as e:
            logger.error(f"Failed to get schema for table {table_name}: {str(e)}")
            raise DatabaseConnectionError(f"Failed to get table schema: {str(e)}")
        
    def get_ddl_statement(self, table_name: str) -> str:
        """
        Get DDL (CREATE TABLE) statement for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            DDL statement as string
        """
        try:
            schema = self.get_table_schema(table_name)
            
            # Build CREATE TABLE statement
            ddl_parts = [f"CREATE TABLE {table_name} ("]
            
            column_definitions = []
            for col in schema["columns"]:
                col_def = f"    {col['name']} {col['type']}"
                
                # Add length/precision if applicable
                if col['max_length']:
                    col_def += f"({col['max_length']})"
                elif col['precision'] and col['scale']:
                    col_def += f"({col['precision']},{col['scale']})"
                elif col['precision']:
                    col_def += f"({col['precision']})"
                
                # Add NOT NULL if applicable
                if not col['nullable']:
                    col_def += " NOT NULL"
                
                # Add DEFAULT if applicable
                if col['default']:
                    col_def += f" DEFAULT {col['default']}"
                
                column_definitions.append(col_def)
            
            ddl_parts.append(",\n".join(column_definitions))
            ddl_parts.append(");")
            
            ddl_statement = "\n".join(ddl_parts)
            
            logger.debug(f"Generated DDL for table {table_name}")
            return ddl_statement
            
        except Exception as e:
            logger.error(f"Failed to generate DDL for table {table_name}: {str(e)}")
            raise DatabaseConnectionError(f"Failed to generate DDL: {str(e)}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get PostgreSQL database information."""
        try:
            info_queries = {
                "version": "SELECT version()",
                "current_database": "SELECT current_database()",
                "current_user": "SELECT current_user",
                "current_schema": "SELECT current_schema()"
            }
            
            info = {}
            for key, query in info_queries.items():
                try:
                    result = self.execute_query(query)
                    info[key] = result["data"][0][0] if result["data"] else None
                except Exception as e:
                    logger.warning(f"Failed to get {key}: {str(e)}")
                    info[key] = None
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get database info: {str(e)}")
            return {} 