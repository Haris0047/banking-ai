"""
MySQL database connector implementation.
"""

import pymysql
from typing import Dict, Any, List, Optional
from app.utils.logger import logger
from config.settings import settings, get_database_url
from .base_connector import DatabaseConnector
from app.utils.exceptions import DatabaseConnectionError


class MySQLConnector(DatabaseConnector):
    """MySQL database connector."""
    
    def __init__(self, connection_params: Optional[Dict[str, Any]] = None):
        """Initialize MySQL connector."""
        super().__init__(connection_params)
        
        # Build connection parameters
        if connection_params:
            self.conn_params = connection_params
        else:
            self.conn_params = {
                'host': settings.mysql_host if hasattr(settings, 'mysql_host') else 'localhost',
                'port': settings.mysql_port if hasattr(settings, 'mysql_port') else 3306,
                'database': settings.mysql_db if hasattr(settings, 'mysql_db') else None,
                'user': settings.mysql_user if hasattr(settings, 'mysql_user') else None,
                'password': settings.mysql_password if hasattr(settings, 'mysql_password') else None,
                'charset': 'utf8mb4',
                'autocommit': True
            }
        
        # Remove None values
        self.conn_params = {k: v for k, v in self.conn_params.items() if v is not None}
        
        logger.info(f"Initialized MySQL connector for: {self.conn_params.get('host', 'localhost')}")
    
    def connect(self):
        """Establish MySQL connection."""
        try:
            self.connection = pymysql.connect(**self.conn_params)
            logger.info("Connected to MySQL database")
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {str(e)}")
            raise DatabaseConnectionError(f"MySQL connection failed: {str(e)}")
    
    def close(self):
        """Close MySQL connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Closed MySQL connection")
    
    def execute_query(self, sql: str) -> Dict[str, Any]:
        """Execute SQL query and return results."""
        if not self.connection:
            self.connect()
        
        try:
            logger.debug(f"Executing MySQL query: {sql[:100]}...")
            cursor = self.connection.cursor()
            cursor.execute(sql)
            
            # Handle different query types
            if sql.strip().upper().startswith(('SELECT', 'WITH', 'SHOW', 'DESCRIBE')):
                # Query returns data
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                data = [list(row) for row in rows]
                
                result = {
                    "data": data,
                    "columns": columns,
                    "row_count": len(data),
                    "query_type": "SELECT"
                }
            else:
                # Query modifies data
                result = {
                    "data": [],
                    "columns": [],
                    "row_count": cursor.rowcount,
                    "query_type": "MODIFY"
                }
            
            cursor.close()
            logger.info(f"MySQL query executed successfully, returned {result['row_count']} rows")
            return result
            
        except Exception as e:
            logger.error(f"MySQL query execution failed: {str(e)}")
            raise DatabaseConnectionError(f"Query execution failed: {str(e)}")
    
    def get_table_names(self) -> List[str]:
        """Get list of table names in the MySQL database."""
        sql = "SHOW TABLES"
        
        result = self.execute_query(sql)
        table_names = [row[0] for row in result["data"]]
        logger.info(f"Found {len(table_names)} tables in MySQL database")
        return table_names
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a MySQL table."""
        sql = f"""
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            CHARACTER_MAXIMUM_LENGTH,
            NUMERIC_PRECISION,
            NUMERIC_SCALE,
            COLUMN_KEY,
            EXTRA
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION
        """
        
        cursor = self.connection.cursor()
        cursor.execute(sql, (table_name,))
        rows = cursor.fetchall()
        cursor.close()
        
        columns = []
        for row in rows:
            columns.append({
                "column_name": row[0],
                "data_type": row[1],
                "is_nullable": row[2] == 'YES',
                "default_value": row[3],
                "character_maximum_length": row[4],
                "numeric_precision": row[5],
                "numeric_scale": row[6],
                "column_key": row[7],
                "extra": row[8]
            })
        
        schema_info = {
            "table_name": table_name,
            "columns": columns,
            "column_count": len(columns)
        }
        
        logger.debug(f"Retrieved schema for table {table_name}: {len(columns)} columns")
        return schema_info
    
    def get_ddl_statement(self, table_name: str) -> str:
        """Get DDL statement for a MySQL table."""
        sql = f"SHOW CREATE TABLE {table_name}"
        
        result = self.execute_query(sql)
        
        if result["data"]:
            ddl = result["data"][0][1]  # Second column contains the CREATE TABLE statement
            logger.debug(f"Retrieved DDL for table {table_name}")
            return ddl
        else:
            logger.error(f"Table '{table_name}' not found")
            raise DatabaseConnectionError(f"Table '{table_name}' not found")
    
    def get_foreign_keys(self, table_name: str) -> List[Dict[str, Any]]:
        """Get foreign key information for a MySQL table."""
        sql = f"""
        SELECT
            COLUMN_NAME,
            REFERENCED_TABLE_NAME,
            REFERENCED_COLUMN_NAME,
            UPDATE_RULE,
            DELETE_RULE
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = DATABASE() 
        AND TABLE_NAME = %s
        AND REFERENCED_TABLE_NAME IS NOT NULL
        """
        
        cursor = self.connection.cursor()
        cursor.execute(sql, (table_name,))
        rows = cursor.fetchall()
        cursor.close()
        
        foreign_keys = []
        for row in rows:
            foreign_keys.append({
                "column_name": row[0],
                "referenced_table": row[1],
                "referenced_column": row[2],
                "on_update": row[3],
                "on_delete": row[4]
            })
        
        logger.debug(f"Found {len(foreign_keys)} foreign keys for table {table_name}")
        return foreign_keys
    
    def get_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Get index information for a MySQL table."""
        sql = f"SHOW INDEX FROM {table_name}"
        
        result = self.execute_query(sql)
        
        # Group indexes by name
        indexes_dict = {}
        for row in result["data"]:
            index_name = row[2]  # Key_name
            column_name = row[4]  # Column_name
            is_unique = row[1] == 0  # Non_unique (0 means unique)
            
            if index_name not in indexes_dict:
                indexes_dict[index_name] = {
                    "index_name": index_name,
                    "is_unique": is_unique,
                    "columns": []
                }
            
            indexes_dict[index_name]["columns"].append(column_name)
        
        indexes = list(indexes_dict.values())
        logger.debug(f"Found {len(indexes)} indexes for table {table_name}")
        return indexes 