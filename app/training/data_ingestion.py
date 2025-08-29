"""
Data ingestion module for training Vanna from various sources.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from app.utils.logger import logger
from app.core.vector_store import VectorStore
from app.database.connectors.base_connector import DatabaseConnector
from app.utils.exceptions import TrainingDataError


class DataIngestion:
    """Handle ingestion of training data from various sources."""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize data ingestion with vector store."""
        self.vector_store = vector_store
        logger.info("Data ingestion module initialized")
    
    def ingest_from_database(self, 
                           connector: DatabaseConnector,
                           tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Ingest training data from a database connection.
        
        Args:
            connector: Database connector instance
            tables: Optional list of specific tables to process
            
        Returns:
            Summary of ingested data
        """
        try:
            logger.info("Starting database ingestion...")
            
            with connector:
                # Get table names
                if tables:
                    table_names = tables
                    logger.info(f"Processing specified tables: {tables}")
                else:
                    table_names = connector.get_table_names()
                    logger.info(f"Processing all tables: {len(table_names)} found")
                
                ingestion_summary = {
                    "tables_processed": 0,
                    "ddl_statements_added": 0,
                    "documentation_added": 0,
                    "errors": []
                }
                
                for table_name in table_names:
                    try:
                        logger.debug(f"Processing table: {table_name}")
                        
                        # Add DDL statement
                        ddl = connector.get_ddl_statement(table_name)
                        self.vector_store.add_ddl(ddl, table_name)
                        ingestion_summary["ddl_statements_added"] += 1
                        
                        # Add basic documentation from schema
                        schema = connector.get_table_schema(table_name)
                        self._add_schema_documentation(table_name, schema)
                        ingestion_summary["documentation_added"] += 1
                        
                        ingestion_summary["tables_processed"] += 1
                        logger.info(f"Successfully processed table: {table_name}")
                        
                    except Exception as e:
                        error_msg = f"Failed to process table {table_name}: {str(e)}"
                        ingestion_summary["errors"].append(error_msg)
                        logger.error(error_msg)
                
                logger.info(f"Database ingestion completed: {ingestion_summary}")
                return ingestion_summary
                
        except Exception as e:
            logger.error(f"Database ingestion failed: {str(e)}")
            raise TrainingDataError(f"Database ingestion failed: {str(e)}")
    
    def ingest_from_json_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Ingest training data from JSON file.
        
        Expected JSON format:
        {
            "ddl_statements": [
                {"statement": "CREATE TABLE...", "table_name": "users"}
            ],
            "sql_pairs": [
                {"question": "How many users?", "sql": "SELECT COUNT(*) FROM users", "explanation": "..."}
            ],
            "documentation": [
                {"table_name": "users", "description": "...", "column_descriptions": {...}}
            ]
        }
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Summary of ingested data
        """
        try:
            file_path = Path(file_path)
            logger.info(f"Starting JSON ingestion from: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            summary = {
                "ddl_added": 0,
                "sql_pairs_added": 0,
                "documentation_added": 0,
                "errors": []
            }
            
            # Process DDL statements
            if "ddl_statements" in data:
                logger.debug(f"Processing {len(data['ddl_statements'])} DDL statements")
                for ddl_item in data["ddl_statements"]:
                    try:
                        self.vector_store.add_ddl(
                            ddl_item["statement"], 
                            ddl_item.get("table_name")
                        )
                        summary["ddl_added"] += 1
                    except Exception as e:
                        summary["errors"].append(f"DDL error: {str(e)}")
            
            # Process SQL pairs
            if "sql_pairs" in data:
                logger.debug(f"Processing {len(data['sql_pairs'])} SQL pairs")
                for pair in data["sql_pairs"]:
                    try:
                        self.vector_store.add_sql_pair(
                            pair["question"],
                            pair["sql"],
                            pair.get("explanation")
                        )
                        summary["sql_pairs_added"] += 1
                    except Exception as e:
                        summary["errors"].append(f"SQL pair error: {str(e)}")
            
            # Process documentation
            if "documentation" in data:
                logger.debug(f"Processing {len(data['documentation'])} documentation entries")
                for doc in data["documentation"]:
                    try:
                        self.vector_store.add_documentation(
                            doc["table_name"],
                            doc["description"],
                            doc.get("column_descriptions")
                        )
                        summary["documentation_added"] += 1
                    except Exception as e:
                        summary["errors"].append(f"Documentation error: {str(e)}")
            
            logger.info(f"JSON ingestion completed: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"JSON ingestion failed: {str(e)}")
            raise TrainingDataError(f"JSON ingestion failed: {str(e)}")
    
    def ingest_from_csv_file(self, 
                           file_path: Union[str, Path],
                           data_type: str = "sql_pairs") -> Dict[str, Any]:
        """
        Ingest training data from CSV file.
        
        For SQL pairs, expected columns: question, sql, explanation (optional)
        For documentation, expected columns: table_name, description, column_descriptions (JSON string, optional)
        
        Args:
            file_path: Path to CSV file
            data_type: Type of data ("sql_pairs" or "documentation")
            
        Returns:
            Summary of ingested data
        """
        try:
            file_path = Path(file_path)
            logger.info(f"Starting CSV ingestion from: {file_path} (type: {data_type})")
            
            summary = {
                "rows_processed": 0,
                "items_added": 0,
                "errors": []
            }
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    summary["rows_processed"] += 1
                    
                    try:
                        if data_type == "sql_pairs":
                            self.vector_store.add_sql_pair(
                                row["question"],
                                row["sql"],
                                row.get("explanation")
                            )
                        elif data_type == "documentation":
                            column_descriptions = None
                            if "column_descriptions" in row and row["column_descriptions"]:
                                column_descriptions = json.loads(row["column_descriptions"])
                            
                            self.vector_store.add_documentation(
                                row["table_name"],
                                row["description"],
                                column_descriptions
                            )
                        
                        summary["items_added"] += 1
                        
                    except Exception as e:
                        summary["errors"].append(f"Row {summary['rows_processed']}: {str(e)}")
            
            logger.info(f"CSV ingestion completed: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"CSV ingestion failed: {str(e)}")
            raise TrainingDataError(f"CSV ingestion failed: {str(e)}")
    
    def ingest_sql_pairs_batch(self, sql_pairs: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Ingest a batch of SQL pairs.
        
        Args:
            sql_pairs: List of dictionaries with keys: question, sql, explanation (optional)
            
        Returns:
            Summary of ingested data
        """
        try:
            logger.info(f"Starting batch ingestion of {len(sql_pairs)} SQL pairs")
            
            summary = {
                "pairs_processed": 0,
                "pairs_added": 0,
                "errors": []
            }
            
            for pair in sql_pairs:
                summary["pairs_processed"] += 1
                
                try:
                    self.vector_store.add_sql_pair(
                        pair["question"],
                        pair["sql"],
                        pair.get("explanation")
                    )
                    summary["pairs_added"] += 1
                    
                except Exception as e:
                    summary["errors"].append(f"Pair {summary['pairs_processed']}: {str(e)}")
            
            logger.info(f"Batch SQL pairs ingestion completed: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Batch ingestion failed: {str(e)}")
            raise TrainingDataError(f"Batch ingestion failed: {str(e)}")
    
    def _add_schema_documentation(self, table_name: str, schema: Dict[str, Any]):
        """Add documentation based on database schema information."""
        try:
            logger.debug(f"Adding schema documentation for table: {table_name}")
            
            # Create basic table description
            column_count = schema.get("column_count", 0)
            description = f"Database table with {column_count} columns"
            
            # Create column descriptions from schema
            column_descriptions = {}
            for col in schema.get("columns", []):
                col_name = col["column_name"]
                col_type = col["data_type"]
                nullable = "nullable" if col.get("is_nullable", True) else "not null"
                
                col_desc = f"{col_type} column ({nullable})"
                
                if col.get("is_primary_key"):
                    col_desc += ", primary key"
                
                column_descriptions[col_name] = col_desc
            
            # Add to vector store as documentation
            self.vector_store.add_documentation(
                title=f"Table: {table_name}",
                body=f"{description}\n\nColumns: {json.dumps(column_descriptions, indent=2)}"
            )
            
            logger.debug(f"Schema documentation added for table: {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to add schema documentation for {table_name}: {str(e)}")
    
    def export_training_data(self, output_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Export current training data to JSON file.
        
        Args:
            output_path: Path for output JSON file
            
        Returns:
            Summary of exported data
        """
        try:
            logger.info(f"Starting training data export to: {output_path}")
            
            # Get all training data from vector store
            ddl_results = self.vector_store.search_similar("CREATE TABLE", n_results=1000, data_types=["ddl"])
            sql_results = self.vector_store.search_similar("SELECT", n_results=1000, data_types=["sql_pair"])
            doc_results = self.vector_store.search_similar("Table:", n_results=1000, data_types=["documentation"])
            
            export_data = {
                "ddl_statements": [],
                "sql_pairs": [],
                "documentation": []
            }
            
            # Process DDL results
            for result in ddl_results:
                export_data["ddl_statements"].append({
                    "statement": result["content"],
                    "table_name": result["metadata"].get("table_name")
                })
            
            # Process SQL pair results
            for result in sql_results:
                metadata = result["metadata"]
                export_data["sql_pairs"].append({
                    "question": metadata.get("question"),
                    "sql": metadata.get("sql"),
                    "explanation": metadata.get("explanation")
                })
            
            # Process documentation results
            for result in doc_results:
                metadata = result["metadata"]
                export_data["documentation"].append({
                    "table_name": metadata.get("table_name"),
                    "description": metadata.get("description"),
                    "column_descriptions": metadata.get("column_descriptions")
                })
            
            # Write to file
            output_path = Path(output_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            summary = {
                "ddl_exported": len(export_data["ddl_statements"]),
                "sql_pairs_exported": len(export_data["sql_pairs"]),
                "documentation_exported": len(export_data["documentation"]),
                "output_file": str(output_path)
            }
            
            logger.info(f"Training data exported successfully: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            raise TrainingDataError(f"Export failed: {str(e)}")