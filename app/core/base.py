"""
Main VannaAI class providing the public API for text-to-SQL generation.
"""

from typing import Dict, Any, Optional, List, Union
from app.utils.logger import logger
from config.settings import settings, ensure_directories
from app.core.sql_generator import SQLGenerator
from app.core.vector_store import VectorStore
from app.core.llm_interface import BaseLLM, get_default_llm
from app.database.connectors.base_connector import DatabaseConnector
from app.database.connectors import get_connector
from app.training.data_ingestion import DataIngestion
from app.utils.exceptions import VannaException, ConfigurationError


class VannaAI:
    """
    Main VannaAI class for text-to-SQL generation.
    
    This class provides the primary interface for:
    1. Training with DDL statements, SQL pairs, and documentation
    2. Generating SQL from natural language questions
    3. Managing database connections
    4. Providing explanations and confidence scores
    """
    
    def __init__(self, 
                 llm: Optional[BaseLLM] = None,
                 vector_store: Optional[VectorStore] = None,
                 database_connector: Optional[DatabaseConnector] = None):
        """
        Initialize VannaAI instance.
        
        Args:
            llm: Language model instance (defaults to OpenAI)
            vector_store: Vector store instance (defaults to Qdrant)
            database_connector: Database connector instance
        """
        logger.info("Initializing VannaAI instance...")
        
        # Ensure required directories exist
        ensure_directories()
        
        # Initialize components
        try:
            self.vector_store = vector_store or VectorStore()
            logger.debug("Vector store initialized")
            
            self.database_connector = database_connector
            
            self.llm = llm or get_default_llm(database_connector)
            logger.debug("LLM initialized")
            
            self.sql_generator = SQLGenerator(self.llm, self.vector_store, database_connector)
            logger.debug("SQL generator initialized")
            self.data_ingestion = DataIngestion(self.vector_store)
            logger.debug("Data ingestion initialized")
            
            logger.info("VannaAI initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VannaAI: {str(e)}")
            raise VannaException(f"VannaAI initialization failed: {str(e)}")
    
    # === Training Methods ===
    
    def train_ddl(self, ddl_statement: str, table_name: Optional[str] = None) -> str:
        """
        Train with DDL statement.
        
        Args:
            ddl_statement: CREATE TABLE or other DDL statement
            table_name: Optional table name for metadata
            
        Returns:
            Document ID of the added training data
        """
        try:
            logger.debug(f"Training DDL for table: {table_name or 'unknown'}")
            doc_id = self.vector_store.add_ddl(ddl_statement, table_name)
            logger.info(f"Added DDL training data for table: {table_name or 'unknown'}")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to train DDL: {str(e)}")
            raise VannaException(f"DDL training failed: {str(e)}")
    
    def train_sql_pair(self, 
                      question: str, 
                      sql: str, 
                      explanation: Optional[str] = None) -> str:
        """
        Train with question-SQL pair.
        
        Args:
            question: Natural language question
            sql: Corresponding SQL query
            explanation: Optional explanation of the query
            
        Returns:
            Document ID of the added training data
        """
        try:
            logger.debug(f"Training SQL pair: {question[:50]}...")
            doc_id = self.vector_store.add_sql_pair(question, sql, explanation)
            logger.info(f"Added SQL pair training data: {question[:50]}...")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to train SQL pair: {str(e)}")
            raise VannaException(f"SQL pair training failed: {str(e)}")
    
    def train_documentation(self, 
                          table_name: str, 
                          description: str,
                          column_descriptions: Optional[Dict[str, str]] = None) -> str:
        """
        Train with table documentation.
        
        Args:
            table_name: Name of the table
            description: Description of the table's purpose
            column_descriptions: Optional descriptions for each column
            
        Returns:
            Document ID of the added training data
        """
        try:
            logger.debug(f"Training documentation for table: {table_name}")
            # Format documentation for the new API
            body = f"Table Description: {description}"
            if column_descriptions:
                body += "\n\nColumn Descriptions:\n"
                for col, desc in column_descriptions.items():
                    body += f"- {col}: {desc}\n"
            
            doc_id = self.vector_store.add_documentation(
                title=f"Table: {table_name}",
                body=body,
                tags=[table_name, "table_documentation"]
            )
            logger.info(f"Added documentation for table: {table_name}")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to train documentation: {str(e)}")
            raise VannaException(f"Documentation training failed: {str(e)}")
    
    def train_from_database(self, 
                          db_type: str,
                          connection_params: Optional[Dict[str, Any]] = None,
                          tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Automatically train from database schema.
        
        Args:
            db_type: Database type ('mysql', 'sqlite')
            connection_params: Database connection parameters
            tables: Optional list of specific tables to train on
            
        Returns:
            Summary of training data added
        """
        try:
            logger.info(f"Training from {db_type} database...")
            connector = get_connector(db_type, connection_params)
            result = self.data_ingestion.ingest_from_database(connector, tables)
            logger.info(f"Database training completed: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to train from database: {str(e)}")
            raise VannaException(f"Database training failed: {str(e)}")
    
    # === Query Methods ===
    
    def ask(self, 
            question: str, 
            execute_sql: bool = True,
            generate_summary: bool = True,
            max_context_length: int = 4000) -> Dict[str, Any]:
        """
        Generate SQL from natural language question and optionally execute it.
        
        Args:
            question: Natural language question
            execute_sql: Whether to execute the generated SQL (requires database connection)
            generate_summary: Whether to generate natural language summary of results
            max_context_length: Maximum length of context to retrieve from vector store
            
        Returns:
            Dictionary containing SQL, explanation, confidence, and optionally query results
            
        Raises:
            ConfigurationError: If execute_sql=True but no database is connected
            VannaException: If SQL generation or execution fails
        """
        if execute_sql and not self.database_connector:
            logger.error("Cannot execute SQL: No database connector configured")
            raise ConfigurationError("Database connection required to execute SQL. Use connect_to_database() first.")
        
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Generate SQL with context retrieval
            result = self.sql_generator.generate_sql(
                question=question,
                max_context_length=max_context_length
            )
            
            logger.info(f"Generated SQL for question: {question[:50]}...")
            print(result)
            
            # Execute SQL if requested and database is connected
            if execute_sql and self.database_connector:
                try:
                    query_result = self.run_sql(result["corrected_sql"])
                    result["query_results"] = query_result
                    result["executed"] = True
                    
                    # Generate natural language summary if requested and we have results
                    if generate_summary and query_result.get('data') and len(query_result['data']) > 0:
                        try:
                            summary = self.llm.generate_summary(
                                question=question,
                                sql=result["sql"],
                                results=query_result
                            )
                            result["summary"] = summary
                            logger.info("Generated natural language summary for results")
                        except Exception as e:
                            logger.warning(f"Failed to generate summary: {str(e)}")
                            result["summary"] = "Summary generation failed, but query executed successfully."
                    elif generate_summary:
                        result["summary"] = "The query executed successfully but returned no data."
                    
                    logger.info("Question processed and SQL executed successfully")
                except Exception as e:
                    result["execution_error"] = str(e)
                    result["executed"] = False
                    logger.error(f"SQL execution failed: {str(e)}")
            else:
                result["executed"] = False
                if not execute_sql:
                    logger.debug("SQL generated but not executed - execution not requested")
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to process question: {str(e)}")
            raise VannaException(f"Question processing failed: {str(e)}")
    
    def run_sql(self, sql: str) -> Dict[str, Any]:
        """
        Execute SQL query on connected database.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Query results and metadata
        """
        if not self.database_connector:
            logger.error("No database connector configured for SQL execution")
            raise ConfigurationError("No database connector configured")
        
        try:
            logger.debug(f"Executing SQL: {sql[:100]}...")
            result = self.database_connector.execute_query(sql)
            logger.info("SQL executed successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to execute SQL: {str(e)}")
            raise VannaException(f"SQL execution failed: {str(e)}")
    

    
    # === Database Connection Methods ===
    
    def connect_to_database(self, 
                          db_type: str,
                          connection_params: Optional[Dict[str, Any]] = None):
        """
        Connect to a database.
        
        Args:
            db_type: Database type ('mysql', 'sqlite')
            connection_params: Connection parameters (host, port, etc.)
        """
        try:
            logger.info(f"Connecting to {db_type} database...")
            self.database_connector = get_connector(db_type, connection_params)
            
            # Update the database connector in existing components
            if hasattr(self, 'llm') and hasattr(self.llm, 'db_connector'):
                self.llm.db_connector = self.database_connector
                logger.debug("Updated LLM with database connector")
            
            if hasattr(self, 'sql_generator'):
                self.sql_generator.db_connector = self.database_connector
                logger.debug("Updated SQL generator with database connector")
            
            logger.info(f"Successfully connected to {db_type} database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise VannaException(f"Database connection failed: {str(e)}")
    
    def disconnect_database(self):
        """Disconnect from the current database."""
        if self.database_connector:
            try:
                self.database_connector.close()
                self.database_connector = None
                logger.info("Disconnected from database")
            except Exception as e:
                logger.error(f"Error during database disconnection: {str(e)}")
        else:
            logger.debug("No database connection to disconnect")
    
    # === Utility Methods ===
    
    def explain_sql(self, sql: str) -> str:
        """
        Get explanation for a SQL query.
        
        Args:
            sql: SQL query to explain
            
        Returns:
            Natural language explanation
        """
        try:
            logger.debug(f"Explaining SQL: {sql[:100]}...")
            explanation = self.llm.explain_sql(sql)
            logger.info("SQL explanation generated successfully")
            return explanation
        except Exception as e:
            logger.error(f"Failed to explain SQL: {str(e)}")
            raise VannaException(f"SQL explanation failed: {str(e)}")
    
    def generate_summary(self, question: str, sql: str, results: Dict[str, Any]) -> str:
        """
        Generate a natural language summary of query results.
        
        Args:
            question: Original natural language question
            sql: SQL query that was executed
            results: Query results dictionary
            
        Returns:
            Natural language summary of the results
        """
        try:
            logger.debug(f"Generating summary for question: {question[:50]}...")
            summary = self.llm.generate_summary(question, sql, results)
            logger.info("Results summary generated successfully")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            raise VannaException(f"Summary generation failed: {str(e)}")
    
    def get_similar_questions(self, question: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar questions in training data.
        
        Args:
            question: Question to find similar examples for
            n_results: Number of results to return
            
        Returns:
            List of similar questions with their SQL queries
        """
        try:
            logger.debug(f"Finding similar questions for: {question[:50]}...")
            results = self.sql_generator.get_similar_examples(question, n_results)
            logger.info(f"Found {len(results)} similar questions")
            return results
        except Exception as e:
            logger.error(f"Failed to get similar questions: {str(e)}")
            return []
    
    def add_feedback(self, 
                    question: str, 
                    generated_sql: str, 
                    correct_sql: str,
                    explanation: Optional[str] = None):
        """
        Add feedback to improve future generations.
        
        Args:
            question: Original question
            generated_sql: SQL that was generated (for reference)
            correct_sql: Correct SQL query
            explanation: Optional explanation
        """
        try:
            logger.debug(f"Adding feedback for question: {question[:50]}...")
            self.sql_generator.improve_with_feedback(
                question, generated_sql, correct_sql, explanation
            )
            logger.info("Feedback added successfully for model improvement")
        except Exception as e:
            logger.error(f"Failed to add feedback: {str(e)}")
            raise VannaException(f"Feedback addition failed: {str(e)}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get statistics about training data.
        
        Returns:
            Statistics about the vector store and training data
        """
        try:
            logger.debug("Retrieving training statistics...")
            stats = self.vector_store.get_collection_stats()
            logger.info("Training statistics retrieved successfully")
            return stats
        except Exception as e:
            logger.error(f"Failed to get training stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_training_data(self):
        """Clear all training data from the vector store."""
        try:
            logger.info("Clearing all training data...")
            self.vector_store.clear_all_collections()
            logger.info("All training data cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear training data: {str(e)}")
            raise VannaException(f"Training data clearing failed: {str(e)}")
    
    # === Context Manager Support ===
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect_database() 