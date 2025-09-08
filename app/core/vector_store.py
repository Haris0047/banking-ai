"""
Vector store implementation using Qdrant with Vanna's three-collection approach.
"""

import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import openai
from app.utils.logger import logger
from config.settings import settings
from app.utils.exceptions import VectorStoreError
import pandas as pd


class VectorStore:
    """Qdrant-based vector store implementing Vanna's three-collection approach."""
    
    def __init__(self, 
                 url: Optional[str] = None,
                 base_collection_name: Optional[str] = None,
                 api_key: Optional[str] = None):
        """Initialize Qdrant vector store with three collections."""
        
        self.url = url or settings.qdrant_url
        self.base_collection_name = base_collection_name or settings.qdrant_collection_name
        self.api_key = api_key or settings.qdrant_api_key
        
        # Define the three collections as per Vanna principles
        self.collections = {
            'query_pairs': f"{self.base_collection_name}_query_pairs",
            'ddl_definitions': f"{self.base_collection_name}_ddl_definitions", 
            'docs': f"{self.base_collection_name}_docs"
        }
        
        logger.info(f"Initializing Qdrant vector store at: {self.url}")
        
        # Initialize Qdrant client
        try:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key
            )
            
            # Initialize OpenAI client for embeddings
            if not settings.openai_api_key:
                raise VectorStoreError("OpenAI API key is required for embeddings")
            
            self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
            self.embedding_model_name = settings.embedding_model
            self.vector_size = settings.embedding_dimensions
            
            # Create collections if they don't exist
            self._ensure_collections_exist()
            
            logger.info(f"Qdrant vector store initialized with 3 collections using OpenAI {self.embedding_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant vector store: {str(e)}")
            raise VectorStoreError(f"Vector store initialization failed: {str(e)}")
    
    def _ensure_collections_exist(self):
        """Ensure all three collections exist."""
        try:
            existing_collections = [col.name for col in self.client.get_collections().collections]
            
            for collection_type, collection_name in self.collections.items():
                if collection_name not in existing_collections:
                    logger.info(f"Creating collection: {collection_name}")
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=self.vector_size,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Created {collection_type} collection: {collection_name}")
                else:
                    logger.debug(f"Collection {collection_name} already exists")
                    
        except Exception as e:
            logger.error(f"Failed to ensure collections exist: {str(e)}")
            raise VectorStoreError(f"Collection creation failed: {str(e)}")
    
    def add_sql_pair(self, question: str, sql: str, explanation: Optional[str] = None) -> str:
        """
        Add a question-SQL pair to the query_pairs collection.
        
        Args:
            question: Natural language question
            sql: Corresponding SQL query
            explanation: Optional explanation of the query
            
        Returns:
            Document ID
        """
        try:
            # Extract tables and columns used (simple parsing)
            tables_used = self._extract_tables_from_sql(sql)
            columns_used = self._extract_columns_from_sql(sql)
            
            # Create embedding from the question
            embedding = self._generate_embedding(question)
            
            # Create point
            point_id = str(uuid.uuid4())
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "id": point_id,
                    "question": question,
                    "sql": sql,
                    "explanation": explanation,
                    "tables_used": tables_used,
                    "columns_used": columns_used,
                    "timestamp": datetime.now().isoformat(),
                    "collection_type": "query_pairs"
                }
            )
            
            # Insert into query_pairs collection
            self.client.upsert(
                collection_name=self.collections['query_pairs'],
                points=[point]
            )
            
            logger.info(f"Added SQL pair to query_pairs collection: {point_id}")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to add SQL pair: {str(e)}")
            raise VectorStoreError(f"Failed to add SQL pair: {str(e)}")
    
    def add_ddl(self, ddl_data: dict, table_name: Optional[str] = None) -> str:
        """
        Add structured DDL data to the ddl_definitions collection.
        
        Args:
            ddl_data: Dictionary containing structured DDL data with keys:
                     - table: table name
                     - ddl: DDL statement  
                     - description: table description
                     - columns: list of column dictionaries
                     - sample_data: list of sample data records
                    - join_info: list of join information dictionaries
                     
            table_name: Optional table name override
            
        Returns:
            Document ID
        """
        try:
            # Extract structured data
            name = table_name or ddl_data.get('table', '')
            ddl_statement = ddl_data.get('ddl', '')
            description = ddl_data.get('description', '')
            column_descriptions = ddl_data.get('columns', [])
            sample_data = ddl_data.get('sample_data', [])
            join_info = ddl_data.get('join_info', [])
            
            if not name or not ddl_statement:
                raise ValueError("Missing required table name or DDL statement")
            
            # Parse DDL to extract basic information
            object_type = self._extract_object_type_from_ddl(ddl_statement)
            columns = self._extract_columns_from_ddl(ddl_statement) if object_type == "table" else []
            
            # Create enhanced content for embedding (preserving rich context)
            # Create enhanced content for embedding (only table, DDL, and description)
            enhanced_content = f"Table: {name}\n\n{ddl_statement}"
            if description:
                enhanced_content += f"\n\nTable Description: {description}"
            # Debug prints
            print(f"DEBUG: Processing table: {name}")
            print(f"DEBUG: DDL length: {len(ddl_statement)}")
            print(f"DEBUG: Enhanced content length: {len(enhanced_content)}")
            # enhanced_content = ddl_statement
            # if description:
            #     enhanced_content += f"\n\nTable Description: {description}"
            # if column_descriptions:
            #     enhanced_content += f"\n\nColumn Descriptions:"
            #     for col in column_descriptions:
            #         col_name = col.get('name', '')
            #         col_desc = col.get('description', '')
            #         if col_name and col_desc:
            #             enhanced_content += f"\n- {col_name}: {col_desc}"
            # if sample_data:
            #     enhanced_content += f"\n\nSample Data:"
            #     for sample in sample_data[:3]:  # Limit to first 3 samples
            #         if 'merchant_name' in sample:
            #             enhanced_content += f"\n- {sample.get('merchant_name', '')}, {sample.get('merchant_category', '')}, {sample.get('txn_type', '')}, {sample.get('direction', '')}, {sample.get('amount_aed', '')} AED"
            #         elif 'full_name' in sample:
            #             enhanced_content += f"\n- User: {sample.get('full_name', '')}, Email: {sample.get('email', '')}"
            #         elif 'account_name' in sample:
            #             enhanced_content += f"\n- Account: {sample.get('account_name', '')}, Type: {sample.get('account_type', '')}, Currency: {sample.get('currency', '')}"
            # ADD THIS SECTION:
            if join_info:
                enhanced_content += f"\n\nJoin Information:"
                for join in join_info:
                    join_type = join.get('type', '')
                    join_table = join.get('table', '')
                    join_condition = join.get('condition', '')
                    if join_type and join_table and join_condition:
                        enhanced_content += f"\n- {join_type} JOIN {join_table} ON {join_condition}"
            # # Generate embedding from enhanced content
            embedding = self._generate_embedding(enhanced_content)
            
            # Create point with structured payload matching JSON format
            point_id = str(uuid.uuid4())
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "id": point_id,
                    "object_type": object_type,
                    "table": name,
                    "ddl": ddl_statement,
                    "description": description,
                    "columns": column_descriptions,  # Structured column info
                    "sample_data": sample_data,
                    "join_info": join_info,
                    "ddl_columns": columns,  # Extracted column names for backward compatibility
                    "timestamp": datetime.now().isoformat(),
                    "collection_type": "ddl_definitions"
                }
            )
            
            # Insert into ddl_definitions collection
            self.client.upsert(
                collection_name=self.collections['ddl_definitions'],
                points=[point]
            )
            
            logger.info(f"Added structured DDL to ddl_definitions collection: {point_id}")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to add DDL: {str(e)}")
            raise VectorStoreError(f"Failed to add DDL: {str(e)}")
    
    def add_documentation(self, title: str, body: str, tags: Optional[List[str]] = None) -> str:
        """
        Add documentation to the docs collection.
        
        Args:
            title: Document title
            body: Document body/content
            tags: Optional tags for categorization
            
        Returns:
            Document ID
        """
        try:
            # Create embedding from title + body
            embedding_text = f"{title} {body}"
            embedding = self._generate_embedding(embedding_text)
            
            # Create point
            point_id = str(uuid.uuid4())
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "id": point_id,
                    "title": title,
                    "body": body,
                    "tags": tags or [],
                    "timestamp": datetime.now().isoformat(),
                    "collection_type": "docs"
                }
            )
            
            # Insert into docs collection
            self.client.upsert(
                    collection_name=self.collections['docs'],
                points=[point]
            )
            
            logger.info(f"Added documentation to docs collection: {point_id}")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to add documentation: {str(e)}")
            raise VectorStoreError(f"Failed to add documentation: {str(e)}")
    
    def search_similar(self, 
                      query: str, 
                      collection_types: Optional[List[str]] = None,
                      n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar content across collections.
        
        Args:
            query: Search query
            collection_types: List of collection types to search ('query_pairs', 'ddl_definitions', 'docs')
            n_results: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        try:
            # Default to searching all collections
            if collection_types is None:
                collection_types = ['query_pairs', 'ddl_definitions', 'docs']
            
            # Create query embedding
            query_embedding = self._generate_embedding(query)
            
            all_results = []
            
            # Search each specified collection
            for collection_type in collection_types:
                if collection_type not in self.collections:
                    logger.warning(f"Unknown collection type: {collection_type}")
                    continue
                
                collection_name = self.collections[collection_type]
                
                try:
                    # Search in this collection
                    search_results = self.client.search(
                        collection_name=collection_name,
                        query_vector=query_embedding,
                        limit=n_results
                    )
                    
                    # Process results
                    for result in search_results:
                        all_results.append({
                            "id": result.payload["id"],
                            "score": result.score,
                            "collection_type": collection_type,
                            "payload": result.payload
                        })
                        
                except Exception as e:
                    logger.warning(f"Search failed for collection {collection_name}: {str(e)}")
                    continue
            
            # Sort by score but don't limit - return all results from all collections
            all_results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Found {len(all_results)} similar results for query: {query[:50]}... ({n_results} per collection)")
            return all_results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise VectorStoreError(f"Search failed: {str(e)}")
    
    def get_context_for_question(self, 
                                   question: str, 
                                   collection_types: Optional[List[str]] = None,
                                   max_context_length: int = 4000) -> str:
        """
        Get relevant context for a question by searching all collections.
        
        Args:
            question: Natural language question
            collection_types: List of collection types to search ('query_pairs', 'ddl_definitions', 'docs')
            max_context_length: Maximum length of context to return
            
        Returns:
            Formatted context string
        """
        try:
            # Search all collections for relevant context
            results = self.search_similar(question, collection_types=collection_types, n_results=10)
            
            context_parts = []
            current_length = 0
            
            # Process results by collection type
            ddl_parts = []
            query_parts = []
            doc_parts = []
            
            for result in results:
                payload = result["payload"]
                collection_type = result["collection_type"]
                
                if collection_type == "ddl_definitions":
                    ddl_text = f"-- Table: {payload['table']}\n{payload['ddl']}\n"
                    ddl_parts.append(ddl_text)
                    
                elif collection_type == "query_pairs":
                    query_text = f"-- Example: {payload['question']}\n{payload['sql']}\n"
                    query_parts.append(query_text)
                    
                elif collection_type == "docs":
                    doc_text = f"-- Documentation: {payload['title']}\n{payload['body'][:200]}...\n"
                    doc_parts.append(doc_text)
            
            # Build context in order of importance: DDL, Examples, Documentation
            if ddl_parts:
                context_parts.append("-- DATABASE SCHEMA --")
                context_parts.extend(ddl_parts[:3])  # Limit DDL statements
                
            if query_parts:
                context_parts.append("\n-- EXAMPLE QUERIES --")
                context_parts.extend(query_parts[:3])  # Limit examples
                
            if doc_parts:
                context_parts.append("\n-- DOCUMENTATION --")
                context_parts.extend(doc_parts[:2])  # Limit documentation
            
            # Join and truncate if necessary
            context = "\n".join(context_parts)
            
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            logger.debug(f"Generated context of length {len(context)} for question")
            return context
            
        except Exception as e:
            logger.error(f"Failed to get context: {str(e)}")
            return ""
    
    def get_table_specific_context(self, table_names: List[str], max_context_length: int = 4000) -> str:
        """
        Get DDL definitions and documentation for specific tables.
        
        Args:
            table_names: List of table names to retrieve context for
            max_context_length: Maximum length of context to return
            
        Returns:
            Formatted context string with DDL and docs for specified tables
        """
        try:
            if not table_names:
                return ""
            
            context_parts = []
            
            # Search for DDL definitions for each table
            ddl_parts = []
            doc_parts = []
            
            for table_name in table_names:
                # Search for DDL definitions
                ddl_results = self.search_similar(
                    query=table_name,
                    collection_types=['ddl_definitions'],
                    n_results=3
                )
                
                # Filter results that actually match the table name
                for result in ddl_results:
                    payload = result['payload']
                    if payload.get('table', '').lower() == table_name.lower():
                        ddl_text = f"-- Table: {payload['table']}\n{payload.get('ddl', '')}\n"
                        if ddl_text not in ddl_parts:
                            ddl_parts.append(ddl_text)
                        break
                
                # Search for documentation for this table
                doc_results = self.search_similar(
                    query=f"{table_name} table",
                    collection_types=['docs'],
                    n_results=2
                )
                
                # Filter docs that are relevant to this table
                for result in doc_results:
                    payload = result['payload']
                    title = payload.get('title', '').lower()
                    body = payload.get('body', '').lower()
                    
                    if table_name.lower() in title or table_name.lower() in body:
                        doc_text = f"-- Documentation: {payload.get('title', '')}\n{payload.get('body', '')[:300]}...\n"
                        if doc_text not in doc_parts:
                            doc_parts.append(doc_text)
            
            # Build context in order: DDL first, then documentation
            if ddl_parts:
                context_parts.append("-- DATABASE SCHEMA --")
                context_parts.extend(ddl_parts)
            
            if doc_parts:
                context_parts.append("\n-- TABLE DOCUMENTATION --")
                context_parts.extend(doc_parts)
            
            # Join and truncate if necessary
            context = "\n".join(context_parts)
            
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            logger.debug(f"Generated table-specific context of length {len(context)} for tables: {table_names}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to get table-specific context: {str(e)}")
            return ""
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        try:
            stats = {}
            total_documents = 0
            
            for collection_type, collection_name in self.collections.items():
                try:
                    info = self.client.get_collection(collection_name)
                    count = info.points_count
                    stats[collection_type] = count
                    total_documents += count
                except Exception as e:
                    logger.warning(f"Failed to get stats for {collection_name}: {str(e)}")
                    stats[collection_type] = 0
            
            stats["total_documents"] = total_documents
            stats["collections"] = list(self.collections.keys())
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_collection(self, collection_type: str):
        """Clear a specific collection."""
        if collection_type not in self.collections:
            raise VectorStoreError(f"Unknown collection type: {collection_type}")
        
        try:
            collection_name = self.collections[collection_type]
            self.client.delete_collection(collection_name)
            
            # Recreate the collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"Cleared collection: {collection_type}")
            
        except Exception as e:
            logger.error(f"Failed to clear collection {collection_type}: {str(e)}")
            raise VectorStoreError(f"Failed to clear collection: {str(e)}")
    
    def clear_all_collections(self):
        """Clear all collections."""
        for collection_type in self.collections.keys():
            self.clear_collection(collection_type)
        logger.info("Cleared all collections")
    
    # Helper methods for parsing SQL and DDL
    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL (simple parsing)."""
        import re
        # Simple regex to find table names after FROM and JOIN
        tables = []
        sql_upper = sql.upper()
        
        # Find tables after FROM
        from_matches = re.findall(r'FROM\s+(\w+)', sql_upper)
        tables.extend(from_matches)
        
        # Find tables after JOIN
        join_matches = re.findall(r'JOIN\s+(\w+)', sql_upper)
        tables.extend(join_matches)
        
        return list(set(tables))  # Remove duplicates
    
    def _extract_columns_from_sql(self, sql: str) -> List[str]:
        """Extract column names from SQL (simple parsing)."""
        import re
        columns = []
        
        # This is a simplified extraction - in practice, you'd want more sophisticated parsing
        # Find column names in SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql.upper())
        if select_match:
            select_clause = select_match.group(1)
            if select_clause != '*':
                # Split by comma and clean up
                cols = [col.strip().split('.')[-1] for col in select_clause.split(',')]
                columns.extend(cols)
        
        return columns
    
    def _extract_object_type_from_ddl(self, ddl: str) -> str:
        """Extract object type from DDL statement."""
        ddl_upper = ddl.upper().strip()
        if ddl_upper.startswith('CREATE TABLE'):
            return 'table'
        elif ddl_upper.startswith('CREATE VIEW'):
            return 'view'
        elif ddl_upper.startswith('CREATE INDEX'):
            return 'index'
        else:
            return 'unknown'
    
    def _extract_name_from_ddl(self, ddl: str) -> str:
        """Extract object name from DDL statement."""
        import re
        # Simple regex to extract table/view name
        match = re.search(r'CREATE\s+(?:TABLE|VIEW)\s+(\w+)', ddl.upper())
        if match:
            return match.group(1).lower()
        return 'unknown'
    
    def _extract_columns_from_ddl(self, ddl: str) -> List[str]:
        """Extract column names from CREATE TABLE statement."""
        import re
        columns = []
        
        # Find content between parentheses
        match = re.search(r'\((.*)\)', ddl, re.DOTALL)
        if match:
            content = match.group(1)
            # Split by comma and extract column names
            for line in content.split(','):
                line = line.strip()
                if line and not line.upper().startswith(('PRIMARY', 'FOREIGN', 'UNIQUE', 'CHECK', 'CONSTRAINT')):
                    # Extract first word as column name
                    col_match = re.match(r'(\w+)', line)
                    if col_match:
                        columns.append(col_match.group(1).lower())
        
        return columns 

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI's text-embedding-3-small model."""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=text,
                dimensions=self.vector_size
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise VectorStoreError(f"Embedding generation failed: {str(e)}") 