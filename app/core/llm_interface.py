"""
LLM interface for different providers (OpenAI, Anthropic, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import re
import sqlparse
from sqlparse import sql, tokens
import sqlglot
from sqlglot import exp
from openai import OpenAI
from app.utils.logger import logger
from config.settings import settings
from app.utils.exceptions import LLMError, ConfigurationError


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_sql(self, prompt: str, context: str, user_id: Optional[int] = None) -> str:
        """Generate SQL from natural language prompt and context."""
        pass
    
    @abstractmethod
    def explain_sql(self, sql: str, relevant_tables: List[str], db_connector=None) -> Dict[str, Any]:
        """Explain what a SQL query does and extract string literal mappings."""
        pass
    
    @abstractmethod
    def generate_summary(self, question: str, sql: str, results: Dict[str, Any]) -> str:
        """Generate a natural language summary of query results."""
        pass
    
    @abstractmethod
    def extract_relevant_tables(self, question: str, context: str) -> List[str]:
        """Extract relevant table names from context based on the question."""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI GPT implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, db_connector=None, 
                 interactive_threshold: float = 5.0):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self.db_connector = db_connector
        self.interactive_threshold = interactive_threshold  # Score difference threshold for interactive selection
        
        if not self.api_key:
            logger.error("OpenAI API key is required but not provided")
            raise ConfigurationError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI LLM with model: {self.model}")
    
    def generate_sql(self, prompt: str, context: str, user_id: Optional[int] = None) -> str:
        """Generate SQL using OpenAI GPT."""
        try:
            system_prompt = self._build_system_prompt(user_id)
            user_prompt = self._build_user_prompt(prompt, context, user_id)
            
            logger.debug(f"Generating SQL for prompt: {prompt[:100]}... (User ID: {user_id})")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            sql = response.choices[0].message.content.strip()
            logger.info(f"Successfully generated SQL for prompt: {prompt[:50]}... (User ID: {user_id})")
            return self._extract_sql(sql)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise LLMError(f"Failed to generate SQL: {str(e)}")
    
    def explain_sql(self, sql: str, relevant_tables: List[str], db_connector=None) -> Dict[str, Any]:
        """Explain SQL query using OpenAI GPT and extract string literal mappings."""
        try:
            prompt = f"""
            Explain the following SQL query in simple, business-friendly language:
            
            {sql}
            
            Focus on:
            1. What data is being retrieved
            2. Any filters or conditions applied
            3. How the data is organized or grouped
            4. The business purpose of the query
            """
            
            logger.debug(f"Generating explanation for SQL: {sql[:100]}...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            explanation = response.choices[0].message.content.strip()
            
            # Extract string literals and their mappings
            string_mappings = self._extract_string_literal_mappings(sql, relevant_tables, db_connector)
            print(f"String mappings: {string_mappings}")
            # Execute fuzzy queries and find best matches if database connector is available
            best_matches = []
            if db_connector and string_mappings:
                best_matches = self._execute_fuzzy_queries_and_find_best_matches(string_mappings, db_connector)
            logger.info("Successfully generated SQL explanation with string mappings")
            return {
                "explanation": explanation,
                "string_mappings": string_mappings,
                "best_matches": best_matches
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error during explanation: {str(e)}")
            raise LLMError(f"Failed to explain SQL: {str(e)}")
    
    def _extract_string_literal_mappings(self, sql: str, relevant_tables: List[str], db_connector=None) -> List[Dict[str, str]]:
        """Extract string literals from SQL and map them to their table/column context using enhanced AST parsing."""
        try:
            # Use enhanced AST-based approach with cardinality analysis if db_connector is provided
            if db_connector:
                ast_results = self.find_string_literals_ast_enhanced(sql, relevant_tables, db_connector)
            else:
                # Fallback to original method if no database connector
                ast_results = self.find_string_literals_ast(sql, relevant_tables)
            print(f"AST results: {ast_results}")
            # Convert AST results to the expected format for compatibility
            mappings = []
            processed_literals = set()  # Track processed literals to avoid duplicates
            
            for result in ast_results:
                literal_value = result.get("value", "")  # Cleaned value (without %)
                table_name = result.get("table", "unknown")
                column_name = result.get("column", "")
                
                # Skip summary entries - they're not real database tables
                if table_name == "SUMMARY" or result.get("kind") == "summary":
                    continue
                
                # Skip if we've already processed this literal for this table/column
                key = (literal_value, table_name, column_name)
                if key in processed_literals:
                    continue
                processed_literals.add(key)
                
                # Only generate fuzzy search variations for high cardinality columns
                is_high_cardinality = result.get("is_high_cardinality", False)
                fuzzy_variations = []
                if is_high_cardinality:
                    fuzzy_variations = self._generate_fuzzy_variations(literal_value, table_name, column_name)
                else:
                    # For low cardinality columns, still provide basic info but no fuzzy matching
                    fuzzy_variations = [{
                        "type": "skipped_low_cardinality",
                        "pattern": literal_value,
                        "sql": f"-- Skipped: {column_name} has ≤{result.get('cardinality_threshold', 10)} unique values",
                        "priority": 999,
                        "reason": f"Column {table_name}.{column_name} filtered out due to low cardinality"
                    }]
                
                # Convert the AST result format to the expected mapping format
                mapping = {
                    "literal": literal_value,  # This is now the cleaned value (without %)
                    "table": table_name,
                    "column": column_name,
                    "alias": result.get("alias", ""),
                    "operator": result.get("operator", "="),
                    "kind": result.get("kind", "compare"),
                    "values": result.get("values", []),
                    "raw_value": result.get("raw_value", literal_value),  # Original with %
                    "fuzzy_variations": fuzzy_variations  # Required by _execute_fuzzy_queries_and_find_best_matches
                }
                
                # For backward compatibility, also add the context field
                alias = result.get("alias", "")
                operator = result.get("operator", "=")
                
                if result.get("kind") == "inlist":
                    values_str = "', '".join(result.get("values", []))
                    col_ref = f"{alias}.{column_name}" if alias else column_name
                    mapping["context"] = f"{col_ref} {operator} ('{values_str}')"
                else:
                    col_ref = f"{alias}.{column_name}" if alias else column_name
                    mapping["context"] = f"{col_ref} {operator} '{literal_value}'"
                
                mappings.append(mapping)
            
            logger.debug(f"AST-based extraction found {len(mappings)} string literals")
            print(f"Relevant tables: {relevant_tables}")
            print(f"AST Mappings: {mappings}")
            
            return mappings
            
        except Exception as e:
            logger.error(f"Error extracting string literal mappings with AST: {str(e)}")
            # Fallback to regex approach if AST fails
            try:
                logger.info("Falling back to regex-based extraction")
                mappings = []
                self._find_string_literals_with_context(sql, mappings, relevant_tables)
                return mappings
            except Exception as fallback_error:
                logger.error(f"Fallback regex extraction also failed: {str(fallback_error)}")
                return []
    
    def _extract_table_info(self, sql: str) -> Dict[str, str]:
        """Extract table names and their aliases from parsed SQL using AST."""
        table_info = {}  # alias -> table_name mapping
        
        try:
            # Parse SQL using sqlglot
            tree = sqlglot.parse_one(sql, read="postgres")
            
            # Find all table references in the SQL
            for table in tree.find_all(exp.Table):
                table_name = table.name
                alias = table.alias if table.alias else table_name
                
                # Store the alias -> table mapping
                table_info[alias] = table_name
                
                # Also store lowercase versions for case-insensitive lookup
                table_info[alias.lower()] = table_name.lower()
                if alias != table_name:
                    table_info[table_name.lower()] = table_name.lower()
            
            logger.debug(f"Extracted table info: {table_info}")
            print(f"AST-based table_info: {table_info}")
            
        except Exception as e:
            logger.debug(f"Failed to extract table info with AST: {e}")
            # Fallback to empty dict - the AST-based string extraction will handle "unknown" tables
            table_info = {}
        
        return table_info
    
    def get_table_column_analysis(self, db_connector, table_name: str) -> Dict[str, int]:
        """Get unique count analysis for all columns in a table."""
        try:
            print(f"Analyzing table: {table_name}")
            # Get table schema first
            schema = db_connector.get_table_schema(table_name)
            print(f"Schema for {table_name}: {schema}")
            columns = []
            
            # Extract column names based on database type
            if 'columns' in schema:
                if isinstance(schema['columns'], list) and len(schema['columns']) > 0:
                    # Handle different schema formats
                    first_col = schema['columns'][0]
                    print(f"First column structure: {first_col}")
                    if 'column_name' in first_col:
                        columns = [col['column_name'] for col in schema['columns']]
                    elif 'name' in first_col:
                        columns = [col['name'] for col in schema['columns']]
            
            print(f"Extracted columns for {table_name}: {columns}")
            if not columns:
                logger.warning(f"No columns found for table {table_name}")
                return {}
            
            # Get unique counts for each column
            column_unique_counts = {}
            for column in columns:
                try:
                    # Use COUNT(DISTINCT column_name) to get unique values
                    sql = f"SELECT COUNT(DISTINCT {column}) as unique_count FROM {table_name}"
                    print(f"Executing: {sql}")
                    result = db_connector.execute_query(sql)
                    print(f"Result for {column}: {result}")
                    
                    if result and 'data' in result and len(result['data']) > 0:
                        unique_count = result['data'][0][0]
                        column_unique_counts[column] = unique_count
                        print(f"Column {column}: {unique_count} unique values")
                    else:
                        column_unique_counts[column] = 0
                        print(f"Column {column}: 0 unique values (no data)")
                        
                except Exception as e:
                    logger.warning(f"Failed to get unique count for {table_name}.{column}: {e}")
                    print(f"Error analyzing {column}: {e}")
                    column_unique_counts[column] = 0
            
            print(f"Final column analysis for {table_name}: {column_unique_counts}")
            logger.info(f"Column analysis for {table_name}: {column_unique_counts}")
            return column_unique_counts
            
        except Exception as e:
            logger.error(f"Failed to analyze table {table_name}: {e}")
            print(f"Failed to analyze table {table_name}: {e}")
            return {}
    
    def filter_high_cardinality_columns(self, column_analysis: Dict[str, int], threshold: int = 10) -> List[str]:
        """Filter columns to keep only those with unique count > threshold."""
        high_cardinality_columns = [
            column for column, unique_count in column_analysis.items() 
            if unique_count > threshold
        ]
        logger.info(f"High cardinality columns (>{threshold}): {high_cardinality_columns}")
        return high_cardinality_columns
    
    def find_string_literals_ast_enhanced(self, sql: str, relevant_tables: List[str], db_connector) -> List[dict]:
        """
        Enhanced version: Extract column ↔ string literal predicates using SQL AST with cardinality analysis.
        Only processes columns with unique count > 10.
        """
        print(f"SQL: {sql}")
        print(f"Relevant tables: {relevant_tables}")
        
        # First, analyze all tables and get high cardinality columns
        table_columns_map = {}
        for table in relevant_tables:
            column_analysis = self.get_table_column_analysis(db_connector, table)
            high_cardinality_columns = self.filter_high_cardinality_columns(column_analysis)
            table_columns_map[table] = high_cardinality_columns
            print(f"Table {table} high cardinality columns: {high_cardinality_columns}")
        
        def is_high_cardinality_column(table_name: str, column_name: str) -> bool:
            """Check if a column is high cardinality (>10 unique values)."""
            return column_name in table_columns_map.get(table_name, [])
        
        try:
            tree = sqlglot.parse_one(sql, read="postgres")
        except Exception as e:
            logger.debug(f"sqlglot parse failed: {e}")
            return []

        out, seen = [], set()

        def resolve_table_name(alias: str) -> str:
            """Resolve alias to actual table name using relevant_tables."""
            if not alias:
                # If no alias and we have relevant tables, use the first one as default
                if relevant_tables:
                    return relevant_tables[0]
                return "unknown"
            
            # Check if alias matches any relevant table (case-insensitive)
            if relevant_tables:
                for table in relevant_tables:
                    if alias.lower() == table.lower():
                        return table
                # If no exact match, return first relevant table as fallback
                return relevant_tables[0]
            
            return "unknown"

        def extract_column_info(column_expr):
            """Extract alias and column name from a column expression."""
            if isinstance(column_expr, exp.Column):
                alias = column_expr.table if column_expr.table else None
                column = column_expr.name
                return alias, column
            
            # Try to find a column within the expression (for functions, etc.)
            for column in column_expr.find_all(exp.Column):
                alias = column.table if column.table else None
                column_name = column.name
                return alias, column_name
            
            return None, None

        # Find all comparison operations with string literals
        for node in tree.find_all((exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.Like, exp.ILike, exp.In)):
            if isinstance(node, (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)):
                # Binary comparisons: column = 'value'
                left, right = node.left, node.right
                alias, column = extract_column_info(left)
                
                if isinstance(right, exp.Literal) and right.is_string:
                    table_name = resolve_table_name(alias)
                    
                    # Map expression types to operators
                    op_map = {
                        exp.EQ: "=", exp.NEQ: "!=", exp.LT: "<", 
                        exp.LTE: "<=", exp.GT: ">", exp.GTE: ">="
                    }
                    operator = op_map.get(type(node), "=")
                    
                    key = (alias, column, operator, right.this)
                    if key not in seen and column:
                        seen.add(key)
                        is_high_cardinality = is_high_cardinality_column(table_name, column)
                        out.append({
                            "table": table_name,
                            "alias": alias,
                            "column": column,
                            "operator": operator,
                            "value": right.this,
                            "values": [right.this],
                            "kind": "compare",
                            "high_cardinality_columns": table_columns_map.get(table_name, []),
                            "column_analysis": "included" if is_high_cardinality else "filtered_out",
                            "is_high_cardinality": is_high_cardinality,
                            "cardinality_threshold": 10
                        })
            
            elif isinstance(node, (exp.Like, exp.ILike)):
                # LIKE/ILIKE operations
                left, right = node.this, node.expression
                alias, column = extract_column_info(left)
                
                if isinstance(right, exp.Literal) and right.is_string:
                    table_name = resolve_table_name(alias)
                    
                    operator = "ILIKE" if isinstance(node, exp.ILike) else "LIKE"
                    
                    # Extract the actual search term by removing % wildcards
                    raw_value = right.this
                    clean_value = raw_value.strip('%')  # Remove leading/trailing %
                    
                    key = (alias, column, operator, raw_value)
                    if key not in seen and column:
                        seen.add(key)
                        is_high_cardinality = is_high_cardinality_column(table_name, column)
                        out.append({
                            "table": table_name,
                            "alias": alias,
                            "column": column,
                            "operator": operator,
                            "value": clean_value,  # Use cleaned value (without %)
                            "raw_value": raw_value,  # Keep original for reference
                            "values": [clean_value],
                            "kind": "compare",
                            "high_cardinality_columns": table_columns_map.get(table_name, []),
                            "column_analysis": "included" if is_high_cardinality else "filtered_out",
                            "is_high_cardinality": is_high_cardinality,
                            "cardinality_threshold": 10
                        })
            
            elif isinstance(node, exp.In):
                # IN operations: column IN ('val1', 'val2')
                alias, column = extract_column_info(node.this)
                
                values = []
                for expr in node.expressions:
                    if isinstance(expr, exp.Literal) and expr.is_string:
                        values.append(expr.this)
                
                if values and column:
                    table_name = resolve_table_name(alias)
                    
                    key = (alias, column, "IN", tuple(values))
                    if key not in seen:
                        seen.add(key)
                        is_high_cardinality = is_high_cardinality_column(table_name, column)
                        out.append({
                            "table": table_name,
                            "alias": alias,
                            "column": column,
                            "operator": "IN",
                            "value": values[0] if len(values) == 1 else None,
                            "values": values,
                            "kind": "inlist",
                            "high_cardinality_columns": table_columns_map.get(table_name, []),
                            "column_analysis": "included" if is_high_cardinality else "filtered_out",
                            "is_high_cardinality": is_high_cardinality,
                            "cardinality_threshold": 10
                        })
        
        # Add summary information about all tables and their high cardinality columns
        summary_info = {
            "table": "SUMMARY",
            "alias": None,
            "column": "ANALYSIS_SUMMARY",
            "operator": "INFO",
            "value": f"Analyzed {len(relevant_tables)} tables",
            "values": [],
            "kind": "summary",
            "tables_analysis": {
                table: {
                    "high_cardinality_columns": columns,
                    "column_count": len(columns)
                } for table, columns in table_columns_map.items()
            },
            "total_high_cardinality_columns": sum(len(columns) for columns in table_columns_map.values())
        }
        out.append(summary_info)
        
        print(f"Enhanced AST results with cardinality analysis: {out}")
        return out

    def find_string_literals_ast(self, sql: str, relevant_tables: List[str]) -> List[dict]:
        """
        Robustly extract column ↔ string literal predicates using SQL AST.
        """
        print(f"SQL: {sql}")
        print(f"Relevant tables: {relevant_tables}")
        try:
            tree = sqlglot.parse_one(sql, read="postgres")
        except Exception as e:
            logger.debug(f"sqlglot parse failed: {e}")
            return []

        out, seen = [], set()

        def resolve_table_name(alias: str) -> str:
            """Resolve alias to actual table name using relevant_tables."""
            if not alias:
                # If no alias and we have relevant tables, use the first one as default
                if relevant_tables:
                    return relevant_tables[0]
                return "unknown"
            
            # Check if alias matches any relevant table (case-insensitive)
            if relevant_tables:
                for table in relevant_tables:
                    if alias.lower() == table.lower():
                        return table
                # If no exact match, return first relevant table as fallback
                return relevant_tables[0]
            
            return "unknown"

        def extract_column_info(column_expr):
            """Extract alias and column name from a column expression."""
            if isinstance(column_expr, exp.Column):
                alias = column_expr.table if column_expr.table else None
                column = column_expr.name
                return alias, column
            
            # Try to find a column within the expression (for functions, etc.)
            for column in column_expr.find_all(exp.Column):
                alias = column.table if column.table else None
                column_name = column.name
                return alias, column_name
            
            return None, None

        # Find all comparison operations with string literals
        for node in tree.find_all((exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.Like, exp.ILike, exp.In)):
            if isinstance(node, (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)):
                # Binary comparisons: column = 'value'
                left, right = node.left, node.right
                alias, column = extract_column_info(left)
                
                if isinstance(right, exp.Literal) and right.is_string:
                    table_name = resolve_table_name(alias)
                    
                    # Map expression types to operators
                    op_map = {
                        exp.EQ: "=", exp.NEQ: "!=", exp.LT: "<", 
                        exp.LTE: "<=", exp.GT: ">", exp.GTE: ">="
                    }
                    operator = op_map.get(type(node), "=")
                    
                    key = (alias, column, operator, right.this)
                    if key not in seen and column:
                        seen.add(key)
                        out.append({
                            "table": table_name,
                            "alias": alias,
                            "column": column,
                            "operator": operator,
                            "value": right.this,
                            "values": [right.this],
                            "kind": "compare",
                        })
            
            elif isinstance(node, (exp.Like, exp.ILike)):
                # LIKE/ILIKE operations
                left, right = node.this, node.expression
                alias, column = extract_column_info(left)
                
                if isinstance(right, exp.Literal) and right.is_string:
                    table_name = resolve_table_name(alias)
                    operator = "ILIKE" if isinstance(node, exp.ILike) else "LIKE"
                    
                    # Extract the actual search term by removing % wildcards
                    raw_value = right.this
                    clean_value = raw_value.strip('%')  # Remove leading/trailing %
                    
                    key = (alias, column, operator, raw_value)
                    if key not in seen and column:
                        seen.add(key)
                        out.append({
                            "table": table_name,
                            "alias": alias,
                            "column": column,
                            "operator": operator,
                            "value": clean_value,  # Use cleaned value (without %)
                            "raw_value": raw_value,  # Keep original for reference
                            "values": [clean_value],
                            "kind": "compare",
                        })
            
            elif isinstance(node, exp.In):
                # IN operations: column IN ('val1', 'val2')
                alias, column = extract_column_info(node.this)
                
                values = []
                for expr in node.expressions:
                    if isinstance(expr, exp.Literal) and expr.is_string:
                        values.append(expr.this)
                
                if values and column:
                    table_name = resolve_table_name(alias)
                    key = (alias, column, "IN", tuple(values))
                    if key not in seen:
                        seen.add(key)
                        out.append({
                            "table": table_name,
                            "alias": alias,
                            "column": column,
                            "operator": "IN",
                            "value": values[0] if len(values) == 1 else None,
                            "values": values,
                            "kind": "inlist",
                        })
        
        return out
    
    def _find_string_literals_with_context(self, sql: str, mappings: List[Dict[str, str]], relevant_tables: List[str]):
        """Find string literals and their context using regex patterns."""
        try:
            processed_literals = set()  # Track processed literals to avoid duplicates
            print(f"SQL: {sql}")
            print(f"relevant_tables: {relevant_tables}")
            
            def resolve_table_name_regex(alias: str) -> str:
                """Resolve table name using relevant_tables."""
                # Check if alias matches any relevant table (case-insensitive)
                if relevant_tables:
                    for table in relevant_tables:
                        if alias.lower() == table.lower():
                            return table
                    # Return first relevant table as fallback
                    return relevant_tables[0]
                
                return alias  # Return original alias if no resolution found
            
            # Pattern 1: table.column = 'value' or table.column LIKE 'value'
            pattern1 = r'(\w+)\.(\w+)\s*(?:[=<>!]+|LIKE|ILIKE|IN)\s*[\'"]([^\'"]+)[\'"]'
            matches1 = re.finditer(pattern1, sql, re.IGNORECASE)
            for match in matches1:
                table_alias = match.group(1).lower()
                column_name = match.group(2).lower()
                literal_value = match.group(3)
                table_name = resolve_table_name_regex(table_alias)
                
                print(f"Pattern 1 match: {table_alias}.{column_name} = '{literal_value}'")
                
                # Generate fuzzy search variations
                fuzzy_variations = self._generate_fuzzy_variations(literal_value, table_name, column_name)
                
                key = (literal_value, table_name, column_name)
                if key not in processed_literals:
                    processed_literals.add(key)
                    mappings.append({
                        "literal": literal_value,
                        "table": table_name,
                        "column": column_name,
                        "fuzzy_variations": fuzzy_variations
                    })
            
            # Pattern 2: LOWER(table.column) = 'value' or similar function calls
            pattern2 = r'(?:LOWER|UPPER)\s*\(\s*(\w+)\.(\w+)\s*\)\s*(?:[=<>!]+|LIKE|ILIKE)\s*[\'"]([^\'"]+)[\'"]'
            matches2 = re.finditer(pattern2, sql, re.IGNORECASE)
            for match in matches2:
                table_alias = match.group(1).lower()
                column_name = match.group(2).lower()
                literal_value = match.group(3)
                table_name = resolve_table_name_regex(table_alias)
                
                print(f"Pattern 2 match: FUNC({table_alias}.{column_name}) = '{literal_value}'")
                
                # Generate fuzzy search variations
                fuzzy_variations = self._generate_fuzzy_variations(literal_value, table_name, column_name)
                
                key = (literal_value, table_name, column_name)
                if key not in processed_literals:
                    processed_literals.add(key)
                    mappings.append({
                        "literal": literal_value,
                        "table": table_name,
                        "column": column_name,
                        "fuzzy_variations": fuzzy_variations
                    })
            
            # Pattern 3: column = 'value' (without table prefix)
            pattern3 = r'(?<!\.)\b(\w+)\s*(?:[=<>!]+|LIKE|ILIKE|IN)\s*[\'"]([^\'"]+)[\'"]'
            matches3 = re.finditer(pattern3, sql, re.IGNORECASE)
            for match in matches3:
                column_name = match.group(1).lower()
                literal_value = match.group(2)
                
                # Skip if this looks like a function or keyword
                if column_name.upper() in ['AND', 'OR', 'WHERE', 'SELECT', 'FROM', 'JOIN', 'ON', 'ORDER', 'GROUP', 'HAVING', 'LIMIT', 'OFFSET']:
                    continue
                
                # Use the first relevant table for columns without table prefix
                table_name = "unknown"
                if relevant_tables:
                    table_name = relevant_tables[0]
                    print(f"Using first relevant table: {table_name}")
                
                print(f"Pattern 3 match: {column_name} = '{literal_value}' (table: {table_name})")
                
                # Generate fuzzy search variations
                fuzzy_variations = self._generate_fuzzy_variations(literal_value, table_name, column_name)
                
                key = (literal_value, table_name, column_name)
                if key not in processed_literals:
                    processed_literals.add(key)
                    mappings.append({
                        "literal": literal_value,
                        "table": table_name,
                        "column": column_name,
                        "fuzzy_variations": fuzzy_variations
                    })

            
        except Exception as e:
            logger.debug(f"Error finding string literals with context: {str(e)}")
    
    def _generate_fuzzy_variations(self, literal_value: str, table_name: str, column_name: str) -> List[Dict[str, str]]:
        """Generate optimized fuzzy search variations for a string literal."""
        variations = []
        
        try:
            # Split the literal into words for analysis
            words = literal_value.split()
            literal_lower = literal_value.lower()
            
            # 1. Exact matches (highest priority)
            variations.append({
                "type": "exact_match",
                "pattern": literal_value,
                "sql": f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} = '{literal_value}' LIMIT 5",
                "priority": 0  # Highest priority
            })
            
            # 2. Case-insensitive exact match
            variations.append({
                "type": "case_insensitive_exact",
                "pattern": literal_value,
                "sql": f"SELECT DISTINCT {column_name} FROM {table_name} WHERE UPPER({column_name}) = UPPER('{literal_value}') LIMIT 5",
                "priority": 0
            })
            
            # 3. Full literal match (contains)
            variations.append({
                "type": "full_literal_match",
                "pattern": f"%{literal_lower}%",
                "sql": f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} ILIKE '%{literal_lower}%' LIMIT 10",
                "priority": 1
            })
            
            # 4. SOUNDEX matching for phonetic similarity
            variations.append({
                "type": "soundex_match",
                "pattern": f"SOUNDEX('{literal_value}')",
                "sql": f"SELECT DISTINCT {column_name} FROM {table_name} WHERE SOUNDEX({column_name}) = SOUNDEX('{literal_value}') LIMIT 10",
                "priority": 3
            })
            
            # 5. Levenshtein distance matching (typo tolerance)
            variations.append({
                "type": "levenshtein_match_1",
                "pattern": f"levenshtein <= 1",
                "sql": f"SELECT DISTINCT {column_name} FROM {table_name} WHERE levenshtein(LOWER({column_name}), LOWER('{literal_value}')) <= 1 LIMIT 10",
                "priority": 4
            })
            
            variations.append({
                "type": "levenshtein_match_2",
                "pattern": f"levenshtein <= 2",
                "sql": f"SELECT DISTINCT {column_name} FROM {table_name} WHERE levenshtein(LOWER({column_name}), LOWER('{literal_value}')) <= 2 LIMIT 15",
                "priority": 5
            })
            
            # 6. Trigram similarity matching
            variations.append({
                "type": "similarity_match",
                "pattern": f"similarity >= 0.3",
                "sql": f"SELECT DISTINCT {column_name} FROM {table_name} WHERE similarity({column_name}, '{literal_value}') >= 0.3 LIMIT 10",
                "priority": 6
            })
            
            # 7. Word similarity matching
            variations.append({
                "type": "word_similarity_match",
                "pattern": f"word_similarity >= 0.4",
                "sql": f"SELECT DISTINCT {column_name} FROM {table_name} WHERE word_similarity({column_name}, '{literal_value}') >= 0.4 LIMIT 10",
                "priority": 7
            })
            
            # 8. If multi-word, try primary word matching
            if len(words) > 1:
                # Use the longest word as it's usually the most distinctive
                longest_word = max(words, key=len)
                if len(longest_word) > 3:  # Only meaningful words
                    variations.append({
                        "type": "primary_word_match", 
                        "word": longest_word,
                        "pattern": f"%{longest_word.lower()}%",
                        "sql": f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} ILIKE '%{longest_word.lower()}%' LIMIT 10",
                        "priority": 2
                    })
                    
                    # SOUNDEX for primary word
                    variations.append({
                        "type": "soundex_word_match",
                        "word": longest_word,
                        "pattern": f"SOUNDEX('{longest_word}')",
                        "sql": f"SELECT DISTINCT {column_name} FROM {table_name} WHERE SOUNDEX({column_name}) = SOUNDEX('{longest_word}') LIMIT 10",
                        "priority": 4
                    })
                    
                    # Levenshtein for primary word
                    variations.append({
                        "type": "levenshtein_word_match",
                        "word": longest_word,
                        "pattern": f"levenshtein <= 1",
                        "sql": f"SELECT DISTINCT {column_name} FROM {table_name} WHERE levenshtein(LOWER({column_name}), LOWER('{longest_word}')) <= 1 LIMIT 10",
                        "priority": 5
                    })
            
        except Exception as e:
            logger.debug(f"Error generating fuzzy variations: {str(e)}")
        
        # Sort by priority (lower number = higher priority)
        variations.sort(key=lambda x: x.get("priority", 999))
        return variations
    

    
    def _execute_fuzzy_queries_and_find_best_matches(self, string_mappings: List[Dict[str, Any]], db_connector) -> List[Dict[str, Any]]:
        """Execute fuzzy search queries and find the best matching results."""
        best_matches = []
        
        try:
            for mapping in string_mappings:
                literal = mapping["literal"]
                table = mapping["table"]
                column = mapping["column"]
                fuzzy_variations = mapping["fuzzy_variations"]
                raw_value = mapping["raw_value"]
                
                print(f"Processing literal: '{literal}' in {table}.{column}")
                
                # Execute each variation and collect unique merchant names
                all_merchant_names = set()
                variation_results = []
                
                for variation in fuzzy_variations:
                    # Skip execution for 'skipped_low_cardinality' variations
                    if variation.get("type") == "skipped_low_cardinality":
                        print(f"Skipping execution for low cardinality column: {variation.get('reason', 'Unknown reason')}")
                        continue
                    
                    try:
                        result = db_connector.execute_query(variation["sql"])
                        row_count = len(result["data"])
                        
                        # Extract unique merchant names from results
                        merchant_names = set()
                        if result["data"] and result["columns"]:
                            # Find the column index for merchant names
                            col_index = 0
                            if column in result["columns"]:
                                col_index = result["columns"].index(column)
                            
                            for row in result["data"]:
                                if row and len(row) > col_index and row[col_index]:
                                    merchant_names.add(str(row[col_index]))
                        
                        all_merchant_names.update(merchant_names)
                        
                        variation_results.append({
                                    "variation": variation,
                                    "row_count": row_count,
                            "merchant_names": merchant_names,
                            "sample_data": result["data"][:5]  # Keep only 5 samples
                        })
                        
                        print(f"Variation {variation['type']}: {row_count} rows, {len(merchant_names)} unique names")
                        
                        # Stop early if we found exact matches
                        if variation["type"] in ["exact_match", "case_insensitive_exact"] and row_count > 0:
                            print(f"Found exact match, stopping search")
                            break
                    
                    except Exception as e:
                        print(f"Error executing variation {variation['type']}: {str(e)}")
                        continue
                
                if not variation_results:
                    continue
                
                # Score merchant names based on similarity to original literal
                def score_merchant_name(name: str) -> float:
                    """Score a merchant name based on similarity to the original literal."""
                    name_lower = name.lower()
                    literal_lower = literal.lower()
                    
                    # Exact match (highest score)
                    if name_lower == literal_lower:
                        return 100.0
                    
                    # Case-insensitive exact match
                    if name.lower() == literal.lower():
                        return 95.0
                    
                    # Contains the full literal
                    if literal_lower in name_lower:
                        return 80.0 + (len(literal) / len(name)) * 10  # Bonus for higher ratio
                    
                    # Literal contains the name (partial match)
                    if name_lower in literal_lower:
                        return 70.0 + (len(name) / len(literal)) * 10
                    
                    # Word-level matching
                    literal_words = set(literal_lower.split())
                    name_words = set(name_lower.split())
                    
                    if literal_words & name_words:  # Has common words
                        common_ratio = len(literal_words & name_words) / len(literal_words | name_words)
                        return 50.0 + common_ratio * 30
                    
                    return 0.0
                
                # Score all merchant names and check for ambiguity
                scored_names = []
                print("all_merchant_names", all_merchant_names)
                for name in all_merchant_names:
                    print("name", name)
                    score = score_merchant_name(name)
                    print("score", score)
                    scored_names.append((name, score))
                
                # Sort by score (highest first)
                scored_names.sort(key=lambda x: x[1], reverse=True)
                
                # Check for ambiguity - if top scores are too close, ask user
                best_merchant_name = None
                best_score = 0.0
                
                if scored_names:
                    # Filter out very low scores (< 50)
                    good_matches = [(name, score) for name, score in scored_names if score > 50.0]
                    
                    if len(good_matches) > 1:
                        # Check if top matches are too close (within threshold points)
                        top_score = good_matches[0][1]
                        close_matches = [(name, score) for name, score in good_matches if top_score - score <= self.interactive_threshold]
                        
                        if len(close_matches) > 1:
                            print(f"\nAmbiguous matches found for '{literal}' in {table}.{column}:")
                            selected_names = self._get_user_selection_for_ambiguous_matches(literal, close_matches)
                            
                            if selected_names:
                                # Process all selected names
                                for selected_name in selected_names:
                                    selected_score = next(score for name, score in close_matches if name == selected_name)
                                    
                                    # Find which variation found this merchant name
                                    best_variation = None
                                    for var_result in variation_results:
                                        if selected_name in var_result["merchant_names"]:
                                            best_variation = var_result
                                            break
                                    
                                    best_matches.append({
                                        "original_literal": literal,
                                        "table": table,
                                        "column": column,
                                        "raw_value": raw_value,
                                        "best_match_value": selected_name,
                                        "match_score": selected_score,
                                        "variation_used": best_variation["variation"]["type"] if best_variation else "unknown",
                                        "total_variations_tested": len(fuzzy_variations),
                                        "unique_names_found": len(all_merchant_names),
                                        "user_selected": True  # Flag to indicate user selection
                                    })
                                
                                # Skip the normal processing since we handled it above
                                continue
                            else:
                                # User chose none, skip this mapping
                                continue
                        else:
                            # Clear winner
                            best_merchant_name = good_matches[0][0]
                            best_score = good_matches[0][1]
                    elif len(good_matches) == 1:
                        # Single good match
                        best_merchant_name = good_matches[0][0]
                        best_score = good_matches[0][1]
                
                print(f"Best match: '{best_merchant_name}' (score: {best_score})")
                
                if best_merchant_name and best_score > 50.0:  # Only accept good matches
                    # Find which variation found this merchant name
                    best_variation = None
                    for var_result in variation_results:
                        if best_merchant_name in var_result["merchant_names"]:
                            best_variation = var_result
                            break
                    
                    best_matches.append({
                        "original_literal": literal,
                        "table": table,
                        "column": column,
                        "raw_value": raw_value,
                        "best_match_value": best_merchant_name,
                        "match_score": best_score,
                        "variation_used": best_variation["variation"]["type"] if best_variation else "unknown",
                        "total_variations_tested": len(fuzzy_variations),
                        "unique_names_found": len(all_merchant_names),
                        "user_selected": False  # Flag to indicate automatic selection
                    })
        
        except Exception as e:
            logger.error(f"Error in fuzzy matching: {str(e)}")
        
        return best_matches
    
    def _get_user_selection_for_ambiguous_matches(self, original_literal: str, close_matches: List[tuple]) -> List[str]:
        """
        Present ambiguous matches to user and get their selection.
        
        Args:
            original_literal: The original string literal from the SQL
            close_matches: List of (name, score) tuples with similar scores
            
        Returns:
            List of selected merchant names (can be empty if user chooses none)
        """
        try:
            print(f"\n{'='*60}")
            print(f"AMBIGUOUS MATCH RESOLUTION")
            print(f"{'='*60}")
            print(f"Original search term: '{original_literal}'")
            print(f"Found {len(close_matches)} similar matches:")
            print()
            
            # Display options
            for i, (name, score) in enumerate(close_matches, 1):
                print(f"{i}. {name} (similarity: {score:.1f}%)")
            
            print(f"{len(close_matches) + 1}. All of the above")
            print(f"{len(close_matches) + 2}. None of the above (skip this match)")
            print()
            
            # Get user input
            while True:
                try:
                    user_input = input("Select option(s) (e.g., '1', '1,3', 'all', or 'none'): ").strip().lower()
                    
                    if user_input in ['none', 'skip', str(len(close_matches) + 2)]:
                        print("Skipping this match as requested.")
                        return []
                    
                    if user_input in ['all', str(len(close_matches) + 1)]:
                        selected_names = [name for name, _ in close_matches]
                        print(f"Selected all matches: {selected_names}")
                        return selected_names
                    
                    # Parse individual selections
                    selections = []
                    for part in user_input.replace(' ', '').split(','):
                        if part.isdigit():
                            idx = int(part) - 1
                            if 0 <= idx < len(close_matches):
                                selections.append(idx)
                            else:
                                print(f"Invalid option: {part}. Please try again.")
                                break
                        else:
                            print(f"Invalid input: {part}. Please use numbers, 'all', or 'none'.")
                            break
                    else:
                        # All parts were valid
                        if selections:
                            selected_names = [close_matches[idx][0] for idx in selections]
                            print(f"Selected: {selected_names}")
                            return selected_names
                        else:
                            print("No valid selections made. Please try again.")
                
                except (ValueError, KeyboardInterrupt):
                    print("Invalid input. Please try again or press Ctrl+C to skip.")
                    continue
                except EOFError:
                    print("\nInput interrupted. Skipping this match.")
                    return []
        
        except Exception as e:
            logger.error(f"Error in user selection: {str(e)}")
            print(f"Error getting user input: {str(e)}. Using first match as fallback.")
            return [close_matches[0][0]] if close_matches else []
    
    def generate_summary(self, question: str, sql: str, results: Dict[str, Any]) -> str:
        """Generate a natural language summary of query results using OpenAI GPT."""
        try:
            # Format the results data for the prompt
            results_text = self._format_results_for_summary(results)
            
            prompt = f"""
            Based on the user's question and the query results, generate a natural, conversational summary.
            
            Original Question: {question}
            
            SQL Query: {sql}
            
            Query Results:
            {results_text}
            
            Please provide a natural language summary that:
            1. Directly answers the user's question
            2. Mentions specific data points from the results
            3. Uses a conversational tone (like "Allison Hill has made these..." as mentioned)
            4. Highlights key insights or patterns if applicable
            5. Keeps it concise but informative
            
            Start your response with a direct answer to the question.
            """
            
            logger.debug(f"Generating summary for question: {question[:100]}...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=800
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info("Successfully generated results summary")
            return summary
            
        except Exception as e:
            logger.error(f"OpenAI API error during summary generation: {str(e)}")
            raise LLMError(f"Failed to generate summary: {str(e)}")
    
    def _format_results_for_summary(self, results: Dict[str, Any]) -> str:
        """Format query results for inclusion in the summary prompt."""
        try:
            if not results.get('data'):
                return "No data returned from the query."
            
            columns = results.get('columns', [])
            data = results.get('data', [])
            row_count = results.get('row_count', len(data))
            
            # Limit the number of rows to include in the prompt (to avoid token limits)
            max_rows = min(20, len(data))
            
            formatted_text = f"Total rows: {row_count}\n"
            formatted_text += f"Columns: {', '.join(columns)}\n\n"
            
            if columns and data:
                formatted_text += "Sample data:\n"
                # Add header
                formatted_text += " | ".join(columns) + "\n"
                formatted_text += "-" * (len(" | ".join(columns))) + "\n"
                
                # Add data rows
                for i, row in enumerate(data[:max_rows]):
                    formatted_text += " | ".join([str(cell) if cell is not None else "NULL" for cell in row]) + "\n"
                
                if len(data) > max_rows:
                    formatted_text += f"... and {len(data) - max_rows} more rows\n"
            
            return formatted_text
            
        except Exception as e:
            logger.error(f"Error formatting results for summary: {str(e)}")
            return f"Error formatting results: {str(e)}"
    
    def _build_system_prompt(self, user_id: Optional[int]) -> str:
        """Build the system prompt for SQL generation."""
        user_context = ""
        if user_id is not None and user_id > 0:
            user_context = f"""
        
        IMPORTANT USER CONTEXT:
        - Current user ID: {user_id}
        - ALWAYS include user_id = {user_id} in WHERE clauses for user-specific data
        - For tables that have user_id column, filter by user_id = {user_id}
        - Use exact equality (=) instead of LIKE when you have exact values
        - Avoid unnecessary LOWER() functions when doing exact matches
        """
        
        return f"""
        You are an expert PostgreSQL developer. Generate accurate PostgreSQL SQL queries from natural language questions.
        
        PostgreSQL Guidelines:
        1. Use PostgreSQL syntax and functions only
        2. For exact string matches: Use = operator (e.g., merchant_name = 'PlayStation Plus')
        3. For pattern matching: Use ILIKE operator (e.g., merchant_name ILIKE '%playstation%')
        4. NEVER use LOWER() with LIKE for exact string matches
        5. NEVER use LIKE without wildcards (%, _) - use = instead
        6. Include appropriate JOINs when needed
        7. Return only the SQL query, no explanations
        8. Use the provided schema and context information
        9. For date operations, use PostgreSQL date functions like EXTRACT(), DATE(), NOW(), etc.
        
        MANDATORY TABLE ALIASING RULES:
        - ALWAYS use table aliases for ALL tables in your queries
        - Use short, meaningful aliases (e.g., transactions → t, accounts → a, users → u)
        - ALWAYS qualify column names with table aliases (e.g., t.merchant_name, a.account_type)
        - Example: SELECT t.amount FROM transactions t WHERE t.direction = 'debit'
        - NEVER use unqualified column names like 'merchant_category' - always use 't.merchant_category'
        
        CRITICAL RULES FOR STRING MATCHING:
        - If you have an exact merchant name like 'PlayStation Plus', use: merchant_name = 'PlayStation Plus'
        - If you need case-insensitive exact match, use: UPPER(merchant_name) = UPPER('PlayStation Plus')
        - Only use LIKE with wildcards: merchant_name ILIKE '%playstation%'
        - NEVER combine LOWER() with LIKE for exact values
        
        DATE HANDLING RULES:
        - Use current date information provided in the user prompt
        - ONLY add date filters if the user explicitly mentions dates or time periods
        - If user says "this month": EXTRACT(YEAR FROM date_col) = current_year AND EXTRACT(MONTH FROM date_col) = current_month
        - If user says "last month": Use appropriate month/year calculation
        - If user says "today": DATE(date_col) = 'YYYY-MM-DD'
        - If user says "this year": EXTRACT(YEAR FROM date_col) = current_year
        - If NO date/time mentioned: DO NOT add any date filters to the query
        - Always use proper PostgreSQL date functions and formats when date filtering is needed
        {user_context}
        
        Always wrap your SQL response in ```sql and ``` tags.
        """
    
    def _build_user_prompt(self, question: str, context: str, user_id: Optional[int]) -> str:
        """Build the user prompt with question and context."""
        from datetime import datetime
        
        # Get current date information
        current_date = datetime.now()
        current_date_str = current_date.strftime("%Y-%m-%d")
        current_datetime_str = current_date.strftime("%Y-%m-%d %H:%M:%S")
        current_year = current_date.year
        current_month = current_date.month
        current_day = current_date.day
        
        user_info = ""
        if user_id is not None and user_id > 0:
            user_info = f"\nCurrent User ID: {user_id} (include user_id = {user_id} in WHERE clause for user-specific data)\n"
        
        date_context = f"""
Current Date Information:
- Today's Date: {current_date_str}
- Current DateTime: {current_datetime_str}
- Current Year: {current_year}
- Current Month: {current_month}
- Current Day: {current_day}

IMPORTANT DATE FILTERING RULES:
- ONLY add date/time filters if the user explicitly mentions dates or time periods
- Examples that NEED date filters:
  * "How much did I spend this month?" → Add month filter
  * "Show transactions from last week" → Add week filter  
  * "What did I buy today?" → Add today filter
  * "My expenses this year" → Add year filter
- Examples that DO NOT need date filters:
  * "How much did I spend on PlayStation?" → No date filter
  * "Show all my transactions" → No date filter
  * "What stores do I shop at?" → No date filter
  * "My total spending" → No date filter

Use this date information for relative date queries like:
- "this month" = WHERE EXTRACT(YEAR FROM date_column) = {current_year} AND EXTRACT(MONTH FROM date_column) = {current_month}
- "this year" = WHERE EXTRACT(YEAR FROM date_column) = {current_year}
- "today" = WHERE DATE(date_column) = '{current_date_str}'
- "last month" = WHERE EXTRACT(YEAR FROM date_column) = {current_year} AND EXTRACT(MONTH FROM date_column) = {current_month - 1 if current_month > 1 else 12}
"""
        
        return f"""
        Context (Database Schema, Documentation, and Examples):
        {context}
        {user_info}
        {date_context}
        Question: {question}
        
        IMPORTANT: When generating SQL:
        - ALWAYS use table aliases and qualify ALL column names (e.g., t.merchant_name, not merchant_name)
        - Use = for exact matches (t.merchant_name = 'PlayStation Plus')
        - Use ILIKE with wildcards for pattern matching (t.merchant_name ILIKE '%playstation%')
        - DO NOT use LOWER() with LIKE for exact strings
        - DO NOT use LIKE without wildcards
        - Use the current date information above for relative date queries
        - ONLY add date/time filters if user explicitly mentions dates or time periods
        - If no date/time mentioned in question, do NOT add any date filters
        
        TABLE ALIASING EXAMPLES:
        ✅ CORRECT: SELECT t.amount FROM transactions t WHERE t.direction = 'debit'
        ❌ WRONG: SELECT amount FROM transactions WHERE direction = 'debit'
        ✅ CORRECT: SELECT t.merchant_name FROM transactions t WHERE t.merchant_category = 'Groceries'
        ❌ WRONG: SELECT merchant_name FROM transactions WHERE merchant_category = 'Groceries'
        
        Generate a SQL query to answer this question using the provided context.
        """
    
    def _extract_sql(self, response: str) -> str:
        """Extract SQL from the LLM response."""
        # Remove markdown code blocks if present
        if "```sql" in response:
            start = response.find("```sql") + 6
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        return response.strip()
    
    def extract_relevant_tables(self, question: str, context: str) -> List[str]:
        """Extract relevant table names from context based on the question."""
        try:
            prompt = f"""
            Given the following question and database context, identify which tables are used to answer the question.
            
            Question: {question}
            
            Context:
            {context}
            
            Analyze the question and context, then return ONLY a JSON list of table names that are needed to answer this question.
            Focus on the most relevant tables - don't include every table mentioned in the context.
            
            Example response format:
            ["users", "orders", "products"]
            
            Response:
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a database expert. Analyze questions and identify the minimum set of tables needed to answer them. Return only a JSON list of table names."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            logger.debug(f"LLM table extraction result: {result}")
            
            # Parse the JSON response
            import json
            try:
                tables = json.loads(result)
                if isinstance(tables, list):
                    # Clean table names (remove quotes, whitespace, etc.)
                    cleaned_tables = [str(table).strip().strip('"\'') for table in tables if table]
                    logger.info(f"Extracted relevant tables: {cleaned_tables}")
                    return cleaned_tables
                else:
                    logger.warning(f"LLM returned non-list result: {result}")
                    return []
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM table extraction result as JSON: {result}")
                # Fallback: try to extract table names from the response text
                return self._fallback_table_extraction(result, context)
                
        except Exception as e:
            logger.error(f"Failed to extract relevant tables: {str(e)}")
            # Fallback: extract all table names from context
            return self._fallback_table_extraction("", context)
    
    def _fallback_table_extraction(self, llm_result: str, context: str) -> List[str]:
        """Fallback method to extract table names from context."""
        import re
        tables = []
        
        # Look for table names in DDL statements
        ddl_pattern = r'CREATE TABLE\s+(\w+)'
        ddl_matches = re.findall(ddl_pattern, context, re.IGNORECASE)
        tables.extend(ddl_matches)
        
        # Look for table names in "Table: name" format
        table_pattern = r'-- Table:\s+(\w+)'
        table_matches = re.findall(table_pattern, context, re.IGNORECASE)
        tables.extend(table_matches)
        
        # Look for table names in FROM clauses
        from_pattern = r'FROM\s+(\w+)'
        from_matches = re.findall(from_pattern, context, re.IGNORECASE)
        tables.extend(from_matches)
        
        # Remove duplicates and return
        unique_tables = list(set(tables))
        logger.info(f"Fallback table extraction found: {unique_tables}")
        return unique_tables


class LLMFactory:
    """Factory class for creating LLM instances."""
    
    @staticmethod
    def create_llm(provider: str = "openai", **kwargs) -> BaseLLM:
        """Create an LLM instance based on the provider."""
        logger.debug(f"Creating LLM instance for provider: {provider}")
        
        if provider.lower() == "openai":
            return OpenAILLM(**kwargs)
        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            raise ConfigurationError(f"Unsupported LLM provider: {provider}")


# Default LLM instance
def get_default_llm(db_connector=None) -> BaseLLM:
    """Get the default LLM instance."""
    logger.debug("Getting default LLM instance")
    return LLMFactory.create_llm("openai", db_connector=db_connector) 