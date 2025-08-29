"""
LLM interface for different providers (OpenAI, Anthropic, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import re
import sqlparse
from sqlparse import sql, tokens
from openai import OpenAI
from app.utils.logger import logger
from config.settings import settings
from app.utils.exceptions import LLMError, ConfigurationError


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_sql(self, prompt: str, context: str) -> str:
        """Generate SQL from natural language prompt and context."""
        pass
    
    @abstractmethod
    def explain_sql(self, sql: str) -> Dict[str, Any]:
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
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, db_connector=None):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self.db_connector = db_connector
        
        if not self.api_key:
            logger.error("OpenAI API key is required but not provided")
            raise ConfigurationError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI LLM with model: {self.model}")
    
    def generate_sql(self, prompt: str, context: str) -> str:
        """Generate SQL using OpenAI GPT."""
        try:
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(prompt, context)
            
            logger.debug(f"Generating SQL for prompt: {prompt[:100]}...")
            
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
            logger.info(f"Successfully generated SQL for prompt: {prompt[:50]}...")
            return self._extract_sql(sql)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise LLMError(f"Failed to generate SQL: {str(e)}")
    
    def explain_sql(self, sql: str) -> Dict[str, Any]:
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
            string_mappings = self._extract_string_literal_mappings(sql)
            # Execute fuzzy queries and find best matches if database connector is available
            best_matches = []
            if self.db_connector and string_mappings:
                best_matches = self._execute_fuzzy_queries_and_find_best_matches(string_mappings)
            logger.info("Successfully generated SQL explanation with string mappings")
            return {
                "explanation": explanation,
                "string_mappings": string_mappings,
                "best_matches": best_matches
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error during explanation: {str(e)}")
            raise LLMError(f"Failed to explain SQL: {str(e)}")
    
    def _extract_string_literal_mappings(self, sql: str) -> List[Dict[str, str]]:
        """Extract string literals from SQL and map them to their table/column context."""
        try:
            mappings = []
            
            # Parse the SQL
            parsed = sqlparse.parse(sql)[0]
            
            # Extract table aliases and names
            table_info = self._extract_table_info(sql)
            print(f"Table info: {table_info}")
            # Find string literals using a simpler regex approach
            self._find_string_literals_with_context(sql, table_info, mappings)
            print(f"Mappings: {mappings}")
            
            return mappings
            
        except Exception as e:
            logger.error(f"Error extracting string literal mappings: {str(e)}")
            return []
    
    def _extract_table_info(self, sql: str) -> Dict[str, str]:
        """Extract table names and their aliases from parsed SQL."""
        table_info = {}  # alias -> table_name
        
        # Simple extraction - look for patterns like "FROM table_name alias" or "FROM table_name"
        sql_upper = sql.upper()
        from_matches = re.finditer(r'FROM\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?', sql_upper, re.IGNORECASE)
        
        for match in from_matches:
            table_name = match.group(1).lower()
            alias = match.group(2).lower() if match.group(2) else table_name
            table_info[alias] = table_name
        
        # Also look for JOIN patterns
        join_matches = re.finditer(r'JOIN\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?', sql_upper, re.IGNORECASE)
        for match in join_matches:
            table_name = match.group(1).lower()
            alias = match.group(2).lower() if match.group(2) else table_name
            table_info[alias] = table_name
        
        return table_info
    
    def _find_string_literals_with_context(self, sql: str, table_info: Dict[str, str], mappings: List[Dict[str, str]]):
        """Find string literals and their context using regex patterns."""
        try:
            processed_literals = set()  # Track processed literals to avoid duplicates
            print(f"SQL: {sql}")
            
            # Pattern 1: table.column = 'value' or table.column LIKE 'value'
            pattern1 = r'(\w+)\.(\w+)\s*(?:[=<>!]+|LIKE|ILIKE|IN)\s*[\'"]([^\'"]+)[\'"]'
            matches1 = re.finditer(pattern1, sql, re.IGNORECASE)
            for match in matches1:
                table_alias = match.group(1).lower()
                column_name = match.group(2).lower()
                literal_value = match.group(3)
                table_name = table_info.get(table_alias, table_alias)
                
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
                table_name = table_info.get(table_alias, table_alias)
                
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
                
                # Try to find the table for this column from table_info
                table_name = "unknown"
                for alias, tbl in table_info.items():
                    table_name = tbl
                    break  # Use first table as default
                
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
        """Generate fuzzy search variations for a string literal."""
        variations = []
        
        try:
            # Split the literal into words for individual word matching
            words = literal_value.split()
            
            # Base pattern for the full literal (search anywhere) - PostgreSQL
            literal_lower = literal_value.lower()
            
            # PostgreSQL ILIKE for case-insensitive search
            variations.append({
                "type": "full_match",
                "pattern": f"%{literal_lower}%",
                "sql": f"SELECT * FROM {table_name} WHERE {column_name} ILIKE '%{literal_lower}%'"
            })
            
            # Individual word patterns (search anywhere)
            for word in words:
                if len(word) > 2:  # Only for words longer than 2 characters
                    word_lower = word.lower()
                    
                    # PostgreSQL ILIKE for case-insensitive word search
                    variations.append({
                        "type": "word_match",
                        "word": word,
                        "pattern": f"%{word_lower}%",
                        "sql": f"SELECT * FROM {table_name} WHERE {column_name} ILIKE '%{word_lower}%'"
                    })
            
            # Add SOUNDEX matching for phonetic similarity (PostgreSQL fuzzystrmatch extension)
            variations.append({
                "type": "soundex_match",
                "pattern": f"SOUNDEX('{literal_value}')",
                "sql": f"SELECT * FROM {table_name} WHERE SOUNDEX({column_name}) = SOUNDEX('{literal_value}')"
            })
            
            # Individual word SOUNDEX matching
            for word in words:
                if len(word) > 2:
                    variations.append({
                        "type": "soundex_word_match",
                        "word": word,
                        "pattern": f"SOUNDEX('{word}')",
                        "sql": f"SELECT * FROM {table_name} WHERE SOUNDEX({column_name}) = SOUNDEX('{word}')"
                    })
            
            # Add Levenshtein distance matching for typo tolerance (PostgreSQL fuzzystrmatch extension)
            # Distance <= 1 for single character errors
            variations.append({
                "type": "levenshtein_match_1",
                "pattern": f"levenshtein <= 1",
                "sql": f"SELECT * FROM {table_name} WHERE levenshtein(LOWER({column_name}), LOWER('{literal_value}')) <= 1"
            })
            
            # Distance <= 2 for up to two character errors
            variations.append({
                "type": "levenshtein_match_2",
                "pattern": f"levenshtein <= 2",
                "sql": f"SELECT * FROM {table_name} WHERE levenshtein(LOWER({column_name}), LOWER('{literal_value}')) <= 2"
            })
            
            # Individual word Levenshtein matching
            for word in words:
                if len(word) > 3:  # Only for words longer than 3 characters
                    variations.append({
                        "type": "levenshtein_word_match",
                        "word": word,
                        "pattern": f"levenshtein <= 1",
                        "sql": f"SELECT * FROM {table_name} WHERE levenshtein(LOWER({column_name}), LOWER('{word}')) <= 1"
                    })
            
            # Add similarity matching (PostgreSQL pg_trgm extension)
            # Trigram similarity (good for partial matches)
            variations.append({
                "type": "similarity_match",
                "pattern": f"similarity >= 0.3",
                "sql": f"SELECT * FROM {table_name} WHERE similarity({column_name}, '{literal_value}') >= 0.3"
            })
            
            # Word similarity for individual words
            for word in words:
                if len(word) > 3:
                    variations.append({
                        "type": "word_similarity_match",
                        "word": word,
                        "pattern": f"word_similarity >= 0.4",
                        "sql": f"SELECT * FROM {table_name} WHERE word_similarity({column_name}, '{word}') >= 0.4"
                    })
            

            
        except Exception as e:
            logger.debug(f"Error generating fuzzy variations: {str(e)}")
        
        return variations
    

    
    def _execute_fuzzy_queries_and_find_best_matches(self, string_mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute fuzzy search queries and find the best matching results."""
        best_matches = []
        
        try:
            for mapping in string_mappings:
                literal = mapping.get("literal", "")
                table = mapping.get("table", "")
                column = mapping.get("column", "")
                fuzzy_variations = mapping.get("fuzzy_variations", [])
                
                if not fuzzy_variations or table == "unknown":
                    continue
                
                logger.debug(f"Testing fuzzy variations for literal: {literal}")
                
                # Execute each fuzzy variation and collect results
                variation_results = []
                for variation in fuzzy_variations:
                    try:
                        query_sql = variation.get("sql", "")
                        if query_sql:
                            result = self.db_connector.execute_query(query_sql)
                            print(f"Result: {result}")
                            
                            # Count the number of matching rows
                            row_count = result.get("row_count", 0)
                            if row_count > 0:
                                variation_results.append({
                                    "variation": variation,
                                    "row_count": row_count,
                                    "sample_data": result.get("data", [])[:5],  # First 5 rows as sample
                                    "columns": result.get("columns", [])
                                })
                                logger.debug(f"Query '{variation.get('pattern')}' found {row_count} matches")
                    
                    except Exception as e:
                        logger.debug(f"Error executing fuzzy query: {str(e)}")
                        continue
                
                # Find the best variation (most specific with results)
                if variation_results:
                    # Sort by specificity: word matches are more specific than full matches
                    # and prefer variations with reasonable result counts (not too many, not too few)
                    def score_variation(var_result):
                        variation = var_result["variation"]
                        row_count = var_result["row_count"]
                        
                        # Base score
                        score = 0
                        
                        # Prefer word matches over full matches (more specific)
                        if variation.get("type") == "word_match":
                            score += 10
                        
                        # Prefer reasonable result counts (1-100 results)
                        if 1 <= row_count <= 10:
                            score += 20
                        elif 11 <= row_count <= 50:
                            score += 15
                        elif 51 <= row_count <= 100:
                            score += 10
                        elif row_count > 100:
                            score += 5
                        
                        return score
                    
                    # Sort by score (highest first)
                    variation_results.sort(key=score_variation, reverse=True)
                    best_variation = variation_results[0]
                    
                    # Find the best matching value from the sample data
                    best_match_value = self._find_best_matching_value(
                        literal, 
                        best_variation["sample_data"], 
                        best_variation["columns"], 
                        column
                    )
                    
                    best_matches.append({
                        "original_literal": literal,
                        "table": table,
                        "column": column,
                        "best_variation": best_variation["variation"],
                        "best_match_value": best_match_value,
                        "match_count": best_variation["row_count"],
                        "sample_matches": best_variation["sample_data"],
                        "all_variations_tested": len(fuzzy_variations),
                        "successful_variations": len(variation_results)
                    })
                    
                    logger.info(f"Best match for '{literal}': {best_variation['variation']['pattern']} ({best_variation['row_count']} results)")
        
        except Exception as e:
            logger.error(f"Error executing fuzzy queries: {str(e)}")
        
        return best_matches
    
    def _find_best_matching_value(self, original_literal: str, sample_data: List, columns: List[str], target_column: str) -> Optional[str]:
        """Find the best matching value from the sample data."""
        try:
            if not sample_data or not columns:
                return None
            
            # Find the index of the target column
            try:
                column_index = columns.index(target_column)
            except ValueError:
                # If exact match not found, try case-insensitive search
                column_index = None
                for i, col in enumerate(columns):
                    if col.lower() == target_column.lower():
                        column_index = i
                        break
                
                if column_index is None:
                    return None
            
            # Extract all values from the target column
            column_values = []
            for row in sample_data:
                if len(row) > column_index and row[column_index] is not None:
                    column_values.append(str(row[column_index]))
            
            if not column_values:
                return None
            
            # Find the best match using simple string similarity
            original_lower = original_literal.lower()
            best_match = None
            best_score = -1
            
            for value in column_values:
                value_lower = value.lower()
                
                # Calculate similarity score
                score = self._calculate_string_similarity(original_lower, value_lower)
                
                if score > best_score:
                    best_score = score
                    best_match = value
            
            return best_match
            
        except Exception as e:
            logger.debug(f"Error finding best matching value: {str(e)}")
            return None
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings."""
        try:
            # Simple similarity based on common characters and length
            if str1 == str2:
                return 1.0
            
            if not str1 or not str2:
                return 0.0
            
            # Check if one string contains the other
            if str1 in str2 or str2 in str1:
                return 0.8
            
            # Count common characters
            common_chars = 0
            str1_chars = set(str1.lower())
            str2_chars = set(str2.lower())
            
            for char in str1_chars:
                if char in str2_chars:
                    common_chars += 1
            
            # Calculate similarity based on common characters and length difference
            max_len = max(len(str1), len(str2))
            char_similarity = common_chars / max_len if max_len > 0 else 0
            
            # Penalize for length difference
            len_diff = abs(len(str1) - len(str2))
            len_penalty = len_diff / max_len if max_len > 0 else 0
            
            similarity = char_similarity - (len_penalty * 0.3)
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.debug(f"Error calculating string similarity: {str(e)}")
            return 0.0
    
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
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for SQL generation."""
        return """
        You are an expert PostgreSQL developer. Generate accurate PostgreSQL SQL queries from natural language questions.
        
        PostgreSQL Guidelines:
        1. Use PostgreSQL syntax and functions only
        2. For case-insensitive comparisons: LOWER() function or ILIKE operator
        3. Use ILIKE for case-insensitive pattern matching
        4. Include appropriate JOINs when needed
        5. Use meaningful table aliases
        6. Return only the SQL query, no explanations
        7. Use the provided schema and context information
        
        Always wrap your SQL response in ```sql and ``` tags.
        """
    
    def _build_user_prompt(self, question: str, context: str) -> str:
        """Build the user prompt with question and context."""
        return f"""
        Context (Database Schema, Documentation, and Examples):
        {context}
        
        Question: {question}
        
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