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
    def generate_sql(self, prompt: str, context: str, user_id: Optional[int] = None) -> str:
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
            # Find string literals using a simpler regex approach
            self._find_string_literals_with_context(sql, table_info, mappings)
            print(f"Table info: {table_info}")
            print(f"Mappings: {mappings}")
            
            return mappings
            
        except Exception as e:
            logger.error(f"Error extracting string literal mappings: {str(e)}")
            return []
    
    def _extract_table_info(self, sql: str) -> Dict[str, str]:
        """Extract table names and their aliases from parsed SQL."""
        table_info = {}  # unique_key -> table_name
        
        # Simple extraction - look for patterns like "FROM table_name alias" or "FROM table_name"
        sql_upper = sql.upper()
        from_matches = re.finditer(r'FROM\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?', sql_upper, re.IGNORECASE)
        print(f"from_matches: {from_matches}")
        
        table_counter = 0
        for match in from_matches:
            print(f"match: {match}")
            table_name = match.group(1).lower()
            print(f"table_name: {table_name}")
            alias = match.group(2).lower() if match.group(2) else table_name
            print(f"alias: sa{alias}, table_name: sa{table_name}")
            
            # Create unique key to avoid overwrites
            unique_key = f"{alias}_{table_counter}"
            table_info[unique_key] = table_name
            # Also store the alias -> table mapping for easy lookup
            table_info[alias] = table_name
            table_counter += 1
        
        # Also look for JOIN patterns
        join_matches = re.finditer(r'JOIN\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?', sql_upper, re.IGNORECASE)
        for match in join_matches:
            table_name = match.group(1).lower()
            alias = match.group(2).lower() if match.group(2) else table_name
            
            # Create unique key to avoid overwrites
            unique_key = f"{alias}_{table_counter}"
            table_info[unique_key] = table_name
            # Also store the alias -> table mapping for easy lookup
            table_info[alias] = table_name
            table_counter += 1
        
        print(f"Final table_info: {table_info}")
        return table_info
    
    def _find_string_literals_with_context(self, sql: str, table_info: Dict[str, str], mappings: List[Dict[str, str]]):
        """Find string literals and their context using regex patterns."""
        try:
            processed_literals = set()  # Track processed literals to avoid duplicates
            print(f"SQL: {sql}")
            print(f"table_info: s{table_info}")
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
            print("table_info", table_info)
            for match in matches3:
                column_name = match.group(1).lower()
                literal_value = match.group(2)
                
                # Skip if this looks like a function or keyword
                if column_name.upper() in ['AND', 'OR', 'WHERE', 'SELECT', 'FROM', 'JOIN', 'ON', 'ORDER', 'GROUP', 'HAVING', 'LIMIT', 'OFFSET']:
                    continue
                
                # Try to find the table for this column from table_info
                table_name = "unknown"
                for alias, tbl in table_info.items():
                    print(f"alias: {alias}, tbl: {tbl}")
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
    

    
    def _execute_fuzzy_queries_and_find_best_matches(self, string_mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute fuzzy search queries and find the best matching results."""
        best_matches = []
        
        try:
            for mapping in string_mappings:
                literal = mapping["literal"]
                table = mapping["table"]
                column = mapping["column"]
                fuzzy_variations = mapping["fuzzy_variations"]
                
                print(f"Processing literal: '{literal}' in {table}.{column}")
                
                # Execute each variation and collect unique merchant names
                all_merchant_names = set()
                variation_results = []
                
                for variation in fuzzy_variations:
                    try:
                        result = self.db_connector.execute_query(variation["sql"])
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
                
                # Find the best merchant name
                best_merchant_name = None
                best_score = 0.0
                print("all_merchant_names", all_merchant_names)
                for name in all_merchant_names:
                    print("name", name)
                    score = score_merchant_name(name)
                    print("score", score)
                    if score > best_score:
                        print("best_score", best_score)
                        best_score = score
                        best_merchant_name = name
                
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
                        "best_match_value": best_merchant_name,
                        "match_score": best_score,
                        "variation_used": best_variation["variation"]["type"] if best_variation else "unknown",
                        "total_variations_tested": len(fuzzy_variations),
                        "unique_names_found": len(all_merchant_names)
                    })
        
        except Exception as e:
            logger.error(f"Error in fuzzy matching: {str(e)}")
        
        return best_matches
    
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
        7. Use meaningful table aliases
        8. Return only the SQL query, no explanations
        9. Use the provided schema and context information
        10. For date operations, use PostgreSQL date functions like EXTRACT(), DATE(), NOW(), etc.
        
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
        - Use = for exact matches (merchant_name = 'PlayStation Plus')
        - Use ILIKE with wildcards for pattern matching (merchant_name ILIKE '%playstation%')
        - DO NOT use LOWER() with LIKE for exact strings
        - DO NOT use LIKE without wildcards
        - Use the current date information above for relative date queries
        - ONLY add date/time filters if user explicitly mentions dates or time periods
        - If no date/time mentioned in question, do NOT add any date filters
        
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