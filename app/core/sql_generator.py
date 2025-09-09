"""
SQL Generator that combines LLM and vector store for text-to-SQL generation.
"""

from typing import Dict, Any, Optional, List
import sqlparse
from app.utils.logger import logger
from app.core.llm_interface import BaseLLM, get_default_llm
from app.core.vector_store import VectorStore
from app.utils.exceptions import SQLGenerationError, VectorStoreError, LLMError


class SQLGenerator:
    """Main SQL generation engine combining LLM and RAG."""
    
    def __init__(self, 
                 llm: Optional[BaseLLM] = None,
                 vector_store: Optional[VectorStore] = None,
                 db_connector=None,
                 user_id: Optional[int] = None):
        """Initialize SQL generator with LLM, vector store, database connector, and user ID."""
        self.db_connector = db_connector
        self.llm = llm or get_default_llm(db_connector)
        self.vector_store = vector_store or VectorStore()
        self.user_id = user_id if user_id is not None else 0
        
        logger.info(f"SQL Generator initialized successfully for user ID: {self.user_id}")
    
    def generate_sql(self, 
                     question: str, 
                     max_context_length: int = 4000) -> Dict[str, Any]:
        """
        Generate SQL from natural language question.
        
        Args:
            question: Natural language question
            max_context_length: Maximum length of context to retrieve from vector store
            
        Returns:
            Dictionary containing SQL, explanation, and metadata
        """
        try:
            logger.info(f"Generating SQL for question: {question[:100]}...")
            
            # Get relevant context from vector store
            context = self.vector_store.get_context_for_question(
                question=question,
                collection_types=['query_pairs', 'ddl_definitions'],
                max_context_length=max_context_length
            )
            print(f"Context: {context}")
            
            if not context.strip():
                logger.debug("No relevant context found in vector store")
                context = "No specific schema or examples available."
                relevant_tables = []
            else:
                # Extract relevant tables from context using LLM
                relevant_tables = self.llm.extract_relevant_tables(question, context)
                print(f"Relevant tables identified: {relevant_tables}")
                
                # Get specific DDL and documentation for identified tables
                if relevant_tables:
                    table_specific_context = self.vector_store.get_table_specific_context(
                        table_names=relevant_tables,
                        max_context_length=max_context_length // 2  # Reserve half for general context
                    )
                    print(f"Table-specific context retrieved: {len(table_specific_context)} characters")
                    print(f"Table-specific context: {table_specific_context}")
                    # Combine general context with table-specific context
                    if table_specific_context.strip():
                        # Use table-specific context as primary, with some general context
                        filtered_general_context = self._filter_context_by_tables(context, relevant_tables)
                        
                        combined_context_parts = [table_specific_context]
                        if filtered_general_context.strip() and len(filtered_general_context) < max_context_length // 2:
                            combined_context_parts.append("\n-- ADDITIONAL CONTEXT --")
                            combined_context_parts.append(filtered_general_context)
                        
                        context = "\n".join(combined_context_parts)
                        
                        # Ensure we don't exceed max length
                        if len(context) > max_context_length:
                            context = context[:max_context_length] + "..."
                        
                        print(f"Combined context length: {len(context)} characters")
                    else:
                        # Fallback to filtered general context
                        filtered_context = self._filter_context_by_tables(context, relevant_tables)
                        if filtered_context.strip():
                            context = filtered_context
                            print(f"Using filtered general context: {len(context)} characters")
            print(f"Context: {context}")
            # Generate SQL using LLM
            raw_sql = self.llm.generate_sql(question, context, user_id=self.user_id)
            logger.info(f"Raw SQL generated: {raw_sql}")
            
            # Clean and validate SQL
            cleaned_sql = self._clean_sql(raw_sql)
            logger.info(f"Cleaned SQL: {cleaned_sql}")
            validation_result = self._validate_sql(cleaned_sql)
            logger.info(f"Validation result: {validation_result}")
            
            # Generate explanation
            explanation_result = self._generate_explanation(cleaned_sql, question,relevant_tables)
            logger.info(f"Explanation result: {explanation_result}")
            
            # Create corrected SQL with best match values
            corrected_sql = self._create_corrected_sql(cleaned_sql, explanation_result.get("best_matches", []))
            print(f"Corrected SQL: {corrected_sql}")
            
            result = {
                "sql": cleaned_sql,
                "corrected_sql": corrected_sql,
                "explanation": explanation_result.get("explanation", ""),
                "string_mappings": explanation_result.get("string_mappings", []),
                "best_matches": explanation_result.get("best_matches", []),
                "confidence": self._calculate_confidence(validation_result, explanation_result),
                "context_used": len(context) > 0,
                "context_length": len(context),
                "relevant_tables": relevant_tables if 'relevant_tables' in locals() else []
            }
            
            logger.info(f"SQL generation completed with confidence: {result['confidence']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate SQL: {str(e)}")
            raise SQLGenerationError(f"SQL generation failed: {str(e)}")
    
    def _clean_sql(self, raw_sql: str) -> str:
        """Clean and format the generated SQL."""
        try:
            logger.debug("Cleaning and formatting SQL...")
            
            # Remove common prefixes/suffixes
            sql = raw_sql.strip()
            
            # Remove markdown code blocks
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.startswith("```"):
                sql = sql[3:]
            if sql.endswith("```"):
                sql = sql[:-3]
            
            # Remove explanatory text before/after SQL
            lines = sql.split('\n')
            sql_lines = []
            in_sql = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line looks like SQL
                if any(keyword in line.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']):
                    in_sql = True
                
                if in_sql:
                    sql_lines.append(line)
                    
                # Stop if we hit explanatory text after SQL
                if in_sql and line.endswith(';'):
                    break
            
            if sql_lines:
                sql = '\n'.join(sql_lines)
            
            # Format using sqlparse
            formatted_sql = sqlparse.format(
                sql,
                reindent=True,
                keyword_case='upper',
                identifier_case='lower',
                strip_comments=False
            )
            
            logger.debug("SQL cleaned and formatted successfully")
            return formatted_sql.strip()
            
        except Exception as e:
            logger.error(f"SQL cleaning failed, returning raw SQL: {str(e)}")
            return raw_sql.strip()
    
    def _validate_sql(self, sql: str) -> Dict[str, Any]:
        """Validate the generated SQL syntax."""
        try:
            logger.debug("Validating SQL syntax...")
            
            # Parse SQL to check syntax
            parsed = sqlparse.parse(sql)
            
            if not parsed:
                return {"valid": False, "error": "Empty or invalid SQL"}
            
            # Basic validation checks
            sql_upper = sql.upper()
            
            # Check for dangerous operations (optional safety check)
            dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER']
            has_dangerous = any(keyword in sql_upper for keyword in dangerous_keywords)
            
            # Check for basic SQL structure
            has_select = 'SELECT' in sql_upper
            has_from = 'FROM' in sql_upper if has_select else True
            
            validation_result = {
                "valid": True,
                "parsed_successfully": True,
                "has_dangerous_operations": has_dangerous,
                "structure_valid": has_from if has_select else True,
                "warnings": []
            }
            
            if has_dangerous:
                validation_result["warnings"].append("Contains potentially dangerous operations")
            
            logger.debug("SQL validation completed")
            return validation_result
            
        except Exception as e:
            logger.error(f"SQL validation failed: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "parsed_successfully": False
            }
    
    def _generate_explanation(self, sql: str, question: str, relevant_tables: List[str]) -> Dict[str, Any]:
        """Generate explanation for the SQL query."""
        try:
            logger.debug("Generating SQL explanation...")
            explanation_result = self.llm.explain_sql(sql, relevant_tables, self.db_connector)
            logger.debug("SQL explanation generated successfully")
            return explanation_result
        except Exception as e:
            logger.error(f"Failed to generate explanation: {str(e)}")
            return {
                "explanation": f"This query answers the question: {question}",
                "string_mappings": []
            }
    
    def _calculate_confidence(self, validation_result: Dict[str, Any], explanation_result: Dict[str, Any]) -> float:
        """Calculate confidence score for the generated SQL."""
        try:
            confidence = 0.5  # Base confidence
            
            # Increase confidence if SQL is valid
            if validation_result.get("valid"):
                confidence += 0.3
            
            # Increase confidence if we have a clear explanation
            if explanation_result.get("explanation") and "This query answers the question" not in explanation_result["explanation"]:
                confidence += 0.2
            
            # Increase confidence if we have string mappings
            if explanation_result.get("string_mappings"):
                confidence += 0.1
            
            # Increase confidence if we have best matches
            if explanation_result.get("best_matches"):
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception:
            return 0.5
    
    def _create_corrected_sql(self, original_sql: str, best_matches: List[Dict[str, Any]]) -> str:
        """Create a corrected SQL query using the best matching values from the database."""
        try:
            if not best_matches:
                return original_sql
            
            corrected_sql = original_sql
            
            # Group matches by original literal to handle multiple selections for the same literal
            matches_by_literal = {}
            for match in best_matches:
                original_literal = match.get("original_literal", "")
                if original_literal:
                    if original_literal not in matches_by_literal:
                        matches_by_literal[original_literal] = []
                    matches_by_literal[original_literal].append(match)
            
            print(f"Matches grouped by literal: {matches_by_literal}")
            
            # Sort by original literal length (longest first) to avoid partial replacements
            sorted_literals = sorted(matches_by_literal.keys(), key=len, reverse=True)
            
            for original_literal in sorted_literals:
                matches = matches_by_literal[original_literal]
                
                if len(matches) == 1:
                    # Single match - use simple replacement
                    match = matches[0]
                    best_match_value = match.get("best_match_value", "")
                    raw_value = match.get("raw_value", "")
                    
                    if best_match_value and original_literal != best_match_value:
                        corrected_sql = self._replace_single_value(corrected_sql, raw_value, best_match_value, original_literal)
                
                elif len(matches) > 1:
                    # Multiple matches - convert to IN clause or OR conditions
                    match_values = [match.get("best_match_value", "") for match in matches if match.get("best_match_value")]
                    raw_value = matches[0].get("raw_value", "")
                    
                    if match_values:
                        corrected_sql = self._replace_with_multiple_values(corrected_sql, raw_value, match_values, original_literal)
            
            return corrected_sql
            
        except Exception as e:
            logger.error(f"Error creating corrected SQL: {str(e)}")
            return original_sql
    
    def _replace_single_value(self, sql: str, raw_value: str, best_match_value: str, original_literal: str) -> str:
        """Replace a single value in the SQL."""
        try:
            print(f"Replacing '{original_literal}' with '{best_match_value}' in SQL")
            
            # Handle both single and double quotes
            patterns_to_replace = [
                f"'{raw_value}'",
                f'"{raw_value}"',
            ]
            
            print(f"Patterns to replace: {patterns_to_replace}")
            
            for pattern in patterns_to_replace:
                if pattern in sql:
                    # Use the same quote style as the original
                    quote_char = pattern[0]
                    replacement = f"{quote_char}{best_match_value}{quote_char}"
                    sql = sql.replace(pattern, replacement)
                    logger.info(f"Replaced '{original_literal}' with '{best_match_value}' in SQL")
                    
                    # Fix SQL structure for exact matches
                    sql = self._fix_sql_structure_for_exact_matches(sql, best_match_value)
                    break
            
            return sql
            
        except Exception as e:
            logger.error(f"Error replacing single value: {str(e)}")
            return sql
    
    def _replace_with_multiple_values(self, sql: str, raw_value: str, match_values: List[str], original_literal: str) -> str:
        """Replace a single value with multiple values using IN clause or OR conditions."""
        try:
            print(f"Replacing '{original_literal}' with multiple values: {match_values}")
            
            # Handle both single and double quotes
            patterns_to_replace = [
                f"'{raw_value}'",
                f'"{raw_value}"',
            ]
            
            for pattern in patterns_to_replace:
                if pattern in sql:
                    quote_char = pattern[0]
                    
                    # Create IN clause with all selected values
                    in_values = [f"{quote_char}{value}{quote_char}" for value in match_values]
                    in_clause = f"({', '.join(in_values)})"
                    
                    # Find the column and operator context
                    import re
                    
                    # Look for patterns like "column = 'value'" or "column LIKE 'value'"
                    column_pattern = r'(\w+(?:\.\w+)?)\s*(=|!=|<>|LIKE|ILIKE|NOT LIKE|NOT ILIKE)\s*' + re.escape(pattern)
                    match = re.search(column_pattern, sql, re.IGNORECASE)
                    
                    if match:
                        column_ref = match.group(1)
                        operator = match.group(2).upper()
                        
                        # Replace the entire condition
                        old_condition = match.group(0)
                        
                        if operator in ['=', '!=', '<>', 'LIKE', 'ILIKE', 'NOT LIKE', 'NOT ILIKE']:
                            # Use IN clause for equality and LIKE operations
                            if operator.startswith('NOT'):
                                new_condition = f"{column_ref} NOT IN {in_clause}"
                            else:
                                new_condition = f"{column_ref} IN {in_clause}"
                        else:
                            # For other operators, use OR conditions
                            or_conditions = []
                            for value in match_values:
                                or_conditions.append(f"{column_ref} {operator} {quote_char}{value}{quote_char}")
                            new_condition = f"({' OR '.join(or_conditions)})"
                        
                        sql = sql.replace(old_condition, new_condition)
                        logger.info(f"Replaced '{original_literal}' with IN clause: {in_clause}")
                        
                    else:
                        # Fallback: simple replacement with first value
                        replacement = f"{quote_char}{match_values[0]}{quote_char}"
                        sql = sql.replace(pattern, replacement)
                        logger.warning(f"Could not find column context, used simple replacement with first value")
                    
                    break
            
            return sql
            
        except Exception as e:
            logger.error(f"Error replacing with multiple values: {str(e)}")
            # Fallback to simple replacement with first value
            return self._replace_single_value(sql, raw_value, match_values[0] if match_values else "", original_literal)
    
    def _fix_sql_structure_for_exact_matches(self, sql: str, exact_value: str) -> str:
        """Fix SQL structure to use exact matches instead of LIKE when we have exact values."""
        try:
            import re
            
            # Escape the exact value for regex safety
            escaped_value = re.escape(exact_value)
            
            # Pattern 1: LOWER(column) LIKE/ILIKE 'exact_value' → column = 'exact_value'
            pattern1 = rf"LOWER\s*\(\s*([^)]+)\s*\)\s+(I?LIKE)\s+(['\"]){escaped_value}\3"
            replacement1 = rf"\1 = \3{exact_value}\3"
            sql = re.sub(pattern1, replacement1, sql, flags=re.IGNORECASE)
            
            # Pattern 2: UPPER(column) LIKE/ILIKE 'EXACT_VALUE' → column = 'exact_value'
            pattern2 = rf"UPPER\s*\(\s*([^)]+)\s*\)\s+(I?LIKE)\s+(['\"]){escaped_value.upper()}\3"
            replacement2 = rf"\1 = \3{exact_value}\3"
            sql = re.sub(pattern2, replacement2, sql, flags=re.IGNORECASE)
            
            # Pattern 3: column LIKE/ILIKE 'exact_value' (only if no wildcards) → column = 'exact_value'
            if '%' not in exact_value and '_' not in exact_value:
                pattern3 = rf"([a-zA-Z_][a-zA-Z0-9_.]*)\s+(I?LIKE)\s+(['\"]){escaped_value}\3"
                replacement3 = rf"\1 = \3{exact_value}\3"
                sql = re.sub(pattern3, replacement3, sql, flags=re.IGNORECASE)
            
            return sql
            
        except Exception as e:
            logger.error(f"Error fixing SQL structure: {str(e)}")
            return sql
    
    def improve_with_feedback(self, 
                            question: str, 
                            generated_sql: str, 
                            correct_sql: str,
                            explanation: Optional[str] = None):
        """Improve the model by adding feedback as training data."""
        try:
            logger.info(f"Adding feedback for question: {question[:50]}...")
            # Add the correct SQL pair to training data
            self.vector_store.add_sql_pair(question, correct_sql, explanation)
            logger.info("Feedback added successfully to training data")
            
        except Exception as e:
            logger.error(f"Failed to add feedback: {str(e)}")
            raise SQLGenerationError(f"Failed to add feedback: {str(e)}")
    
    def get_similar_examples(self, question: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Get similar SQL examples for a question."""
        try:
            logger.debug(f"Getting similar examples for: {question[:50]}...")
            results = self.vector_store.search_similar(
                question, 
                n_results=n_results, 
                data_types=["sql_pair"]
            )
            logger.info(f"Found {len(results)} similar examples")
            return results
        except Exception as e:
            logger.error(f"Failed to get similar examples: {str(e)}")
            return []
    
    def _filter_context_by_tables(self, context: str, relevant_tables: List[str]) -> str:
        """Filter context to only include information about relevant tables."""
        if not relevant_tables:
            return context
        
        try:
            filtered_parts = []
            context_lines = context.split('\n')
            current_section = None
            include_section = False
            
            for line in context_lines:
                line_lower = line.lower()
                
                # Check if this is a section header
                if line.startswith('-- '):
                    current_section = line
                    # Check if this section is about a relevant table
                    include_section = any(table.lower() in line_lower for table in relevant_tables)
                    if include_section:
                        filtered_parts.append(line)
                elif include_section:
                    # Include this line if we're in a relevant section
                    filtered_parts.append(line)
                elif any(table.lower() in line_lower for table in relevant_tables):
                    # Include lines that mention relevant tables
                    if current_section and current_section not in filtered_parts:
                        filtered_parts.append(current_section)
                    filtered_parts.append(line)
                    include_section = True
            
            filtered_context = '\n'.join(filtered_parts)
            
            # If filtering resulted in very little content, return original context
            if len(filtered_context.strip()) < 100:
                logger.debug("Filtered context too short, returning original context")
                return context
            
            logger.debug(f"Filtered context from {len(context)} to {len(filtered_context)} characters")
            return filtered_context
            
        except Exception as e:
            logger.warning(f"Failed to filter context by tables: {str(e)}")
            return context 