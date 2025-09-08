#!/usr/bin/env python3
"""
Test script to verify the modified string literal extraction functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.llm_interface import OpenAILLM

def test_string_literal_extraction():
    """Test the modified _extract_string_literal_mappings method."""
    
    # Create a mock LLM instance (we don't need actual API calls for this test)
    class MockOpenAILLM(OpenAILLM):
        def __init__(self):
            # Skip the parent __init__ to avoid API key requirements
            self.db_connector = None
    
    llm = MockOpenAILLM()
    
    # Test SQL with string literals
    test_sql = """
    SELECT t.merchant_name, t.amount 
    FROM transactions t 
    WHERE t.merchant_name = 'PlayStation Plus' 
    AND t.category ILIKE '%gaming%'
    AND t.user_id = 123
    """
    
    # Test with relevant tables list
    relevant_tables = ['transactions', 'accounts', 'users']
    
    print("Testing string literal extraction with relevant_tables...")
    print(f"SQL: {test_sql}")
    print(f"Relevant tables: {relevant_tables}")
    print("Note: Now using relevant_tables directly without extracting table_info from SQL!")
    print("-" * 50)
    
    try:
        # Call the modified method
        mappings = llm._extract_string_literal_mappings(test_sql, relevant_tables)
        
        print(f"Found {len(mappings)} string literal mappings:")
        for i, mapping in enumerate(mappings, 1):
            print(f"\nMapping {i}:")
            print(f"  Literal: '{mapping.get('literal', 'N/A')}'")
            print(f"  Table: {mapping.get('table', 'N/A')}")
            print(f"  Column: {mapping.get('column', 'N/A')}")
            print(f"  Operator: {mapping.get('operator', 'N/A')}")
            print(f"  Context: {mapping.get('context', 'N/A')}")
            
            # Show fuzzy variations count
            fuzzy_vars = mapping.get('fuzzy_variations', [])
            print(f"  Fuzzy variations: {len(fuzzy_vars)} generated")
        
        print("\n" + "="*50)
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_string_literal_extraction() 