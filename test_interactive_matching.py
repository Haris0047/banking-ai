#!/usr/bin/env python3
"""
Test script to demonstrate interactive ambiguous match resolution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.llm_interface import OpenAILLM

def test_interactive_matching():
    """Test the interactive ambiguous match resolution."""
    
    # Create a mock LLM instance with mock database connector
    class MockDBConnector:
        def execute_query(self, sql):
            """Mock database connector that returns sample merchant names."""
            # Simulate different merchant names with similar scores
            if "merchant_name" in sql.lower():
                return {
                    "data": [
                        ["Lulu Hypermarket"],
                        ["Safeer Hypermarket"], 
                        ["Viva Supermarket"],
                        ["Istanbul Supermarket"],
                        ["Nesto Hypermarket"],
                        ["Fresh Market"]
                    ],
                    "columns": ["merchant_name"]
                }
            return {"data": [], "columns": []}
    
    class MockOpenAILLM(OpenAILLM):
        def __init__(self):
            # Skip the parent __init__ to avoid API key requirements
            self.db_connector = MockDBConnector()
    
    llm = MockOpenAILLM()
    
    # Test SQL with a string literal that will match multiple merchants
    test_sql = """
    SELECT t.merchant_name, t.amount 
    FROM transactions t 
    WHERE t.merchant_name ILIKE '%market%'
    AND t.user_id = 123
    """
    
    # Test with relevant tables list
    relevant_tables = ['transactions']
    
    print("Testing interactive ambiguous match resolution...")
    print(f"SQL: {test_sql}")
    print(f"Relevant tables: {relevant_tables}")
    print("\nThis will simulate finding multiple similar merchant names")
    print("and prompt you to select which ones to use.")
    print("-" * 60)
    
    try:
        # Call the string literal extraction method
        mappings = llm._extract_string_literal_mappings(test_sql, relevant_tables)
        
        print(f"\nFound {len(mappings)} string literal mappings:")
        for i, mapping in enumerate(mappings, 1):
            print(f"\nMapping {i}:")
            print(f"  Literal: '{mapping.get('literal', 'N/A')}'")
            print(f"  Table: {mapping.get('table', 'N/A')}")
            print(f"  Column: {mapping.get('column', 'N/A')}")
            
            # Execute fuzzy matching to trigger interactive selection
            if mapping.get('fuzzy_variations'):
                print(f"  Testing fuzzy matching with {len(mapping['fuzzy_variations'])} variations...")
                
                # This will trigger the interactive selection
                best_matches = llm._execute_fuzzy_queries_and_find_best_matches([mapping])
                
                print(f"\nFinal results:")
                for match in best_matches:
                    print(f"  - Selected: '{match['best_match_value']}' (score: {match['match_score']:.1f})")
                    print(f"    User selected: {match.get('user_selected', False)}")
        
        print("\n" + "="*60)
        print("Interactive matching test completed!")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_interactive_matching() 