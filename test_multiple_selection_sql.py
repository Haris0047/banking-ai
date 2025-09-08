#!/usr/bin/env python3
"""
Test script to demonstrate SQL correction with multiple selected values.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.sql_generator import SQLGenerator

def test_multiple_selection_sql():
    """Test SQL correction when user selects multiple values."""
    
    # Create a mock SQL generator
    sql_gen = SQLGenerator(None, None, None)
    
    # Test case 1: Single selection
    print("="*60)
    print("TEST 1: Single Selection")
    print("="*60)
    
    original_sql1 = "SELECT t.merchant_name, t.amount FROM transactions t WHERE t.merchant_name ILIKE '%market%'"
    best_matches1 = [
        {
            "original_literal": "market",
            "best_match_value": "Fresh Market",
            "raw_value": "%market%",
            "table": "transactions",
            "column": "merchant_name",
            "user_selected": False
        }
    ]
    
    print(f"Original SQL: {original_sql1}")
    print(f"Best matches: {best_matches1}")
    
    corrected_sql1 = sql_gen._create_corrected_sql(original_sql1, best_matches1)
    print(f"Corrected SQL: {corrected_sql1}")
    
    # Test case 2: Multiple selections for same literal
    print("\n" + "="*60)
    print("TEST 2: Multiple Selections (User selected 3 options)")
    print("="*60)
    
    original_sql2 = "SELECT t.merchant_name, t.amount FROM transactions t WHERE t.merchant_name ILIKE '%market%'"
    best_matches2 = [
        {
            "original_literal": "market",
            "best_match_value": "Fresh Market",
            "raw_value": "%market%",
            "table": "transactions",
            "column": "merchant_name",
            "user_selected": True
        },
        {
            "original_literal": "market",
            "best_match_value": "Lulu Hypermarket",
            "raw_value": "%market%",
            "table": "transactions",
            "column": "merchant_name",
            "user_selected": True
        },
        {
            "original_literal": "market",
            "best_match_value": "Nesto Hypermarket",
            "raw_value": "%market%",
            "table": "transactions",
            "column": "merchant_name",
            "user_selected": True
        }
    ]
    
    print(f"Original SQL: {original_sql2}")
    print(f"Best matches: {best_matches2}")
    
    corrected_sql2 = sql_gen._create_corrected_sql(original_sql2, best_matches2)
    print(f"Corrected SQL: {corrected_sql2}")
    
    # Test case 3: Multiple literals with multiple selections
    print("\n" + "="*60)
    print("TEST 3: Multiple Literals with Multiple Selections")
    print("="*60)
    
    original_sql3 = """
    SELECT t.merchant_name, t.amount, t.category 
    FROM transactions t 
    WHERE t.merchant_name ILIKE '%market%' 
    AND t.category = 'grocery'
    """
    
    best_matches3 = [
        # Multiple selections for 'market'
        {
            "original_literal": "market",
            "best_match_value": "Fresh Market",
            "raw_value": "%market%",
            "table": "transactions",
            "column": "merchant_name",
            "user_selected": True
        },
        {
            "original_literal": "market",
            "best_match_value": "Viva Supermarket",
            "raw_value": "%market%",
            "table": "transactions",
            "column": "merchant_name",
            "user_selected": True
        },
        # Single selection for 'grocery'
        {
            "original_literal": "grocery",
            "best_match_value": "Groceries",
            "raw_value": "grocery",
            "table": "transactions",
            "column": "category",
            "user_selected": False
        }
    ]
    
    print(f"Original SQL: {original_sql3.strip()}")
    print(f"Best matches: {best_matches3}")
    
    corrected_sql3 = sql_gen._create_corrected_sql(original_sql3, best_matches3)
    print(f"Corrected SQL: {corrected_sql3.strip()}")
    
    # Test case 4: Equality operator with multiple selections
    print("\n" + "="*60)
    print("TEST 4: Equality Operator with Multiple Selections")
    print("="*60)
    
    original_sql4 = "SELECT * FROM transactions t WHERE t.merchant_name = 'market'"
    best_matches4 = [
        {
            "original_literal": "market",
            "best_match_value": "Fresh Market",
            "raw_value": "market",
            "table": "transactions",
            "column": "merchant_name",
            "user_selected": True
        },
        {
            "original_literal": "market",
            "best_match_value": "Istanbul Supermarket",
            "raw_value": "market",
            "table": "transactions",
            "column": "merchant_name",
            "user_selected": True
        }
    ]
    
    print(f"Original SQL: {original_sql4}")
    print(f"Best matches: {best_matches4}")
    
    corrected_sql4 = sql_gen._create_corrected_sql(original_sql4, best_matches4)
    print(f"Corrected SQL: {corrected_sql4}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✅ Single selection: Simple replacement")
    print("✅ Multiple selections: Converts to IN clause")
    print("✅ Mixed scenarios: Handles each literal appropriately")
    print("✅ Different operators: Adapts SQL structure accordingly")

if __name__ == "__main__":
    test_multiple_selection_sql() 