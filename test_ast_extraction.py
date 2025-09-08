#!/usr/bin/env python3
"""
Test script to verify AST-based string literal extraction.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.llm_interface import OpenAILLM

def test_ast_extraction():
    """Test the AST-based string literal extraction."""
    
    # Sample SQL with qualified column references
    sql = """
    SELECT coalesce(merchant_category, '(uncategorized)') AS category,
       sum(amount_aed) AS total_aed
FROM transactions t
JOIN accounts a ON t.account_id = a.account_id
WHERE a.user_id = 1
  AND t.direction = 'debit'
  AND merchant_category = 'Groceries'
GROUP BY category
ORDER BY total_aed DESC;
    """
    
    # Table info mapping (alias -> table_name)
    table_info = {
        't': 'transactions',
        'a': 'accounts'
    }
    
    # Create LLM instance and test extraction
    llm = OpenAILLM()
    
    print("🧪 Testing AST-based string literal extraction")
    print("=" * 60)
    print(f"SQL: {sql.strip()}")
    print(f"Table Info: {table_info}")
    print()
    
    # Test the AST extraction directly
    ast_results = llm.find_string_literals_ast(sql, table_info)
    print("AST Results:")
    for i, result in enumerate(ast_results, 1):
        print(f"  {i}. {result}")
    print()
    
    # Test the full extraction pipeline
    mappings = llm._extract_string_literal_mappings(sql)
    print("Final Mappings:")
    for i, mapping in enumerate(mappings, 1):
        print(f"  {i}. {mapping}")
    print()
    
    # Verify expected results
    expected_debit = {
        'literal': 'debit',
        'table': 'transactions',
        'column': 'direction',
        'alias': 't',
        'operator': '=',
        'context': "t.direction = 'debit'"
    }
    
    expected_market = {
        'literal': 'market',  # Cleaned value without %
        'table': 'transactions',
        'column': 'merchant_name',
        'alias': 't',
        'operator': 'ILIKE',
        'raw_value': '%market%',  # Original with %
        'context': "t.merchant_name ILIKE 'market'"
    }
    
    print("✅ Expected Results:")
    print(f"  - Debit: {expected_debit}")
    print(f"  - Market: {expected_market}")
    
    # Check if we got the expected results
    found_debit = any(
        m.get('literal') == 'debit' and 
        m.get('table') == 'transactions' and
        m.get('alias') == 't' and
        m.get('column') == 'direction'
        for m in mappings
    )
    
    found_market = any(
        m.get('literal') == 'market' and  # Now checking for cleaned value
        m.get('table') == 'transactions' and
        m.get('alias') == 't' and
        m.get('column') == 'merchant_name'
        for m in mappings
    )
    
    print(f"\n🔍 Verification:")
    print(f"  - Found 'debit' extraction: {'✅' if found_debit else '❌'}")
    print(f"  - Found '%market%' extraction: {'✅' if found_market else '❌'}")
    
    if found_debit and found_market:
        print("\n🎉 All tests passed! AST extraction is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Check the extraction logic.")
        return False

if __name__ == "__main__":
    test_ast_extraction() 