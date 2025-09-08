#!/usr/bin/env python3
"""
Test script to verify improved SQL generation prompting.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_improved_prompting():
    """Test that the improved prompting generates properly aliased SQL."""
    
    # Sample context (schema information)
    context = """
    Database Schema:
    
    Table: transactions
    - txn_id (INTEGER PRIMARY KEY)
    - merchant_name (TEXT)
    - merchant_category (TEXT) 
    - amount_aed (NUMERIC)
    - direction (TEXT) - 'debit' or 'credit'
    
    Table: accounts
    - account_id (INTEGER PRIMARY KEY)
    - user_id (INTEGER)
    - account_type (TEXT)
    
    Example Queries:
    SELECT t.amount_aed FROM transactions t WHERE t.direction = 'debit'
    """
    
    # Test questions that should generate properly aliased SQL
    test_questions = [
        """SELECT coalesce(merchant_category, '(uncategorized)') AS category,
       sum(amount_aed) AS total_aed
FROM transactions t
JOIN accounts a ON t.account_id = a.account_id
WHERE a.user_id = 1
  AND t.direction = 'debit'
  AND merchant_category = 'Groceries'
GROUP BY category
ORDER BY total_aed DESC;"""
    ]
    
    print("🧪 Testing Improved SQL Generation Prompting")
    print("=" * 60)
    print(f"Context: {context.strip()}")
    print()
    
    from app.core.llm_interface import OpenAILLM
    
    try:
        llm = OpenAILLM()
        
        for i, question in enumerate(test_questions, 1):
            print(f"{i}️⃣ Question: {question}")
            
            try:
                # Generate SQL using the improved prompting
                sql = llm.generate_sql(question, context, user_id=1)
                print(f"   Generated SQL: {sql}")
                
                # Check if SQL uses proper aliasing
                has_table_alias = any(
                    alias in sql.lower() 
                    for alias in ['t.', 'a.', 'u.', 'm.']  # Common aliases
                )
                
                # Check if it avoids unqualified column names for common columns
                unqualified_columns = [
                    'merchant_name', 'merchant_category', 'amount_aed', 
                    'direction', 'account_type'
                ]
                has_unqualified = any(
                    f" {col} " in sql or f"({col} " in sql or sql.startswith(col)
                    for col in unqualified_columns
                )
                
                status = "✅" if has_table_alias and not has_unqualified else "❌"
                print(f"   Aliasing Check: {status}")
                
                if not has_table_alias:
                    print(f"   ⚠️  Missing table aliases")
                if has_unqualified:
                    print(f"   ⚠️  Contains unqualified column names")
                    
            except Exception as e:
                print(f"   ❌ Error generating SQL: {str(e)}")
            
            print()
    
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {str(e)}")
        print("This test requires OpenAI API key to be configured")
        return False
    
    print("🎯 Test completed! Check the generated SQL for proper table aliasing.")
    return True

if __name__ == "__main__":
    test_improved_prompting() 