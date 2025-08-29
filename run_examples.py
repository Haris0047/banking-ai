#!/usr/bin/env python3
"""
Simple script to run Vanna.AI examples and test the system.
"""

import sys
import os
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app import VannaAI
from app.utils.exceptions import VannaException


def main():
    """Main function to run examples."""
    print("🤖 Vanna.AI Custom Implementation - Example Runner")
    print("=" * 60)
    
    try:
        # Initialize Vanna
        print("Initializing Vanna.AI...")
        vanna = VannaAI()
        print("✅ VannaAI initialized successfully!")
        
        # Load sample data
        print("\n📚 Loading sample training data...")
        load_sample_data(vanna)
        
        # Test queries
        print("\n🔍 Testing SQL generation...")
        test_queries(vanna)
        
        # Show stats
        print("\n📊 Training data statistics:")
        stats = vanna.get_training_stats()
        print(json.dumps(stats, indent=2))
        
        print("\n✅ Example run completed successfully!")
        print("\nNext steps:")
        print("- Set your OPENAI_API_KEY environment variable")
        print("- Run the web interface: streamlit run frontend/streamlit_app.py")
        print("- Run the API server: python api/main.py")
        print("- Check the getting started notebook: examples/notebooks/getting_started.ipynb")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0


def load_sample_data(vanna: VannaAI):
    """Load sample training data."""
    
    # Sample DDL
    ddl_statements = [
        ("CREATE TABLE users (id SERIAL PRIMARY KEY, username VARCHAR(50), email VARCHAR(100), is_active BOOLEAN)", "users"),
        ("CREATE TABLE orders (id SERIAL PRIMARY KEY, user_id INTEGER, total_amount DECIMAL(10,2), status VARCHAR(20))", "orders"),
        ("CREATE TABLE products (id SERIAL PRIMARY KEY, name VARCHAR(200), price DECIMAL(10,2), stock_quantity INTEGER)", "products")
    ]
    
    for ddl, table_name in ddl_statements:
        vanna.train_ddl(ddl, table_name)
        print(f"  ✅ Added DDL for {table_name}")
    
    # Sample SQL pairs
    sql_pairs = [
        ("How many users are there?", "SELECT COUNT(*) FROM users"),
        ("How many active users?", "SELECT COUNT(*) FROM users WHERE is_active = TRUE"),
        ("What's the total revenue?", "SELECT SUM(total_amount) FROM orders WHERE status = 'completed'"),
        ("Show expensive products", "SELECT name, price FROM products ORDER BY price DESC LIMIT 10")
    ]
    
    for question, sql in sql_pairs:
        vanna.train_sql_pair(question, sql)
        print(f"  ✅ Added SQL pair: {question[:30]}...")
    
    # Sample documentation
    docs = [
        ("users", "Customer information and accounts", {"id": "User ID", "username": "Login name", "email": "Contact email"}),
        ("orders", "Customer orders and transactions", {"id": "Order ID", "user_id": "Customer ID", "total_amount": "Order value"}),
        ("products", "Product catalog and inventory", {"id": "Product ID", "name": "Product name", "price": "Selling price"})
    ]
    
    for table_name, description, columns in docs:
        vanna.train_documentation(table_name, description, columns)
        print(f"  ✅ Added documentation for {table_name}")


def test_queries(vanna: VannaAI):
    """Test SQL generation with sample queries."""
    
    test_questions = [
        "How many users do we have?",
        "What's the average order value?",
        "Show me the most expensive products",
        "How many pending orders are there?"
    ]
    
    for question in test_questions:
        print(f"\n  Question: {question}")
        try:
            result = vanna.ask(question)
            print(f"  SQL: {result['sql']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Valid: {'Yes' if result.get('validation', {}).get('valid', False) else 'No'}")
        except VannaException as e:
            print(f"  Error: {str(e)}")
        except Exception as e:
            print(f"  Unexpected error: {str(e)}")


if __name__ == "__main__":
    sys.exit(main()) 