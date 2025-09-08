#!/usr/bin/env python3
"""
Script to load all training data (DDL, SQL pairs) from separate JSON files into Vanna's vector database.
This script loads data into the collections without requiring a database connection.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app import VannaAI
from app.utils.logger import logger
from app.utils.exceptions import VannaException


def load_all_training_data_from_files(ddl_file_path: str, query_pairs_file_path: str):
    """
    Load all training data from separate JSON files into the vector database collections.
    
    Args:
        ddl_file_path: Path to the JSON file containing DDL statements
        query_pairs_file_path: Path to the JSON file containing SQL query pairs
    """
    try:
        print("🤖 Loading All Training Data into Vanna.AI Vector Database")
        print("=" * 65)
        
        # Initialize VannaAI (without database connection)
        print("Initializing Vanna.AI...")
        vanna = VannaAI()
        print("✅ Vanna.AI initialized! (No database connection required)")
        
        # Load DDL data
        print(f"📁 Loading DDL statements from: {ddl_file_path}")
        with open(ddl_file_path, 'r', encoding='utf-8') as f:
            ddl_data = json.load(f)
        
        # Load Query Pairs data
        print(f"📁 Loading SQL query pairs from: {query_pairs_file_path}")
        with open(query_pairs_file_path, 'r', encoding='utf-8') as f:
            query_data = json.load(f)
        
        print(f"✅ Training data loaded successfully!")
        
        # Statistics
        schema_count = len(ddl_data.get('ddl_statements', []))
        query_pairs_count = len(query_data.get('query_pairs', []))
        
        print(f"\n📊 Training Data Summary:")
        print(f"  - DDL Statements: {schema_count} tables")
        print(f"  - Text-to-SQL Pairs: {query_pairs_count} examples")
        
        # Load DDL statements into ddl_definitions collection
        print(f"\n🏗️  Loading enhanced DDL statements into ddl_definitions collection...")
        ddl_success = 0
        ddl_failed = 0
        
        for i, ddl_item in enumerate(ddl_data.get('ddl_statements', []), 1):
            try:
                table_name = ddl_item.get('table', '')
                ddl_statement = ddl_item.get('ddl', '')
                description = ddl_item.get('description', '')
                columns = ddl_item.get('columns', [])
                sample_data = ddl_item.get('sample_data', [])
                
                if not table_name or not ddl_statement:
                    print(f"  ⚠️  [{i}/{schema_count}] Skipping: Missing table name or DDL")
                    ddl_failed += 1
                    continue
                
                                # Pass the complete structured data directly to the vector store
                # instead of going through the train_ddl method which expects a string
                doc_id = vanna.vector_store.add_ddl(ddl_item, table_name)
                print(f"  ✅ [{i}/{schema_count}] Added enhanced DDL for table: {table_name}")
                ddl_success += 1
                
            except Exception as e:
                print(f"  ❌ [{i}/{schema_count}] Failed to add DDL for {ddl_item.get('table', 'unknown')}: {str(e)}")
                ddl_failed += 1
        
        print(f"✅ Enhanced DDL Loading Complete: {ddl_success} successful, {ddl_failed} failed")
        
        # Load query pairs into query_pairs collection
        print(f"\n🔍 Loading text-to-SQL pairs into query_pairs collection...")
        query_success = 0
        query_failed = 0
        
        for i, query_pair in enumerate(query_data.get('query_pairs', []), 1):
            try:
                question = query_pair.get('question', '')
                sql = query_pair.get('sql', '')
                explanation = query_pair.get('explanation', None)
                
                if not question or not sql:
                    print(f"  ⚠️  [{i}/{query_pairs_count}] Skipping: Missing question or SQL")
                    query_failed += 1
                    continue
                
                doc_id = vanna.train_sql_pair(question, sql, explanation)
                print(f"  ✅ [{i}/{query_pairs_count}] Added: {question[:60]}...")
                query_success += 1
                
            except Exception as e:
                print(f"  ❌ [{i}/{query_pairs_count}] Failed to add query pair: {str(e)}")
                query_failed += 1
        
        print(f"✅ Text-to-SQL Pairs Loading Complete: {query_success} successful, {query_failed} failed")
        
        # Get final statistics from vector store
        print(f"\n📈 Final Vector Database Statistics:")
        try:
            stats = vanna.get_training_stats()
            print(f"  - DDL Definitions Collection: {stats.get('ddl_definitions', 0)} documents")
            print(f"  - Query Pairs Collection: {stats.get('query_pairs', 0)} documents")
            print(f"  - Total Documents: {stats.get('total_documents', 0)}")
        except Exception as e:
            print(f"  ⚠️  Could not retrieve statistics: {str(e)}")
        
        # Summary
        total_success = ddl_success + query_success
        total_failed = ddl_failed + query_failed
        total_items = schema_count + query_pairs_count
        
        print(f"\n🎉 Training data loading completed!")
        print(f"  - Total Successful: {total_success}/{total_items}")
        print(f"  - Total Failed: {total_failed}/{total_items}")
        
        if query_pairs_count > 0:
            print(f"\nYou can now ask questions like:")
            for pair in query_data.get('query_pairs', [])[:3]:  # Show first 3 examples
                print(f"  - '{pair.get('question', '')}'")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ Error: JSON file not found: {str(e)}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format: {str(e)}")
        return False
    except VannaException as e:
        print(f"❌ Vanna Error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")
        return False


def main():
    """Main function."""
    # Configuration - edit these paths as needed
    ddl_file_path = "training_data.json"
    query_pairs_file_path = "sql_query_pairs.json"
    clear_existing = False  # Set to True to clear existing training data before loading
    
    # Check if JSON files exist
    missing_files = []
    if not Path(ddl_file_path).exists():
        missing_files.append(ddl_file_path)
    if not Path(query_pairs_file_path).exists():
        missing_files.append(query_pairs_file_path)
    
    if missing_files:
        print(f"❌ Error: Files not found: {', '.join(missing_files)}")
        print(f"Please make sure the JSON files exist with the following structures:")
        print(f"\n{ddl_file_path}:")
        print(f"""{{
  "ddl_statements": [
    {{
      "table": "table_name",
      "ddl": "CREATE TABLE statement...",
      "description": "Table description",
      "columns": [
        {{
          "name": "column_name",
          "description": "Column description"
        }}
      ],
      "sample_data": [...]
    }}
  ]
}}""")
        print(f"\n{query_pairs_file_path}:")
        print(f"""{{
  "query_pairs": [
    {{
      "question": "Natural language question",
      "sql": "SELECT * FROM table_name;",
      "explanation": "Optional explanation"
    }}
  ]
}}""")
        return 1
    
    try:
        # Clear existing training data if requested
        if clear_existing:
            print("🧹 Clearing existing training data...")
            vanna = VannaAI()
            print("⚠️  Warning: This will clear ALL existing training data")
            confirm = input("Are you sure? (y/N): ").lower().strip()
            if confirm == 'y':
                vanna.clear_training_data()
                print("✅ Existing training data cleared!")
            else:
                print("❌ Operation cancelled")
                return 1
        
        # Load all training data
        success = load_all_training_data_from_files(ddl_file_path, query_pairs_file_path)
        
        if success:
            print(f"\n✅ All done! Your vector database is now ready for queries.")
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 