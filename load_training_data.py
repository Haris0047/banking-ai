#!/usr/bin/env python3
"""
Script to load training data from JSON file into Vanna's three collections.
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


def load_training_data_from_json(json_file_path: str, db_path: str = "./data/uae_banking.db"):
    """
    Load training data from JSON file into the three Qdrant collections.
    
    Args:
        json_file_path: Path to the training data JSON file
        db_path: Path to the database file
    """
    try:
        print("🤖 Loading Training Data into Vanna.AI Collections")
        print("=" * 60)
        
        # Initialize VannaAI
        print("Initializing Vanna.AI...")
        vanna = VannaAI()
        
        # Connect to database (without auto-training to avoid duplicates)
        print(f"🔌 Connecting to database: {db_path}")
        vanna.connect_to_database("sqlite", {"path": db_path})
        print("✅ Database connected! (No auto-training - will load from JSON only)")
        
        # Load JSON data
        print(f"📁 Loading training data from: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        print(f"✅ Training data loaded successfully!")
        
        # Statistics
        schema_count = len(training_data.get('schema', []))
        query_pairs_count = len(training_data.get('query_pairs', []))
        docs_count = len(training_data.get('docs', []))
        
        print(f"\n📊 Training Data Summary:")
        print(f"  - Schema (DDL): {schema_count} tables")
        print(f"  - Query Pairs: {query_pairs_count} examples")
        print(f"  - Documentation: {docs_count} documents")
        
        # Load DDL statements into ddl_definitions collection
        print(f"\n🏗️  Loading DDL statements into ddl_definitions collection...")
        ddl_success = 0
        for schema_item in training_data.get('schema', []):
            try:
                table_name = schema_item['table']
                ddl_statement = schema_item['ddl']
                
                doc_id = vanna.train_ddl(ddl_statement, table_name)
                print(f"  ✅ Added DDL for table: {table_name}")
                ddl_success += 1
                
            except Exception as e:
                print(f"  ❌ Failed to add DDL for {schema_item.get('table', 'unknown')}: {str(e)}")
        
        print(f"✅ DDL Loading Complete: {ddl_success}/{schema_count} successful")
        
        # Load query pairs into query_pairs collection
        print(f"\n🔍 Loading query pairs into query_pairs collection...")
        query_success = 0
        for query_pair in training_data.get('query_pairs', []):
            try:
                question = query_pair['question']
                sql = query_pair['sql']
                
                doc_id = vanna.train_sql_pair(question, sql)
                print(f"  ✅ Added query pair: {question[:50]}...")
                query_success += 1
                
            except Exception as e:
                print(f"  ❌ Failed to add query pair: {str(e)}")
        
        print(f"✅ Query Pairs Loading Complete: {query_success}/{query_pairs_count} successful")
        
        # Load documentation into docs collection
        print(f"\n📚 Loading documentation into docs collection...")
        docs_success = 0
        for doc in training_data.get('docs', []):
            try:
                title = doc['title']
                content = doc['content']
                
                doc_id = vanna.train_documentation(
                    table_name=title,  # Using title as table_name for compatibility
                    description=content
                )
                print(f"  ✅ Added documentation: {title}")
                docs_success += 1
                
            except Exception as e:
                print(f"  ❌ Failed to add documentation for {doc.get('title', 'unknown')}: {str(e)}")
        
        print(f"✅ Documentation Loading Complete: {docs_success}/{docs_count} successful")
        
        # Get final statistics
        print(f"\n📈 Final Statistics:")
        stats = vanna.get_training_stats()
        print(f"  - Query Pairs Collection: {stats.get('query_pairs', 0)} documents")
        print(f"  - DDL Definitions Collection: {stats.get('ddl_definitions', 0)} documents")
        print(f"  - Documentation Collection: {stats.get('docs', 0)} documents")
        print(f"  - Total Documents: {stats.get('total_documents', 0)}")
        
        print(f"\n🎉 Training data loading completed successfully!")
        print(f"You can now ask questions like:")
        print(f"  - 'How many users are registered?'")
        print(f"  - 'List all merchant names in the Groceries category'")
        print(f"  - 'What is the total amount of all transactions?'")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ Error: Training data file not found: {json_file_path}")
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
    json_file_path = r"training_data.json"
    db_path = "./data/uae_banking.db"
    clear_first = True  # Set to True to clear existing data before loading
    
    # Check if JSON file exists
    if not Path(json_file_path).exists():
        print(f"❌ Error: File not found: {json_file_path}")
        return 1
    
    # Check if database file exists
    if not Path(db_path).exists():
        print(f"❌ Error: Database file not found: {db_path}")
        return 1
    
    try:
        # Clear existing data if requested
        if clear_first:
            print("🧹 Clearing existing training data...")
            vanna = VannaAI()
            vanna.clear_training_data()
            print("✅ Existing training data cleared!")
        
        # Load training data
        success = load_training_data_from_json(json_file_path, db_path)
        
        if success:
            print(f"\n✅ All done! You can now run: python run.py")
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