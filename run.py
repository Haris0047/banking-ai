#!/usr/bin/env python3
"""
Simple interactive Vanna.AI bot - just ask questions and get SQL answers!
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app import VannaAI
from app.utils.logger import logger
from app.utils.exceptions import VannaException


class VannaBot:
    """Simple interactive Vanna.AI bot."""
    
    def __init__(self):
        """Initialize the bot."""
        print("🤖 Initializing Vanna.AI Bot...")
        try:
            self.vanna = VannaAI()
            self.connected_databases = []
            self.current_db = None
            print("✅ Vanna.AI Bot ready!")
        except Exception as e:
            print(f"❌ Failed to initialize: {str(e)}")
            sys.exit(1)
    
    def connect_database(self, auto_train: bool = False):
        """Connect to database using DATABASE_URL from environment."""
        try:
            from config.settings import settings
            
            if not settings.database_url:
                print("❌ No DATABASE_URL configured in environment")
                return False
            
            # Ensure it's a PostgreSQL URL
            if not (settings.database_url.startswith('postgresql://') or settings.database_url.startswith('postgres://')):
                print(f"❌ Only PostgreSQL databases are supported. URL must start with 'postgresql://' or 'postgres://'")
                return False
            
            db_type = "postgresql"
            
            connection_params = {"connection_url": settings.database_url}
            
            print(f"🔌 Connecting to {db_type} database...")
            self.vanna.connect_to_database(db_type, connection_params)
            
            # Extract database name from URL for display
            from urllib.parse import urlparse
            parsed = urlparse(settings.database_url)
            db_name = parsed.path.lstrip('/') or parsed.hostname or "database"
            
            if db_name not in [db['name'] for db in self.connected_databases]:
                self.connected_databases.append({
                    'name': db_name,
                    'type': db_type,
                    'url': settings.database_url,
                    'tables': 0
                })
            
            self.current_db = db_name
            print("✅ Database connected!")
            
            # Auto-train from database (optional)
            if auto_train:
                print("📚 Auto-training from database schema...")
                summary = self.vanna.train_from_database(db_type, connection_params)
                print(f"✅ Training completed: {summary['tables_processed']} tables processed")
                
                # Update table count
                for db in self.connected_databases:
                    if db['name'] == db_name:
                        db['tables'] = summary['tables_processed']
            else:
                print("⏭️ Skipping auto-training (training data will be loaded separately)")
            
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            return False
        return True
    
    def connect_all_databases(self, db_files):
        """Connect to all database files and train from them."""
        print(f"🔌 Connecting to all {len(db_files)} databases...")
        
        total_tables = 0
        successful_connections = 0
        
        for db_file in db_files:
            try:
                print(f"\n📁 Processing: {db_file.name}")
                
                # Connect and train from this database
                if self.connect_database(str(db_file), auto_train=True):
                    successful_connections += 1
                    # Get the table count from the last connected database
                    if self.connected_databases:
                        total_tables += self.connected_databases[-1]['tables']
                
            except Exception as e:
                print(f"❌ Failed to process {db_file.name}: {str(e)}")
        
        print(f"\n🎉 Multi-database setup complete!")
        print(f"✅ Connected to {successful_connections}/{len(db_files)} databases")
        print(f"📊 Total tables trained: {total_tables}")
        
        # Set current database to the last one for queries
        if self.connected_databases:
            self.current_db = self.connected_databases[-1]['path']
            print(f"🎯 Current active database: {Path(self.current_db).stem}")
        
        return successful_connections > 0
    
    def ask_question(self, question: str):
        """Ask a question and get SQL + results."""
        try:
            print(f"\n🔍 Processing: {question}")
            print("⏳ Generating SQL...")
            
            if self.connected_databases:
                # Generate and execute SQL
                result = self.vanna.ask(question, execute_sql=True, generate_summary=True)
                
                print(f"\n📝 Generated SQL:")
                print(f"```sql\n{result['sql']}\n```")
                
                print(f"\n💡 Explanation:")
                print(result['explanation'])
                
                print(f"\n📊 Confidence: {result['confidence']:.1%}")
                
                if result.get('executed') and result.get('query_results'):
                    query_results = result['query_results']
                    if query_results['data']:
                        print(f"\n📋 Results ({query_results['row_count']} rows):")
                        
                        # Display results in a simple table format
                        if query_results['columns']:
                            # Print headers
                            print("| " + " | ".join(query_results['columns']) + " |")
                            print("|" + "|".join(["-" * (len(col) + 2) for col in query_results['columns']]) + "|")
                            
                            # Print data (limit to first 10 rows)
                            for i, row in enumerate(query_results['data'][:10]):
                                print("| " + " | ".join([str(cell) for cell in row]) + " |")
                            
                            if len(query_results['data']) > 10:
                                print(f"... and {len(query_results['data']) - 10} more rows")
                    else:
                        print("📋 Query executed successfully but returned no data.")
                elif result.get('execution_error'):
                    print(f"⚠️ SQL generated but execution failed: {result['execution_error']}")
                
                # Display natural language summary if available
                if result.get('summary'):
                    print(f"\n🗣️ Summary:")
                    print(result['summary'])
            else:
                # No database connected - this will raise an error
                print("❌ No database connected. Please connect to a database first using /connect or /connect-all")
                return
            
        except VannaException as e:
            print(f"❌ Vanna Error: {str(e)}")
        except Exception as e:
            print(f"❌ Unexpected Error: {str(e)}")
    
    def show_help(self):
        """Show help information."""
        print("""
🤖 Vanna.AI Bot Commands:

Basic Usage:
  - Just type your question in natural language
  - Example: "How many users are there?"
  - Example: "Show me the top 5 products by price"

Special Commands:
  /connect         - Connect to database using DATABASE_URL
  /list-dbs        - List all connected databases
  /help           - Show this help
  /stats          - Show training data statistics
  /quit           - Exit the bot

Examples:
  /connect
  How many orders were placed last month?
  What are the most popular products?
  
Note: Set DATABASE_URL in your .env file:
  DATABASE_URL=postgresql://localhost:5432/uae_banking
        """)
    
    def show_stats(self):
        """Show training data statistics."""
        try:
            stats = self.vanna.get_training_stats()
            print(f"\n📊 Training Data Statistics:")
            print(f"Total Documents: {stats.get('total_documents', 0)}")
            
            if 'data_type_distribution' in stats:
                print("Data Types:")
                for data_type, count in stats['data_type_distribution'].items():
                    print(f"  - {data_type}: {count}")
            
            print(f"\n💾 Connected Databases: {len(self.connected_databases)}")
            for db in self.connected_databases:
                status = "🎯 (active)" if db['path'] == self.current_db else ""
                print(f"  - {db['name']}: {db['tables']} tables {status}")
            
        except Exception as e:
            print(f"❌ Failed to get stats: {str(e)}")
    
    def list_databases(self):
        """List all connected databases."""
        if not self.connected_databases:
            print("❌ No databases connected")
            return
        
        print(f"\n💾 Connected Databases ({len(self.connected_databases)}):")
        for i, db in enumerate(self.connected_databases, 1):
            status = "🎯 (active)" if db['path'] == self.current_db else ""
            print(f"  {i}. {db['name']}: {db['tables']} tables {status}")
    
    def switch_database(self, db_name: str):
        """Switch to a different connected database."""
        for db in self.connected_databases:
            if db['name'].lower() == db_name.lower():
                try:
                    print(f"🔄 Switching to database: {db['name']}")
                    self.vanna.connect_to_database("sqlite", {"path": db['path']})
                    self.current_db = db['path']
                    print(f"✅ Switched to {db['name']}")
                    return True
                except Exception as e:
                    print(f"❌ Failed to switch: {str(e)}")
                    return False
        
        print(f"❌ Database '{db_name}' not found in connected databases")
        print("Use /list-dbs to see available databases")
        return False
    
    def run(self):
        """Run the interactive bot."""
        print("""
🎉 Welcome to Vanna.AI Interactive Bot!

Type your questions in natural language and get SQL answers.
Type /help for commands or /quit to exit.
        """)
        
        # Try to connect to database using DATABASE_URL
        from config.settings import settings
        
        if settings.database_url:
            print(f"🔗 Found DATABASE_URL: {settings.database_url}")
            # choice = input("Connect to database? (y/N): ").strip().lower()
            # if choice == 'y':
            #     self.connect_database(auto_train=False)
            # else:
            #     print("Skipping database connection")
            self.connect_database(auto_train=False)
        else:
            print("❌ No DATABASE_URL configured in environment")
            print("💡 Set DATABASE_URL in your .env file (e.g., DATABASE_URL=postgresql://localhost:5432/uae_banking)")
        
        print("\n💬 Ready for questions! (Type /help for commands)")
        
        while True:
            try:
                user_input = input("\n🤖 Ask me: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.startswith('/'):
                    command_parts = user_input.split(' ', 1)
                    command = command_parts[0].lower()
                    
                    if command == '/quit' or command == '/exit':
                        print("👋 Goodbye!")
                        break
                    elif command == '/help':
                        self.show_help()
                    elif command == '/stats':
                        self.show_stats()
                    elif command == '/list-dbs':
                        self.list_databases()
                    elif command == '/connect':
                        self.connect_database(auto_train=True)
                    else:
                        print(f"❌ Unknown command: {command}")
                        print("Type /help for available commands")
                else:
                    # Regular question
                    self.ask_question(user_input)
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")


def main():
    """Main function."""
    try:
        bot = VannaBot()
        bot.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 