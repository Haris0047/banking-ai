#!/usr/bin/env python3
"""
Simple database connection test script.
Test PostgreSQL connection directly using a connection URL.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.database.connectors.postgres_connector import PostgreSQLConnector
from app.utils.logger import logger


def test_connection(connection_url: str):
    """Test database connection with the provided URL."""
    print(f"🔌 Testing connection to: {connection_url}")
    
    try:
        # Create connector with connection URL
        connection_params = {"connection_url": connection_url}
        connector = PostgreSQLConnector(connection_params)
        
        print("⏳ Attempting to connect...")
        connector.connect()
        print("✅ Connection successful!")
        
        # Test basic query
        print("⏳ Testing basic query...")
        result = connector.execute_query("SELECT 1 as test_column")
        print(f"✅ Query successful! Result: {result}")
        
        # Get database info
        print("⏳ Getting database info...")
        db_info = connector.get_database_info()
        print(f"📊 Database Info:")
        for key, value in db_info.items():
            print(f"  - {key}: {value}")
        
        # Get table names
        print("⏳ Getting table names...")
        tables = connector.get_table_names()
        print(f"📋 Found {len(tables)} tables:")
        for i, table in enumerate(tables[:10], 1):  # Show first 10 tables
            print(f"  {i}. {table}")
        if len(tables) > 10:
            print(f"  ... and {len(tables) - 10} more tables")
        
        # Close connection
        connector.close()
        print("✅ Connection closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False


def main():
    """Main function."""
    print("🧪 Database Connection Test Tool")
    print("=" * 50)
    
    # Get connection URL from user
    connection_url = "postgresql://postgres:p0st..r3s%231.%3C%3CS%21%3C.3l@core-data.ckzwkyqo4bz1.us-east-1.rds.amazonaws.com:5432/postgres"
    
    if not connection_url:
        print("❌ No connection URL provided")
        return 1
    
    # Validate URL format
    if not (connection_url.startswith('postgresql://') or connection_url.startswith('postgres://')):
        print("❌ Invalid URL format. Must start with 'postgresql://' or 'postgres://'")
        return 1
    
    # Test connection
    success = test_connection(connection_url)
    
    if success:
        print("\n🎉 All tests passed! Your database connection is working.")
        return 0
    else:
        print("\n💡 Connection failed. Please check:")
        print("  1. Database server is running and accessible")
        print("  2. Username and password are correct")
        print("  3. Database name exists")
        print("  4. Network connectivity (firewall, security groups)")
        print("  5. PostgreSQL pg_hba.conf allows your IP address")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 