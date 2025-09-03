#!/usr/bin/env python3
"""
Direct PostgreSQL connection test using psycopg2 - similar to user's approach.
"""

import os
import sys
import psycopg2
import psycopg2.extras
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_direct_connection(connection_url: str):
    """Test direct PostgreSQL connection using psycopg2."""
    print(f"🔌 Testing direct connection to PostgreSQL...")
    print(f"URL: {connection_url[:50]}...")
    
    try:
        print("⏳ Connecting directly with psycopg2...")
        
        # Direct connection using the URL string (like your example)
        connection = psycopg2.connect(connection_url)
        connection.autocommit = True
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        print("✅ Direct connection successful!")
        
        # Test basic query
        print("⏳ Testing basic query...")
        cursor.execute("SELECT 1 as test_column, current_database() as db_name, current_user as user_name")
        result = cursor.fetchone()
        print(f"✅ Query successful! Result: {dict(result)}")
        
        # Get table count
        print("⏳ Getting table count...")
        cursor.execute("""
            SELECT COUNT(*) as table_count 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        table_result = cursor.fetchone()
        print(f"📊 Found {table_result['table_count']} tables in public schema")
        
        # Get first few table names
        if table_result['table_count'] > 0:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
                LIMIT 5
            """)
            tables = cursor.fetchall()
            print("📋 Sample tables:")
            for i, table in enumerate(tables, 1):
                print(f"  {i}. {table['table_name']}")
        
        # Close connection
        cursor.close()
        connection.close()
        print("✅ Connection closed successfully")
        
        return True
        
    except psycopg2.Error as e:
        print(f"❌ PostgreSQL Error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")
        return False


def test_with_connector():
    """Test using our PostgreSQL connector with direct URL approach."""
    print(f"\n" + "="*60)
    print("🧪 Testing with PostgreSQL Connector (Direct URL)")
    print("="*60)
    
    try:
        from app.database.connectors.postgres_connector import PostgreSQLConnector
        
        # Your connection URL
        connection_url = "postgresql://postgres:p0st..r3s%231.%3C%3CS%21%3C.3l@core-data.ckzwkyqo4bz1.us-east-1.rds.amazonaws.com:5432/postgres"
        
        print("⏳ Creating PostgreSQL connector...")
        connector = PostgreSQLConnector({"connection_url": connection_url})
        
        print("⏳ Connecting...")
        connector.connect()
        print("✅ Connector connection successful!")
        
        # Test query
        print("⏳ Testing query through connector...")
        result = connector.execute_query("SELECT 1 as test_column, current_database() as db_name")
        print(f"✅ Connector query successful! Result: {result}")
        
        # Get tables
        print("⏳ Getting table names...")
        tables = connector.get_table_names()
        print(f"📋 Found {len(tables)} tables:")
        for i, table in enumerate(tables[:5], 1):
            print(f"  {i}. {table}")
        if len(tables) > 5:
            print(f"  ... and {len(tables) - 5} more tables")
        
        # Close
        connector.close()
        print("✅ Connector closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Connector test failed: {str(e)}")
        return False


def main():
    """Main function."""
    print("🧪 Direct PostgreSQL Connection Test")
    print("="*60)
    
    # Your connection URL (from the test file you modified)
    connection_url = "postgresql://postgres:p0st..r3s%231.%3C%3CS%21%3C.3l@core-data.ckzwkyqo4bz1.us-east-1.rds.amazonaws.com:5432/postgres"
    
    # Test 1: Direct psycopg2 connection (like your example)
    print("🧪 Test 1: Direct psycopg2 connection")
    print("-" * 40)
    success1 = test_direct_connection(connection_url)
    
    # Test 2: Using our connector with direct URL
    success2 = test_with_connector()
    
    print(f"\n" + "="*60)
    print("📊 Test Results:")
    print(f"  Direct psycopg2: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"  Connector:       {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 or success2:
        print("\n🎉 At least one connection method worked!")
        return 0
    else:
        print("\n💡 Both methods failed. Check:")
        print("  1. Database credentials are correct")
        print("  2. Network connectivity (VPN, firewall)")
        print("  3. Database server allows your IP")
        print("  4. SSL/TLS requirements")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 