#!/usr/bin/env python3
"""
AWS RDS PostgreSQL connection test with SSL support.
"""

import sys
import os
from pathlib import Path
import psycopg2
from urllib.parse import urlparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_direct_psycopg2(connection_url: str, ssl_modes: list = None):
    """Test connection directly with psycopg2 using different SSL modes."""
    if ssl_modes is None:
        ssl_modes = ['require', 'prefer', 'allow', 'disable']
    
    parsed = urlparse(connection_url)
    
    base_params = {
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username,
        'password': parsed.password
    }
    
    print(f"🔍 Testing connection to: {parsed.hostname}")
    print(f"📊 Connection details:")
    print(f"  - Host: {base_params['host']}")
    print(f"  - Port: {base_params['port']}")
    print(f"  - Database: {base_params['database']}")
    print(f"  - User: {base_params['user']}")
    print(f"  - Password: {'*' * len(base_params['password']) if base_params['password'] else 'None'}")
    
    for ssl_mode in ssl_modes:
        print(f"\n🔐 Trying SSL mode: {ssl_mode}")
        
        conn_params = base_params.copy()
        conn_params['sslmode'] = ssl_mode
        
        try:
            conn = psycopg2.connect(**conn_params)
            print(f"✅ Connection successful with sslmode={ssl_mode}!")
            
            # Test a simple query
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            print(f"📋 PostgreSQL version: {version[:50]}...")
            
            cursor.close()
            conn.close()
            return ssl_mode
            
        except psycopg2.OperationalError as e:
            error_msg = str(e)
            print(f"❌ Failed with sslmode={ssl_mode}: {error_msg}")
            
            # Provide specific guidance based on error
            if "password authentication failed" in error_msg:
                print("💡 Issue: Wrong username or password")
            elif "no pg_hba.conf entry" in error_msg:
                print("💡 Issue: Your IP address is not allowed to connect")
                print("   - Check AWS RDS security groups")
                print("   - Verify inbound rules allow your IP on port 5432")
            elif "SSL connection has been closed unexpectedly" in error_msg:
                print("💡 Issue: SSL connection problem")
            elif "server closed the connection unexpectedly" in error_msg:
                print("💡 Issue: Network connectivity problem")
        except Exception as e:
            print(f"❌ Unexpected error with sslmode={ssl_mode}: {str(e)}")
    
    return None

def test_with_our_connector(connection_url: str):
    """Test using our PostgreSQL connector."""
    print(f"\n🧪 Testing with our PostgreSQL connector...")
    
    try:
        from app.database.connectors.postgres_connector import PostgreSQLConnector
        
        connection_params = {"connection_url": connection_url}
        connector = PostgreSQLConnector(connection_params)
        
        print("⏳ Attempting to connect...")
        connector.connect()
        print("✅ Our connector works!")
        
        # Test basic query
        result = connector.execute_query("SELECT 1 as test")
        print(f"📋 Test query result: {result}")
        
        connector.close()
        return True
        
    except Exception as e:
        print(f"❌ Our connector failed: {str(e)}")
        return False

def main():
    """Main function."""
    print("🔗 AWS RDS PostgreSQL Connection Tester")
    print("=" * 50)
    
    # Your connection URL (with URL encoding issues fixed)
    connection_url = "postgresql://postgres:p0st..r3s%231.%3C%3CS%21%3C.3l@core-data.ckzwkyqo4bz1.us-east-1.rds.amazonaws.com:5432/postgres"
    
    print(f"🎯 Testing connection URL...")
    
    # Test direct psycopg2 connection
    working_ssl_mode = test_direct_psycopg2(connection_url)
    
    if working_ssl_mode:
        print(f"\n🎉 Found working SSL mode: {working_ssl_mode}")
        
        # Test with our connector
        test_with_our_connector(connection_url)
    else:
        print(f"\n💔 No SSL mode worked. Common issues:")
        print("1. 🔐 Wrong credentials - Check username/password")
        print("2. 🌐 Network access - Check AWS security groups")
        print("3. 🏠 IP whitelist - Your IP might not be allowed")
        print("4. 🔒 SSL requirements - Server might require specific SSL config")
        
        print(f"\n🛠️ Troubleshooting steps:")
        print("1. Verify credentials in AWS RDS console")
        print("2. Check security group inbound rules")
        print("3. Try connecting from AWS EC2 instance in same VPC")
        print("4. Use AWS RDS proxy if available")

if __name__ == "__main__":
    main() 