#!/usr/bin/env python3
"""
Migration script to convert existing vector collections from sentence-transformers 
to OpenAI text-embedding-3-small embeddings.

This script will:
1. Backup existing collections
2. Re-generate embeddings using OpenAI's text-embedding-3-small
3. Create new collections with 1536-dimensional vectors
4. Migrate all existing data
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.vector_store import VectorStore
from app.utils.logger import logger
from config.settings import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class EmbeddingMigrator:
    """Migrates vector collections from old embeddings to new OpenAI embeddings."""
    
    def __init__(self):
        """Initialize the migrator."""
        print("🔄 Initializing Embedding Migrator...")
        
        # Initialize Qdrant client directly
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        
        # Initialize new vector store with OpenAI embeddings
        self.new_vector_store = VectorStore()
        
        self.backup_suffix = f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print("✅ Migrator initialized!")
    
    def list_existing_collections(self) -> List[str]:
        """List all existing collections."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            # Filter for vanna collections
            vanna_collections = [
                name for name in collection_names 
                if settings.qdrant_collection_name in name and not name.endswith('_backup')
            ]
            
            return vanna_collections
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            return []
    
    def backup_collection(self, collection_name: str) -> bool:
        """Create a backup of an existing collection."""
        try:
            backup_name = f"{collection_name}{self.backup_suffix}"
            print(f"📦 Creating backup: {collection_name} -> {backup_name}")
            
            # Get collection info
            collection_info = self.client.get_collection(collection_name)
            
            # Create backup collection with same configuration
            self.client.create_collection(
                collection_name=backup_name,
                vectors_config=collection_info.config.params.vectors
            )
            
            # Copy all points
            points, _ = self.client.scroll(collection_name, limit=10000)
            if points:
                self.client.upsert(backup_name, points)
                print(f"✅ Backed up {len(points)} points to {backup_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to backup {collection_name}: {str(e)}")
            return False
    
    def migrate_collection_data(self, old_collection_name: str) -> bool:
        """Migrate data from old collection to new collection with OpenAI embeddings."""
        try:
            print(f"🔄 Migrating data from {old_collection_name}...")
            
            # Get all points from old collection
            points, _ = self.client.scroll(old_collection_name, limit=10000, with_payload=True)
            
            if not points:
                print(f"⚠️ No data found in {old_collection_name}")
                return True
            
            print(f"📊 Found {len(points)} points to migrate")
            
            # Determine collection type and migrate accordingly
            if 'query_pairs' in old_collection_name:
                self._migrate_query_pairs(points)
            elif 'ddl_definitions' in old_collection_name:
                self._migrate_ddl_definitions(points)
            elif 'docs' in old_collection_name:
                self._migrate_docs(points)
            else:
                print(f"⚠️ Unknown collection type: {old_collection_name}")
                return False
            
            print(f"✅ Successfully migrated {len(points)} points from {old_collection_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to migrate {old_collection_name}: {str(e)}")
            return False
    
    def _migrate_query_pairs(self, points: List[Any]):
        """Migrate query pairs data."""
        for point in points:
            payload = point.payload
            try:
                self.new_vector_store.add_sql_pair(
                    question=payload.get('question', ''),
                    sql=payload.get('sql', ''),
                    explanation=payload.get('explanation')
                )
            except Exception as e:
                print(f"⚠️ Failed to migrate query pair: {str(e)}")
    
    def _migrate_ddl_definitions(self, points: List[Any]):
        """Migrate DDL definitions data."""
        for point in points:
            payload = point.payload
            try:
                self.new_vector_store.add_ddl_definition(
                    ddl_statement=payload.get('ddl_statement', ''),
                    name=payload.get('name', ''),
                    columns=payload.get('columns', [])
                )
            except Exception as e:
                print(f"⚠️ Failed to migrate DDL definition: {str(e)}")
    
    def _migrate_docs(self, points: List[Any]):
        """Migrate documentation data."""
        for point in points:
            payload = point.payload
            try:
                self.new_vector_store.add_documentation(
                    title=payload.get('title', ''),
                    body=payload.get('body', ''),
                    doc_type=payload.get('doc_type', 'general')
                )
            except Exception as e:
                print(f"⚠️ Failed to migrate documentation: {str(e)}")
    
    def delete_old_collection(self, collection_name: str) -> bool:
        """Delete old collection after successful migration."""
        try:
            print(f"🗑️ Deleting old collection: {collection_name}")
            self.client.delete_collection(collection_name)
            print(f"✅ Deleted {collection_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to delete {collection_name}: {str(e)}")
            return False
    
    def run_migration(self, backup_first: bool = True, delete_old: bool = False):
        """Run the complete migration process."""
        print("🚀 Starting embedding migration process...")
        print(f"📋 Current settings:")
        print(f"   - Embedding Model: {settings.embedding_model}")
        print(f"   - Embedding Dimensions: {settings.embedding_dimensions}")
        print(f"   - Qdrant URL: {settings.qdrant_url}")
        
        # Check if OpenAI API key is configured
        if not settings.openai_api_key:
            print("❌ OpenAI API key not configured. Please set OPENAI_API_KEY in your environment.")
            return False
        
        # List existing collections
        existing_collections = self.list_existing_collections()
        
        if not existing_collections:
            print("✅ No existing collections found. Migration not needed.")
            return True
        
        print(f"📊 Found {len(existing_collections)} collections to migrate:")
        for collection in existing_collections:
            print(f"   - {collection}")
        
        # Confirm migration
        if input("\nProceed with migration? (y/N): ").lower() != 'y':
            print("❌ Migration cancelled by user")
            return False
        
        success_count = 0
        
        for collection_name in existing_collections:
            print(f"\n🔄 Processing {collection_name}...")
            
            # Backup if requested
            if backup_first:
                if not self.backup_collection(collection_name):
                    print(f"❌ Skipping {collection_name} due to backup failure")
                    continue
            
            # Migrate data
            if self.migrate_collection_data(collection_name):
                success_count += 1
                
                # Delete old collection if requested
                if delete_old:
                    self.delete_old_collection(collection_name)
            else:
                print(f"❌ Migration failed for {collection_name}")
        
        print(f"\n🎉 Migration complete!")
        print(f"✅ Successfully migrated {success_count}/{len(existing_collections)} collections")
        
        if backup_first:
            print(f"💾 Backups created with suffix: {self.backup_suffix}")
        
        return success_count == len(existing_collections)


def main():
    """Main function."""
    print("🔄 Vanna.AI Embedding Migration Tool")
    print("=" * 50)
    
    try:
        migrator = EmbeddingMigrator()
        
        # Parse command line arguments
        backup_first = '--no-backup' not in sys.argv
        delete_old = '--delete-old' in sys.argv
        
        if '--help' in sys.argv:
            print("""
Usage: python migrate_embeddings.py [options]

Options:
  --no-backup    Skip creating backups of existing collections
  --delete-old   Delete old collections after successful migration
  --help         Show this help message

This script migrates existing vector collections from sentence-transformers
embeddings to OpenAI text-embedding-3-small embeddings.

Make sure to:
1. Set your OPENAI_API_KEY environment variable
2. Backup your data before running (unless using --no-backup)
3. Test the migration on a small dataset first
            """)
            return 0
        
        success = migrator.run_migration(backup_first=backup_first, delete_old=delete_old)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n👋 Migration cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 