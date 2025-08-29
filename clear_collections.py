#!/usr/bin/env python3
"""
Simple script to clear all Qdrant collections.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app import VannaAI


def main():
    """Clear all collections."""
    try:
        print("🧹 Clearing all Vanna.AI collections...")
        
        vanna = VannaAI()
        vanna.clear_training_data()
        
        print("✅ All collections cleared successfully!")
        print("You can now run: python load_training_data.py")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 