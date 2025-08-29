#!/usr/bin/env python3
"""
Script to clean JSONL file and keep only 'question' and 'sql' fields.
"""

import json
import sys
from pathlib import Path


def clean_jsonl_file(input_file: str, output_file: str = None):
    """Clean JSONL file to keep only question and sql fields."""
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_file}")
        return False
    
    # Default output file name
    if output_file is None:
        output_file = input_path.stem + "_cleaned" + input_path.suffix
    
    output_path = Path(output_file)
    
    print(f"🧹 Cleaning JSONL file...")
    print(f"📁 Input:  {input_path}")
    print(f"📁 Output: {output_path}")
    
    cleaned_count = 0
    total_count = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON
                    data = json.loads(line)
                    total_count += 1
                    
                    # Extract only question and sql fields
                    cleaned_data = {}
                    
                    if 'question' in data:
                        cleaned_data['question'] = data['question']
                    else:
                        print(f"⚠️ Line {line_num}: Missing 'question' field")
                        continue
                    
                    if 'sql' in data:
                        # Clean up SQL - preserve structure but remove extra whitespace
                        sql = data['sql'].strip()
                        # Remove leading/trailing whitespace from each line and join with single spaces
                        lines = [line.strip() for line in sql.split('\n') if line.strip()]
                        sql = ' '.join(lines)
                        cleaned_data['sql'] = sql
                    else:
                        print(f"⚠️ Line {line_num}: Missing 'sql' field")
                        continue
                    
                    # Write cleaned JSON
                    json.dump(cleaned_data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    cleaned_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Line {line_num}: Invalid JSON - {str(e)}")
                    continue
                except Exception as e:
                    print(f"❌ Line {line_num}: Error processing - {str(e)}")
                    continue
        
        print(f"✅ Cleaning completed!")
        print(f"📊 Total entries processed: {total_count}")
        print(f"📊 Cleaned entries written: {cleaned_count}")
        print(f"📊 Entries skipped: {total_count - cleaned_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        return False


def main():
    """Main function."""
    # Specify input and output files directly
    input_file = r"c:\Users\CBM\Downloads\t2sql_test_cases_v3.jsonl"
    output_file = "t2sql_test_cases_cleaned.jsonl"
    
    print("🚀 Starting JSONL cleaning process...")
    print(f"📂 Input file: {input_file}")
    print(f"📂 Output file: {output_file}")
    
    success = clean_jsonl_file(input_file, output_file)
    
    if success:
        print("🎉 Process completed successfully!")
    else:
        print("💥 Process failed!")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 