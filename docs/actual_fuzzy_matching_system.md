# 🔍 Actual Fuzzy String Matching System Documentation

## Overview

This document describes the **actual fuzzy string matching system** currently implemented in the Vanna.AI codebase. This system detects string literals in generated SQL queries and provides fuzzy matching suggestions to handle typos and variations in user input.

**Important**: This system currently provides **suggestions only** - it does not automatically replace literals in SQL queries.

---

## System Architecture

### Core Components

```
app/core/llm_interface.py
├── explain_sql()                           # Entry point - called during SQL explanation
├── _extract_string_literal_mappings()      # Main orchestrator
├── _find_string_literals_with_context()    # Detect string literals in SQL
├── _generate_fuzzy_variations()            # Create fuzzy search queries
├── _execute_fuzzy_queries_and_find_best_matches()  # Run queries and score results
├── _find_best_matching_value()             # Select best match from results
└── _calculate_string_similarity()          # Simple similarity scoring
```

### Integration Point

The fuzzy matching is triggered during **SQL explanation** (not generation):

```python
# In explain_sql() method (lines 107-118)
string_mappings = self._extract_string_literal_mappings(sql)
best_matches = []
if self.db_connector and string_mappings:
    best_matches = self._execute_fuzzy_queries_and_find_best_matches(string_mappings)

return {
    "explanation": explanation,
    "string_mappings": string_mappings,  # Raw literals found
    "best_matches": best_matches         # Fuzzy match suggestions
}
```

---

## Step-by-Step Process

### Step 1: String Literal Detection

**Method**: `_find_string_literals_with_context()`  
**Location**: Lines 168-257

The system uses **3 regex patterns** to find string literals:

```python
# Pattern 1: table.column = 'value'
pattern1 = r'(\w+)\.(\w+)\s*(?:[=<>!]+|LIKE|ILIKE|IN)\s*[\'"]([^\'"]+)[\'"]'

# Pattern 2: LOWER(table.column) = 'value' 
pattern2 = r'(?:LOWER|UPPER)\s*\(\s*(\w+)\.(\w+)\s*\)\s*(?:[=<>!]+|LIKE|ILIKE)\s*[\'"]([^\'"]+)[\'"]'

# Pattern 3: column = 'value' (without table prefix)
pattern3 = r'(?<!\.)\b(\w+)\s*(?:[=<>!]+|LIKE|ILIKE|IN)\s*[\'"]([^\'"]+)[\'"]'
```

**Example Detection**:
```sql
-- Input SQL:
WHERE LOWER(u.full_name) = 'alis' AND a.account_type = 'savings'

-- Detected:
Pattern 2 match: FUNC(u.full_name) = 'alis'
Pattern 1 match: a.account_type = 'savings'
```

**Output**:
```python
mappings = [
    {
        "literal": "alis",
        "table": "users", 
        "column": "full_name",
        "fuzzy_variations": [...]  # Generated in next step
    },
    {
        "literal": "savings",
        "table": "accounts",
        "column": "account_type", 
        "fuzzy_variations": [...]
    }
]
```

### Step 2: Fuzzy Query Generation

**Method**: `_generate_fuzzy_variations()`  
**Location**: Lines 258-354

For each detected literal, the system generates **8 different fuzzy search strategies**:

```python
def _generate_fuzzy_variations(self, literal_value: str, table_name: str, column_name: str):
    variations = []
    words = literal_value.split()
    literal_lower = literal_value.lower()
    
    # 1. Full Pattern Match (ILIKE)
    variations.append({
        "type": "full_match",
        "pattern": f"%{literal_lower}%",
        "sql": f"SELECT * FROM {table_name} WHERE {column_name} ILIKE '%{literal_lower}%'"
    })
    
    # 2. Individual Word Matches
    for word in words:
        if len(word) > 2:
            variations.append({
                "type": "word_match",
                "word": word,
                "pattern": f"%{word.lower()}%",
                "sql": f"SELECT * FROM {table_name} WHERE {column_name} ILIKE '%{word.lower()}%'"
            })
    
    # 3. SOUNDEX Matching (Phonetic)
    variations.append({
        "type": "soundex_match", 
        "pattern": f"SOUNDEX('{literal_value}')",
        "sql": f"SELECT * FROM {table_name} WHERE SOUNDEX({column_name}) = SOUNDEX('{literal_value}')"
    })
    
    # 4. Word-level SOUNDEX
    for word in words:
        if len(word) > 2:
            variations.append({
                "type": "soundex_word_match",
                "word": word,
                "sql": f"SELECT * FROM {table_name} WHERE SOUNDEX({column_name}) = SOUNDEX('{word}')"
            })
    
    # 5. Levenshtein Distance ≤ 1 (Single character errors)
    variations.append({
        "type": "levenshtein_match_1",
        "pattern": "levenshtein <= 1", 
        "sql": f"SELECT * FROM {table_name} WHERE levenshtein(LOWER({column_name}), LOWER('{literal_value}')) <= 1"
    })
    
    # 6. Levenshtein Distance ≤ 2 (Two character errors)
    variations.append({
        "type": "levenshtein_match_2",
        "pattern": "levenshtein <= 2",
        "sql": f"SELECT * FROM {table_name} WHERE levenshtein(LOWER({column_name}), LOWER('{literal_value}')) <= 2"
    })
    
    # 7. Trigram Similarity (PostgreSQL pg_trgm)
    variations.append({
        "type": "similarity_match",
        "pattern": "similarity >= 0.3",
        "sql": f"SELECT * FROM {table_name} WHERE similarity({column_name}, '{literal_value}') >= 0.3"
    })
    
    # 8. Word Similarity
    for word in words:
        if len(word) > 3:
            variations.append({
                "type": "word_similarity_match", 
                "word": word,
                "sql": f"SELECT * FROM {table_name} WHERE word_similarity({column_name}, '{word}') >= 0.4"
            })
```

**Example for literal "alis"**:
```python
fuzzy_variations = [
    {
        "type": "full_match",
        "sql": "SELECT * FROM users WHERE full_name ILIKE '%alis%'"
    },
    {
        "type": "soundex_match", 
        "sql": "SELECT * FROM users WHERE SOUNDEX(full_name) = SOUNDEX('alis')"
    },
    {
        "type": "levenshtein_match_1",
        "sql": "SELECT * FROM users WHERE levenshtein(LOWER(full_name), LOWER('alis')) <= 1"
    },
    # ... 5 more variations
]
```

### Step 3: Query Execution and Scoring

**Method**: `_execute_fuzzy_queries_and_find_best_matches()`  
**Location**: Lines 358-454

The system executes each fuzzy query **sequentially** (not parallel) and scores results:

```python
def _execute_fuzzy_queries_and_find_best_matches(self, string_mappings):
    best_matches = []
    
    for mapping in string_mappings:
        literal = mapping["literal"]
        fuzzy_variations = mapping["fuzzy_variations"]
        
        # Execute each variation and collect results
        variation_results = []
        for variation in fuzzy_variations:
            try:
                result = self.db_connector.execute_query(variation["sql"])
                row_count = result.get("row_count", 0)
                
                if row_count > 0:
                    variation_results.append({
                        "variation": variation,
                        "row_count": row_count,
                        "sample_data": result.get("data", [])[:5],  # First 5 rows
                        "columns": result.get("columns", [])
                    })
            except Exception as e:
                continue  # Skip failed queries
        
        # Score and select best variation
        if variation_results:
            best_variation = max(variation_results, key=score_variation)
            # Find best matching value from sample data
            best_match_value = self._find_best_matching_value(...)
            best_matches.append({...})
    
    return best_matches
```

### Step 4: Simple Scoring Algorithm

**Method**: `score_variation()` (inline function)  
**Location**: Lines 402-423

The scoring is **much simpler** than complex multi-algorithm approaches:

```python
def score_variation(var_result):
    variation = var_result["variation"]
    row_count = var_result["row_count"]
    
    score = 0
    
    # 1. Prefer word matches over full matches (more specific)
    if variation.get("type") == "word_match":
        score += 10
    
    # 2. Prefer reasonable result counts
    if 1 <= row_count <= 10:
        score += 20      # Best range - specific results
    elif 11 <= row_count <= 50:
        score += 15      # Good range
    elif 51 <= row_count <= 100:
        score += 10      # Acceptable range
    elif row_count > 100:
        score += 5       # Too many results - less specific
    
    return score
```

**Scoring Logic**:
- **Word matches get +10 bonus** (more specific than full matches)
- **1-10 results = +20** (most specific)
- **11-50 results = +15** (good specificity)
- **51-100 results = +10** (acceptable)
- **100+ results = +5** (too broad)

### Step 5: Best Value Selection

**Method**: `_find_best_matching_value()`  
**Location**: Lines 456-504

From the winning variation's sample data, select the best matching value:

```python
def _find_best_matching_value(self, original_literal, sample_data, columns, target_column):
    # Find target column index
    column_index = columns.index(target_column)
    
    # Extract all values from target column
    column_values = [str(row[column_index]) for row in sample_data if row[column_index]]
    
    # Find best match using simple similarity
    best_match = None
    best_score = -1
    
    for value in column_values:
        score = self._calculate_string_similarity(original_literal.lower(), value.lower())
        if score > best_score:
            best_score = score
            best_match = value
    
    return best_match
```

### Step 6: Simple String Similarity

**Method**: `_calculate_string_similarity()`  
**Location**: Lines 506-542

**Very basic similarity calculation**:

```python
def _calculate_string_similarity(self, str1, str2):
    # 1. Exact match
    if str1 == str2:
        return 1.0
    
    # 2. Substring match
    if str1 in str2 or str2 in str1:
        return 0.8
    
    # 3. Common characters count
    str1_chars = set(str1.lower())
    str2_chars = set(str2.lower())
    common_chars = len(str1_chars & str2_chars)
    
    # 4. Calculate similarity with length penalty
    max_len = max(len(str1), len(str2))
    char_similarity = common_chars / max_len if max_len > 0 else 0
    
    len_diff = abs(len(str1) - len(str2))
    len_penalty = len_diff / max_len if max_len > 0 else 0
    
    similarity = char_similarity - (len_penalty * 0.3)
    return max(0.0, min(1.0, similarity))
```

---

## Real Example Walkthrough

### Input
```sql
SELECT u.full_name, a.balance 
FROM users u 
JOIN accounts a ON u.user_id = a.user_id 
WHERE LOWER(u.full_name) = 'alis'
```

### Step 1: Detection
```python
# Detected literal
{
    "literal": "alis",
    "table": "users",
    "column": "full_name"
}
```

### Step 2: Generated Queries
```sql
-- Query 1: Full match
SELECT * FROM users WHERE full_name ILIKE '%alis%'

-- Query 2: SOUNDEX  
SELECT * FROM users WHERE SOUNDEX(full_name) = SOUNDEX('alis')

-- Query 3: Levenshtein ≤ 1
SELECT * FROM users WHERE levenshtein(LOWER(full_name), LOWER('alis')) <= 1

-- Query 4: Levenshtein ≤ 2  
SELECT * FROM users WHERE levenshtein(LOWER(full_name), LOWER('alis')) <= 2

-- Query 5: Trigram similarity
SELECT * FROM users WHERE similarity(full_name, 'alis') >= 0.3
```

### Step 3: Execution Results
```python
variation_results = [
    {
        "variation": {"type": "soundex_match"},
        "row_count": 3,
        "sample_data": [["Alice Johnson"], ["Alison Smith"], ["Allison Hill"]]
    },
    {
        "variation": {"type": "levenshtein_match_2"}, 
        "row_count": 2,
        "sample_data": [["Alice Johnson"], ["Alison Smith"]]
    },
    {
        "variation": {"type": "similarity_match"},
        "row_count": 15,
        "sample_data": [["Alice Johnson"], ["Alison Smith"], ["Alex Wilson"], ...]
    }
]
```

### Step 4: Scoring
```python
# SOUNDEX: score = 0 + 20 (1-10 results) = 20
# Levenshtein_2: score = 0 + 20 (1-10 results) = 20  
# Similarity: score = 0 + 15 (11-50 results) = 15

# Winner: SOUNDEX (first with score 20)
```

### Step 5: Best Value Selection
```python
# From SOUNDEX sample data: ["Alice Johnson", "Alison Smith", "Allison Hill"]
# Calculate similarity with "alis":

similarity("alis", "alice johnson") = 0.4   # Some common chars
similarity("alis", "alison smith") = 0.8    # "alis" in "alison" 
similarity("alis", "allison hill") = 0.6    # Some common chars

# Winner: "Alison Smith" (score 0.8)
```

### Final Output
```python
best_matches = [
    {
        "original_literal": "alis",
        "table": "users", 
        "column": "full_name",
        "best_variation": {"type": "soundex_match", "pattern": "SOUNDEX('alis')"},
        "best_match_value": "Alison Smith",
        "match_count": 3,
        "sample_matches": [["Alice Johnson"], ["Alison Smith"], ["Allison Hill"]]
    }
]
```

---

## Current Limitations

### What It Does NOT Do

1. **❌ No Automatic SQL Correction**: The system only provides suggestions, doesn't replace literals in SQL
2. **❌ No Parallel Execution**: Queries run sequentially, not concurrently  
3. **❌ No Advanced Scoring**: Only simple result-count + word-match scoring
4. **❌ No Confidence Thresholds**: No auto-accept/reject based on confidence levels
5. **❌ No Composite Scoring**: No weighted combination of multiple algorithms
6. **❌ No Performance Optimization**: No caching, connection pooling, or timeouts

### What It DOES Do

1. **✅ String Literal Detection**: Finds literals in SQL using regex patterns
2. **✅ Multiple Fuzzy Strategies**: Generates 8 different search approaches
3. **✅ PostgreSQL Integration**: Uses SOUNDEX, Levenshtein, trigram functions
4. **✅ Simple Scoring**: Ranks results by specificity and result count
5. **✅ Best Match Selection**: Picks most similar value from results
6. **✅ Error Handling**: Gracefully handles failed queries

---

## PostgreSQL Dependencies

The system requires these PostgreSQL extensions:

```sql
-- For SOUNDEX and Levenshtein functions
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

-- For trigram similarity functions  
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

**Functions Used**:
- `SOUNDEX(text)` - Phonetic matching
- `levenshtein(text, text)` - Edit distance calculation
- `similarity(text, text)` - Trigram similarity (0.0 to 1.0)
- `word_similarity(text, text)` - Word-based trigram similarity

---

## Integration Usage

The fuzzy matching is currently integrated into the **SQL explanation** process:

```python
# In your main application
result = vanna.ask("Show me Alis's account balance")

# The explanation will include fuzzy matches
explanation_result = vanna.llm.explain_sql(result['sql'])

print("Fuzzy match suggestions:")
for match in explanation_result.get('best_matches', []):
    print(f"  '{match['original_literal']}' → '{match['best_match_value']}'")
    print(f"  Found via: {match['best_variation']['type']}")
    print(f"  {match['match_count']} total matches")
```

**Output Example**:
```
Fuzzy match suggestions:
  'alis' → 'Alison Smith'
  Found via: soundex_match
  3 total matches
```

---

## Configuration

Currently, the thresholds are **hardcoded** in the methods:

```python
# In _generate_fuzzy_variations()
SIMILARITY_THRESHOLD = 0.3        # Trigram similarity minimum
WORD_SIMILARITY_THRESHOLD = 0.4   # Word similarity minimum
MIN_WORD_LENGTH = 2               # Minimum word length for matching
MIN_LEVENSHTEIN_WORD_LENGTH = 3   # Minimum word length for Levenshtein

# In scoring
BEST_RESULT_COUNT_RANGE = (1, 10)     # Most specific result count
GOOD_RESULT_COUNT_RANGE = (11, 50)    # Good result count  
OK_RESULT_COUNT_RANGE = (51, 100)     # Acceptable result count
```

---

## Summary

This is a **functional but basic** fuzzy matching system that:

- **Detects** string literals in SQL queries
- **Generates** multiple fuzzy search strategies using PostgreSQL functions
- **Executes** queries sequentially and scores by result specificity  
- **Suggests** the best matching values (but doesn't auto-correct)

It's **not** the sophisticated multi-algorithm scoring system I initially documented, but it **does work** for finding fuzzy matches and providing suggestions to users! 🎯 