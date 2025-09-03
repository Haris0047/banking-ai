"""
Streamlit web interface for Vanna.AI.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import json
from typing import Dict, Any
from app.utils.logger import logger

from app import VannaAI
from app.utils.exceptions import VannaException

# Page configuration
st.set_page_config(
    page_title="Vanna.AI - Text to SQL",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vanna' not in st.session_state:
    # Initialize user ID in session state
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    try:
        # We'll initialize VannaAI after user provides ID
        st.session_state.vanna = None
        st.session_state.connected = False
        st.session_state.query_history = []
    except Exception as e:
        st.error(f"Failed to initialize session: {str(e)}")
        st.stop()

def initialize_vanna(user_id: int):
    """Initialize VannaAI with user ID."""
    try:
        st.session_state.vanna = VannaAI(user_id=user_id)
        st.session_state.user_id = user_id
        return True
    except Exception as e:
        st.error(f"Failed to initialize Vanna.AI: {str(e)}")
        return False

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Vanna.AI - Text to SQL</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # User ID section
        st.subheader("👤 User Identification")
        
        if st.session_state.user_id is None:
            user_id_input = st.number_input(
                "User ID", 
                min_value=0,
                value=0,
                step=1,
                help="Provide a unique integer identifier for your session (0 = anonymous)"
            )
            
            if st.button("Initialize Session"):
                if initialize_vanna(int(user_id_input)):
                    st.success(f"✅ Session initialized for user: {user_id_input}")
                    st.rerun()
                else:
                    st.error("Failed to initialize session")
        else:
            st.success(f"✅ Active User: {st.session_state.user_id}")
            if st.button("Change User"):
                st.session_state.user_id = None
                st.session_state.vanna = None
                st.session_state.connected = False
                st.rerun()
        
        st.markdown("---")
        
        # Only show other options if VannaAI is initialized
        if st.session_state.vanna is not None:
            # Database connection
            st.subheader("Database Connection")
            db_type = st.selectbox("Database Type", ["sqlite", "mysql"])
            
            if db_type == "sqlite":
                db_path = st.text_input("Database Path", value="./data/sample.db")
                if st.button("Connect to SQLite"):
                    connect_database(db_type, {"path": db_path})
            
            elif db_type == "mysql":
                col1, col2 = st.columns(2)
                with col1:
                    mysql_host = st.text_input("Host", value="localhost")
                    mysql_db = st.text_input("Database")
                with col2:
                    mysql_port = st.number_input("Port", value=3306)
                    mysql_user = st.text_input("User")
                mysql_password = st.text_input("Password", type="password")
                
                if st.button("Connect to MySQL"):
                    params = {
                        "host": mysql_host,
                        "port": mysql_port,
                        "database": mysql_db,
                        "user": mysql_user,
                        "password": mysql_password
                    }
                    connect_database(db_type, params)
            
            # Connection status
            if st.session_state.connected:
                st.success("✅ Database Connected")
                if st.button("Disconnect"):
                    disconnect_database()
            else:
                st.warning("⚠️ No Database Connected")
            
            st.markdown("---")
            
            # Training data stats
            st.subheader("Training Data")
            try:
                stats = st.session_state.vanna.get_training_stats()
                st.metric("Total Documents", stats.get("total_documents", 0))
                
                if "data_type_distribution" in stats:
                    for data_type, count in stats["data_type_distribution"].items():
                        st.metric(f"{data_type.title()}", count)
            except Exception as e:
                st.error(f"Failed to get stats: {str(e)}")
        else:
            st.info("👆 Please initialize your session first")
    
    # Main content - only show if VannaAI is initialized
    if st.session_state.vanna is not None:
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query", "📚 Training", "📊 Results", "ℹ️ About"])
        
        with tab1:
            query_interface()
        
        with tab2:
            training_interface()
        
        with tab3:
            results_interface()
        
        with tab4:
            about_interface()
    else:
        st.info("👈 Please provide your User ID in the sidebar to get started!")


def connect_database(db_type: str, params: Dict[str, Any]):
    """Connect to database."""
    try:
        st.session_state.vanna.connect_to_database(db_type, params)
        st.session_state.connected = True
        st.success(f"Successfully connected to {db_type} database!")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to connect: {str(e)}")


def disconnect_database():
    """Disconnect from database."""
    try:
        st.session_state.vanna.disconnect_database()
        st.session_state.connected = False
        st.success("Disconnected from database!")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to disconnect: {str(e)}")


def query_interface():
    """Query interface tab."""
    st.header("Ask Questions in Natural Language")
    
    # Question input
    question = st.text_area(
        "Enter your question:",
        placeholder="e.g., How many users are there? Show me all orders from last month.",
        height=100
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🚀 Generate SQL", type="primary"):
            if question.strip():
                generate_sql(question)
            else:
                st.warning("Please enter a question.")
    
    with col2:
        execute_sql_flag = st.checkbox("Execute SQL", value=False)
    
    with col3:
        if st.button("🔍 Find Similar"):
            if question.strip():
                find_similar_questions(question)
    
    # Display results
    if 'current_result' in st.session_state:
        display_query_result(st.session_state.current_result, execute_sql_flag)


def generate_sql(question: str):
    """Generate SQL from question."""
    try:
        with st.spinner("Generating SQL..."):
            if st.session_state.connected:
                execute_sql = st.checkbox("Execute SQL", value=True)
                result = st.session_state.vanna.ask(
                    question=question, 
                    execute_sql=execute_sql, 
                    generate_summary=execute_sql
                )
            else:
                st.error("❌ No database connected. Please connect to a database first.")
                return
        
        st.session_state.current_result = result
        st.session_state.query_history.append({
            "question": question,
            "result": result,
            "timestamp": pd.Timestamp.now()
        })
        
    except VannaException as e:
        st.error(f"Vanna Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")


def display_query_result(result: Dict[str, Any], execute_flag: bool):
    """Display query result."""
    st.markdown("### 📋 Generated SQL")
    
    # SQL query
    st.code(result["sql"], language="sql")
    
    # Metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        confidence = result.get("confidence", 0)
        st.metric("Confidence", f"{confidence:.2%}")
    with col2:
        context_used = result.get("context_used", False)
        st.metric("Context Used", "✅" if context_used else "❌")
    with col3:
        validation = result.get("validation", {})
        is_valid = validation.get("valid", False)
        st.metric("Valid SQL", "✅" if is_valid else "❌")
    
    # Explanation
    if "explanation" in result:
        st.markdown("### 💡 Explanation")
        st.info(result["explanation"])
    
    # Execution results
    if "query_results" in result:
        st.markdown("### 📊 Query Results")
        query_results = result["query_results"]
        
        if query_results["data"]:
            df = pd.DataFrame(query_results["data"], columns=query_results["columns"])
            st.dataframe(df, use_container_width=True)
            st.caption(f"Returned {len(df)} rows")
        else:
            st.info("Query executed successfully but returned no data.")
    
    elif execute_flag and st.session_state.connected:
        if st.button("▶️ Execute Query"):
            execute_query(result["sql"])
    
    # Validation warnings
    if validation.get("warnings"):
        st.warning("⚠️ Warnings: " + ", ".join(validation["warnings"]))


def execute_query(sql: str):
    """Execute SQL query."""
    try:
        with st.spinner("Executing query..."):
            result = st.session_state.vanna.run_sql(sql)
        
        st.markdown("### 📊 Query Results")
        if result["data"]:
            df = pd.DataFrame(result["data"], columns=result["columns"])
            st.dataframe(df, use_container_width=True)
            st.caption(f"Returned {len(df)} rows")
        else:
            st.info("Query executed successfully but returned no data.")
            
    except Exception as e:
        st.error(f"Query execution failed: {str(e)}")


def find_similar_questions(question: str):
    """Find similar questions."""
    try:
        similar = st.session_state.vanna.get_similar_questions(question, 5)
        
        if similar:
            st.markdown("### 🔍 Similar Questions")
            for i, item in enumerate(similar, 1):
                with st.expander(f"{i}. {item['metadata'].get('question', 'Unknown')}"):
                    st.code(item['metadata'].get('sql', 'No SQL available'), language="sql")
                    st.caption(f"Similarity: {1 - item['distance']:.2%}")
        else:
            st.info("No similar questions found.")
            
    except Exception as e:
        st.error(f"Failed to find similar questions: {str(e)}")


def training_interface():
    """Training interface tab."""
    st.header("Train Vanna.AI")
    
    training_type = st.selectbox(
        "Training Data Type",
        ["DDL Statements", "SQL Pairs", "Documentation", "From Database"]
    )
    
    if training_type == "DDL Statements":
        st.subheader("Add DDL Statement")
        ddl = st.text_area("DDL Statement", placeholder="CREATE TABLE users (id INT PRIMARY KEY, ...)")
        table_name = st.text_input("Table Name (optional)")
        
        if st.button("Add DDL"):
            if ddl.strip():
                try:
                    doc_id = st.session_state.vanna.train_ddl(ddl, table_name or None)
                    st.success(f"DDL added successfully! Document ID: {doc_id}")
                except Exception as e:
                    st.error(f"Failed to add DDL: {str(e)}")
    
    elif training_type == "SQL Pairs":
        st.subheader("Add Question-SQL Pair")
        question = st.text_area("Question", placeholder="How many users are there?")
        sql = st.text_area("SQL Query", placeholder="SELECT COUNT(*) FROM users")
        explanation = st.text_area("Explanation (optional)")
        
        if st.button("Add SQL Pair"):
            if question.strip() and sql.strip():
                try:
                    doc_id = st.session_state.vanna.train_sql_pair(
                        question, sql, explanation or None
                    )
                    st.success(f"SQL pair added successfully! Document ID: {doc_id}")
                except Exception as e:
                    st.error(f"Failed to add SQL pair: {str(e)}")
    
    elif training_type == "Documentation":
        st.subheader("Add Table Documentation")
        table_name = st.text_input("Table Name")
        description = st.text_area("Table Description")
        
        st.write("Column Descriptions (optional):")
        col_desc_text = st.text_area(
            "Column Descriptions", 
            placeholder='{"column1": "Description 1", "column2": "Description 2"}'
        )
        
        if st.button("Add Documentation"):
            if table_name.strip() and description.strip():
                try:
                    col_descriptions = None
                    if col_desc_text.strip():
                        col_descriptions = json.loads(col_desc_text)
                    
                    doc_id = st.session_state.vanna.train_documentation(
                        table_name, description, col_descriptions
                    )
                    st.success(f"Documentation added successfully! Document ID: {doc_id}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON format for column descriptions")
                except Exception as e:
                    st.error(f"Failed to add documentation: {str(e)}")
    
    elif training_type == "From Database":
        st.subheader("Train from Connected Database")
        
        if not st.session_state.connected:
            st.warning("Please connect to a database first.")
        else:
            if st.button("Train from Database Schema"):
                try:
                    with st.spinner("Training from database..."):
                        # This would need the database connection info
                        st.info("Database training feature requires connection parameters.")
                except Exception as e:
                    st.error(f"Failed to train from database: {str(e)}")


def results_interface():
    """Results interface tab."""
    st.header("Query History & Results")
    
    if st.session_state.query_history:
        for i, entry in enumerate(reversed(st.session_state.query_history[-10:]), 1):
            with st.expander(f"{i}. {entry['question'][:50]}... ({entry['timestamp'].strftime('%H:%M:%S')})"):
                st.write("**Question:**", entry['question'])
                st.code(entry['result']['sql'], language="sql")
                
                if 'explanation' in entry['result']:
                    st.write("**Explanation:**", entry['result']['explanation'])
                
                confidence = entry['result'].get('confidence', 0)
                st.write(f"**Confidence:** {confidence:.2%}")
    else:
        st.info("No query history available. Ask some questions to see results here!")


def about_interface():
    """About interface tab."""
    st.header("About Vanna.AI")
    
    st.markdown("""
    ### 🤖 What is Vanna.AI?
    
    Vanna.AI is a professional text-to-SQL generation system that uses three key training mechanisms:
    
    1. **DDL Statements** - Database schema definitions
    2. **SQL Pairs** - Question and SQL query examples  
    3. **Documentation** - Business context and table descriptions
    
    ### 🏗️ Architecture
    
    - **RAG (Retrieval-Augmented Generation)** - Finds relevant context for each question
    - **Vector Store (ChromaDB)** - Stores and searches training data using embeddings
    - **LLM Integration** - Uses OpenAI GPT for SQL generation and explanation
    - **Multi-Database Support** - PostgreSQL, MySQL, SQLite
    
    ### 🚀 Features
    
    - Natural language to SQL conversion
    - Confidence scoring and validation
    - Query explanation in plain English
    - Self-learning through feedback
    - Multiple database connectors
    - REST API and web interface
    
    ### 📖 How to Use
    
    1. **Connect** to your database
    2. **Train** with your schema and examples
    3. **Ask** questions in natural language
    4. **Review** and provide feedback
    
    ### 🔧 Configuration
    
    Set up your environment variables in `.env`:
    - `OPENAI_API_KEY` - Your OpenAI API key
    - Database connection parameters
    - Vector store settings
    """)
    
    # System info
    st.markdown("### 📊 System Information")
    try:
        stats = st.session_state.vanna.get_training_stats()
        st.json(stats)
    except Exception as e:
        st.error(f"Failed to get system info: {str(e)}")


if __name__ == "__main__":
    main() 