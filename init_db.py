from database import get_connection

conn = get_connection()

cursor = conn.cursor()

# =====================================================
# DOCUMENTS TABLE
# =====================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    document_name TEXT UNIQUE,

    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# =====================================================
# CHUNKS TABLE
# =====================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    document_name TEXT,

    page_number INTEGER,

    chunk_number INTEGER,

    chunk_content TEXT
)
""")

# =====================================================
# QUERY LOGS TABLE
# =====================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS query_logs (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    session_code TEXT,

    user_query TEXT,

    retrieved_pages TEXT,

    retrieved_chunks TEXT,

    context_sent TEXT,

    prompt_sent TEXT,

    llm_response TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

conn.close()

print("Database initialized successfully")