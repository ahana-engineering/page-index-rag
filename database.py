import sqlite3

DB_NAME = "page_rag.db"

def get_connection():
    return sqlite3.connect(DB_NAME)