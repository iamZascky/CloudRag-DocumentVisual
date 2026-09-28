import sqlite3
import os
from typing import List, Dict, Optional
from src.utils.helpers import load_config

config = load_config()
DB_PATH = os.path.abspath(config.get("storage", {}).get("sqlite_db", "./storage/doc_archive.db"))

class SqliteDocDatabase:
    """
    Manages document metadata and SQLite FTS5 Full-Text Search index.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Primary documents table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            doc_category TEXT,
            title_or_subject TEXT,
            full_transcription TEXT,
            structured_data TEXT,
            has_stamps_or_signatures BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # FTS5 virtual table
        cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_documents USING fts5(
            title_or_subject,
            full_transcription,
            doc_category
        );
        """)
        conn.commit()
        conn.close()

    def save_document(
        self,
        file_path: str,
        doc_category: str,
        title: str,
        full_text: str,
        structured_json: str,
        has_visuals: bool
    ) -> int:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO documents (file_path, doc_category, title_or_subject, full_transcription, structured_data, has_stamps_or_signatures)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                doc_category=excluded.doc_category,
                title_or_subject=excluded.title_or_subject,
                full_transcription=excluded.full_transcription,
                structured_data=excluded.structured_data,
                has_stamps_or_signatures=excluded.has_stamps_or_signatures;
        """, (file_path, doc_category, title, full_text, structured_json, has_visuals))
        
        cursor.execute("SELECT id FROM documents WHERE file_path = ?", (file_path,))
        rowid = cursor.fetchone()[0]
        
        # Sync FTS
        cursor.execute("DELETE FROM fts_documents WHERE rowid = ?", (rowid,))
        cursor.execute("""
            INSERT INTO fts_documents(rowid, title_or_subject, full_transcription, doc_category)
            VALUES (?, ?, ?, ?)
        """, (rowid, title, full_text, doc_category))
        
        conn.commit()
        conn.close()
        return rowid

    def search_text(self, query: str, limit: int = 5) -> List[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT 
                    documents.file_path, 
                    documents.doc_category, 
                    snippet(fts_documents, 1, '<b>', '</b>', '...', 15) AS highlight
                FROM fts_documents
                JOIN documents ON documents.id = fts_documents.rowid
                WHERE fts_documents MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit))
            results = [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            results = []
        finally:
            conn.close()
        return results

    def is_indexed(self, file_path: str) -> bool:
        conn = self.get_connection()
        existing = conn.execute("SELECT id FROM documents WHERE file_path = ?", (file_path,)).fetchone()
        conn.close()
        return existing is not None
