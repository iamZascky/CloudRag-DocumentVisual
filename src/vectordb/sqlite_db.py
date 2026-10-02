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

        # Multi-turn conversation memory table (per WhatsApp user/session)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_message TEXT,
            bot_reply TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_sessions(session_id, id);")
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

    def get_document_by_path(self, file_path: str) -> Optional[Dict]:
        conn = self.get_connection()
        row = conn.execute("""
            SELECT file_path, doc_category, title_or_subject, full_transcription, structured_data
            FROM documents WHERE file_path = ?
        """, (file_path,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_all_documents(self) -> List[Dict]:
        conn = self.get_connection()
        rows = conn.execute("""
            SELECT id, file_path, doc_category, title_or_subject, created_at
            FROM documents ORDER BY id DESC
        """).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def clear_all(self) -> int:
        """Deletes all documents, FTS index records, and chat history. Returns count of deleted documents."""
        conn = self.get_connection()
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM fts_documents")
        conn.execute("DELETE FROM chat_sessions")
        conn.commit()
        conn.close()
        return count

    def add_chat_history(self, session_id: str, user_message: str, bot_reply: str, max_turns: int = 5):
        """Saves a conversational turn and retains only the latest max_turns for the session."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_sessions (session_id, user_message, bot_reply)
            VALUES (?, ?, ?)
        """, (session_id, user_message, bot_reply))
        
        # Prune old turns beyond max_turns to keep DB lightweight and queries fast
        cursor.execute("""
            DELETE FROM chat_sessions 
            WHERE session_id = ? AND id NOT IN (
                SELECT id FROM chat_sessions 
                WHERE session_id = ? 
                ORDER BY id DESC LIMIT ?
            )
        """, (session_id, session_id, max_turns))
        conn.commit()
        conn.close()

    def get_chat_history(self, session_id: str, limit: int = 3) -> List[Dict[str, str]]:
        """Retrieves the recent conversation history in chronological order."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_message, bot_reply 
            FROM chat_sessions 
            WHERE session_id = ? 
            ORDER BY id DESC LIMIT ?
        """, (session_id, limit))
        rows = cursor.fetchall()
        conn.close()
        # Return in chronological order (oldest to newest)
        return [{"user_message": row["user_message"], "bot_reply": row["bot_reply"]} for row in reversed(rows)]
