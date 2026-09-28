import os
import sys
import sqlite3
import time

sys.path.insert(0, os.path.abspath("."))
from src.vectordb.vector_store import ChromaVectorStore

DB_PATH = os.path.abspath("./storage/doc_archive.db")

def backfill():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database {DB_PATH} not found.")
        return

    print("=" * 60)
    print("  Backfilling SQLite Documents into ChromaDB Vector Store")
    print("  Model: intfloat/multilingual-e5-small (CPU, 0 MB GPU VRAM)")
    print("=" * 60)

    # Initialize Chroma vector store
    vector_store = ChromaVectorStore()
    existing_paths = vector_store.get_indexed_paths()
    print(f"ChromaDB already has {len(existing_paths)} indexed pages.")

    # Read from SQLite
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT file_path, doc_category, title_or_subject, full_transcription FROM documents")
    rows = cursor.fetchall()
    conn.close()

    total_rows = len(rows)
    print(f"Found {total_rows} pages in SQLite database.\n")

    added_count = 0
    start_time = time.time()

    for idx, row in enumerate(rows, start=1):
        file_path = row["file_path"]
        norm_path = os.path.normpath(file_path)

        if norm_path in existing_paths:
            print(f"[{idx}/{total_rows}] Already indexed: {os.path.basename(file_path)}")
            continue

        category = row["doc_category"] or "OTHER"
        title = row["title_or_subject"] or "Unknown"
        text = row["full_transcription"] or ""

        t0 = time.time()
        vector_store.upsert_page(
            file_path=norm_path,
            text=text,
            category=category,
            title=title
        )
        elapsed = time.time() - t0
        added_count += 1
        print(f"[{idx}/{total_rows}] Embedded ({elapsed*1000:.1f}ms): {os.path.basename(file_path)} [{category}]")

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Backfill Complete!")
    print(f"Successfully added: {added_count} pages in {total_elapsed:.2f}s")
    print(f"Total pages in ChromaDB: {vector_store.count()}")
    print("=" * 60)

if __name__ == "__main__":
    backfill()
