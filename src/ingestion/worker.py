import os
import json
import time
from typing import Dict, Any, List
from src.ingestion.loader import render_pdf, enhance_scan
from src.vectordb.sqlite_db import SqliteDocDatabase
from src.vectordb.vector_store import ChromaVectorStore
from src.utils.helpers import validate_extraction
from src.api.whatsapp import WhatsAppClient

class IngestionWorker:
    """
    Background worker that ingests documents (PDF/images) uploaded via WhatsApp or API:
    1. Downloads or accepts file path
    2. Renders pages (if PDF)
    3. Enhances contrast via CLAHE
    4. Extracts structured data & OCR via Vision AI (Gemini Flash or Qwen2.5-VL)
    5. Indexes into SQLite FTS5 and ChromaDB vector store
    6. Saves formatted .txt dumps into ./storage/outputtext/
    7. Sends completion message to WhatsApp recipient
    """

    def __init__(
        self,
        reader: Any,
        sqlite_db: SqliteDocDatabase,
        vector_store: ChromaVectorStore,
        whatsapp_client: WhatsAppClient
    ):
        self.reader = reader
        self.db = sqlite_db
        self.vector_store = vector_store
        self.wa = whatsapp_client

    def ingest_file(self, file_path: str, recipient_number: str = None) -> Dict[str, Any]:
        """
        Executes full ingestion pipeline for a given document file.
        Sends live progress and completion updates via WhatsApp if recipient_number is provided.
        """
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        t0 = time.time()

        print(f"\n[IngestionWorker] 🚀 Starting ingestion for: '{filename}'...")

        # 1. Render pages
        image_paths: List[str] = []
        if ext == ".pdf":
            try:
                image_paths = render_pdf(file_path, dpi=300)
            except Exception as e:
                err_msg = f"❌ Gagal memproses file PDF '{filename}': {str(e)}"
                print(f"[IngestionWorker] {err_msg}")
                if recipient_number:
                    self.wa.send_text_message(recipient_number, err_msg)
                return {"status": "error", "error": str(e)}
        elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".webp"]:
            image_paths = [file_path]
        else:
            msg = f"⚠️ Format file '{ext}' tidak didukung. Mohon kirim file PDF atau Gambar (PNG/JPG)."
            if recipient_number:
                self.wa.send_text_message(recipient_number, msg)
            return {"status": "unsupported", "error": msg}

        total_pages = len(image_paths)
        print(f"[IngestionWorker] 📄 Extracted {total_pages} page(s). Processing visual OCR & indexing...")

        if recipient_number and total_pages > 1:
            self.wa.send_text_message(
                recipient_number,
                f"📄 File terdeteksi memiliki *{total_pages} halaman*.\nSedang membaca dan mengindeks seluruh halaman..."
            )

        processed_count = 0
        output_text_dir = os.path.abspath("./storage/outputtext")
        os.makedirs(output_text_dir, exist_ok=True)

        for idx, img_path in enumerate(image_paths, start=1):
            enhanced_path = enhance_scan(img_path)

            # Cek apakah sudah pernah diindeks
            if self.db.is_indexed(enhanced_path):
                print(f"[IngestionWorker] Page {idx}/{total_pages} already indexed, skipping duplicate...")
                processed_count += 1
                continue

            try:
                print(f"[IngestionWorker] 👁️ Reading Page {idx}/{total_pages} via Vision AI...")
                result = self.reader.process_document(enhanced_path)

                doc_category = result.get("doc_category", "OTHER")
                structured_data = result.get("structured_data", {})
                title = result.get("title_or_subject", f"{filename} - Hal {idx}")
                full_text = result.get("full_transcription", "")

                # Validation
                validation_status = validate_extraction(doc_category, structured_data)
                structured_data["validation_status"] = validation_status

                # Save to SQLite FTS5
                self.db.save_document(
                    file_path=enhanced_path,
                    doc_category=doc_category,
                    title=title,
                    full_text=full_text,
                    structured_json=json.dumps(structured_data, ensure_ascii=False),
                    has_visuals=result.get("has_stamps_or_signatures", False)
                )

                # Sync to vector store
                self.vector_store.upsert_page(
                    file_path=enhanced_path,
                    text=full_text,
                    category=doc_category,
                    title=title
                )

                # Save readable text export
                txt_filename = os.path.splitext(os.path.basename(enhanced_path))[0] + ".txt"
                txt_path = os.path.join(output_text_dir, txt_filename)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"=== DOCUMENT METADATA ===\n")
                    f.write(f"Source Page: {enhanced_path}\n")
                    f.write(f"Category: {doc_category}\n")
                    f.write(f"Title / Subject: {title}\n")
                    f.write(f"Validation: {validation_status}\n\n")
                    f.write(f"=== STRUCTURED DATA (JSON) ===\n")
                    f.write(json.dumps(structured_data, indent=2, ensure_ascii=False))
                    f.write(f"\n\n=== FULL TRANSCRIPTION ===\n")
                    f.write(full_text)

                processed_count += 1
            except Exception as page_err:
                import traceback
                print(f"[IngestionWorker] ❌ Error processing Page {idx}/{total_pages}: {page_err}")
                traceback.print_exc()

        elapsed = time.time() - t0
        print(f"[IngestionWorker] ✅ Ingestion finished for '{filename}' ({processed_count}/{total_pages} pages) in {elapsed:.2f}s!")

        if recipient_number:
            if processed_count > 0:
                done_msg = (
                    f"✅ *Dokumen Berhasil Diindeks!*\n\n"
                    f"📁 *File:* `{filename}`\n"
                    f"📄 *Berhasil Diindeks:* {processed_count} dari {total_pages} halaman\n"
                    f"⏱️ *Waktu Proses:* {elapsed:.1f} detik\n\n"
                    f"💡 _Sekarang Anda bisa langsung mencari dokumen ini dengan:_ \n"
                    f"• `cari: {os.path.splitext(filename)[0]}`\n"
                    f"• Atau ajukan pertanyaan langsung mengenai isi dokumen ini!"
                )
            else:
                done_msg = (
                    f"⚠️ Gagal mengekstrak isi dokumen `{filename}`.\n"
                    f"Mohon periksa log atau pastikan dokumen dapat dibaca dengan jelas."
                )
            self.wa.send_text_message(recipient_number, done_msg)

        return {
            "status": "success",
            "file": filename,
            "pages": total_pages,
            "elapsed_seconds": round(elapsed, 2)
        }
