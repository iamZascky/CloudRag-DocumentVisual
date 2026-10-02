import os
import json
import time
from typing import Dict, Any, List
from src.ingestion.loader import render_pdf, enhance_scan, analyze_pdf_page_content
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

        # 1. Render pages (use 150 DPI for optimal speed and low memory on shared hosting)
        image_paths: List[str] = []
        if ext == ".pdf":
            try:
                image_paths = render_pdf(file_path, dpi=150)
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
        fast_path_count = 0
        vision_ai_count = 0

        from src.utils.helpers import load_config
        cfg = load_config()
        output_text_dir = cfg.get("storage", {}).get("output_text_dir", os.path.abspath("./storage/outputtext"))
        os.makedirs(output_text_dir, exist_ok=True)

        for idx, img_path in enumerate(image_paths, start=1):
            enhanced_path = enhance_scan(img_path)

            # Cek apakah sudah pernah diindeks
            if self.db.is_indexed(enhanced_path):
                print(f"[IngestionWorker] Page {idx}/{total_pages} already indexed, skipping duplicate...")
                processed_count += 1
                continue

            try:
                # 1. Hybrid Completeness Detection:
                # If PDF, check if this page is pure digital text (NO embedded receipt/photo)
                page_analysis = {}
                is_pure_digital = False
                if is_pdf:
                    page_analysis = analyze_pdf_page_content(file_path, idx - 1)
                    is_pure_digital = page_analysis.get("is_pure_digital", False)

                if is_pure_digital:
                    # FAST-PATH: Pure digital text with zero embedded receipts/photos
                    full_text = page_analysis.get("extracted_text", "")
                    doc_category = "TABLE_REPORT" if ("total" in full_text.lower() or "\t" in full_text) else "DOCUMENT"
                    title = f"{filename} - Hal {idx} (Digital)"
                    structured_data = {
                        "extraction_engine": "PYPDFIUM2_FAST_PATH",
                        "ocr_cost": "0 API Call (Instant)",
                        "text_length": len(full_text)
                    }
                    has_visuals = False
                    fast_path_count += 1
                    print(f"[IngestionWorker] 📄 Page {idx}/{total_pages}: [Pure Digital Text] ➔ Instant Fast-Path ({len(full_text)} chars)")
                else:
                    # VISION AI PATH: Mixed page (contains embedded receipts/photos) or scanned page
                    reason = "Embedded Receipt/Photo" if page_analysis.get("has_embedded_images") else "Visual Document"
                    print(f"[IngestionWorker] 👁️ Page {idx}/{total_pages}: [{reason}] ➔ Vision AI...")
                    result = self.reader.process_document(enhanced_path)

                    doc_category = result.get("doc_category", "OTHER")
                    structured_data = result.get("structured_data", {})
                    structured_data["extraction_engine"] = "GEMINI_VISION_AI"
                    structured_data["ocr_cost"] = "1 API Call"
                    title = result.get("title_or_subject", f"{filename} - Hal {idx}")
                    full_text = result.get("full_transcription", "")
                    has_visuals = result.get("has_stamps_or_signatures", False)
                    vision_ai_count += 1

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
                    has_visuals=has_visuals
                )

                # Sync to vector store
                self.vector_store.upsert_page(
                    file_path=enhanced_path,
                    text=full_text,
                    category=doc_category,
                    title=title
                )

                # Save readable text export with clear audit engine header
                txt_filename = os.path.splitext(os.path.basename(enhanced_path))[0] + ".txt"
                txt_path = os.path.join(output_text_dir, txt_filename)
                engine_tag = structured_data.get("extraction_engine", "GEMINI_VISION_AI")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"=== DOCUMENT METADATA ===\n")
                    f.write(f"Source Page: {enhanced_path}\n")
                    f.write(f"Category: {doc_category}\n")
                    f.write(f"Title / Subject: {title}\n")
                    f.write(f"Extraction Engine: {engine_tag}\n")
                    f.write(f"Validation: {validation_status}\n\n")
                    f.write(f"=== STRUCTURED DATA (JSON) ===\n")
                    f.write(json.dumps(structured_data, indent=2, ensure_ascii=False))
                    f.write(f"\n\n=== FULL TRANSCRIPTION ===\n")
                    f.write(full_text)

                processed_count += 1

                # Adaptive progress updates via WhatsApp:
                # - For page 1 of multi-page docs (>= 4 pages): sends confirmation that page 1 is successfully read
                # - For 4 to 9 pages: sends 1 update at halfway mark (50%)
                # - For >= 10 pages: sends updates at 25%, 50%, and 75%
                if recipient_number and total_pages >= 4:
                    if idx == 1:
                        try:
                            self.whatsapp_client.send_text_message(
                                recipient_number,
                                f"🚀 *Mulai Membaca Dokumen:*\nHalaman 1/{total_pages} berhasil dibaca. Melanjutkan ekstraksi halaman berikutnya..."
                            )
                        except Exception:
                            pass
                    else:
                        if total_pages < 10:
                            milestones = [total_pages // 2]
                        else:
                            milestones = [int(total_pages * 0.25), int(total_pages * 0.50), int(total_pages * 0.75)]

                        if idx in milestones:
                            progress_pct = int((idx / total_pages) * 100)
                            try:
                                self.whatsapp_client.send_text_message(
                                    recipient_number,
                                    f"⏳ *Progres Ekstraksi:* {idx}/{total_pages} halaman ({progress_pct}%)\n_Sedang membaca data visual dokumen..._"
                                )
                            except Exception:
                                pass

                # Solusi 1: Pacing jeda 2.5 detik antar halaman agar tidak terkena limit 15 RPM Google Gemini
                if idx < total_pages:
                    time.sleep(2.5)

            except Exception as page_err:
                import traceback
                print(f"[IngestionWorker] ❌ Error processing Page {idx}/{total_pages}: {page_err}")
                traceback.print_exc()
                # Jika terkena error berat, beri jeda lebih lama sebelum lanjut halaman berikutnya
                time.sleep(5)

        elapsed = time.time() - t0
        print(f"[IngestionWorker] ✅ Ingestion finished for '{filename}' ({processed_count}/{total_pages} pages) in {elapsed:.2f}s!")

        if recipient_number:
            if processed_count > 0:
                engine_breakdown = ""
                if is_pdf and (fast_path_count > 0 or vision_ai_count > 0):
                    engine_breakdown = (
                        f"⚡ *Fast-Path Digital:* {fast_path_count} halaman (Hemat Kuota)\n"
                        f"👁️ *Gemini AI Vision:* {vision_ai_count} halaman (Struk/Visual)\n\n"
                    )

                done_msg = (
                    f"✅ *Dokumen Berhasil Diindeks!*\n\n"
                    f"📁 *File:* `{filename}`\n"
                    f"📄 *Total:* {processed_count} dari {total_pages} halaman\n"
                    f"{engine_breakdown}"
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
