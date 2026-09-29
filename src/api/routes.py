import os
import sys

# Limit OpenBLAS / NumPy / OpenMP threads to 1 to stay safely within cPanel nproc limit (40)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["AI_MODE"] = os.getenv("AI_MODE", "cloud")

from fastapi import FastAPI, HTTPException, Request, Response, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from src.utils.helpers import StructuredQAResponse
from src.api.whatsapp import WhatsAppClient
import time

app = FastAPI(
    title="Visual-RAG API",
    description="REST API for Visual-RAG with Hybrid Search (FTS5 + ChromaDB RRF), Multi-Page Context & WhatsApp Integration"
)

# Global singletons
_retriever = None
_reader = None
_sqlite_db = None
_vector_store = None
_whatsapp_client = None
_ingestion_worker = None

def get_retriever():
    global _retriever
    if _retriever is None:
        from src.retrieval.retriever import HybridRetriever
        _retriever = HybridRetriever()
    return _retriever

def get_reader():
    """
    Returns either Cloud GeminiVisualReader or Local QwenVisualReader based on AI_MODE env var.
    Default: 'cloud' if GEMINI_API_KEY is present or AI_MODE=cloud, else 'local'.
    """
    global _reader
    if _reader is None:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        ai_mode = os.getenv("AI_MODE", "cloud").lower()
        gemini_key = os.getenv("GEMINI_API_KEY", "")

        # Always default to cloud mode on hosting or when Gemini key exists
        if ai_mode != "local" or gemini_key:
            from src.llm.gemini_client import GeminiVisualReader
            print("[LLM Factory] [Cloud] Initializing Cloud GeminiVisualReader (Zero GPU/Torch)...")
            _reader = GeminiVisualReader(api_key=gemini_key)
        else:
            try:
                from src.llm.llm_client import QwenVisualReader
                print("[LLM Factory] [Local] Initializing Local QwenVisualReader (GPU BF16)...")
                _reader = QwenVisualReader()
            except Exception as e:
                from src.llm.gemini_client import GeminiVisualReader
                print(f"[LLM Factory] Local GPU loader failed ({e}), falling back to Cloud Gemini...")
                _reader = GeminiVisualReader(api_key=gemini_key)
    return _reader

def get_sqlite_db():
    global _sqlite_db
    if _sqlite_db is None:
        from src.vectordb.sqlite_db import SqliteDocDatabase
        _sqlite_db = SqliteDocDatabase()
    return _sqlite_db

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        from src.vectordb.vector_store import ChromaVectorStore
        _vector_store = ChromaVectorStore()
    return _vector_store

def get_whatsapp_client() -> WhatsAppClient:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    return WhatsAppClient(
        token=os.getenv("WHATSAPP_TOKEN", ""),
        phone_number_id=os.getenv("PHONE_NUMBER_ID", "1382298338296594"),
        verify_token=os.getenv("WHATSAPP_VERIFY_TOKEN", "docuvisual_secret_token_2026")
    )

def get_ingestion_worker():
    global _ingestion_worker
    if _ingestion_worker is None:
        from src.ingestion.worker import IngestionWorker
        _ingestion_worker = IngestionWorker(
            reader=get_reader(),
            sqlite_db=get_sqlite_db(),
            vector_store=get_vector_store(),
            whatsapp_client=get_whatsapp_client()
        )
    return _ingestion_worker

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    mode: Optional[str] = "hybrid"

class SearchResult(BaseModel):
    final_rank: int
    rrf_score: float
    file_path: str
    doc_category: str
    fts_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    vector_similarity: Optional[float] = None
    highlight: Optional[str] = None

class AskRequest(BaseModel):
    question: str
    structured: Optional[bool] = True

class APIAskResponse(BaseModel):
    question: str
    candidate_page: str
    doc_category: str
    rrf_score: float
    fts_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    data: StructuredQAResponse
    latency_seconds: float

@app.get("/health")
def health_check():
    return {"status": "ok", "system": "Local Windows-Native Visual-RAG with WhatsApp Webhook"}

@app.post("/search", response_model=List[SearchResult])
def search_endpoint(req: SearchRequest):
    retriever = get_retriever()
    results = retriever.retrieve(req.query, limit=req.limit)
    return results

@app.post("/ask", response_model=APIAskResponse)
def ask_endpoint(req: AskRequest):
    t0 = time.time()
    retriever = get_retriever()
    results = retriever.retrieve(req.question, limit=7)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found for question.")
    
    top = results[0]
    reader = get_reader()
    
    if req.structured:
        qa_data = reader.answer_question_structured(top["file_path"], req.question)
    else:
        raw_ans = reader.answer_question(top["file_path"], req.question, stream=False)
        qa_data = StructuredQAResponse(
            direct_answer=raw_ans,
            raw_answer=raw_ans
        )
        
    elapsed = time.time() - t0
    return APIAskResponse(
        question=req.question,
        candidate_page=top["file_path"],
        doc_category=top.get("doc_category", "DOCUMENT"),
        rrf_score=top["rrf_score"],
        fts_rank=top.get("fts_rank"),
        vector_rank=top.get("vector_rank"),
        data=qa_data,
        latency_seconds=round(elapsed, 2)
    )

# -------------------------------------------------------------
# WhatsApp Webhook Endpoints
# -------------------------------------------------------------

def process_and_reply_whatsapp(sender_number: str, question: str, message_id: str):
    """Background worker to answer visual RAG query and reply on WhatsApp."""
    try:
        print(f"\n[WhatsApp Bot] 📥 Processing question from {sender_number}: '{question}'...")
        client = get_whatsapp_client()
        client.mark_as_read(message_id)

        # Cek apakah pengguna mengirim salam / meminta menu bantuan
        clean_text = question.strip().lower()
        greeting_keywords = [
            "halo", "hallo", "hello", "hai", "hi", "hey", "p", 
            "menu", "help", "bantuan", "info", "mulai", "start",
            "tes", "test", "ping", "assalamualaikum", "selamat"
        ]
        is_greeting = any(clean_text == kw or clean_text.startswith(kw + " ") for kw in greeting_keywords)
        if is_greeting:
            welcome_msg = (
                "👋 *Halo! Selamat datang di Bot Asisten Dokumen Visual.*\n\n"
                "Saya dapat membantu Anda mencari dan menganalisis arsip dokumen/kwitansi secara cerdas.\n\n"
                "📌 *Panduan Penggunaan:*\n"
                "1. *Cari Cepat Dokumen (< 1 detik):*\n"
                "   Ketik: `cari: <kata kunci>` atau `/cari <kata kunci>`\n"
                "   Contoh: `cari: SPBU Balung Lor`\n\n"
                "2. *Tanya Jawab Dokumen (AI Visual RAG):*\n"
                "   Ketik langsung pertanyaan Anda.\n"
                "   Contoh: `Berapa total belanja bbm di kwitansi?`\n\n"
                "Silakan ketik pertanyaan atau kata kunci yang ingin Anda cari!"
            )
            client.send_text_message(sender_number, welcome_msg)
            print(f"[WhatsApp Bot] 📤 Sent welcome/help menu to {sender_number}!")
            return

        # Cek apakah pengguna meminta mode pencarian cepat (search mode)
        if clean_text.startswith(("/cari", "cari:", "search:", "/search")):
            # Extract keyword setelah command
            if ":" in question:
                search_query = question.split(":", 1)[1].strip()
            else:
                parts = question.split(maxsplit=1)
                search_query = parts[1].strip() if len(parts) > 1 else ""

            if not search_query:
                client.send_text_message(
                    sender_number,
                    "⚠️ Format salah. Contoh penggunaan:\n*cari: SPBU Balung Lor* atau */cari BBM September*"
                )
                return

            print(f"[WhatsApp Bot] 🔎 Executing Fast Search Mode for query: '{search_query}'...")
            retriever = get_retriever()
            search_results = retriever.retrieve(search_query, limit=5)
            
            if not search_results:
                client.send_text_message(
                    sender_number,
                    f"❌ Tidak ditemukan dokumen untuk kata kunci: *{search_query}*"
                )
                return

            response_lines = [f"📂 *Hasil Pencarian Arsip Dokumen:*", f"Kata Kunci: _{search_query}_\n"]
            for idx, item in enumerate(search_results, start=1):
                base_name = os.path.basename(item["file_path"])
                sim_str = f"{item['vector_similarity'] * 100:.1f}%" if item.get('vector_similarity') else "Kecocokan Kata"
                highlight = item.get("highlight", "").replace("<b>", "*").replace("</b>", "*")
                if len(highlight) > 120:
                    highlight = highlight[:120] + "..."
                response_lines.append(f"*{idx}. {base_name}*")
                response_lines.append(f"   • Relevansi: {sim_str}")
                if highlight:
                    response_lines.append(f"   • Cuplikan: _{highlight}_")
                response_lines.append("")

            response_lines.append("💡 _Ketik pertanyaan lengkap jika ingin rincian/analisis dari dokumen di atas._")
            reply_text = "\n".join(response_lines).strip()
            client.send_text_message(sender_number, reply_text)
            print(f"[WhatsApp Bot] 📤 Search results sent to {sender_number}!")
            return

        # Send immediate acknowledgment so user knows bot is thinking (QA mode)
        client.send_text_message(
            sender_number,
            "🔍 *Sedang membaca dan menganalisis dokumen...*\nMohon tunggu beberapa detik, jawaban sedang diproses."
        )

        retriever = get_retriever()
        reader = get_reader()
        db = get_sqlite_db()
        
        # 1. Retrieve top 7 candidates
        results = retriever.retrieve(question, limit=7)
        if not results:
            client.send_text_message(
                sender_number,
                "Maaf, tidak ditemukan data dokumen yang relevan untuk pertanyaan Anda."
            )
            return

        # 2. Gather multi-page context and smart primary page selection
        context_blocks = []
        primary_idx = 0
        for rank_idx, doc in enumerate(results):
            doc_path = doc["file_path"]
            base_name = os.path.basename(doc_path)
            doc_details = db.get_document_by_path(doc_path)
            if doc_details and doc_details.get("full_transcription"):
                transcription = doc_details["full_transcription"]
                context_blocks.append(f"--- [PAGE {rank_idx+1}: {base_name}] ---\n{transcription[:1200]}")
                if primary_idx == 0 and any(kw in transcription.lower() for kw in ["permohonan pencairan", "kode rekening", "rekapitulasi", "belanja bahan bakar"]):
                    primary_idx = rank_idx

        multi_page_context = "\n\n".join(context_blocks)
        target_page = results[primary_idx]["file_path"]
        print(f"[WhatsApp Bot] 📄 Primary document selected: {os.path.basename(target_page)}")

        # 3. Generate answer via Qwen2.5-VL
        answer = reader.answer_question(
            image_path=target_page,
            question=question,
            stream=False,
            context_text=multi_page_context
        )
        print(f"[WhatsApp Bot] 💡 Answer generated:\n{answer}")

        # 4. Send reply back to user with source attribution
        final_reply = (
            f"{answer}\n\n"
            f"─────────────────────\n"
            f"📄 *Halaman Referensi Utama:* `{os.path.basename(target_page)}`\n"
            f"🤖 _Dianalisis oleh Qwen2.5-VL Multi-Page Visual RAG_"
        )
        send_res = client.send_text_message(sender_number, final_reply)
        print(f"[WhatsApp Bot] 📤 Send response status: {send_res}")
    except Exception as e:
        import traceback
        print(f"[WhatsApp Bot Error] ❌ Exception: {e}")
        traceback.print_exc()

def process_incoming_media(sender_number: str, media_id: str, filename: str, mime_type: str, caption: str, message_id: str):
    """
    Background worker to download media from Meta Graph API and trigger IngestionWorker.
    """
    try:
        client = get_whatsapp_client()
        client.mark_as_read(message_id)

        # Notify user that download has begun
        client.send_text_message(
            sender_number,
            f"📥 *Menerima Dokumen:*\n`{filename}`\n\nSedang mengunduh dan menyiapkan proses ekstraksi visual..."
        )

        data_dir = os.path.abspath("./data")
        os.makedirs(data_dir, exist_ok=True)
        save_path = os.path.join(data_dir, filename)

        print(f"[WhatsApp Ingestion] 📥 Downloading media {media_id} to {save_path}...")
        download_success = client.download_media(media_id, save_path)

        if not download_success or not os.path.exists(save_path):
            client.send_text_message(
                sender_number,
                f"❌ Gagal mengunduh file `{filename}` dari server WhatsApp. Mohon coba kirim ulang."
            )
            return

        print(f"[WhatsApp Ingestion] 🚀 Passing {save_path} to IngestionWorker...")
        worker = get_ingestion_worker()
        worker.ingest_file(save_path, recipient_number=sender_number)

    except Exception as e:
        import traceback
        print(f"[WhatsApp Ingestion Error] ❌ Exception: {e}")
        traceback.print_exc()

@app.get("/webhook")
@app.get("/rag-documentvisual/webhook")
async def whatsapp_verify(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token")
):
    """Verification handshake required by Meta Cloud API."""
    client = get_whatsapp_client()
    verified_challenge = client.verify_challenge(hub_mode, hub_verify_token, hub_challenge)
    if verified_challenge:
        return Response(content=verified_challenge, media_type="text/plain")
    return Response(content="Verification failed", status_code=403)

@app.post("/webhook")
@app.post("/rag-documentvisual/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    """Inbound webhook receiver for WhatsApp messages."""
    payload = await request.json()
    client = get_whatsapp_client()
    parsed = client.parse_incoming_message(payload)
    
    if not parsed or not parsed.get("sender"):
        return {"status": "ok"}

    sender = parsed["sender"]
    msg_id = parsed["msg_id"]
    msg_type = parsed.get("type")

    # Skenario 1: Pengguna mengirim dokumen (PDF) atau gambar
    if msg_type in ["document", "image"] and parsed.get("media_id"):
        background_tasks.add_task(
            process_incoming_media,
            sender,
            parsed["media_id"],
            parsed.get("filename") or f"doc_{msg_id}.pdf",
            parsed.get("mime_type", ""),
            parsed.get("body", ""),
            msg_id
        )
    # Skenario 2: Pengguna mengirim pesan teks (Tanya Jawab / Cari / Salam)
    elif parsed.get("body"):
        background_tasks.add_task(
            process_and_reply_whatsapp,
            sender,
            parsed["body"],
            msg_id
        )
        
    return {"status": "ok"}
