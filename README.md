# Document Visual-RAG & WhatsApp Bot (Cloud AI & Local GPU)

An enterprise-ready Visual-RAG (Retrieval-Augmented Generation) pipeline and conversational WhatsApp Bot for scanned documents, financial reports, and receipts. Features **Dual AI Engines** (Cloud Google Gemini or Local Qwen2.5-VL), paired with hybrid retrieval (**SQLite FTS5 + ChromaDB**) and multi-modal WhatsApp integration.

---

## ⚡ Dual AI Engine Modes

Switch between modes anytime by setting `AI_MODE` in your `.env` file:

### 1. Mode 1: Cloud AI (Google Gemini Flash)
Set `AI_MODE=cloud` in `.env`

- **Does NOT use a local GPU**: Requires zero CUDA, zero PyTorch, and zero VRAM.
- **Model**: Google Gemini Flash (via Cloud REST API).
- **RAM Footprint**: Under **100 MB** of system RAM.
- **Latency**: **1.0 – 2.0 seconds** per query.
- **Extras**: Transcribes WhatsApp voice notes directly and can be hosted on cheap cloud VPS or Shared Hosting (cPanel).

### 2. Mode 2: Local GPU (Qwen Model)
Set `AI_MODE=local` in `.env`

- **Uses Local GPU & Qwen**: Runs **`Qwen/Qwen2.5-VL-3B-Instruct`** in native `torch.bfloat16`.
- **Model Pre-download**: Run `python scripts/download_models.py` to cache models locally.
- **Hardware Needed**: NVIDIA RTX GPU with at least **8 GB VRAM**.
- **Privacy**: 100% offline on-premise execution with zero external network calls or cloud API costs.

### Comparison Table

| Feature | ☁️ Cloud AI Mode (`AI_MODE=cloud`) | 🖥️ Local GPU Mode (`AI_MODE=local`) |
| :--- | :--- | :--- |
| **Model** | **Google Gemini Flash** (2.5 / 1.5 Flash via REST API) | **Qwen2.5-VL-3B-Instruct** (`bfloat16`) |
| **Hardware** | **Zero GPU / CUDA / PyTorch required** | NVIDIA RTX GPU (8 GB+ VRAM recommended) |
| **RAM Footprint** | **< 100 MB RAM** (ultra-lightweight) | ~8 GB VRAM + 4 GB System RAM |
| **Response Latency** | **1.0 – 2.5 seconds** per query | ~9 – 35 seconds per page extraction |
| **WhatsApp Voice Notes** | Direct audio transcription via Gemini Multimodal | Requires local Whisper (or cloud fallback) |
| **Deployment Target** | Shared Hosting (cPanel), Low-spec VPS, Railway, Local | Dedicated Workstation, GPU Cloud (RunPod, Lambda) |
| **Data Privacy** | Cloud API (processed via Google AI Studio) | 100% On-Premise, zero external network calls |

---

## 📱 WhatsApp Bot Capabilities (Meta WhatsApp Cloud API)

The built-in FastAPI webhook (`/webhook`) turns your RAG pipeline into an interactive WhatsApp assistant:

1. **Visual Grounding & Proof-of-Answer**: 
   When answering questions (e.g., *"Berapa total belanja BBM di kwitansi?"*), the bot replies with the precise textual answer **and immediately sends the relevant receipt/page crop as an image proof**.
2. **In-Chat Document Ingestion**: 
   Upload PDF files or receipt photos directly into the WhatsApp chat. The bot automatically downloads, renders, pre-processes (CLAHE contrast), extracts text/tables, and indexes them into SQLite & ChromaDB in the background.
3. **Voice Note (Audio) Q&A**: 
   Send WhatsApp voice notes; the system transcribes the speech via Gemini Cloud and routes the query directly through the RAG engine.
4. **Built-in Quick Commands**:
   - `cari: <kata kunci>` or `/cari <keyword>`: Sub-second lexical & semantic search.
   - `/status`: Real-time report of indexed documents and vector count.
   - `/clear database`: Cleans and resets the database and image storage.

---

## ⚙️ Environment Configuration (`.env`)

Create or update your `.env` file in the project root:

```dotenv
# ========================================================
# 1. AI Engine Selection ('cloud' or 'local')
# ========================================================
AI_MODE=cloud

# ========================================================
# 2. Cloud AI Credentials (Required if AI_MODE=cloud)
# ========================================================
# Free API key from Google AI Studio: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# ========================================================
# 3. Meta WhatsApp Cloud API (Required for WhatsApp Bot)
# ========================================================
WHATSAPP_TOKEN=your_permanent_meta_system_user_token
PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_VERIFY_TOKEN=docuvisual_secret_token_2026

# ========================================================
# 4. Optional: Hugging Face Token (for downloading Qwen)
# ========================================================
HF_TOKEN=

# ========================================================
# 5. Server Bind Address
# ========================================================
API_HOST=0.0.0.0
API_PORT=8000
```

---

## 🚀 How to Run: Step-by-Step Guides

### Track 1: ☁️ Running in Cloud AI Mode (Zero GPU)

Ideal for shared hosting, cloud VPS, low-spec laptops, or lightweight servers.

1. **Create and activate virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install lightweight dependencies** (no PyTorch / CUDA, < 50 MB total):
   ```powershell
   pip install -r requirements-cloud.txt
   ```

3. **Configure `.env`**:
   Ensure `AI_MODE=cloud` and paste your `GEMINI_API_KEY`.

4. **Index documents via Cloud AI**:
   ```powershell
   python main.py --index --input_dir ./data/
   ```

5. **Ask questions via CLI**:
   ```powershell
   python main.py --ask "Berapa total belanja di kwitansi?"
   ```

6. **Start the WhatsApp Bot & REST API**:
   ```powershell
   python -m uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --reload
   ```

---

### Track 2: 🖥️ Running in Local GPU Mode (Qwen Model)

Ideal for workstations with NVIDIA RTX GPUs needing 100% offline privacy.

1. **Create and activate virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install GPU dependencies** (PyTorch with CUDA support):
   ```powershell
   pip install -r requirements.txt
   ```
   *(For newer RTX 50-series / Blackwell GPUs needing CUDA 12.8+: `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128`)*

3. **Pre-download Qwen & Embedding Models**:
   Run the dedicated download script to fetch and verify the models (~6.5 GB):
   ```powershell
   python scripts/download_models.py
   ```
   *(This downloads `Qwen/Qwen2.5-VL-3B-Instruct` and `intfloat/multilingual-e5-small` into local cache).*

4. **Configure `.env`**:
   Set:
   ```dotenv
   AI_MODE=local
   ```

5. **Index documents using local GPU**:
   ```powershell
   python main.py --index --input_dir ./data/
   ```

6. **Ask questions or start interactive chat (VRAM persistent)**:
   ```powershell
   # Single query:
   python main.py --ask "What is the total expenditure?"

   # Interactive chat session (model stays loaded in VRAM):
   python main.py --chat
   ```

7. **Start the WhatsApp Bot & REST API using Local GPU**:
   ```powershell
   python -m uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## 🌐 Connecting the WhatsApp Bot (Meta Webhook Setup)

1. **Start the FastAPI server**:
   ```powershell
   python -m uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Expose Localhost with Cloudflare Tunnel**:
   Meta requires an `https://` webhook endpoint. Run the included `cloudflared.exe` in a separate terminal:
   ```powershell
   .\cloudflared.exe tunnel --url http://localhost:8000
   ```
   *(Copy the generated HTTPS URL, e.g., `https://example-subdomain.trycloudflare.com`)*

3. **Configure Meta Developer Dashboard**:
   - Go to **Meta for Developers** -> Your App -> **WhatsApp** -> **Configuration**.
   - **Callback URL**: `https://<your-tunnel-url>.trycloudflare.com/webhook`
   - **Verify Token**: Must match `WHATSAPP_VERIFY_TOKEN` in your `.env`.
   - Click **Verify and Save**.
   - Under **Webhook Fields**, click **Subscribe** on the **`messages`** event.

---

## 📊 Benchmark (Local GPU: 41-Page Scanned PDF Batch)

**Setup**:
- **Workload**: 41-page PDF containing mixed financial tables, SPBU receipts, collages, and official forms.
- **GPU**: NVIDIA RTX (16 GB VRAM)
- **OS**: Windows 10, PyTorch (CUDA build, native SDPA)
- **Model**: `Qwen/Qwen2.5-VL-3B-Instruct` (`bfloat16`)

### Summary

| Metric | Result | Notes |
| :--- | :---: | :--- |
| **Pages Processed** | 41 pages | 0 failures / crashes |
| **Total Run Time** | 1,443.3 s (~24 min) | Render + CLAHE + VLM extraction + DB indexing |
| **Average Latency** | ~34.3 s / page | Typical range: 9s (single receipt) to 54s (dense collage) |
| **Peak VRAM** | ~7.56 GB | Consistent memory footprint, leaves ~8.4 GB free on a 16 GB GPU |
| **VRAM Drift** | < 25 MB | No memory leaks across the entire batch |
| **Validation** | 100% verified | Line items and calculated totals matched |

<details>
<summary><b>Page-by-Page Latency & Memory Log</b></summary>

<br>

| Page # | File | Category | Status | Latency (s) | Peak VRAM |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **1** | `page_1.jpg` | TABLE_REPORT | VERIFIED | 52.81 s | 7,549.2 MB |
| **2** | `page_2.jpg` | TABLE_REPORT | VERIFIED | 19.82 s | 7,558.6 MB |
| **3** | `page_3.jpg` | TABLE_REPORT | VERIFIED | 51.45 s | 7,563.1 MB |
| **4** | `page_4.jpg` | RECEIPT_COLLAGE | VERIFIED | 53.12 s | 7,546.4 MB |
| **5** | `page_5.jpg` | RECEIPT_COLLAGE | VERIFIED | 13.50 s | 7,539.6 MB |
| **6** | `page_6.jpg` | RECEIPT_COLLAGE | VERIFIED | 31.20 s | 7,561.3 MB |
| **7** | `page_7.jpg` | RECEIPT_COLLAGE | VERIFIED | 53.97 s | 7,549.1 MB |
| **8** | `page_8.jpg` | RECEIPT_COLLAGE | VERIFIED | 33.46 s | 7,539.6 MB |
| **9** | `page_9.jpg` | RECEIPT_COLLAGE | VERIFIED | 53.81 s | 7,549.1 MB |
| **10** | `page_10.jpg` | RECEIPT_COLLAGE | VERIFIED | 53.59 s | 7,549.1 MB |
| **11** | `page_11.jpg` | RECEIPT_COLLAGE | VERIFIED | 24.54 s | 7,549.1 MB |
| **12** | `page_12.jpg` | RECEIPT_COLLAGE | VERIFIED | 16.63 s | 7,549.1 MB |
| **13** | `page_13.jpg` | RECEIPT_COLLAGE | VERIFIED | 53.63 s | 7,550.4 MB |
| **14** | `page_14.jpg` | RECEIPT_COLLAGE | VERIFIED | 22.75 s | 7,549.1 MB |
| **15** | `page_15.jpg` | RECEIPT_COLLAGE | VERIFIED | 53.66 s | 7,549.1 MB |
| **16** | `page_16.jpg` | RECEIPT_COLLAGE | VERIFIED | 15.72 s | 7,549.1 MB |
| **17** | `page_17.jpg` | RECEIPT_COLLAGE | VERIFIED | 15.63 s | 7,557.0 MB |
| **18** | `page_18.jpg` | RECEIPT_COLLAGE | VERIFIED | 46.90 s | 7,539.6 MB |
| **19** | `page_19.jpg` | RECEIPT_COLLAGE | VERIFIED | **9.39 s** | 7,550.4 MB |
| **20** | `page_20.jpg` | RECEIPT_COLLAGE | VERIFIED | 25.06 s | 7,539.6 MB |
| **21** | `page_21.jpg` | RECEIPT_COLLAGE | VERIFIED | 23.50 s | 7,559.8 MB |
| **22** | `page_22.jpg` | RECEIPT_COLLAGE | VERIFIED | 38.36 s | 7,539.6 MB |
| **23** | `page_23.jpg` | RECEIPT_COLLAGE | VERIFIED | 53.58 s | 7,539.6 MB |
| **24** | `page_24.jpg` | RECEIPT_COLLAGE | VERIFIED | 27.01 s | 7,538.4 MB |
| **25** | `page_25.jpg` | RECEIPT_COLLAGE | VERIFIED | 26.77 s | 7,539.6 MB |
| **26** | `page_26.jpg` | RECEIPT_COLLAGE | VERIFIED | 32.58 s | 7,546.4 MB |
| **27** | `page_27.jpg` | RECEIPT_COLLAGE | VERIFIED | 30.69 s | 7,549.1 MB |
| **28** | `page_28.jpg` | RECEIPT_COLLAGE | VERIFIED | 29.09 s | 7,543.0 MB |
| **29** | `page_29.jpg` | RECEIPT_COLLAGE | VERIFIED | 19.95 s | 7,549.2 MB |
| **30** | `page_30.jpg` | RECEIPT_COLLAGE | VERIFIED | 23.54 s | 7,539.6 MB |
| **31** | `page_31.jpg` | RECEIPT_COLLAGE | VERIFIED | 37.67 s | 7,539.6 MB |
| **32** | `page_32.jpg` | RECEIPT_COLLAGE | VERIFIED | 29.98 s | 7,546.4 MB |
| **33** | `page_33.jpg` | RECEIPT_COLLAGE | VERIFIED | 53.87 s | 7,539.6 MB |
| **34** | `page_34.jpg` | RECEIPT_COLLAGE | VERIFIED | 31.42 s | 7,549.1 MB |
| **35** | `page_35.jpg` | RECEIPT_COLLAGE | VERIFIED | 15.05 s | 7,549.1 MB |
| **36** | `page_36.jpg` | RECEIPT_COLLAGE | VERIFIED | 53.90 s | 7,546.4 MB |
| **37** | `page_37.jpg` | RECEIPT_COLLAGE | VERIFIED | 40.37 s | 7,549.1 MB |
| **38** | `page_38.jpg` | RECEIPT_COLLAGE | VERIFIED | 32.78 s | 7,539.6 MB |
| **39** | `page_39.jpg` | RECEIPT_COLLAGE | VERIFIED | 44.75 s | 7,539.6 MB |
| **40** | `page_40.jpg` | RECEIPT_COLLAGE | VERIFIED | 53.93 s | 7,539.6 MB |
| **41** | `page_41.jpg` | RECEIPT_COLLAGE | VERIFIED | 20.40 s | 7,539.6 MB |

</details>

### Architecture & Memory Notes
- **Bounded Latency**: Dynamic pixel limits (`min_pixels` / `max_pixels`) prevent token blowup on dense receipt collages. Peak per-page latency remained under 54s.
- **Stable VRAM Footprint**: VRAM stayed at **7.54–7.56 GB** (< 25 MB drift) across the entire 24-minute batch.
- **Data Integrity**: Extractions passed arithmetic validation against listed line items.

*(Note: In **Cloud AI Mode**, indexing and query answering take **1.0 – 2.0s per request** with **0 MB GPU VRAM** and < 100 MB system RAM).*

---

## 📁 Directory Structure

```text
RAG-DocumentVisual/
├── config.yaml            # Centralized configuration (models, resolutions, paths)
├── .env                   # Active environment variables
├── .env.example           # Environment variables template
├── requirements.txt       # Dependencies for Local GPU Mode (PyTorch, Transformers, CUDA)
├── requirements-cloud.txt # Lightweight dependencies for Cloud AI & Shared Hosting
├── passenger_wsgi.py      # cPanel Phusion Passenger WSGI entrypoint
├── cloudflared.exe        # Cloudflare Tunnel binary for instant HTTPS webhook exposure
├── main.py                # CLI entry point (index, search, ask, chat - supports both modes)
├── scripts/
│   ├── download_models.py # Pre-downloads Qwen2.5-VL and Multilingual-E5 for local mode
│   └── backfill_vectors.py# Backfill SQLite documents into ChromaDB
├── src/
│   ├── api/               # FastAPI REST endpoints & WhatsApp Bot integration
│   │   ├── routes.py      # REST routes & WhatsApp webhook handlers
│   │   ├── whatsapp.py    # Modular WhatsAppClient (handshake, parser, media, sender)
│   │   └── queue_processor.py # Background task processing
│   ├── llm/               # Document Vision-Language Model interfaces
│   │   ├── gemini_client.py # Cloud Google Gemini Flash client (Zero GPU, Multimodal)
│   │   └── llm_client.py   # Local Qwen2.5-VL GPU BF16 client
│   ├── ingestion/         # PDF rendering & CLAHE contrast enhancement
│   │   ├── loader.py      # Page rendering (Poppler / pypdfium2) & CLAHE filter
│   │   └── worker.py      # Background IngestionWorker for WhatsApp uploads
│   ├── retrieval/         # Hybrid Search with Reciprocal Rank Fusion (RRF)
│   │   └── retriever.py
│   ├── vectordb/          # ChromaDB vector store & SQLite FTS5 database
│   │   ├── sqlite_db.py   # SQLite storage & full-text search index
│   │   └── vector_store.py# ChromaDB dense vector store
│   ├── embeddings/        # multilingual-e5-small CPU embedding engine
│   ├── chunking/          # Page and text chunking logic
│   ├── prompts/           # Structured extraction & QA prompt templates
│   └── utils/             # Pydantic schemas, image helpers, arithmetic validators
├── storage/               # Databases, page caches, and rendered artifacts
│   ├── doc_archive.db     # SQLite database with FTS5 search index
│   ├── chroma/            # ChromaDB vector collection
│   ├── pages/             # Rendered PDF page images
│   ├── enhanced_pages/    # CLAHE contrast-enhanced images
│   └── outputtext/        # Plain-text exports for all pages
└── data/                  # Raw input PDFs and scans
```

---

## 📜 License & Citation

Designed for intelligent document processing, financial auditing, and automated receipt verification. Free to adapt and use for personal or commercial projects.
