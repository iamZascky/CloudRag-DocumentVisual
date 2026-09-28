# Local Document Visual-RAG (Windows)

A local Visual-RAG pipeline for scanned documents, receipts, and tabular reports. Runs `Qwen/Qwen2.5-VL-3B-Instruct` locally in native `torch.bfloat16` on NVIDIA RTX GPUs, paired with hybrid retrieval (SQLite FTS5 + ChromaDB) on CPU.

## Architecture & Features
- **Hybrid Search (RRF)**: Combines SQLite FTS5 lexical search (exact invoice codes, reference numbers) and ChromaDB dense vectors (`intfloat/multilingual-e5-small`) using Reciprocal Rank Fusion ($k = 60$).
- **Multilingual Retrieval**: Handles queries and document text in English, Indonesian, and mixed terminology out of the box.
- **CPU Embedding Engine**: Embeddings run on CPU using `sentence-transformers`, leaving GPU VRAM dedicated to the vision-language model.
- **Isolated Local Storage**: Model checkpoints, SQLite databases, ChromaDB collections, and rendered pages stay inside `./storage/`.
- **Pre-processing Pipeline**: Automatically renders PDFs via Poppler, runs contour cropping, and applies CLAHE contrast enhancement for faint receipts.
- **Validation**: Pydantic schema validation checks extracted line items and totals against transcribed values.

---

## Benchmark (41-Page Scanned PDF Batch)

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
| :---: | :--- | :--- | :---: | :---: | :---: |
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

### 2. Architecture Notes

- **Bounded Latency**: Dynamic pixel limits (`min_pixels` / `max_pixels`) prevent token blowup on dense receipt collages. Peak per-page latency remained under 54s.
- **Stable VRAM Footprint**: VRAM stayed at **7.54–7.56 GB** (< 25 MB drift) across the entire 24-minute batch.
- **Data Integrity**: Extractions passed arithmetic validation against listed line items.

---

## Setup

1. **Virtual Environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
   *(For RTX 50-series / Blackwell GPUs needing CUDA 12.8+: `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128`)*

3. **Poppler (Windows PDF Rendering)**
   - Download Poppler binaries for Windows.
   - Extract to `./tools/poppler/` or ensure `pdftoppm.exe` is in your system `PATH`.

---

## Usage

### 1. Indexing
Place PDFs or scan images into `./data/`:
```powershell
python main.py --index --input_dir ./data/
```
Renders PDF pages, applies CLAHE contrast, extracts structured JSON via Qwen2.5-VL, writes `.txt` dumps into `./storage/outputtext/`, and indexes text into SQLite FTS5 + ChromaDB.

### 2. Hybrid Search
```powershell
# Hybrid (FTS5 keyword + Vector E5 semantic via Reciprocal Rank Fusion):
python main.py --search "operational and maintenance expenses"

# Exact reference matching (e.g. invoice codes, dates, account numbers):
python main.py --search "INV-2024-001" --search_mode fts

# Semantic concept matching:
python main.py --search "annual hardware procurement costs" --search_mode vector
```

### 3. Visual Question Answering
```powershell
# Standard streaming answer in terminal:
python main.py --ask "What is the total expenditure stated in the document?"

# Enforce Pydantic-validated Structured JSON Output:
python main.py --ask "What is the total expenditure stated in the document?" --json
```
Retrieves the top candidate page and runs visual inference. Using `--json` guarantees structured fields (`direct_answer`, `numeric_value`, `currency`, `source_citation`, `breakdown`, `confidence`).

### 4. Interactive Chat (Persistent VRAM)
```powershell
python main.py --chat
```
Keeps Qwen2.5-VL and the embedding model resident in memory to avoid reloading overhead between queries.

### 5. Running the REST API & WhatsApp Bot Server (FastAPI)

The project includes an enterprise-ready WhatsApp Bot powered by Meta WhatsApp Cloud API with multi-page visual RAG.

#### A. Configure Credentials (`.env`)
Create a `.env` file based on `.env.example`:
```ini
WHATSAPP_TOKEN=your_permanent_system_user_token_here
PHONE_NUMBER_ID=your_phone_number_id_here
WHATSAPP_VERIFY_TOKEN=your_custom_secret_verify_token
```

> [!TIP]
> **Permanent Token (Anti-Expire):** Do not use the temporary 24-hour token from the "API Setup" tab. Instead:
> 1. Go to **Business Settings** (`business.facebook.com/settings`) -> **Users** -> **System Users**.
> 2. Create an **Admin System User** (or use existing).
> 3. Assign your WhatsApp App asset with Full Control.
> 4. Click **Generate New Token**, select your App, set expiration to **Never**, and check `whatsapp_business_messaging` and `whatsapp_business_management`.
> 5. Paste this permanent token into `WHATSAPP_TOKEN` in `.env`.

#### B. Start the Webhook Server
```powershell
uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --reload
```

#### C. Expose via Public HTTPS Tunnel
Meta requires a valid `https://` webhook endpoint. Run Cloudflare Tunnel (or ngrok) in a separate terminal:
```powershell
.\cloudflared.exe tunnel --url http://localhost:8000
```
*(Or if installed globally: `cloudflared tunnel --url http://localhost:8000`)*

#### D. Connect Webhook in Meta Developer Dashboard
1. Go to your Meta App Dashboard -> **WhatsApp** -> **Configuration**.
2. Set **Callback URL**: `https://<your-tunnel-url>.trycloudflare.com/webhook`
3. Set **Verify Token**: Enter the same secret token you set in `WHATSAPP_VERIFY_TOKEN` (in `.env`).
4. Click **Verify and Save**.
5. Under **Webhook Fields**, click **Subscribe** on the **`messages`** event.

#### E. Receiving Messages from Other Users (Production / Multi-User)
- **In Development Mode:** Only numbers explicitly registered under **API Setup -> To (Manage Phone Number List)** can send and receive messages.
- **For Any User (Live Public Bot):**
  1. Toggle the App Mode in the top navigation bar from **In Development** to **Live**.
  2. (Meta requires adding a basic Privacy Policy URL in **App Settings -> Basic** to go Live).
  3. Once Live, any WhatsApp user worldwide can message your bot number and receive responses automatically!

#### F. WhatsApp Bot Capabilities & Commands
Once running, users can interact with your bot directly on WhatsApp:
- **Live Document Ingestion (PDF / Images)**: 
  Send any PDF document or scan image directly to the WhatsApp chat! The bot's background **`IngestionWorker`** will:
  1. Download the file from Meta Cloud API to `./data/`.
  2. Send progress messages to the user.
  3. Render pages with Poppler and enhance contrast with CLAHE.
  4. Perform OCR & structured field extraction via Qwen2.5-VL.
  5. Index all pages into SQLite FTS5 and ChromaDB vector store.
  6. Notify user once ready for searching and asking questions!
- **Fast Search Mode (< 1s)**: Send `/cari <keyword>` or `cari: <keyword>` (e.g. `cari: SPBU Balung Lor`). Returns instant lexical + vector matches with relevance scores and snippets.
- **AI Visual QA Mode**: Ask any free-form question (e.g. `Berapa total belanja bbm?`). Automatically triggers Top-7 multi-page hybrid context retrieval, smart table prioritization, and Qwen2.5-VL vision-language inference with instant acknowledgement.

---

## Directory Structure
```text
rag-DocumentVisual/
├── config.yaml          # Centralized configuration (models, resolutions, paths)
├── .env.example         # Environment variables template
├── requirements.txt     # Python dependencies (Local GPU mode)
├── requirements-cloud.txt # Lightweight Python dependencies for Shared Hosting
├── passenger_wsgi.py    # cPanel Phusion Passenger WSGI entrypoint
├── update.md            # Production RAG architecture & real-life strategy notes
├── recommendation.md    # Shared hosting vs LangGraph deployment analysis
├── main.py              # CLI entry point (index, search, ask, chat)
├── src/
│   ├── api/             # FastAPI REST endpoints & Meta WhatsApp Cloud client
│   │   ├── routes.py    # REST routes & WhatsApp webhook handlers
│   │   └── whatsapp.py  # Modular WhatsAppClient (handshake, parser, sender)
│   ├── ingestion/       # PDF rendering & CLAHE contrast enhancement
│   │   ├── loader.py    # Page rendering (Poppler / pypdfium2 fallback) & CLAHE
│   │   └── worker.py    # Background IngestionWorker for WhatsApp PDF upload
│   ├── chunking/        # Page and text chunking logic
│   │   └── chunker.py
│   ├── embeddings/      # multilingual-e5-small CPU embedding engine
│   │   └── embedder.py
│   ├── vectordb/        # ChromaDB vector store & SQLite FTS5 database
│   │   ├── vector_store.py
│   │   └── sqlite_db.py
│   ├── retrieval/       # Hybrid Search with Reciprocal Rank Fusion (RRF)
│   │   └── retriever.py
│   ├── prompts/         # Structured extraction & QA prompt templates
│   │   └── prompt_templates.py
│   ├── llm/             # Document Vision-Language Model interfaces
│   │   ├── llm_client.py     # Local Qwen2.5-VL GPU BF16 client
│   │   └── gemini_client.py  # Cloud Google Gemini Flash REST client (Zero GPU)
│   └── utils/           # Configuration loader & Pydantic arithmetic validation
│       └── helpers.py
├── scripts/
│   └── backfill_vectors.py  # Backfill SQLite documents into ChromaDB
├── tests/
│   └── test_app.py          # Modular architecture verification test suite
├── logs/                    # Application and daemon logs
├── data/                    # Raw input PDFs and scans
└── storage/                 # Project-isolated models, databases, and dumps
    ├── models/              # Local Hugging Face & PyTorch weights cache
    ├── doc_archive.db       # SQLite database with FTS5 search index
    ├── chroma/              # ChromaDB vector database index
    ├── pages/               # Rendered uncompressed PDF page images
    ├── enhanced_pages/      # Preprocessed, contrast-enhanced scan images
    ├── outputtext/          # Clean .txt plain-text exports for all pages
    └── result.json          # Latest query response output
```
asdas