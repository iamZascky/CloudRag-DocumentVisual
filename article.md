# Engineering Visual-RAG on Sub-$5 Shared Hosting: How We Built a Production Document Assistant Under Extreme LVE Limits

*By [Your Name / Engineering Team] • 10 min read • AI Systems Architecture, Information Retrieval & Edge Engineering*

---

A common assumption in enterprise AI is that Retrieval-Augmented Generation (RAG) with visual documents demands dedicated compute. The standard recipe calls for Kubernetes pods, heavy vector databases running HNSW indexes, Docker containers mounting 24GB GPUs, and orchestrators like LangChain or LangGraph coordinating multi-step reflection loops.

That stack works when infrastructure budgets are unlimited. It collapses immediately when your target deployment is **cPanel on shared hosting**—running CloudLinux OS with hard LVE (Lightweight Virtual Environment) guardrails: **1 CPU core, 512 MB to 1 GB RAM, and a strict 40-process ceiling (`nproc <= 40`)**.

Over the past few months, we designed and deployed a full **Visual-RAG WhatsApp bot** capable of ingesting messy, multi-page financial dossiers (SPJ/Nota Dinas) containing official letters, mixed tabular summaries, and physical dot-matrix receipts. 

This article details the real engineering decisions, search methodologies, mathematical fusion formulas, failures, and performance optimizations required to achieve sub-2-second answers on sub-$5 shared hosting without crashing the kernel.

---

```
                       INBOUND WHATSAPP DOCUMENT / QUERY
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │ cPanel Cron Job Queue Processor (50s) │
                   │  - Atomic File Renaming (.processing) │
                   │  - Single Process (nproc: 1 / 40)     │
                   └───────────────────┬───────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │                                             │
      [Document Upload: PDF]                        [Interactive Query]
                │                                             │
                ▼                                             ▼
  ┌───────────────────────────┐                 ┌───────────────────────────┐
  │  Streaming Page Ingestion │                 │ Universal Query Expansion │
  │  (pypdfium2 Generator)    │                 │ (Pure Python / 0ms)       │
  │  RAM Footprint: ~30MB     │                 └─────────────┬─────────────┘
  └─────────────┬─────────────┘                               │
                ▼                                             ▼
  ┌───────────────────────────┐                 ┌───────────────────────────┐
  │ Hybrid Completeness Check │                 │ FTS5 Lexical Search First │
  │ (Digital Text vs Photos)  │                 │ Sub-1ms Retrieval Hits    │
  └──────┬─────────────┬──────┘                 └─────────────┬─────────────┘
         │             │                                      │
  [Pure Digital] [Mixed/Scans]                                ▼
         │             │                        ┌───────────────────────────┐
         │             ▼                        │ Universal Context Scoring │
         │   ┌───────────────────┐              │ (Grand Totals & Summary)  │
         │   │ Gemini 3.5 Flash  │              └─────────────┬─────────────┘
         │   │ 8192 Token OCR    │                            │
         │   └─────────┬─────────┘                            ▼
         ▼             ▼                        ┌───────────────────────────┐
  ┌───────────────────────────────┐             │ Downsampled Image Ground  │
  │ SQLite FTS5 + Document Store  │◄────────────│ (Lanczos 1400px, 350KB)   │
  └───────────────────────────────┘             └─────────────┬─────────────┘
                                                              ▼
                                                ┌───────────────────────────┐
                                                │ Direct WhatsApp Answer    │
                                                │ Latency: < 2.2 Seconds    │
                                                └───────────────────────────┘
```

---

## 1. The Core Dilemma: The "Executive Summary vs. Receipt" Trap

Most toy RAG tutorials use chunk-based similarity search. If a user asks:

> *"What is the total expenditure for fuel in this period?"*

A standard vector database indexes every paragraph independently. When evaluated against a 41-page PDF:
1. **The Trap**: 30 pages are individual fuel receipts from petrol stations (e.g., `Rp 134.000`, `Rp 250.000`). Keyword and vector similarity on words like *"BBM"*, *"Pertamina"*, and *"Beli"* heavily favors those individual transaction slips.
2. **The Truth**: Page 2 of the document contains the official *Rekapitulasi Permohonan Pencairan Dana* with the executive grand total: `Rp 45.562.550`.
3. **The Result**: A naïve Top-1 retriever serves a random receipt snippet. The model answers `$134.000`, hallucinating that this single receipt is the entire project's budget.

Simply increasing retrieval depth (`top_k = 7`) creates a secondary bottleneck: sending 7 uncompressed document images over HTTP creates a 25MB payload, triggering web timeouts or massive API bills.

---

## 2. Advanced Search & Retrieval Methodology: Deep Dive

To solve the trap above without melting low-spec hosting CPUs, we engineered a multi-tiered information retrieval pipeline combining lexical precision, semantic vector search, rank fusion, and structural heuristic reranking.

### A. Universal Query Expansion & Intent Enrichment (Zero-Latency)
Human queries via WhatsApp are notoriously terse:
```text
"jumlah bbm"
```
A keyword search for `"jumlah bbm"` will miss Page 2 because the official document actually reads: *"Rekapitulasi Belanja Bahan Bakar Minyak dan Pelumas - Permohonan Pencairan Dana"*.

Instead of spinning up an auxiliary LLM agent (which adds 800ms–1.5s latency), we designed a deterministic, pure-Python expansion engine that intercepts queries in **< 0.05 milliseconds**:

```python
# Pure-Python query expansion (helpers.py)
def expand_general_document_query(query: str) -> str:
    clean = query.lower().strip()
    expansions = []
    # Financial & Calculation Intent
    if any(k in clean for k in ["total", "jumlah", "biaya", "harga", "anggaran", "belanja", "pencairan"]):
        expansions.extend(["grand total", "rekapitulasi", "jumlah total", "permohonan pencairan", "subtotal"])
    # Administrative & Official Letters Intent
    if any(k in clean for k in ["surat", "nomor", "perihal", "lampiran", "dinas", "keputusan"]):
        expansions.extend(["nota dinas", "nomor surat", "lampiran", "perihal", "kepada yth"])
    ...
```
*Result*: The query `"jumlah bbm"` seamlessly becomes `"jumlah bbm grand total rekapitulasi jumlah total permohonan pencairan"`. Both Page 2 and Page 1 instantly surge into candidate consideration.

---

### B. Hybrid Retrieval Fusion: Lexical FTS5 + Pure-Python Vector Cosine
Rather than relying solely on keyword matching or solely on semantic vectors, we fuse two distinct retrieval engines:

1. **Lexical Path (SQLite FTS5)**:
   - Evaluates exact phrases and prefix token queries (`"rekapitulasi"* OR "bbm"*`).
   - Uses the BM25 probabilistic relevance algorithm natively built into SQLite.
   - Executes in **under 1 millisecond** with zero external RAM footprint.

2. **Semantic Path (PureVectorStore)**:
   - Computes cosine similarity between document vectors and query vectors.
   - Built with pure Python `math.sqrt` and dot products to avoid C++ threading explosions (`nproc > 40`) caused by ChromaDB or HNSWlib on CloudLinux.

3. **Reciprocal Rank Fusion (RRF)**:
   The ranked outputs of both searchers are fused using the mathematical formula:
   $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{FTS}, \text{Vector}\}} \frac{1}{k + \text{rank}_m(d)}$$
   Where $k = 60$ (smoothing constant). Documents appearing high in *both* keyword matches and semantic meaning receive the highest composite score.

4. **Lexical Fast-Path Circuit Breaker**:
   If the SQLite FTS5 search returns high-confidence lexical matches ($\ge \text{limit}$), the retriever **bypasses the CPU vector search entirely**. This drops retrieval time from 50 seconds (CPU sentence-transformers) to **0.99 ms**!

---

### C. Universal Document Priority Scoring (Heuristic Reranking)
After retrieving top-7 candidate pages, the system runs an audit-grade scoring pass to pick the single best **primary visual page** while compiling the rest into supporting context:

| Structural Term / Keyword in Page | Weight | Rationale |
|---|:---:|---|
| **Rekapitulasi / Permohonan Pencairan** | `+20 / +18` | Executive summary table holding the definitive grand total |
| **Grand Total / Total Pembayaran** | `+18 / +15` | Critical aggregation row |
| **Nota Dinas / Surat Keputusan / Surat Perjanjian** | `+18` | Official administrative cover letter / executive authorization |
| **Page 1 / Hal 1 Indicator** | `+15` | Document front page / cover letter priority |
| **Daftar Isi / Lembar Pengesahan** | `+10 / +12` | High-level index or official endorsement |

This guarantees that whether processing government SPJ, corporate invoices, legal contracts, or travel reimbursements, the executive summary page is crowned as the primary visual focus for Gemini Vision.

---

## 3. Resolving the 5 Critical Production Bottlenecks

### Bottleneck 1: The OOM Crash on Multi-Page Rendering
Standard libraries like `pdf2image` (Poppler) convert multi-page documents by loading every rendered page into RAM simultaneously. For a 41-page scan at 150 DPI, memory usage quickly spiked beyond 700 MB, causing the CloudLinux kernel to kill the process silently (`kill -9`).

**Implementation**:
We engineered a **streaming generator** using `pypdfium2`. The renderer compiles one page at a time, writes the compressed JPEG directly to disk, and explicitly closes the bitmap and page handles before touching the next:

```python
# Low-memory streaming PDF renderer (loader.py)
pdf = pdfium.PdfDocument(pdf_path)
scale = dpi / 72.0
for i in range(len(pdf)):
    page = pdf[i]
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    pil_image.save(page_path, 'JPEG', quality=90)
    pil_image.close()
    bitmap.close()
    page.close()
pdf.close()
```
*Result*: Memory consumption remains completely flat at **~30 MB**, whether the document has 2 pages or 200 pages.

---

### Bottleneck 2: High-Resolution Visual Payloads (10MB → 350KB)
When asking questions across multi-page receipts, the model requires visual grounding to distinguish handwritten annotations, official stamps, and cashier signatures. However, uploading 3 full-resolution scans caused 15–20 second transfer lags over hosting outbound bandwidth.

**Implementation**:
We built an in-flight downsampler into `GeminiVisualReader`. Using Lanczos resampling via Pillow, images exceeding 1400px along their longest edge are downscaled and re-compressed to JPEG quality 80%:

* **Raw Page Image**: 3.25 MB base64 payload.
* **Optimized Payload**: 336 KB (a **90.3% payload reduction**).
* **Processing Overhead**: ~170 ms.
* **Visual Fidelity**: Dot-matrix text, dates, and currency stamps remain sharp and unambiguous.

---

### Bottleneck 3: Truncated OCR Transcriptions
In initial tests, the output text files saved in `storage/outputtext/` contained only partial metadata and truncated lines. 

**Root Cause**: The vision model's extraction call was capped at `maxOutputTokens: 2048`. On dense administrative forms with dozens of itemized rows, the JSON output hit the ceiling and cut off the verbatim markdown transcription.

**Implementation**:
We decoupled the extraction budget from QA answering:
* Document ingestion extraction allows up to **8,192 tokens**, capturing complete verbatim transcriptions of receipts, account codes, and signatory blocks.
* Interactive chat QA operates at **2,048 tokens**, enforcing direct, focused responses back to the user.

---

### Bottleneck 4: The 5-Minute Rate Limit Freeze
Shared hosting environments using free-tier API keys frequently encounter transient HTTP 429 (Rate Limit) or 503 (Overloaded) statuses. The original client implemented an exponential retry backoff (15s, 30s, 45s, 60s). If two candidate models failed consecutively, a WhatsApp user was left waiting for over 4 minutes.

**Implementation**:
We introduced **asymmetric retry profiles**:
* **Ingestion Worker**: Retains patient retry loops with 2.5-second pacing delays between pages to process long batch documents reliably without exhausting quota.
* **Interactive WhatsApp QA**: Employs a strict **fail-fast budget** (`max_attempts=2`, `backoff_step=5s`). If a model experiences delays, it cascades immediately to the lighter fallback model (`gemini-3.5-flash-lite`) within 5 seconds.

---

### Bottleneck 5: Race Conditions in Cron Queue Processing
Because cPanel environments cannot maintain persistent daemon background workers due to process pruning, we implemented an autonomous queue processor triggered via cron every minute (running an internal 50-second poll loop).

When a 41-page PDF took 1.5 minutes to process, the subsequent cron trigger would read the same pending JSON payload and spawn a duplicate ingestion task—resulting in duplicated database records and confusing double replies on WhatsApp.

**Implementation**:
We implemented an **atomic file-system claim lock**:
```python
# queue_processor.py
claimed_path = file_path + ".processing"
try:
    os.rename(file_path, claimed_path)
except OSError:
    continue  # Already claimed by another worker instance
```
By utilizing POSIX atomic rename semantics, simultaneous cron workers gracefully skip tasks currently in flight.

---

## 4. End-to-End Performance Benchmarks

All benchmarks were conducted on a production cPanel shared hosting account (1 vCPU, 1 GB LVE RAM quota, CloudLinux kernel):

| Performance Dimension | Standard Heavy RAG (LangChain + Chroma) | Our Optimized Pure-Python Architecture | Gain / Optimization Factor |
|---|---|---|---|
| **RAM Footprint (Idle / Boot)** | ~350 MB | **~42 MB** | **88% Less RAM Usage** |
| **PDF Page Render (41 Pages)** | ~620 MB peak (Kernel OOM risk) | **Flat 32 MB** (Streaming pypdfium2) | **Safe on 512MB RAM Hosting** |
| **Retrieval Speed (Top-7)** | 480 ms – 1,200 ms | **0.99 ms – 2.0 ms** (FTS5 Fast-Path) | **200x Faster Candidate Fetch** |
| **Outbound Image Payload** | ~10 MB (3 raw pages) | **~700 KB** (Top-2 Lanczos Resized) | **93% Bandwidth Reduction** |
| **OCR Completeness** | Truncated at 2,048 tokens | **Complete at 8,192 tokens** | **100% Verbatim Capture** |
| **Chat QA Latency (End-to-End)** | 18.0s – 35.0s | **1.8s – 2.4s** | **~10x Faster WhatsApp Reply** |

---

## 5. Architectural Takeaways for Edge & Budget RAG

1. **Avoid Framework Dogmatism**: Frameworks like LangChain or LangGraph are phenomenal for rapid prototyping on high-spec cloud machines. But when engineering for resource-constrained environments, standard Python data structures and SQLite primitives deliver significantly higher uptime, lower latency, and absolute determinism.
2. **Lexical Fast-Path Outperforms Naïve Vectors for Structured Documents**: For receipts, official letters, and invoices, exact keyword matching (via FTS5 with prefix queries) regularly scores higher in relevance than semantic embeddings, while consuming zero GPU or CPU compute.
3. **Payload Optimization is Latency Optimization**: In multimodal RAG, network transfer often dwarfs LLM time-to-first-token. Spending 150 ms downsampling an image locally saves 5,000 ms of socket transmission time.
4. **Hierarchical Awareness Over Flat Chunking**: When dealing with multi-page financial archives, recognizing executive summaries versus transaction evidence is the single biggest factor in preventing hallucinations.

---

### Tech Stack Summary
* **Language & Runtime**: Python 3.12 (cPanel Passenger WSGI / Cron Worker)
* **Search & Indexing**: SQLite FTS5 (BM25) + Pure-Python Vector Cosine Engine + RRF Fusion
* **Vision & Extraction Engine**: Google Gemini 3.5 Flash & Flash-Lite (8192 Token Window)
* **PDF & Image Processing**: pypdfium2 (Streaming Generator) + Pillow (Lanczos Resampling)
* **Interface & Messaging**: FastAPI REST API + Meta WhatsApp Cloud API
