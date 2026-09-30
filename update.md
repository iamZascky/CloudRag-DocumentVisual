# Arsitektur & Strategi Solusi Real-Life Production RAG

Catatan ini merangkum perbandingan antara solusi saat ini (Top-k Multi-Page Context) dan arsitektur standar industri (*enterprise-grade*) untuk menangani dokumen keuangan, arsip dinas, dan struk berskala ratusan hingga ribuan halaman.

---

## 1. Analisis Kasus (Current Case Study)

### Pertanyaan User
> `"jumlah bbm yang dibeli"`

### Perilaku Sistem
- **Top-1 Halaman**: Sistem menangkap halaman struk eceran SPBU (contoh: Page 38) dengan total `134.000 Rupiah` karena skor kemiripan kata kunci (*keyword match*) pada struk satuan sangat dominan.
- **Tabel Rekapitulasi (Page 2)**: Berisi *"Belanja Bahan Bakar dan Pelumas - Permohonan Pencairan"* dengan grand total `Rp 45.562.550`. Pada pencarian naif kata kunci pendek, lembar rekap sempat berada di peringkat bawah (Rank 6).

---

## 2. Solusi Jangka Pendek (Current Implementation)

1. **Top-7 Multi-Page Retrieval**:
   - Mengambil 7 kandidat teratas sekaligus menggunakan kombinasi **SQLite FTS5 + ChromaDB Vector (Multilingual-E5) + RRF (Reciprocal Rank Fusion)**.
   - Tetap cepat (< 10 ms di database, penambahan latensi GPU < 0.5s karena hanya 1 gambar visual utama dan teks halaman pendukung).
2. **Rekapitulasi Heuristics**:
   - Sistem secara cerdas mendeteksi kata kunci permohonan/rekap/kode rekening di antara 7 kandidat teratas dan memprioritaskannya sebagai acuan visual utama.
3. **Cross-Page Context Injection**:
   - Memberikan transkripsi halaman-halaman relevan ke Vision LLM (Qwen2.5-VL) sehingga AI memahami gambaran besar (grand total) sekaligus rincian struk.

---

## 3. Solusi Praktis di Dunia Nyata / Enterprise Production

Untuk skala dokumen ribuan halaman (misal 500 halaman laporan keuangan/SPJ), Top-7 tidak selalu cukup karena lembar rekapitulasi bisa terlempar ke peringkat puluhan. Industri AI Engineering menerapkan **3 pilar utama**:

### 1. Hierarchical / Multi-Level Indexing (Paling Fundamental)
Dokumen tidak diperlakukan datar (*flat*), melainkan berstruktur hierarkis:
- **Tingkat 1 (Document Level / Executive Summary)**:
  - Halaman rekapitulasi, surat pengantar permohonan, atau lembar disposisi diberi label/metadata khusus: `doc_type: "SUMMARY_REPORT"`.
- **Tingkat 2 (Evidence Level / Lampiran Transaksi)**:
  - Struk eceran, nota kecil, foto fisik diberi metadata: `doc_type: "RECEIPT_ITEM"`.
- **Routing Logis**:
  - Saat user bertanya *"Berapa total anggaran/belanja..."*, router sistem otomatis memprioritaskan pencarian di `doc_type: "SUMMARY_REPORT"`.

### 2. Query Rewriting / Expansion (Mengubah Pertanyaan User)
User manusia (khususnya melalui WhatsApp / Chatbot) sering mengetik pesan singkat dan ambigu seperti:
```text
"jumlah bbm"
```
Di sistem produksi, sebelum dilempar ke database pencari, terdapat modul **Query Rewriter** (LLM ringan/cepat, latensi ~50ms):
- **Query asli**: `"jumlah bbm yang dibeli"`
- **Query yang di-expand oleh sistem**: `"total rekapitulasi belanja bahan bakar minyak permohonan pencairan SPBU volume liter harga"`
- Hasilnya, halaman rekapitulasi langsung melompat ke **Rank 1**.

### 3. Two-Stage Retrieval with Cross-Encoder Reranker
Setelah mengambil 20–50 kandidat kasar dari database (FTS5 + Vector), sistem melewatkannya ke model **Reranker** (seperti `bge-reranker-v2-m3`):
- Model reranker membaca pasangan *(pertanyaan user, konten halaman)* secara bersamaan (*deep cross-attention*).
- Reranker memahami maksud tersirat: *"User bertanya kata 'jumlah/total', halaman 2 adalah tabel penjumlahan, maka halaman 2 harus dipaksa ke nomor 1"*.

---

## 4. Key Takeaways untuk Portofolio & LinkedIn

> *"In this project, I addressed the classic multi-page RAG retrieval challenge (recapitulation vs. itemized receipts) by implementing Hybrid RRF Fusion and Cross-Page Context Injection. In future enterprise iterations, this can be further augmented with Hierarchical Indexing, Query Expansion, and Two-Stage Cross-Encoder Reranking."*

---

## 5. Analisis Performa & Fitur Pengembangan Lanjutan (Production Roadmap)

Berikut adalah daftar optimasi performa dan fitur baru yang dirancang untuk meningkatkan skalabilitas dan pengalaman pengguna (khususnya integrasi WhatsApp Bot dan migrasi ke Cloud/Shared Hosting):

### A. Optimasi Performa & Stabilitas VRAM (Performance & Reliability)

1. **Dual AI Mode: Local GPU vs Cloud AI Fallback (`AI_MODE=local|cloud`)**
   - **Kondisi Saat Ini**: Bot 100% bergantung pada inferensi lokal Qwen2.5-VL di GPU lokal (butuh VRAM ~7–8 GB).
   - **Rencana Peningkatan**: Menambahkan flag `AI_MODE=cloud` (Google Gemini 1.5/2.0 Flash atau OpenAI Vision API).
   - **Manfaat**:
     - Memungkinkan deploy server ke **Shared Hosting / cPanel biasa** (tanpa GPU dedicated dan tanpa Docker).
     - Menghilangkan latensi GPU lokal, inferensi cloud hanya ~1–2 detik per query.

2. **Async Concurrency Lock untuk GPU (VRAM OOM Protection)**
   - **Kondisi Saat Ini**: Jika beberapa pengguna WhatsApp mengirim pertanyaan atau dokumen secara bersamaan, beberapa background task dapat memicu inferensi model visi paralel yang berpotensi menyebabkan *CUDA Out of Memory (OOM)*.
   - **Rencana Peningkatan**: Mengimplementasikan `asyncio.Semaphore(1)` atau `threading.Lock()` khusus pada pipeline GPU.
   - **Manfaat**: Bot menangani antrean dengan anggun (*graceful queueing*) dan dapat mengirim pesan tunggu status posisi antrean ke pengguna WhatsApp.

3. **Fast-Path Page Pre-Filtering untuk Dokumen Besar (> 10 Halaman)**
   - **Kondisi Saat Ini**: `IngestionWorker` memproses setiap halaman PDF satu per satu ke Vision LLM. Pada dokumen 30+ halaman, proses bisa memakan waktu 2–3 menit.
   - **Rencana Peningkatan**: Ekstraksi teks cepat menggunakan `PyMuPDF / fitz` di awal.
     - Halaman teks digital murni langsung diindeks ke SQLite & Vector (< 0.05 detik).
     - Hanya halaman dengan gambar, stempel, tabel rumit, atau tulisan tangan yang dialirkan ke Qwen2.5-VL.
   - **Manfaat**: Mempercepat proses indexing dokumen hingga **10x lebih cepat**.

---

### B. Fitur Fungsional & User Experience (UX & Features)

4. **Multi-Turn Conversation Memory (WhatsApp Chat History)**
   - **Kondisi Saat Ini**: Setiap pesan WhatsApp bersifat *stateless* (tidak mengingat pertanyaan sebelumnya).
   - **Rencana Peningkatan**: Menyimpan 3–5 interaksi terakhir per nomor telepon di SQLite (`session_history`).
   - **Manfaat**: Pengguna bisa berdiskusi interaktif secara bertahap (contoh: *"Berapa totalnya?"* lalu disusul *"Tolong rincikan nomor kwitansinya"*).

5. **Pengiriman Foto/Gambar Halaman Bukti ke WhatsApp (`send_image_message`)**
   - **Kondisi Saat Ini**: Bot hanya membalas dengan teks jawaban dan menyebutkan nama file referensi.
   - **Rencana Peningkatan**: Menggunakan WhatsApp Cloud API Media Messages (`type: image`) untuk mengirim gambar halaman kwitansi fisik / tabel rekap yang relevan langsung ke chat WhatsApp.
   - **Manfaat**: Pengguna mendapatkan bukti visual otentik seketika tanpa harus membuka arsip manual.

6. **Filter Pencarian per Dokumen (`doc_filter`)**
   - **Kondisi Saat Ini**: Perintah `/cari <keyword>` menelusuri seluruh basis data dokumen.
   - **Rencana Peningkatan**: Menambahkan sintaks filter dokumen, contoh: `/cari SPBU doc:BBM_September.pdf`.
   - **Manfaat**: Akurasi pencarian meningkat drastis ketika basis data telah menampung ribuan dokumen dari berbagai divisi atau periode.

---

### C. Matriks Prioritas Implementasi

| Prioritas | Fitur | Kategori | Alasan Utama |
|---|---|---|---|
| **P1** | **VRAM Concurrency Lock** | Reliability | Mencegah crash GPU saat banyak pengguna WA aktif serentak |
| **P2** | **Cloud AI Fallback (Gemini Flash)** | Portability | Fondasi deploy ke shared hosting murah tanpa GPU |
| **P3** | **Kirim Gambar Bukti ke WhatsApp** | User Experience | Nilai jual utama dari *Visual RAG* adalah verifikasi visual langsung |
| **P4** | **Fast-Path Page Pre-Filtering** | Performance | Mempercepat upload PDF tebal (> 20 halaman) |
| **P5** | **Multi-Turn Chat History** | UX | Interaksi natural seperti ChatGPT di WhatsApp |

---

## 6. Arsitektur Lanjutan: LangGraph + Gemini AI untuk Agentic Visual RAG

### A. Apakah LangGraph Menggantikan Gemini AI?
**Sama sekali TIDAK.**
- **Google Gemini 2.5 Flash**: Berfungsi sebagai **Model / Otak Visi (Vision & Reasoning Engine)** yang membaca gambar kwitansi, transkripsi tabel, dan mengekstraksi data.
- **LangGraph**: Berfungsi sebagai **Orkestrator Alur Kerja (Workflow & Decision State Machine)** yang mengatur kapan Gemini dipanggil, kapan harus melakukan verifikasi mandiri (*self-correction*), dan kapan harus membaca halaman rekapitulasi.

LangGraph terintegrasi secara native dengan Gemini melalui paket resmi `langchain-google-genai`.

---

### B. Keamanan Implementasi di Shared Hosting (cPanel / CloudLinux)
LangGraph **100% AMAN** dijalankan di shared hosting dengan batasan ketat CloudLinux (`max 40 nproc`):
1. **Pure Python**: LangGraph murni logika graph state berbasis Python standard (tanpa C++ compilation, tanpa dependensi PyTorch/CUDA).
2. **Single-Thread Execution**: Berjalan di dalam 1 proses tunggal (`nproc: 1 / 40`, CPU 0%–2.5%).
3. **Ukuran Ringan**: Hanya membutuhkan library modular:
   ```bash
   pip install langgraph langchain-core langchain-google-genai
   ```
   *(Menghindari instalasi paket monolitik `langchain` yang berat).*

---

### C. Manfaat Agentic Visual RAG dengan LangGraph

Dibandingkan dengan rantai sekuensial linear biasa, LangGraph mengubah sistem menjadi alur adaptif:

```
[Inbound WA Message]
        │
        ▼
   <Router Node> ──── (Smalltalk / Help) ────► [Kirim Jawaban Singkat]
        │
  (Document QA)
        ▼
[Hybrid Retrieval Node] ──► (FTS5 + Vector Cosine + RRF)
        │
        ▼
<Evaluator Node>
  ├─ Data Tidak Lengkap? ──► [Fallback Query Expansion] ──┐
  │                                                        │ (loop)
  └─ Data Siap                                             ▼
        │                                         [Re-retrieve Docs]
        ▼
[Gemini Vision Analysis Node]
        │
        ▼
<Self-Correction / Verification Node>
  ├─ Angka tidak sinkron dengan cover nota dinas? ──► [Paksa Muat Page 1] ──┐
  │                                                                         │ (loop)
  └─ Angka Valid & Lengkap                                                  ▼
        │                                                           [Re-evaluasi Gemini]
        ▼
[Format & Kirim Reply ke WhatsApp]
```

### D. Solusi Masalah Nyata: "Halaman 1 (Cover Rekap) vs Halaman 4 (Struk Eceran)"
Pada arsitektur linear, jika retriever salah memilih Halaman 4 sebagai acuan visual utama, sistem akan menghasilkan subtotal parsial. 
Dengan **LangGraph**:
1. Node verifikasi mendeteksi bahwa user meminta *"total pencairan belanja"*, tetapi halaman acuan adalah struk eceran.
2. Graph secara otomatis melompat ke branch pemuatan Halaman 1 (*Nota Dinas / Rekapitulasi*) tanpa perlu hardcode kaku.
3. Menjamin jawaban grand total (contoh: `Rp 45.562.550`) selalu akurat dan terverifikasi sebelum dikirimkan ke WhatsApp pengguna.

