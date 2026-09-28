# Rekomendasi Arsitektur & Analisis Deployment: Shared Hosting vs LangGraph / Heavy Frameworks

Dokumen ini mendokumentasikan analisis teknis kelayakan arsitektur, perbandingan framework orkestrasi (seperti **LangGraph / LangChain / LangQuarry**), dan panduan strategis untuk menjalankan sistem **Visual-RAG & WhatsApp Bot** pada infrastruktur **Shared Hosting (cPanel / Cloud Hosting tanpa Docker)**.

---

## 1. Karakteristik & Batasan Ketat Shared Hosting

Lingkungan Shared Hosting (contoh: cPanel Python App di Niagahoster, Hostinger, DomaiNesia, dll.) memiliki batasan ketat (*hard limits*) yang dikontrol oleh kernel CloudLinux LVE:

| Parameter | Batasan Tipikal Shared Hosting | Dampak pada Sistem RAG |
|---|---|---|
| **Alokasi RAM** | 512 MB – 1.024 MB (Max 2 GB) | Library berat langsung memicu *Out of Memory* (OOM) |
| **Batas CPU** | 1 Core (Throttling jika 100% > 3 detik) | Proses OCR lokal/kompilasi berat akan dibunuh (*killed*) |
| **Entry Processes** | Maksimal 20 – 30 proses simultan | Multi-agent looping akan menghabiskan slot proses |
| **Kompilasi C / Rust** | Tidak ada akses `root / sudo` | Pemasangan library dengan *native bindings* rumit sering gagal |
| **Storage / IOPS** | Terbatas (I/O Throttling) | Bobot file model AI lokal (> 5 GB) tidak dapat disimpan |

---

## 2. Mengapa LangGraph / Heavy Frameworks TIDAK Cocok di Shared Hosting?

Framework seperti **LangGraph / LangChain** sangat populer di lingkungan enterprise dengan server mandiri (VPS/Dedicated dengan Docker). Namun, jika dipaksakan ke Shared Hosting, timbul masalah-masalah kritis berikut:

### A. Pembengkakan Dependensi (*Dependency Bloat*)
- Menginstal `langgraph` mewajibkan instalasi dependensi berantai: `langchain`, `langchain-core`, `langsmith`, `pydantic-v2`, `tenacity`, `dataclasses-json`, dll.
- Direktori `.venv` membengkak hingga ratusan megabyte.
- **Memory Footprint**: Perintah `import langgraph` saja dapat mengonsumsi **150 MB – 250 MB RAM** sebelum server melayani permintaan pertama. Hal ini menyisakan sedikit ruang untuk FastAPI dan buffer koneksi.

### B. Latensi Multi-Step & Risiko *Gateway Timeout*
- LangGraph beroperasi dengan siklus graf berulang (*Looping: Generate ➔ Critique ➔ Rewrite Query ➔ Regenerate*).
- Setiap putaran loop membutuhkan 1 panggilan HTTP keluar ke API LLM.
- Jika terjadi 3–4 siklus panggilan, latensi total mencapai **15 – 30 detik**. Server web Shared Hosting (seperti LiteSpeed/Nginx reverse proxy) biasanya menerapkan batas *504 Gateway Timeout* pada 15–30 detik.

### C. Kompleksitas Debugging & Risiko Kernel Kill
- Jika memori melebihi alokasi paket hosting saat membangun StateGraph besar, server web langsung mengembalikan error:  
  `508 Resource Limit Is Reached` atau proses Python langsung di-`kill -9` oleh sistem tanpa meninggalkan stack trace log.

---

## 3. Solusi Terbaik: "Zero-Dependency Lightweight Agentic Pattern" (Pure Python)

Solusi paling optimal untuk Shared Hosting adalah **mengadopsi konsep/logikanya (Routing & Query Expansion) tanpa mengimpor library-nya**:

| Aspek | LangGraph / LangChain | Pure Python Pattern (Arsitektur Kita) |
|---|---|---|
| **Penggunaan RAM** | ~300 MB – 500 MB (Tinggi) | **< 60 MB – 90 MB** (Sangat aman di paket 512MB RAM) |
| **Waktu Booting Server** | 3.0 – 5.0 detik | **< 0.4 detik** (Instant startup) |
| **Kecepatan Respon (Latency)** | 10 – 25 detik (Multi-step loop) | **1.0 – 2.5 detik** (Single-shot via Gemini Flash API) |
| **Instalasi cPanel** | Rawan gagal kompilasi | **100% mulus** (`fastapi`, `uvicorn`, `requests`, `sqlite3`) |
| **Arsitektur Database** | Butuh Vector DB terpisah | **SQLite FTS5 + BM25** (Native bawaan Python, zero RAM) |

---

## 4. Implementasi Konsep "LangGraph" Versi Ringan (Pure Python)

Untuk mendapatkan kemampuan adaptif seperti LangGraph tanpa membebani Shared Hosting:

### 1. Simple Router (Pengganti Conditional Edge LangGraph)
Gunakan fungsi Python deterministik sederhana:
```python
def route_query(user_message: str) -> str:
    clean = user_message.lower().strip()
    if clean in ["halo", "menu", "bantuan"]:
        return "GREETING"
    elif clean.startswith(("/cari", "cari:")):
        return "SEARCH_FAST"
    elif any(kw in clean for kw in ["total", "berapa", "rincian", "kwitansi"]):
        return "DEEP_QA"
    return "GENERAL_QA"
```

### 2. Query Rewriting / Expansion Sederhana
Alih-alih agen LLM bertingkat, gunakan kamus sinonim domain (*lexical thesaurus*) atau 1 prompt terpadu:
```python
# Menangani kueri singkat seperti "jumlah bbm"
SYNONYM_MAP = {
    "bbm": "bahan bakar minyak solar pertalite dexlite spbu",
    "pencairan": "permohonan pencairan spj rekapitulasi",
    "makan": "konsumsi jamuan rapat konsumsi makan minum"
}
```

### 3. Single-Call Vision RAG (Cloud API)
Di Shared Hosting, ganti model Qwen2.5-VL lokal dengan **Google Gemini 2.0 Flash / 1.5 Flash**:
- Mengirim gambar kwitansi + pertanyaan ke endpoint API Google.
- Kuota gratis melimpah (15 RPM / 1.500 RPD gratis).
- Eksekusi selesai dalam **1.2 detik** dengan pemakaian RAM server hosting < 10 MB.

---

## 5. Ringkasan Rekomendasi Eksekutif

1. **JANGAN gunakan library LangGraph / LangChain pada Shared Hosting cPanel.**
2. **Pertahankan arsitektur modular Pure Python** (`FastAPI` + `SQLite FTS5` + `WhatsAppClient`).
3. **Aktifkan `AI_MODE=cloud`** (menggunakan Gemini Flash API) ketika mendeploy ke Shared Hosting untuk menghilangkan kebutuhan GPU lokal dan menghemat RAM.
4. **Gunakan LangGraph HANYA JIKA**:
   - Sistem dideploy pada VPS / Dedicated Server / Kubernetes mandiri dengan RAM minimal 8 GB.
   - Bot WhatsApp harus mengeksekusi multi-tool terintegrasi (misal: otomatis update SAP, kirim email dinas, dan manipulasi API perbankan).
