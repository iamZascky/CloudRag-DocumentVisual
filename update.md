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
