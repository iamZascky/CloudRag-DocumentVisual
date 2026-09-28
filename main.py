import argparse
import os
import json
import time
import torch
from src.ingestion.loader import render_pdf, enhance_scan
from src.vectordb.sqlite_db import SqliteDocDatabase
from src.vectordb.vector_store import ChromaVectorStore
from src.retrieval.retriever import HybridRetriever
from src.llm.llm_client import QwenVisualReader
from src.utils.helpers import validate_extraction

db = SqliteDocDatabase()

def index_documents(input_dir: str):
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # Initialize the reader and vector store lazily
    reader = QwenVisualReader()
    vector_store = ChromaVectorStore()
    
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if not os.path.isfile(file_path):
            continue
            
        print(f"Processing: {file_path}")
        ext = os.path.splitext(filename)[1].lower()
        
        image_paths = []
        if ext == '.pdf':
            print("Rendering PDF...")
            image_paths = render_pdf(file_path, dpi=300)
        elif ext in ['.jpg', '.jpeg', '.png', '.tiff']:
            image_paths = [file_path]
        else:
            print(f"Skipping unsupported file type: {filename}")
            continue
            
        for img_path in image_paths:
            enhanced_path = enhance_scan(img_path)
            
            # Check if already indexed in database
            if db.is_indexed(enhanced_path):
                print(f"Page already indexed ({enhanced_path}), skipping...")
                continue
                
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            
            print(f"Processing: {enhanced_path}...")
            result = reader.process_document(enhanced_path)
            
            torch.cuda.synchronize()
            latency = time.time() - t0
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            doc_category = result.get("doc_category", "OTHER")
            structured_data = result.get("structured_data", {})
            
            # Validation
            validation_status = validate_extraction(doc_category, structured_data)
            structured_data["validation_status"] = validation_status
            
            print(f"Validation: {validation_status} | Latency: {latency:.2f}s | Peak VRAM: {peak_vram:.1f}MB")
            
            db.save_document(
                file_path=enhanced_path,
                doc_category=doc_category,
                title=result.get("title_or_subject", "Unknown Title"),
                full_text=result.get("full_transcription", ""),
                structured_json=json.dumps(structured_data, ensure_ascii=False),
                has_visuals=result.get("has_stamps_or_signatures", False)
            )
            print(f"Saved {enhanced_path} to database.")

            # Sync to ChromaDB vector store
            vector_store.upsert_page(
                file_path=enhanced_path,
                text=result.get("full_transcription", ""),
                category=doc_category,
                title=result.get("title_or_subject", "Unknown Title")
            )
            print(f"Indexed {enhanced_path} into ChromaDB vector store.")

            # Save readable text output to ./storage/outputtext/
            output_text_dir = os.path.abspath("./storage/outputtext")
            os.makedirs(output_text_dir, exist_ok=True)
            txt_filename = os.path.splitext(os.path.basename(enhanced_path))[0] + ".txt"
            txt_path = os.path.join(output_text_dir, txt_filename)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"=== DOCUMENT METADATA ===\n")
                f.write(f"Source Page: {enhanced_path}\n")
                f.write(f"Category: {doc_category}\n")
                f.write(f"Title / Subject: {result.get('title_or_subject', 'Unknown Title')}\n")
                f.write(f"Has Stamps / Signatures: {result.get('has_stamps_or_signatures', False)}\n")
                f.write(f"Validation: {validation_status}\n\n")
                f.write(f"=== STRUCTURED DATA (JSON) ===\n")
                f.write(json.dumps(structured_data, indent=2, ensure_ascii=False))
                f.write(f"\n\n=== FULL TRANSCRIPTION ===\n")
                f.write(result.get("full_transcription", ""))
            print(f"Saved text dump: {txt_path}")

def search(query: str, mode: str = "hybrid", vector_store=None):
    print(f"Searching for: '{query}' (Mode: {mode.upper()})...")
    
    if mode == "fts":
        results = db.search_text(query, limit=5)
        if not results:
            print("No matches found.")
            return
        for r in results:
            print(f"\n[{r['doc_category']}] {r['file_path']}")
            print(f"Highlight: {r['highlight']}")
            
    elif mode == "vector":
        if vector_store is None:
            vector_store = ChromaVectorStore()
        results = vector_store.search(query, limit=5)
        if not results:
            print("No matches found.")
            return
        for idx, r in enumerate(results, start=1):
            print(f"\n[Rank {idx} | Sim: {r['similarity']:.4f}] [{r['doc_category']}] {r['file_path']}")
            print(f"Snippet: {r['snippet'][:200]}...")
            
    else:  # Hybrid RRF (default)
        if vector_store is None:
            vector_store = ChromaVectorStore()
        retriever = HybridRetriever(sqlite_db=db, vector_store=vector_store)
        results = retriever.retrieve(query, limit=5)
        if not results:
            print("No matches found.")
            return
        for r in results:
            fts_info = f"Rank {r['fts_rank']}" if r['fts_rank'] else "-"
            vec_info = f"Rank {r['vector_rank']}" if r['vector_rank'] else "-"
            if r['vector_similarity'] is not None:
                vec_info += f" (Sim: {r['vector_similarity']:.4f})"
                
            print(f"\n[Rank {r['final_rank']} | RRF Score: {r['rrf_score']:.5f}] [{r['doc_category']}]")
            print(f"  File: {r['file_path']}")
            print(f"  Breakdown: FTS: {fts_info} | Vector: {vec_info}")
            if r.get('highlight'):
                print(f"  Highlight: {r['highlight'][:250]}...")

def answer_question(question: str, reader=None, vector_store=None, stream: bool = True, structured: bool = False):
    start_time = time.time()
    
    if not structured:
        print("\n" + "═" * 70)
        print(f" 🔍 RETRIEVAL: Hybrid Search (FTS5 + Vector E5 RRF)")
        print(f" ❓ Question: '{question}'")
        print("═" * 70)
    
    if vector_store is None:
        vector_store = ChromaVectorStore()
        
    retriever = HybridRetriever(sqlite_db=db, vector_store=vector_store)
    results = retriever.retrieve(question, limit=1)
    if not results:
        if structured:
            print(json.dumps({"error": "No relevant documents found"}, indent=2))
        else:
            print("❌ Could not find any relevant documents to answer the question.")
        return None
        
    top_candidate = results[0]
    target_page = top_candidate['file_path']
    fts_rank = top_candidate['fts_rank'] if top_candidate['fts_rank'] else "-"
    vec_rank = top_candidate['vector_rank'] if top_candidate['vector_rank'] else "-"
    sim_info = f" (Sim: {top_candidate['vector_similarity']:.4f})" if top_candidate.get('vector_similarity') is not None else ""
    
    if not structured:
        print(f" 📄 Source Doc : {os.path.basename(target_page)}")
        print(f" 🏷️  Category   : {top_candidate.get('doc_category', 'DOCUMENT')}")
        print(f" 🎯 Match Stats: RRF Score: {top_candidate['rrf_score']:.5f} | FTS: Rank {fts_rank} | Vector: Rank {vec_rank}{sim_info}")
        print("─" * 70)
        print(f" 💡 AI VISUAL RESPONSE:")
        print("─" * 70)
    
    if reader is None:
        reader = QwenVisualReader()
        
    if structured:
        structured_resp = reader.answer_question_structured(target_page, question)
        elapsed = time.time() - start_time
        
        result_data = {
            "question": question,
            "candidate_page": target_page,
            "doc_category": top_candidate.get('doc_category', 'DOCUMENT'),
            "rrf_score": top_candidate['rrf_score'],
            "fts_rank": top_candidate['fts_rank'],
            "vector_rank": top_candidate['vector_rank'],
            "vector_similarity": top_candidate.get('vector_similarity'),
            "structured_data": structured_resp.model_dump(),
            "latency_seconds": round(elapsed, 2)
        }
        print(json.dumps(result_data, indent=2, ensure_ascii=False))
        
        os.makedirs("./storage", exist_ok=True)
        with open("./storage/result.json", "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        return structured_resp
    else:
        output_text = reader.answer_question(target_page, question, stream=stream)
        elapsed = time.time() - start_time
        print("\n" + "─" * 70)
        print(f" ⏱️ Latency: {elapsed:.2f}s | Source Path: {target_page}")
        print("═" * 70)
        
        result_data = {
            "question": question,
            "candidate_page": target_page,
            "doc_category": top_candidate.get('doc_category', 'DOCUMENT'),
            "rrf_score": top_candidate['rrf_score'],
            "fts_rank": top_candidate['fts_rank'],
            "vector_rank": top_candidate['vector_rank'],
            "vector_similarity": top_candidate.get('vector_similarity'),
            "answer": output_text,
            "latency_seconds": round(elapsed, 2)
        }
        
        os.makedirs("./storage", exist_ok=True)
        with open("./storage/result.json", "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print("Saved answer to ./storage/result.json\n")
        return output_text

def ask(question: str, structured: bool = False):
    answer_question(question, structured=structured)


def chat_session():
    print("=" * 60)
    print("  Interactive Visual-RAG Chat Session (Hybrid Search Enabled)")
    print("=" * 60)
    print("Loading models (one-time GPU & Vector Store initialization)...")
    
    reader = QwenVisualReader()
    vector_store = ChromaVectorStore()
    
    print("\nModels loaded and ready! Ask questions about your indexed documents.")
    print("Commands: type 'exit', 'quit', or 'q' to end session.")
    print("-" * 60)
    
    while True:
        try:
            prompt = input("\n[Question] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session.")
            break
            
        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            print("Exiting chat session.")
            break
            
        answer_question(prompt, reader=reader, vector_store=vector_store)

def main():
    parser = argparse.ArgumentParser(description="Local Windows-Native General Document Visual-RAG")
    parser.add_argument("--index", action="store_true", help="Index documents in the specified directory")
    parser.add_argument("--input_dir", type=str, default="./data/", help="Directory containing PDFs/Images to index")
    parser.add_argument("--search", type=str, help="Search the indexed documents")
    parser.add_argument("--search_mode", type=str, default="hybrid", choices=["hybrid", "fts", "vector"], help="Search mode (hybrid, fts, vector)")
    parser.add_argument("--ask", type=str, help="Ask a one-off question about the documents")
    parser.add_argument("--json", action="store_true", help="Enforce structured JSON output (Pydantic validated)")
    parser.add_argument("--chat", action="store_true", help="Start an interactive chat session with model kept loaded in GPU memory")
    
    args = parser.parse_args()
    
    if args.index:
        index_documents(args.input_dir)
    elif args.search:
        search(args.search, mode=args.search_mode)
    elif args.ask:
        ask(args.ask, structured=args.json)
    elif args.chat:
        chat_session()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
