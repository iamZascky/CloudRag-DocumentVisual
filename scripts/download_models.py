import os
import argparse
import torch
from dotenv import load_dotenv

load_dotenv()

def download_models(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", embedding_model: str = "intfloat/multilingual-e5-small"):
    print("=" * 60)
    print("  Visual-RAG Local Models Downloader")
    print("=" * 60)

    # 1. Download Embedding Model
    print(f"\n[1/2] Downloading Embedding Model: '{embedding_model}'...")
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(embedding_model, device="cpu")
        print(f"✅ Successfully downloaded & cached embedding model: {embedding_model}")
    except Exception as e:
        print(f"❌ Failed to download embedding model: {e}")

    # 2. Download Qwen2.5-VL Vision-Language Model
    print(f"\n[2/2] Downloading Vision Model: '{model_id}'...")
    print("Note: This model is ~6.5 GB. Download speed depends on your internet connection.")
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        hf_token = os.getenv("HF_TOKEN", None)
        print("  - Downloading AutoProcessor...")
        AutoProcessor.from_pretrained(model_id, token=hf_token)
        print("  ✅ Processor downloaded.")

        print("  - Downloading Model Weights (BF16)...")
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            token=hf_token
        )
        print(f"✅ Successfully downloaded & verified {model_id}!")
    except Exception as e:
        print(f"❌ Failed to download Qwen model: {e}")
        print("Tip: If you encounter rate limits on Hugging Face, add HF_TOKEN=your_token in .env")

    print("\n" + "=" * 60)
    print("  Download Completed! You can now run local mode with AI_MODE=local")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-download Qwen and Embedding models for offline local GPU use.")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Hugging Face model ID for Vision LLM")
    parser.add_argument("--embed_id", default="intfloat/multilingual-e5-small", help="Hugging Face model ID for Embeddings")
    args = parser.parse_args()

    download_models(args.model_id, args.embed_id)
