import os
import sys

# Limit OpenBLAS / OpenCV / NumPy / OpenMP threads to 1 to stay safely within cPanel nproc limit (40)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENCV_FORBID_ONLY_PTHREADS_USAGE"] = "1"
os.environ["OPENCV_CPU_MAX_THREADS"] = "1"
os.environ["AI_MODE"] = os.getenv("AI_MODE", "cloud")

import glob
import json
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "../..")))

from dotenv import load_dotenv
load_dotenv(os.path.join(CURRENT_DIR, "../../.env"))

from src.api.whatsapp import WhatsAppClient
from src.api.routes import process_and_reply_whatsapp, process_incoming_media, get_whatsapp_client

def process_queue():
    project_root = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
    queue_dir = os.path.join(project_root, "storage", "queue")
    if not os.path.exists(queue_dir):
        return

    # Process all incoming payload files in order
    files = sorted(glob.glob(os.path.join(queue_dir, "*.json")))
    if not files:
        return

    client = get_whatsapp_client()

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            parsed = client.parse_incoming_message(payload)
            if parsed and parsed.get("sender"):
                sender = parsed["sender"]
                msg_id = parsed["msg_id"]
                msg_type = parsed.get("type")

                print(f"[QueueProcessor] 📩 Processing {msg_type} from {sender} (ID: {msg_id})...")

                if msg_type in ["document", "image"] and parsed.get("media_id"):
                    process_incoming_media(
                        sender,
                        parsed["media_id"],
                        parsed.get("filename") or f"doc_{msg_id}.pdf",
                        parsed.get("mime_type", ""),
                        parsed.get("body", ""),
                        msg_id
                    )
                elif parsed.get("body"):
                    process_and_reply_whatsapp(
                        sender,
                        parsed["body"],
                        msg_id
                    )
        except Exception as e:
            print(f"[QueueProcessor Error] {e}")
        finally:
            try:
                os.remove(file_path)
            except Exception:
                pass

if __name__ == "__main__":
    process_queue()
