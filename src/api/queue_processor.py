import os
import sys
import glob
import json
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "../..")))

from src.api.whatsapp import WhatsAppClient
from src.api.routes import process_and_reply_whatsapp, process_incoming_media, get_whatsapp_client

def process_queue():
    queue_dir = os.path.abspath("./storage/queue")
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
