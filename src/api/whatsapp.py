import os
import json
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "docuvisual_secret_token_2026")
GRAPH_API_VERSION = "v20.0"

class WhatsAppClient:
    """
    Modular client interface for Meta WhatsApp Cloud API.
    Handles webhook validation, incoming payload parsing, and outbound messaging.
    """
    def __init__(
        self,
        token: str = WHATSAPP_TOKEN,
        phone_number_id: str = PHONE_NUMBER_ID,
        verify_token: str = VERIFY_TOKEN
    ):
        self.token = token
        self.phone_number_id = phone_number_id
        self.verify_token = verify_token
        self.base_url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{self.phone_number_id}/messages"

    def verify_challenge(self, mode: Optional[str], token: Optional[str], challenge: Optional[str]) -> Optional[str]:
        """
        Validates the webhook challenge issued during Meta Webhook subscription.
        Returns the challenge string if successful, else None.
        """
        if mode == "subscribe" and token == self.verify_token:
            return challenge
        return None

    def parse_incoming_message(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parses inbound message data from the Meta webhook payload.
        Returns dict with sender, message_id, type, and body.
        """
        try:
            entry = payload.get("entry", [])[0]
            changes = entry.get("changes", [])[0]
            value = changes.get("value", {})
            messages = value.get("messages", [])

            if not messages:
                return None

            msg = messages[0]
            sender_id = msg.get("from")
            msg_type = msg.get("type")
            msg_id = msg.get("id")

            result = {
                "sender": sender_id,
                "msg_id": msg_id,
                "type": msg_type,
                "body": None,
                "media_id": None
            }

            if msg_type == "text":
                result["body"] = msg.get("text", {}).get("body", "").strip()
            elif msg_type in ["image", "document"]:
                media_obj = msg.get(msg_type, {})
                result["media_id"] = media_obj.get("id")
                result["mime_type"] = media_obj.get("mime_type", "")
                result["filename"] = media_obj.get("filename") or f"doc_{msg_id}.pdf"
                result["body"] = media_obj.get("caption", "").strip()
            elif msg_type in ["audio", "voice"]:
                media_obj = msg.get(msg_type, {})
                result["media_id"] = media_obj.get("id")
                result["mime_type"] = media_obj.get("mime_type", "audio/ogg")
                result["filename"] = f"voice_{msg_id}.ogg"
                result["body"] = None  # Audio will be transcribed by Gemini Multimodal

            return result
        except (IndexError, KeyError, TypeError):
            return None

    def get_media_url(self, media_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the temporary media download URL and metadata from Meta Graph API.
        """
        url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{media_id}"
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    def download_media(self, media_id: str, destination_path: str) -> bool:
        """
        Downloads media binary content given its media_id and saves it locally.
        """
        media_info = self.get_media_url(media_id)
        if not media_info or "url" not in media_info:
            return False

        download_url = media_info["url"]
        headers = {
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "curl/7.68.0"
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)
            with requests.get(download_url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(destination_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"[WhatsAppClient Error] Failed to download media: {e}")
            return False

    def upload_media(self, file_path: str, mime_type: str = "image/jpeg") -> Optional[str]:
        """
        Uploads a local media file to Meta Graph API and returns its media_id.
        """
        if not os.path.exists(file_path):
            return None

        url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{self.phone_number_id}/media"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        try:
            with open(file_path, "rb") as f:
                files = {
                    "file": (os.path.basename(file_path), f, mime_type),
                    "messaging_product": (None, "whatsapp")
                }
                resp = requests.post(url, headers=headers, files=files, timeout=30)
                if resp.status_code in [200, 201]:
                    return resp.json().get("id")
                else:
                    print(f"[WhatsAppClient Error] Upload media failed ({resp.status_code}): {resp.text}")
                    return None
        except Exception as e:
            print(f"[WhatsAppClient Error] Upload exception: {e}")
            return None

    def send_image_message(self, recipient_number: str, image_path: str, caption: str = "") -> Dict[str, Any]:
        """
        Sends an image to recipient via Meta WhatsApp Cloud API:
        1. Uploads image to get media_id
        2. Sends image message with caption
        Falls back to send_text_message if upload fails.
        """
        # Compress / ensure JPEG MIME
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        
        media_id = self.upload_media(image_path, mime_type=mime)
        if not media_id:
            print("[WhatsAppClient] ⚠️ Media upload failed, falling back to text message...")
            fallback_text = f"{caption}\n\n[Lampiran visual: {os.path.basename(image_path)}]" if caption else f"[Lampiran visual: {os.path.basename(image_path)}]"
            return self.send_text_message(recipient_number, fallback_text)

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        data = {
            "messaging_product": "whatsapp",
            "to": recipient_number,
            "type": "image",
            "image": {
                "id": media_id,
                "caption": caption[:1024] if caption else ""
            }
        }
        resp = requests.post(self.base_url, headers=headers, json=data, timeout=15)
        return resp.json()

    def send_text_message(self, recipient_number: str, message: str) -> Dict[str, Any]:
        """
        Sends an outbound free-form text message to the recipient.
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        data = {
            "messaging_product": "whatsapp",
            "to": recipient_number,
            "type": "text",
            "text": {"body": message}
        }
        resp = requests.post(self.base_url, headers=headers, json=data, timeout=15)
        return resp.json()

    def mark_as_read(self, message_id: str) -> bool:
        """
        Marks an incoming message as read (shows blue double checkmark).
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        data = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        }
        try:
            resp = requests.post(self.base_url, headers=headers, json=data, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
