import os
import json
import re
import base64
import requests
from typing import Optional, Dict, Any
from src.utils.helpers import StructuredQAResponse, parse_first_numeric
from src.prompts.prompt_templates import DOCUMENT_EXTRACTION_PROMPT, format_qa_prompt, format_multi_page_qa_prompt, format_structured_qa_prompt

class GeminiVisualReader:
    """
    Cloud AI Vision Reader using Google Gemini 2.0 / 1.5 Flash via REST API.
    Zero local GPU/CUDA/Torch required.
    RAM usage: < 15 MB.
    Latency: ~1.2 - 2.0 seconds.
    Ideal for Shared Hosting (cPanel), Cloud VPS, and low-spec environments.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        # Allow override via GEMINI_MODEL env var or default to stable models
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.candidate_models = [self.model, "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-flash-latest"]

    def _encode_image(self, image_path: str) -> Dict[str, str]:
        """Encodes local image into base64 for Gemini REST payload."""
        ext = os.path.splitext(image_path)[1].lower().replace(".", "")
        mime = f"image/{ext}" if ext in ["jpeg", "jpg", "png", "webp"] else "image/jpeg"
        if mime == "image/jpg":
            mime = "image/jpeg"

        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        return {
            "inline_data": {
                "mime_type": mime,
                "data": b64_data
            }
        }

    def _call_gemini_api(self, prompt: str, image_path: Optional[str] = None) -> str:
        """Executes HTTP request to Gemini REST API with model fallback."""
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set GEMINI_API_KEY in your .env file."
            )

        parts = []
        if image_path and os.path.exists(image_path):
            parts.append(self._encode_image(image_path))
        parts.append({"text": prompt})

        payload = {
            "contents": [
                {
                    "parts": parts
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048
            }
        }

        headers = {"Content-Type": "application/json"}
        last_err = None

        # Try candidate models in order with exponential backoff on 503 / 429
        models_to_try = list(dict.fromkeys(self.candidate_models))
        for m in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={self.api_key}"
            for attempt in range(3):
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=40)
                    if response.status_code == 200:
                        self.model = m  # Keep the working model
                        res_json = response.json()
                        candidates = res_json.get("candidates", [])
                        if candidates:
                            content = candidates[0].get("content", {})
                            parts_resp = content.get("parts", [])
                            if parts_resp:
                                return parts_resp[0].get("text", "").strip()
                        return ""
                    elif response.status_code in [503, 429]:
                        # Transient high-load or rate limit from Google; wait and retry
                        wait_sec = (attempt + 1) * 2
                        print(f"[GeminiClient] Model '{m}' returned {response.status_code}. Retrying in {wait_sec}s (attempt {attempt+1}/3)...")
                        import time
                        time.sleep(wait_sec)
                        continue
                    else:
                        last_err = f"Gemini API Error {response.status_code} on {m}: {response.text}"
                        print(f"[GeminiClient] Model '{m}' returned {response.status_code}, trying next model...")
                        break
                except Exception as req_err:
                    last_err = str(req_err)
                    break

        raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")

    def _clean_and_parse_json(self, raw_output: str) -> dict:
        """Robust parser for JSON output."""
        cleaned = re.sub(r'```json\s*', '', raw_output, flags=re.IGNORECASE)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

        return {
            "doc_category": "UNKNOWN",
            "title_or_subject": "Unknown Document",
            "full_transcription": raw_output,
            "structured_data": {},
            "has_stamps_or_signatures": False,
            "raw_output": raw_output
        }

    def process_document(self, image_path: str) -> dict:
        """Extract structured data and full transcription via Gemini."""
        raw_text = self._call_gemini_api(DOCUMENT_EXTRACTION_PROMPT, image_path=image_path)
        return self._clean_and_parse_json(raw_text)

    def answer_question(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 1024,
        stream: bool = False,
        context_text: str = ""
    ) -> str:
        """Answers visual question with optional cross-page context."""
        if context_text:
            prompt = format_multi_page_qa_prompt(question, context_text)
        else:
            prompt = format_qa_prompt(question)

        return self._call_gemini_api(prompt, image_path=image_path)

    def answer_question_structured(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 1024
    ) -> StructuredQAResponse:
        """Answers visual question enforcing structured Pydantic schema."""
        prompt = format_structured_qa_prompt(question)
        raw_text = self._call_gemini_api(prompt, image_path=image_path)

        data = self._clean_and_parse_json(raw_text)
        if data.get("direct_answer"):
            num_val = data.get("numeric_value")
            if num_val is None:
                num_val = parse_first_numeric(data.get("direct_answer", ""))

            return StructuredQAResponse(
                direct_answer=str(data.get("direct_answer", "")),
                numeric_value=num_val,
                currency=data.get("currency"),
                source_citation=data.get("source_citation"),
                breakdown=data.get("breakdown", []),
                confidence=data.get("confidence", "HIGH"),
                raw_answer=raw_text
            )

        num_val = parse_first_numeric(raw_text)
        return StructuredQAResponse(
            direct_answer=raw_text,
            numeric_value=num_val,
            currency="IDR" if any(w in raw_text.lower() for w in ["rp", "rupiah"]) else None,
            source_citation="Gemini Visual Reader",
            breakdown=[],
            confidence="MEDIUM",
            raw_answer=raw_text
        )
