import os
import json
import re
import base64
import requests
from typing import Optional, Dict, Any, List
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
        # Tested and verified working: gemini-3.5-flash (primary) and gemini-3.5-flash-lite (fallback)
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        self.candidate_models = [self.model, "gemini-3.5-flash", "gemini-3.5-flash-lite"]

    def _encode_image(self, image_path: str, max_dimension: int = 1400, quality: int = 80) -> Dict[str, str]:
        """
        Encodes and optimizes local image for Gemini REST payload:
        - Downsamples images exceeding max_dimension (preserves full text readability while cutting payload by 90%).
        - Re-compresses to JPEG quality 80% to ensure sub-second upload latency over cloud/cPanel network.
        """
        try:
            from PIL import Image
            import io
            with Image.open(image_path) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                
                w, h = img.size
                if max(w, h) > max_dimension:
                    scale = max_dimension / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                return {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64_data
                    }
                }
        except Exception:
            # Fallback to direct raw file read if PIL is unavailable
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

    def _call_gemini_api(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        max_output_tokens: int = 2048,
        max_attempts: int = 4,
        backoff_step: int = 15
    ) -> str:
        """Executes HTTP request to Gemini REST API supporting multiple image inputs and robust retry."""
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set GEMINI_API_KEY in your .env file."
            )

        parts = []
        # Support multiple images or single image
        targets = []
        if image_paths:
            targets.extend(image_paths)
        elif image_path:
            targets.append(image_path)

        for img in targets:
            if img and os.path.exists(img):
                parts.append(self._encode_image(img))

        parts.append({"text": prompt})

        payload = {
            "contents": [
                {
                    "parts": parts
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": max_output_tokens
            }
        }

        headers = {"Content-Type": "application/json"}
        last_err = None

        # Try candidate models in order with exponential backoff on 503 / 429
        models_to_try = list(dict.fromkeys(self.candidate_models))
        for m in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={self.api_key}"
            for attempt in range(max_attempts):
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=45)
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
                    elif response.status_code in [429, 503]:
                        # Rate limit (15 RPM) or temporary service overload from Google
                        last_err = f"Gemini API {response.status_code} on {m}"
                        if attempt + 1 >= max_attempts:
                            break  # move on to the next candidate model immediately
                        wait_sec = (attempt + 1) * backoff_step
                        print(f"[GeminiClient] Model '{m}' hit status {response.status_code} (Rate Limit). Backing off for {wait_sec}s (attempt {attempt+1}/{max_attempts})...")
                        import time
                        time.sleep(wait_sec)
                        continue
                    else:
                        last_err = f"Gemini API Error {response.status_code} on {m}: {response.text}"
                        print(f"[GeminiClient] Model '{m}' returned {response.status_code}, trying next model...")
                        break
                except Exception as req_err:
                    last_err = str(req_err)
                    import time
                    time.sleep(3)

        raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")

    def _clean_and_parse_json(self, raw_output: str) -> dict:
        """Robust parser for extracting structured metadata and complete transcription text."""
        result = {
            "doc_category": "OTHER",
            "title_or_subject": "Unknown Document",
            "structured_data": {},
            "has_stamps_or_signatures": False,
            "full_transcription": "",
            "raw_output": raw_output
        }

        # 1. Extract Full Transcription block
        split_match = re.search(r"-{2,}\s*TRANSCRIPTION\s*-{0,}", raw_output, re.IGNORECASE)
        if split_match:
            meta_part = raw_output[:split_match.start()]
            result["full_transcription"] = raw_output[split_match.end():].strip()
        elif "```json" in raw_output and "```" in raw_output.split("```json", 1)[1]:
            parts = raw_output.split("```", 2)
            if len(parts) >= 3:
                meta_part = parts[0] + "```" + parts[1] + "```"
                result["full_transcription"] = parts[2].strip()
            else:
                meta_part = raw_output
        else:
            meta_part = raw_output
            result["full_transcription"] = raw_output.strip()

        # 2. Extract JSON structured metadata
        json_match = re.search(r'```json\s*(.*?)\s*```', meta_part, re.DOTALL | re.IGNORECASE)
        candidate = json_match.group(1).strip() if json_match else meta_part.strip()

        # Look for outermost JSON object
        brace_start = candidate.find('{')
        brace_end = candidate.rfind('}')
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            try:
                parsed = json.loads(candidate[brace_start:brace_end + 1])
                if isinstance(parsed, dict):
                    result["doc_category"] = parsed.get("doc_category", result["doc_category"])
                    result["title_or_subject"] = parsed.get("title_or_subject", result["title_or_subject"])
                    result["structured_data"] = parsed.get("structured_data", {})
                    result["has_stamps_or_signatures"] = parsed.get("has_stamps_or_signatures", False)
                    # If full_transcription wasn't in split section, check if in JSON
                    if not result["full_transcription"] and "full_transcription" in parsed:
                        result["full_transcription"] = str(parsed["full_transcription"])
            except Exception:
                pass

        if not result["full_transcription"]:
            result["full_transcription"] = raw_output.strip()

        return result

    def process_document(self, image_path: str) -> dict:
        """Extract structured data and full transcription via Gemini."""
        # 8192 tokens: a dense table/receipt page with full verbatim transcription easily exceeds 2048,
        # which previously truncated the stored text mid-page.
        raw_text = self._call_gemini_api(DOCUMENT_EXTRACTION_PROMPT, image_path=image_path, max_output_tokens=8192)
        return self._clean_and_parse_json(raw_text)

    def answer_question(
        self,
        image_path: str = None,
        question: str = "",
        max_new_tokens: int = 1024,
        stream: bool = False,
        context_text: str = "",
        image_paths: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Answers visual question with optional cross-page context, multi-image inspection, and conversation memory."""
        if context_text:
            prompt = format_multi_page_qa_prompt(question, context_text, chat_history=chat_history)
        else:
            prompt = format_qa_prompt(question)

        # Interactive QA must fail fast: worst case ~2 models x (45s + 5s) instead of ~5 minutes of backoff.
        return self._call_gemini_api(
            prompt, image_path=image_path, image_paths=image_paths,
            max_output_tokens=2048, max_attempts=2, backoff_step=5
        )

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
