import os
import json
import re
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TextStreamer
from qwen_vl_utils import process_vision_info
from src.utils.helpers import load_config, StructuredQAResponse, parse_first_numeric
from src.prompts.prompt_templates import DOCUMENT_EXTRACTION_PROMPT, format_qa_prompt, format_structured_qa_prompt

config = load_config()
storage_cfg = config.get("storage", {})
vision_cfg = config.get("models", {}).get("vision", {})

os.environ["HF_HOME"] = os.path.abspath(storage_cfg.get("hf_cache_dir", "./storage/models/huggingface"))
os.environ["TORCH_HOME"] = os.path.abspath(storage_cfg.get("torch_cache_dir", "./storage/models/torch"))

MODEL_ID = vision_cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
DEFAULT_MIN_PIXELS = vision_cfg.get("min_pixels", 256 * 28 * 28)
DEFAULT_MAX_PIXELS = vision_cfg.get("max_pixels", 1024 * 28 * 28)

class QwenVisualReader:
    """
    Local GPU-accelerated Document Vision-Language Model interface
    using Qwen2.5-VL-3B in native BF16.
    """
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.cache_dir = os.path.abspath(storage_cfg.get("hf_cache_dir", "./storage/models/huggingface/hub"))
        
        print(f"Loading {self.model_id} onto GPU...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
            cache_dir=self.cache_dir
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir
        )
        print("Model loaded successfully.")

    @torch.inference_mode()
    def process_document(self, image_path: str) -> dict:
        """Extract metadata and transcription from document image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{os.path.abspath(image_path)}",
                        "min_pixels": DEFAULT_MIN_PIXELS,
                        "max_pixels": DEFAULT_MAX_PIXELS,
                    },
                    {"type": "text", "text": DOCUMENT_EXTRACTION_PROMPT},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        try:
            print(f"Generating extraction for {image_path}...")
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1536, 
                repetition_penalty=1.05,
                do_sample=False
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return self._clean_and_parse_json(output_text)
        finally:
            del inputs
            if "generated_ids" in locals():
                del generated_ids
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def answer_question(self, image_path: str, question: str, max_new_tokens: int = 1024, stream: bool = False) -> str:
        """Answer a question about a document image with optional token streaming."""
        prompt = format_qa_prompt(question)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{os.path.abspath(image_path)}",
                        "min_pixels": DEFAULT_MIN_PIXELS,
                        "max_pixels": DEFAULT_MAX_PIXELS,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        streamer = TextStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

        try:
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                streamer=streamer
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return output_text.strip()
        finally:
            del inputs
            if "generated_ids" in locals():
                del generated_ids
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def answer_question_structured(self, image_path: str, question: str, max_new_tokens: int = 1024) -> StructuredQAResponse:
        """
        Answers question about a document image and strictly enforces a validated Pydantic schema.
        Falls back safely to robust text parsing if model omits optional JSON tags.
        """
        prompt = format_structured_qa_prompt(question)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{os.path.abspath(image_path)}",
                        "min_pixels": DEFAULT_MIN_PIXELS,
                        "max_pixels": DEFAULT_MAX_PIXELS,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        try:
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            # Attempt JSON extraction
            parsed_dict = None
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
            candidate_json = json_match.group(1).strip() if json_match else raw_text

            try:
                parsed_dict = json.loads(candidate_json)
            except Exception:
                pass

            if isinstance(parsed_dict, dict):
                # Ensure numeric_value is float if present
                num_val = parsed_dict.get("numeric_value")
                if num_val is not None:
                    try:
                        num_val = float(num_val)
                    except (ValueError, TypeError):
                        num_val = parse_first_numeric(str(num_val))
                else:
                    num_val = parse_first_numeric(parsed_dict.get("direct_answer", ""))

                return StructuredQAResponse(
                    direct_answer=parsed_dict.get("direct_answer", raw_text),
                    numeric_value=num_val,
                    currency=parsed_dict.get("currency", "IDR"),
                    source_citation=parsed_dict.get("source_citation"),
                    breakdown=parsed_dict.get("breakdown", []),
                    confidence=parsed_dict.get("confidence", "HIGH"),
                    raw_answer=raw_text
                )
            
            # Fallback if raw JSON block wasn't produced
            extracted_num = parse_first_numeric(raw_text)
            return StructuredQAResponse(
                direct_answer=raw_text,
                numeric_value=extracted_num,
                currency="IDR",
                source_citation=None,
                breakdown=[],
                confidence="MEDIUM",
                raw_answer=raw_text
            )
        finally:
            del inputs
            if "generated_ids" in locals():
                del generated_ids
            torch.cuda.empty_cache()


    def _clean_and_parse_json(self, text: str) -> dict:
        result = {
            "doc_category": "OTHER",
            "title_or_subject": "Unknown Document",
            "structured_data": {},
            "has_stamps_or_signatures": False,
            "full_transcription": ""
        }

        # 1. Separate Transcription
        split_match = re.search(r"-{2,}\s*TRANSCRIPTION\s*-{0,}", text, re.IGNORECASE)
        if split_match:
            meta_part = text[:split_match.start()]
            result["full_transcription"] = text[split_match.end():].strip()
        elif "```json" in text and "```" in text.split("```json", 1)[1]:
            parts = text.split("```", 2)
            if len(parts) >= 3:
                meta_part = parts[0] + "```" + parts[1] + "```"
                result["full_transcription"] = parts[2].strip()
            else:
                meta_part = text
        else:
            meta_part = text

        # 2. Extract JSON block
        json_str = ""
        json_block_match = re.search(r"```json\s*(.*?)\s*```", meta_part, re.DOTALL)
        candidate_text = json_block_match.group(1).strip() if json_block_match else meta_part.strip()
        
        start_idx = candidate_text.find('{')
        if start_idx != -1:
            depth = 0
            end_idx = -1
            in_str = False
            escape = False
            for i in range(start_idx, len(candidate_text)):
                c = candidate_text[i]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            end_idx = i + 1
                            break
            if end_idx != -1:
                json_str = candidate_text[start_idx:end_idx].strip()
            else:
                json_str = candidate_text[start_idx:].strip()
        else:
            json_str = candidate_text

        cleaned_json = re.sub(r'(?<=[:,\s\[])([0-9]+/[0-9]+)(?=[,\s\]\}])', r'"\1"', json_str)
        cleaned_json = re.sub(r'(?<=[:,\s\[])0+(\d+)(?=[,\s\]\}])', r'"0\1"', cleaned_json)

        try:
            parsed = json.loads(cleaned_json)
            if isinstance(parsed, dict):
                result["doc_category"] = parsed.get("doc_category", result["doc_category"])
                result["title_or_subject"] = parsed.get("title_or_subject", result["title_or_subject"])
                result["structured_data"] = parsed.get("structured_data", {})
                result["has_stamps_or_signatures"] = parsed.get("has_stamps_or_signatures", False)
                if not result["full_transcription"] and "full_transcription" in parsed:
                    result["full_transcription"] = parsed["full_transcription"]
        except Exception:
            cat_match = re.search(r'"doc_category":\s*"([^"]+)"', meta_part)
            if cat_match:
                result["doc_category"] = cat_match.group(1)
            title_match = re.search(r'"title_or_subject":\s*"([^"]+)"', meta_part)
            if title_match:
                result["title_or_subject"] = title_match.group(1)

        if not result["full_transcription"]:
            result["full_transcription"] = text

        return result
