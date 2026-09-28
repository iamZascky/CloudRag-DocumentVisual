import os
import yaml
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ValidationError

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config.yaml"))

def load_config(config_path: Optional[str] = None) -> dict:
    """Loads configuration from YAML file."""
    path = config_path or CONFIG_PATH
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

class ReceiptStructuredData(BaseModel):
    items: list[dict] = Field(default_factory=list)
    total_price: Optional[float] = None
    handwritten_total_nota: Optional[float] = None
    
    class Config:
        extra = "allow"

class StructuredQAResponse(BaseModel):
    """Pydantic schema enforcing structured question answering output."""
    direct_answer: str = Field(description="Direct, unequivocal summary answer to the user's question")
    numeric_value: Optional[float] = Field(default=None, description="Extracted numerical total or main amount without currency symbols")
    currency: Optional[str] = Field(default="IDR", description="Currency code (IDR, USD, etc.)")
    source_citation: Optional[str] = Field(default=None, description="Account code, table header, or document reference name")
    breakdown: list[dict] = Field(default_factory=list, description="Itemized period or vehicle breakdown components")
    confidence: str = Field(default="HIGH", description="Confidence level: HIGH, MEDIUM, LOW")
    raw_answer: Optional[str] = Field(default=None, description="Full raw textual explanation from the model")

def parse_first_numeric(text: str) -> Optional[float]:
    """
    Extracts first currency/number found in text supporting both:
    - Dot thousands, comma decimal: 45.562.550 or 45.562.550,00
    - Comma thousands, dot decimal: 12,450.00 or 12,450
    """
    import re
    # Match candidate numbers with optional currency prefixes
    pattern = r"(?:(?:Rp|[$€£])\.?\s*)?(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)"
    matches = re.findall(pattern, text)
    for m in matches:
        raw = m.strip()
        # Case 1: Standard international format (e.g. 12,450.00)
        if "," in raw and "." in raw:
            if raw.rfind(".") > raw.rfind(","):  # comma thousand, dot decimal
                cleaned = raw.replace(",", "")
            else:  # dot thousand, comma decimal (e.g. 12.450,00)
                cleaned = raw.replace(".", "").replace(",", ".")
        elif "," in raw and "." not in raw:
            # Check if comma is decimal (2 digits at end) or thousand
            parts = raw.split(",")
            if len(parts[-1]) == 2:
                cleaned = raw.replace(",", ".")
            else:
                cleaned = raw.replace(",", "")
        elif "." in raw and "," not in raw:
            # Check if dot is decimal (2 digits at end and single dot) or thousand
            parts = raw.split(".")
            if len(parts) == 2 and len(parts[-1]) == 2 and len(parts[0]) <= 3:
                cleaned = raw
            else:
                cleaned = raw.replace(".", "")
        else:
            cleaned = raw

        try:
            val = float(cleaned)
            if val > 0:
                return val
        except ValueError:
            continue
    return None


def validate_extraction(doc_category: str, structured_data: Dict[str, Any]) -> str:
    """
    Validates structured data extracted from receipts or invoices.
    Returns status string ("VERIFIED" or "AUDIT_FLAGGED: <reason>").
    """
    if doc_category.upper() in ["RECEIPT", "RECEIPT_COLLAGE", "INVOICE"]:
        try:
            receipt = ReceiptStructuredData(**structured_data)
            
            calculated_sum = 0.0
            if receipt.items:
                for item in receipt.items:
                    price = item.get("price", 0)
                    try:
                        calculated_sum += float(price)
                    except (ValueError, TypeError):
                        pass
                        
            target_total = (
                receipt.handwritten_total_nota 
                if receipt.handwritten_total_nota is not None 
                else receipt.total_price
            )
            
            if target_total is not None and calculated_sum > 0:
                if abs(calculated_sum - target_total) < 0.01:
                    return "VERIFIED"
                else:
                    return f"AUDIT_FLAGGED: Calculated sum ({calculated_sum}) does not match total ({target_total})"
            
            return "VERIFIED"
            
        except ValidationError as e:
            return f"AUDIT_FLAGGED: Schema validation error - {e}"
            
    return "VERIFIED"
