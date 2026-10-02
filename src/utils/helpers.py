import os
import yaml
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ValidationError

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config.yaml"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def load_config(config_path: Optional[str] = None) -> dict:
    """Loads configuration from YAML file and normalizes relative paths to PROJECT_ROOT."""
    path = config_path or CONFIG_PATH
    cfg = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # Anchor any relative storage paths to PROJECT_ROOT
    if "storage" in cfg and isinstance(cfg["storage"], dict):
        for k, v in cfg["storage"].items():
            if isinstance(v, str) and v.startswith("."):
                cfg["storage"][k] = os.path.abspath(os.path.join(PROJECT_ROOT, v))

    return cfg

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

def expand_general_document_query(query: str) -> str:
    """
    Lightweight, Pure-Python Query Expansion & Intent Enrichment for general documents:
    - Invoices, Financial reports, SPJ, and Reimbursements
    - Official letters, Memos, Decrees, and Circulars
    - Legal agreements, Contracts, MoUs, and Terms
    - Form applications, Approvals, and Attendance sheets
    Works universally across Indonesian and English without external heavy libraries.
    """
    if not query:
        return query

    clean = query.lower().strip()
    expansions = []

    # 1. Total / Financial / Calculation Intent
    if any(k in clean for k in ["total", "jumlah", "biaya", "harga", "nilai", "anggaran", "belanja", "pencairan", "bayar", "cost", "sum", "amount"]):
        expansions.extend([
            "grand total", "rekapitulasi", "jumlah total", "permohonan pencairan", "subtotal", "rincian biaya"
        ])

    # 2. Letter / Disposisi / Administration Intent
    if any(k in clean for k in ["surat", "nomor", "no surat", "perihal", "lampiran", "disposisi", "dinas", "keputusan", "letter"]):
        expansions.extend([
            "nota dinas", "nomor surat", "lampiran", "perihal", "kepada yth", "tanggal surat"
        ])

    # 3. Contract / Legal / Agreement Intent
    if any(k in clean for k in ["kontrak", "perjanjian", "pasal", "mou", "pihak", "kesepakatan", "contract", "agreement"]):
        expansions.extend([
            "surat perjanjian", "pihak pertama", "pihak kedua", "pasal", "ketentuan", "tanda tangan"
        ])

    # 4. Receipt / Invoice / Transaction Evidence Intent
    if any(k in clean for k in ["struk", "kwitansi", "nota", "kasir", "invoice", "bukti", "receipt"]):
        expansions.extend([
            "bukti pembayaran", "kwitansi", "tanda terima", "no transaksi", "tanggal transaksi"
        ])

    # 5. Date / Period Intent
    if any(k in clean for k in ["kapan", "tanggal", "periode", "bulan", "tahun", "date", "period"]):
        expansions.extend([
            "periode", "tanggal", "jatuh tempo", "tahun anggaran"
        ])

    if expansions:
        # Deduplicate while preserving order
        unique_added = [w for w in expansions if w not in clean]
        if unique_added:
            return f"{query} " + " ".join(unique_added[:6])

    return query
