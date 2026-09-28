# System and user prompt templates for Visual-RAG

DOCUMENT_EXTRACTION_PROMPT = (
    "Extract structured metadata and transcription from this document image.\n\n"
    "```json\n"
    "{\n"
    '  "doc_category": "INVOICE | RECEIPT_COLLAGE | TABLE_REPORT | OFFICIAL_LETTER | FORM | OTHER",\n'
    '  "title_or_subject": "<string>",\n'
    '  "structured_data": {\n'
    '    "total_amount": "<string>",\n'
    '    "date": "<string>",\n'
    '    "entities": "<string>"\n'
    '  },\n'
    '  "has_stamps_or_signatures": false\n'
    "}\n"
    "```\n\n"
    "---TRANSCRIPTION---\n"
    "<Markdown transcription of tables, items, and text>"
)

def format_qa_prompt(question: str) -> str:
    """Formats visual question answering prompt."""
    return (
        f"Examine this document image and answer the question accurately based on the visible content.\n\n"
        f"Question: {question}\n\n"
        "Requirements:\n"
        "- Provide a complete and informative answer in clear Indonesian/English.\n"
        "- State the final total or amount clearly with currency/units (e.g. Rp ..., Liter).\n"
        "- If the document shows sub-totals, items, or breakdowns, summarize them clearly.\n"
        "- Do not guess numbers not visible in the document."
    )

def format_multi_page_qa_prompt(question: str, context_text: str) -> str:
    """Formats multi-page visual question answering prompt with cross-page transcription context."""
    return (
        f"Examine this primary document image alongside the verified text transcriptions from other relevant retrieved pages of the document set.\n\n"
        f"Context from related retrieved pages:\n"
        f"```\n{context_text}\n```\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- If a summary table or grand total is present (such as total belanja/permohonan pencairan), prioritize and state the overall total clearly.\n"
        "- Provide a structured breakdown where appropriate (e.g., period breakdown, category amounts, or receipt volumes).\n"
        "- State all monetary amounts with currency (Rp) and volume units (Liter).\n"
        "- Answer politely, clearly, and concisely based strictly on the factual evidence provided."
    )

def format_structured_qa_prompt(question: str) -> str:
    """Formats visual question answering prompt enforcing structured JSON Schema output."""
    return (
        f"Examine this document image and answer the question in strict JSON format.\n\n"
        f"Question: {question}\n\n"
        "You MUST return ONLY valid JSON matching this schema:\n"
        "```json\n"
        "{\n"
        '  "direct_answer": "<string: clear summary answer>",\n'
        '  "numeric_value": <float or null: exact numerical total/value if applicable>,\n'
        '  "currency": "<string: IDR, USD, etc.>",\n'
        '  "source_citation": "<string or null: account code, table header, or section name>",\n'
        '  "breakdown": [\n'
        '    {"label": "<string>", "amount": <float>}\n'
        '  ],\n'
        '  "confidence": "HIGH | MEDIUM | LOW"\n'
        "}\n"
        "```\n"
        "Do not include commentary outside the JSON block. Do not hallucinate numbers."
    )


