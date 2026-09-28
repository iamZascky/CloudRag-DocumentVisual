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
        f"Examine this document image and answer the question accurately based only on what is visibly present.\n\n"
        f"Question: {question}\n\n"
        "Requirements:\n"
        "- Answer directly with the key value, total, or finding first.\n"
        "- If the document shows itemized breakdowns, periods, or sub-totals, list them clearly.\n"
        "- Mention any relevant section header, document title, or account code found in the image.\n"
        "- Do not guess or extrapolate numbers not visible in the document."
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


