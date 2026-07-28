"""
Prompt templates.

Design choice: the baseline gets ONE universal prompt that must both classify
AND extract every possible field, every time -- that's what makes it naive
and expensive (long prompt, long output, no assumptions).

The optimized path splits this into two cheaper steps:
  1. Ollama (free, local) does a lightweight classify-only pass.
  2. Once we know the type, we send a SHORT prompt that only asks for the
     fields relevant to that type -- fewer output tokens, and the model
     isn't wasting reasoning deciding "is this a bill or a log" a second time.
"""

SCHEMA_DESCRIPTION = """
Categories and their fields:
- mechanic_log: vehicle_code, date, mechanic_name, work_description, materials_used, language_detected
- vendor_bill: vendor_name, vehicle_no, bill_date, items (list of {description, qty, amount}), gst_amount, total_amount
- meter_reading: reading_type ("odometer" or "adblue_dispenser"), value, unit, price_per_unit, total_amount
- rejected: reason (use this if the image is NOT a document -- e.g. a photo of a battery, tyre, number plate, or anything with no extractable record)
"""

CLASSIFY_ONLY_PROMPT = f"""You are looking at a phone photo from a transport company's operations inbox.
It is one of: a handwritten mechanic work log, a printed vendor bill/invoice,
a digital meter or dashboard reading photo, or NOT a document at all (e.g. a
bare photo of a battery, tyre, or number plate with no structured record to extract).

Do not extract fields yet. Just classify.

Respond with ONLY a JSON object, no other text:
{{"doc_type": "mechanic_log" | "vendor_bill" | "meter_reading" | "rejected", "confidence": 0.0-1.0, "reason": "one short phrase"}}

If you are not confident, still give your best guess but lower the confidence score honestly.
"""

BASELINE_UNIVERSAL_PROMPT = f"""You are extracting structured data from a phone photo sent to a transport
company's document pipeline. The photo could be a handwritten mechanic work
log, a printed vendor bill, a digital meter/dashboard reading, or not a
document at all.
{SCHEMA_DESCRIPTION}
Rules:
- Only extract data that is actually visible in the image. Never invent or guess values.
- If the image is not a document (e.g. a battery, tyre, or plate with no record), return doc_type "rejected" with a reason. Do not fabricate fields for it.
- Text may be in English, Hindi, or Gujarati. Transliterate names/places to English where reasonable but keep numbers exact.
- Output ONLY a single JSON object matching the schema for the doc_type you chose. No markdown, no explanation, no code fences.
"""


def targeted_prompt(doc_type: str) -> str:
    """Short, category-specific extraction prompt used once we already know the type."""
    if doc_type == "mechanic_log":
        return """This is a handwritten mechanic work log. Extract ONLY:
{"doc_type": "mechanic_log", "vehicle_code": "...", "date": "...", "mechanic_name": "...", "work_description": "...", "materials_used": "...", "language_detected": "..."}
Only use what is visible. Use null for anything illegible or absent. Output ONLY the JSON object."""

    if doc_type == "vendor_bill":
        return """This is a printed vendor bill/invoice with handwritten line items. Extract ONLY:
{"doc_type": "vendor_bill", "vendor_name": "...", "vehicle_no": "...", "bill_date": "...", "items": [{"description": "...", "qty": "...", "amount": "..."}], "gst_amount": "...", "total_amount": "..."}
Only use what is visible. Use null for anything illegible or absent. Output ONLY the JSON object."""

    if doc_type == "meter_reading":
        return """This is a digital meter/dashboard reading photo (odometer or AdBlue/DEF dispenser). Extract ONLY:
{"doc_type": "meter_reading", "reading_type": "odometer" or "adblue_dispenser", "value": "...", "unit": "...", "price_per_unit": "...", "total_amount": "..."}
Only use what is visible. Use null for anything illegible or absent. Output ONLY the JSON object."""

    raise ValueError(f"No targeted prompt defined for doc_type={doc_type!r}")
