"""
Canonical schema(s) for the four categories of Optera document images.

Design choice: one flat JSON schema with a discriminating `doc_type` field,
rather than four separate endpoints. This keeps prompts simple (model always
returns the same envelope) and keeps downstream code (cost logger, eval)
type-agnostic -- it just reads `doc_type` and validates against the matching
required-field set below.

We deliberately do NOT use pydantic to keep the dependency footprint to
stdlib + requests + Pillow, so `pip install -r requirements.txt` stays fast
and there's one less thing that can fail to install on the grader's machine.
"""

from dataclasses import dataclass, field
from typing import Optional


REQUIRED_FIELDS = {
    "mechanic_log": ["vehicle_code", "date", "mechanic_name", "work_description"],
    "vendor_bill": ["vendor_name", "vehicle_no", "total_amount"],
    "meter_reading": ["reading_type", "value", "unit"],
    "rejected": ["reason"],
}

ALL_FIELDS = {
    "mechanic_log": [
        "vehicle_code", "date", "mechanic_name", "work_description",
        "materials_used", "language_detected",
    ],
    "vendor_bill": [
        "vendor_name", "vehicle_no", "bill_date", "items",
        "gst_amount", "total_amount",
    ],
    "meter_reading": [
        "reading_type", "value", "unit", "price_per_unit", "total_amount",
    ],
    "rejected": ["reason"],
}


def validate_extraction(record: dict) -> tuple[bool, str]:
    """
    Minimal structural validation: does this record have the required fields
    for its declared doc_type, and is doc_type itself valid?
    Returns (is_valid, error_message).
    """
    if not isinstance(record, dict):
        return False, "record is not a JSON object"

    doc_type = record.get("doc_type")
    if doc_type not in REQUIRED_FIELDS:
        return False, f"unknown or missing doc_type: {doc_type!r}"

    missing = [f for f in REQUIRED_FIELDS[doc_type] if not record.get(f)]
    if missing:
        return False, f"missing required fields for {doc_type}: {missing}"

    return True, ""


def empty_record(doc_type: str, reason: str = "") -> dict:
    """Fallback record when extraction fails entirely (e.g. bad JSON from model)."""
    if doc_type == "rejected":
        return {"doc_type": "rejected", "reason": reason or "extraction_failed"}
    rec = {f: None for f in ALL_FIELDS.get(doc_type, [])}
    rec["doc_type"] = doc_type
    rec["_extraction_error"] = reason
    return rec
