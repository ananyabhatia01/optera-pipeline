"""
Central configuration: model names, pricing, thresholds.

PRICING NOTE (read this before trusting the cost numbers):
Prices below are Gemini Developer API list prices, USD per 1M tokens,
checked July 2026 against https://ai.google.dev/gemini-api/docs/pricing
Gemini 2.0 Flash / Flash-Lite were shut down 1 June 2026 -- do not use them.
Gemini 2.5 Flash-Lite / Flash / Pro are the current stable tier as of this
writing but are scheduled for retirement 16 Oct 2026 per Google's docs.
If you run this after that date, re-check pricing and update this table --
the cost math is only as honest as this table is current.
"""

# (input_price_per_1M_tokens, output_price_per_1M_tokens) in USD
MODEL_PRICING = {
    "gemini-3.5-flash":      (1.50, 7.50),    # baseline + escalation tier (Pro unavailable without billing)
    "gemini-3.1-flash-lite": (0.10, 0.40),    # cheap tier for confident routing
}

BASELINE_MODEL = "gemini-3.5-flash"
ESCALATION_MODEL = "gemini-3.5-flash"
CHEAP_MODEL = "gemini-3.1-flash-lite"

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Local Ollama router (free, no API cost). Must support vision.
# 'llava' is the most commonly available; 'moondream' is smaller/faster.
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llava"

# Router confidence thresholds
REJECT_CONFIDENCE_THRESHOLD = 0.60   # >= this -> trust router's "non_document" call, skip paid API entirely
ROUTE_CONFIDENCE_THRESHOLD = 0.70    # >= this -> use cheap model with targeted prompt
# below ROUTE_CONFIDENCE_THRESHOLD (and not a confident reject) -> escalate to Flash

# Image resizing for the optimized pipeline (baseline uses original, full-res image
# on purpose -- that's what makes it "naive"). Gemini bills image input by tile count,
# which scales with resolution, so downscaling large phone photos is a real, provable
# cost lever, not a trick.
MAX_IMAGE_DIMENSION = 1280
JPEG_QUALITY = 85

DOC_TYPES = ["mechanic_log", "vendor_bill", "meter_reading", "rejected", "uncertain"]
