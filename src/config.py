

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
