"""
Free local pre-classifier using Ollama. This is the piece that makes the
optimized pipeline genuinely cheaper rather than just "smaller paid model":
every image gets classified for $0 before we decide whether -- and which --
paid API call is worth making.

Requires Ollama running locally with a vision-capable model pulled, e.g.:
    ollama pull llava
"""

import base64
import json
import re
import time

import requests

from . import config


def _extract_json(text: str) -> dict:
    """Local models are less reliable about 'JSON only' instructions than
    hosted frontier models -- strip markdown fences and grab the first
    {...} block as a fallback."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"doc_type": "uncertain", "confidence": 0.0, "reason": "router_json_parse_failed"}


def classify(image_bytes: bytes, prompt: str, model: str = None, host: str = None) -> dict:
    """
    Returns a dict with:
      doc_type, confidence, reason, input_tokens, output_tokens, latency_ms
    input_tokens/output_tokens come from Ollama's own eval counts (informational
    only -- these calls cost $0 since they run on your machine).
    """
    model = model or config.OLLAMA_MODEL
    host = host or config.OLLAMA_HOST

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64_image],
        "stream": False,
        "format": "json",
    }

    start = time.time()
    try:
        resp = requests.post(f"{host}/api/generate", json=payload, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {
            "doc_type": "uncertain", "confidence": 0.0,
            "reason": f"ollama_request_failed: {e}",
            "input_tokens": 0, "output_tokens": 0,
            "latency_ms": (time.time() - start) * 1000,
        }
    latency_ms = (time.time() - start) * 1000

    data = resp.json()
    raw_text = data.get("response", "")
    parsed = _extract_json(raw_text)

    return {
        "doc_type": parsed.get("doc_type", "uncertain"),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "reason": parsed.get("reason", ""),
        "input_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
        "latency_ms": latency_ms,
    }
