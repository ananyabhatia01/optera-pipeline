
import base64
import json
import os
import re
import time

import requests

from . import config


class GeminiError(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
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
    raise GeminiError(f"Could not parse JSON from model output: {text[:300]!r}")


def _extract_retry_delay(error_text: str, default: float = 15.0) -> float:
    match = re.search(r"retry in ([\d.]+)s", error_text, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0
    return default


def call_gemini(image_bytes: bytes, prompt: str, model: str, api_key: str = None, max_retries: int = 4) -> dict:
    last_error = None
    for attempt in range(max_retries):
        try:
            return _call_gemini_once(image_bytes, prompt, model, api_key)
        except GeminiError as e:
            last_error = e
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                delay = _extract_retry_delay(msg)
            elif "503" in msg or "UNAVAILABLE" in msg:
                delay = 5.0 * (attempt + 1)
            else:
                raise
            time.sleep(delay)
    raise last_error

def _call_gemini_once(image_bytes: bytes, prompt: str, model: str, api_key: str = None) -> dict:
    """
    Returns dict with: parsed (dict), input_tokens, output_tokens, latency_ms, raw_text
    Raises GeminiError on HTTP failure or unparseable JSON (caller decides
    whether to retry / escalate / fall back to empty_record).
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiError("GEMINI_API_KEY not set in environment")

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    url = f"{config.GEMINI_API_BASE}/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}},
            ]
        }],
        "generationConfig": {
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    }

    start = time.time()
    resp = requests.post(url, json=body, headers=headers, timeout=60)
    latency_ms = (time.time() - start) * 1000

    if resp.status_code != 200:
        raise GeminiError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise GeminiError(f"Unexpected response shape: {data}") from e

    usage = data.get("usageMetadata", {})
    input_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)

    parsed = _extract_json(raw_text)

    return {
        "parsed": parsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "raw_text": raw_text,
    }
