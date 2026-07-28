"""
Orchestrates the baseline and optimized runs over a directory of images.

Baseline: one call per image, full resolution, universal prompt, big model.
This is deliberately wasteful -- it's the 1x we measure savings against.

Optimized:
  1. Ollama classifies (free).
  2. Confident "rejected" -> stop, cost $0.
  3. Confident real doc_type -> resize image, cheap model, targeted prompt.
  4. Low confidence / "uncertain" -> resize image, escalate to mid-tier model
     with the universal prompt (safety net -- don't guess on ambiguous cases).
  5. Cache hit on identical image bytes -> skip the paid call entirely.
"""
import time
import os
from glob import glob

from . import config, image_prep, ollama_router, gemini_client, prompts, schema
from .cache import ExtractionCache
from .cost_logger import CostLogger, compute_cost


def list_images(images_dir: str):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(images_dir, e)))
    return sorted(files)


def run_baseline(images_dir: str, logger: CostLogger, api_key: str = None):
    results = {}
    for path in list_images(images_dir):
        fname = os.path.basename(path)

        try:
            image_bytes = image_prep.load_original_bytes(path)
        except image_prep.ImageLoadError as e:
            record = {"doc_type": "rejected", "reason": f"unreadable_file: {e}"}
            logger.log(
                image=fname, pipeline="baseline", stage="extract",
                model="none", input_tokens=0, output_tokens=0, cost_usd=0.0,
                doc_type_predicted="rejected", confidence=None,
                cache_hit=False, latency_ms=0.0, note="file failed to decode, skipped -- no API call made",
            )
            results[fname] = record
            continue

        try:
            time.sleep(4)
            resp = gemini_client.call_gemini(
                image_bytes, prompts.BASELINE_UNIVERSAL_PROMPT,
                model=config.BASELINE_MODEL, api_key=api_key,
            )
            record = resp["parsed"]
            valid, err = schema.validate_extraction(record)
            if not valid:
                record = schema.empty_record(record.get("doc_type", "uncertain"), err)
            cost = compute_cost(config.BASELINE_MODEL, resp["input_tokens"], resp["output_tokens"])
            logger.log(
                image=fname, pipeline="baseline", stage="extract",
                model=config.BASELINE_MODEL,
                input_tokens=resp["input_tokens"], output_tokens=resp["output_tokens"],
                cost_usd=cost, doc_type_predicted=record.get("doc_type"),
                confidence=None, cache_hit=False, latency_ms=resp["latency_ms"],
            )
        except gemini_client.GeminiError as e:
            record = schema.empty_record("uncertain", str(e))
            logger.log(
                image=fname, pipeline="baseline", stage="extract",
                model=config.BASELINE_MODEL, input_tokens=0, output_tokens=0,
                cost_usd=0.0, doc_type_predicted="uncertain", confidence=None,
                cache_hit=False, latency_ms=0.0, note=f"error: {e}",
            )
        results[fname] = record
    return results


def run_optimized(images_dir: str, logger: CostLogger, cache: ExtractionCache, api_key: str = None):
    results = {}
    for path in list_images(images_dir):
        fname = os.path.basename(path)

        # Step 1: free local classification on the resized image (router doesn't
        # need full resolution to decide "is this a battery or a ledger page").
        try:
            resized_bytes = image_prep.load_and_resize(path, config.MAX_IMAGE_DIMENSION, config.JPEG_QUALITY)
        except image_prep.ImageLoadError as e:
            record = {"doc_type": "rejected", "reason": f"unreadable_file: {e}"}
            logger.log(
                image=fname, pipeline="optimized", stage="extract",
                model="none", input_tokens=0, output_tokens=0, cost_usd=0.0,
                doc_type_predicted="rejected", confidence=None,
                cache_hit=False, latency_ms=0.0, note="file failed to decode, skipped -- no router or API call made",
            )
            results[fname] = record
            continue

        try:
            _process_one(fname, path, resized_bytes, logger, cache, api_key, results)
        except Exception as e:
            # Safety net for anything genuinely unanticipated (not ImageLoadError,
            # not GeminiError -- those are handled above/below already). One bad
            # file must never take down every other result in the batch.
            record = schema.empty_record("uncertain", f"unexpected_error: {e}")
            logger.log(
                image=fname, pipeline="optimized", stage="extract",
                model="none", input_tokens=0, output_tokens=0, cost_usd=0.0,
                doc_type_predicted="uncertain", confidence=None,
                cache_hit=False, latency_ms=0.0, note=f"unexpected error, skipped: {e}",
            )
            results[fname] = record

    return results


def _process_one(fname, path, resized_bytes, logger, cache, api_key, results):
    """The per-image body of run_optimized, split out so the safety-net
    try/except in the caller can wrap it without one giant indented block."""
    cached = cache.get(resized_bytes)
    if cached is not None:
        logger.log(
            image=fname, pipeline="optimized", stage="extract",
            model="cache", input_tokens=0, output_tokens=0, cost_usd=0.0,
            doc_type_predicted=cached.get("doc_type"), confidence=None,
            cache_hit=True, latency_ms=0.0,
        )
        results[fname] = cached
        return

    route = ollama_router.classify(resized_bytes, prompts.CLASSIFY_ONLY_PROMPT)
    logger.log(
        image=fname, pipeline="optimized", stage="router",
        model=f"ollama:{config.OLLAMA_MODEL}",
        input_tokens=route["input_tokens"], output_tokens=route["output_tokens"],
        cost_usd=0.0, doc_type_predicted=route["doc_type"],
        confidence=route["confidence"], cache_hit=False, latency_ms=route["latency_ms"],
    )

    # Step 2: confident rejection -> stop here, zero paid calls.
    if route["doc_type"] == "rejected" and route["confidence"] >= config.REJECT_CONFIDENCE_THRESHOLD:
        record = {"doc_type": "rejected", "reason": route["reason"] or "router_confident_non_document"}
        cache.set(resized_bytes, record)
        results[fname] = record
        return

    # Step 3 / 4: decide cheap targeted call vs. escalation.
    confident_real_type = (
        route["doc_type"] in ("mechanic_log", "vendor_bill", "meter_reading")
        and route["confidence"] >= config.ROUTE_CONFIDENCE_THRESHOLD
    )

    if confident_real_type:
        model = config.CHEAP_MODEL
        prompt = prompts.targeted_prompt(route["doc_type"])
        stage = "extract"
    else:
        model = config.ESCALATION_MODEL
        prompt = prompts.BASELINE_UNIVERSAL_PROMPT
        stage = "escalate"

    try:
        time.sleep(4)
        resp = gemini_client.call_gemini(resized_bytes, prompt, model=model, api_key=api_key)
        record = resp["parsed"]
        # targeted prompts don't ask the model to restate doc_type confidently in
        # ambiguous cases -- make sure it's set for downstream validation.
        record.setdefault("doc_type", route["doc_type"])
        valid, err = schema.validate_extraction(record)
        if not valid:
            record = schema.empty_record(record.get("doc_type", route["doc_type"]), err)
        cost = compute_cost(model, resp["input_tokens"], resp["output_tokens"])
        logger.log(
            image=fname, pipeline="optimized", stage=stage, model=model,
            input_tokens=resp["input_tokens"], output_tokens=resp["output_tokens"],
            cost_usd=cost, doc_type_predicted=record.get("doc_type"),
            confidence=route["confidence"], cache_hit=False, latency_ms=resp["latency_ms"],
        )
    except gemini_client.GeminiError as e:
        record = schema.empty_record(route["doc_type"], str(e))
        logger.log(
            image=fname, pipeline="optimized", stage=stage, model=model,
            input_tokens=0, output_tokens=0, cost_usd=0.0,
            doc_type_predicted="uncertain", confidence=route["confidence"],
            cache_hit=False, latency_ms=0.0, note=f"error: {e}",
        )

    cache.set(resized_bytes, record)
    results[fname] = record
