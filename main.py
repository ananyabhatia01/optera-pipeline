#!/usr/bin/env python3
"""
Optera 24-hour challenge -- document pipeline.

Usage:
    python main.py --images images/ --mode both

Requires:
    - GEMINI_API_KEY set in environment (or a .env file, see README)
    - Ollama running locally with a vision model pulled (default: llava)
      -> ollama pull llava

Outputs (written to results/):
    - baseline_extractions.json / optimized_extractions.json  (the structured data)
    - cost_log.jsonl / cost_log.csv                            (every API call, priced)
    - summary.json                                              (aggregated cost + savings)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src import pipeline
from src.cost_logger import CostLogger
from src.cache import ExtractionCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="images", help="Directory of input images")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--mode", choices=["baseline", "optimized", "both"], default="both")
    parser.add_argument("--api-key", default=None, help="Gemini API key (else reads GEMINI_API_KEY env var)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logger = CostLogger(os.path.join(args.out, "cost_log.jsonl"))
    cache = None

    try:
        if args.mode in ("baseline", "both"):
            print(f"[baseline] running naive one-call-per-image pipeline on {args.images}/ ...")
            baseline_results = pipeline.run_baseline(args.images, logger, api_key=args.api_key)
            with open(os.path.join(args.out, "baseline_extractions.json"), "w") as f:
                json.dump(baseline_results, f, indent=2)
            print(f"[baseline] done: {len(baseline_results)} images processed")

        if args.mode in ("optimized", "both"):
            print(f"[optimized] running routed pipeline (Ollama -> Gemini tiers) on {args.images}/ ...")
            cache = ExtractionCache(os.path.join(args.out, "extraction_cache.json"))
            optimized_results = pipeline.run_optimized(args.images, logger, cache, api_key=args.api_key)
            with open(os.path.join(args.out, "optimized_extractions.json"), "w") as f:
                json.dump(optimized_results, f, indent=2)
            print(f"[optimized] done: {len(optimized_results)} images processed")
    finally:
        # Best-effort persistence no matter what happened above. pipeline.py
        # already catches per-image failures (ImageLoadError, GeminiError,
        # and a final Exception safety net), so this branch is defense in
        # depth for anything that still somehow escapes -- e.g. Ctrl-C
        # mid-run. The cost log's JSONL is already durable per-call (see
        # CostLogger.log), so this mainly protects the cache and CSV/summary.
        if cache is not None:
            cache.save()
        logger.flush()
        summary = logger.summary()
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("\n=== SUMMARY ===")
        print(json.dumps(summary, indent=2))
        print(f"\nFull logs written to {args.out}/")


if __name__ == "__main__":
    main()
