"""
Mocked end-to-end test: no real network calls (Gemini or Ollama). This
validates the orchestration logic itself -- routing decisions, cost
computation, caching, escalation -- using fake responses standing in for
the real APIs. Run this before you spend a single real token, to catch
logic bugs for free.

Run: python3 -m tests.test_pipeline_logic   (from repo root)
"""

import os
import sys
import shutil
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PIL import Image

from src import pipeline, config
from src.cost_logger import CostLogger
from src.cache import ExtractionCache

TEST_DIR = "tests/_tmp_images"
OUT_DIR = "tests/_tmp_results"


def make_fake_images():
    os.makedirs(TEST_DIR, exist_ok=True)
    # 3 fake images: sizes vary to test resize logic too
    specs = {
        "fake_log.jpg": (2000, 3000),
        "fake_bill.jpg": (1080, 1440),
        "fake_battery.jpg": (900, 1200),
    }
    for name, size in specs.items():
        Image.new("RGB", size, color=(200, 200, 200)).save(os.path.join(TEST_DIR, name))

    # Regression fixture for the optera_doc_33.jpg bug: a file with an image
    # extension that isn't actually a decodable image at all (in the real
    # dataset, this was an HTML 404 page saved as .jpg after a broken
    # download link). Both pipelines must reject this gracefully -- log it
    # and move on -- rather than crashing and losing every other result.
    with open(os.path.join(TEST_DIR, "fake_broken.jpg"), "w") as f:
        f.write("<!doctype html><html><body>404 Not Found</body></html>")

    return specs


FAKE_ROUTES = {
    "fake_log.jpg": {"doc_type": "mechanic_log", "confidence": 0.85, "reason": "ruled ledger page",
                      "input_tokens": 300, "output_tokens": 20, "latency_ms": 400},
    "fake_bill.jpg": {"doc_type": "vendor_bill", "confidence": 0.55, "reason": "not sure, faded",
                       "input_tokens": 300, "output_tokens": 20, "latency_ms": 400},
    "fake_battery.jpg": {"doc_type": "rejected", "confidence": 0.9, "reason": "bare battery photo",
                          "input_tokens": 300, "output_tokens": 20, "latency_ms": 400},
}

FAKE_GEMINI = {
    "mechanic_log": {
        "parsed": {"doc_type": "mechanic_log", "vehicle_code": "TCM35", "date": "12/07",
                   "mechanic_name": "Ravi", "work_description": "brake pad change",
                   "materials_used": "brake pads", "language_detected": "English"},
        "input_tokens": 800, "output_tokens": 60, "latency_ms": 900, "raw_text": "{}",
    },
    "vendor_bill_escalated": {
        "parsed": {"doc_type": "vendor_bill", "vendor_name": "Anupam Tyres", "vehicle_no": "GJ01AB1234",
                   "bill_date": "10/07", "items": [{"description": "tyre", "qty": "2", "amount": "4000"}],
                   "gst_amount": "200", "total_amount": "4200"},
        "input_tokens": 1200, "output_tokens": 90, "latency_ms": 1100, "raw_text": "{}",
    },
    "baseline_universal": {
        "parsed": {"doc_type": "mechanic_log", "vehicle_code": "TCM35", "date": "12/07",
                   "mechanic_name": "Ravi", "work_description": "brake pad change",
                   "materials_used": "brake pads", "language_detected": "English"},
        "input_tokens": 1500, "output_tokens": 150, "latency_ms": 1500, "raw_text": "{}",
    },
}


def fake_ollama_classify(image_bytes, prompt, model=None, host=None):
    # crude: identify by testing which fake image size band we resized from
    # (in this fake test we just cycle through based on call count)
    fake_ollama_classify.calls += 1
    # list_images() sorts alphabetically: fake_battery.jpg, fake_bill.jpg, fake_log.jpg
    order = ["fake_battery.jpg", "fake_bill.jpg", "fake_log.jpg"]
    name = order[(fake_ollama_classify.calls - 1) % len(order)]
    return FAKE_ROUTES[name]
fake_ollama_classify.calls = 0


def fake_gemini_call(image_bytes, prompt, model, api_key=None):
    if "vendor_bill" in prompt or "Anupam" in prompt:
        pass
    if config.BASELINE_MODEL == model:
        return FAKE_GEMINI["baseline_universal"]
    if "mechanic_log" in prompt.lower() or "vehicle_code" in prompt:
        return FAKE_GEMINI["mechanic_log"]
    return FAKE_GEMINI["vendor_bill_escalated"]


def run():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    make_fake_images()
    os.makedirs(OUT_DIR, exist_ok=True)

    logger = CostLogger(os.path.join(OUT_DIR, "cost_log.jsonl"))
    cache = ExtractionCache(os.path.join(OUT_DIR, "cache.json"))

    with mock.patch("src.pipeline.ollama_router.classify", side_effect=fake_ollama_classify), \
         mock.patch("src.pipeline.gemini_client.call_gemini", side_effect=fake_gemini_call):

        baseline_results = pipeline.run_baseline(TEST_DIR, logger, api_key="fake")
        fake_ollama_classify.calls = 0
        optimized_results = pipeline.run_optimized(TEST_DIR, logger, cache, api_key="fake")

    logger.flush()
    summary = logger.summary()

    # --- assertions ---
    assert len(baseline_results) == 4, "baseline should process all 4 images, including the unreadable one"
    assert len(optimized_results) == 4, "optimized should process all 4 images, including the unreadable one"

    # Regression check for the optera_doc_33.jpg bug: an unreadable "image"
    # file must not crash the run, and must not burn any paid API cost.
    assert baseline_results["fake_broken.jpg"]["doc_type"] == "rejected", \
        "unreadable file should be rejected, not crash the batch"
    assert "unreadable_file" in baseline_results["fake_broken.jpg"]["reason"]
    assert optimized_results["fake_broken.jpg"]["doc_type"] == "rejected", \
        "unreadable file should be rejected, not crash the batch"
    assert "unreadable_file" in optimized_results["fake_broken.jpg"]["reason"]
    broken_rows = [r for r in logger.records if r.image == "fake_broken.jpg"]
    assert all(r.cost_usd == 0.0 for r in broken_rows), "unreadable file must incur zero cost in either pipeline"

    assert optimized_results["fake_battery.jpg"]["doc_type"] == "rejected", \
        "confident non-document should be rejected"
    # confirm zero paid cost for the rejected image
    battery_rows = [r for r in logger.records if r.image == "fake_battery.jpg" and r.pipeline == "optimized"]
    assert all(r.cost_usd == 0.0 for r in battery_rows), "rejected image must incur zero paid cost"
    assert any(r.model.startswith("ollama") for r in battery_rows), "router should have run"
    assert not any(not r.model.startswith("ollama") for r in battery_rows), "no paid model should have been called"

    assert optimized_results["fake_log.jpg"]["doc_type"] == "mechanic_log"
    log_rows = [r for r in logger.records if r.image == "fake_log.jpg" and r.pipeline == "optimized"]
    assert any(r.model == config.CHEAP_MODEL for r in log_rows), "confident mechanic_log should route to cheap model"

    bill_rows = [r for r in logger.records if r.image == "fake_bill.jpg" and r.pipeline == "optimized"]
    assert any(r.stage == "escalate" and r.model == config.ESCALATION_MODEL for r in bill_rows), \
        "low-confidence route (0.55) should escalate to mid-tier model, not cheap tier"

    assert summary["optimized"]["n_rejected"] == 2, "battery photo + unreadable file should both count as rejected"
    assert summary["optimized"]["cost_per_doc_usd"] < summary["baseline"]["cost_per_doc_usd"], \
        "optimized must be cheaper than baseline on this fake batch"

    print("ALL LOGIC TESTS PASSED")
    print("Summary from mocked run:")
    import json
    print(json.dumps(summary, indent=2))

    shutil.rmtree(TEST_DIR)
    shutil.rmtree(OUT_DIR)


if __name__ == "__main__":
    run()
