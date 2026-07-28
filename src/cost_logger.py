"""
Cost + accuracy logging. This is the artifact the assessment explicitly asks
for: "your numbers ... with the token/cost log that proves them."

Every API call (router or extractor) writes one row here. Nothing is
estimated after the fact -- costs are computed directly from the token
counts each provider returns with its response, multiplied by the pricing
table in config.py.
"""

import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

from . import config


@dataclass
class CallRecord:
    timestamp: float
    image: str
    pipeline: str          # "baseline" | "optimized"
    stage: str             # "router" | "extract" | "escalate"
    model: str              # "ollama:llava" | "gemini-2.5-flash-lite" | etc.
    input_tokens: int
    output_tokens: int
    cost_usd: float
    doc_type_predicted: Optional[str]
    confidence: Optional[float]
    cache_hit: bool
    latency_ms: float
    note: str = ""


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Ollama calls are local/free. Gemini calls are priced from config.MODEL_PRICING."""
    if model.startswith("ollama"):
        return 0.0
    if model not in config.MODEL_PRICING:
        raise ValueError(f"No pricing entry for model '{model}' -- add it to config.MODEL_PRICING")
    in_price, out_price = config.MODEL_PRICING[model]
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


class CostLogger:
    def __init__(self, out_path: str):
        self.out_path = out_path
        self.records: list[CallRecord] = []
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        # Start the JSONL fresh for this run. Records are appended to this
        # file as soon as log() is called (see below) rather than only being
        # written once at the end in flush() -- if the process dies partway
        # through a run, every call made before the crash is still on disk
        # and provable, instead of the whole log vanishing with the process.
        open(self.out_path, "w").close()

    def log(self, **kwargs) -> CallRecord:
        rec = CallRecord(timestamp=time.time(), **kwargs)
        self.records.append(rec)
        with open(self.out_path, "a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        return rec

    def flush(self):
        """
        Write the CSV (easy to skim) view of the log. The JSONL file is
        already durable -- each record was appended to it in log() -- but we
        rewrite it here too from in-memory state as a final consistency pass.
        """
        jsonl_path = self.out_path
        csv_path = self.out_path.rsplit(".", 1)[0] + ".csv"

        with open(jsonl_path, "w") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r)) + "\n")

        if self.records:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(self.records[0]).keys()))
                writer.writeheader()
                for r in self.records:
                    writer.writerow(asdict(r))

    def summary(self) -> dict:
        """Aggregate totals per pipeline -- what actually goes in DESIGN.md."""
        out = {}
        for pipeline in ("baseline", "optimized"):
            rows = [r for r in self.records if r.pipeline == pipeline]
            images = sorted(set(r.image for r in rows))
            total_cost = sum(r.cost_usd for r in rows)
            n_images = len(images) or 1
            out[pipeline] = {
                "n_images": len(images),
                "n_calls": len(rows),
                "total_cost_usd": round(total_cost, 6),
                "cost_per_doc_usd": round(total_cost / n_images, 6),
                "n_rejected": len(set(
                    r.image for r in rows
                    if r.doc_type_predicted == "rejected"
                )),
                "n_paid_calls": len([r for r in rows if not r.model.startswith("ollama")]),
                "n_free_router_calls": len([r for r in rows if r.model.startswith("ollama")]),
                "n_cache_hits": len([r for r in rows if r.cache_hit]),
            }
        if out["baseline"]["cost_per_doc_usd"] > 0:
            out["savings_pct"] = round(
                100 * (1 - out["optimized"]["cost_per_doc_usd"] / out["baseline"]["cost_per_doc_usd"]), 2
            )
        return out
