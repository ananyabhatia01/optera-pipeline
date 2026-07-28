

import argparse
import json


def normalize(v):
    if v is None:
        return None
    return str(v).strip().lower().replace(" ", "")


def score_image(truth: dict, predicted: dict) -> dict:
    """Returns per-field match info for one image."""
    if truth.get("doc_type") != predicted.get("doc_type"):
        return {"doc_type_correct": False, "field_matches": None, "field_total": None}

    fields = [k for k in truth if not k.startswith("_") and k != "doc_type"]
    matches = 0
    total = 0
    for f in fields:
        true_val = truth.get(f)
        if true_val is None:
            continue  # illegible even to a human -- don't penalize or reward either pipeline
        total += 1
        if normalize(true_val) == normalize(predicted.get(f)):
            matches += 1

    return {"doc_type_correct": True, "field_matches": matches, "field_total": total}


def evaluate(ground_truth: dict, extractions: dict, label: str):
    rows = []
    for fname, truth in ground_truth.items():
        if fname.startswith("_"):
            continue
        predicted = extractions.get(fname)
        if predicted is None:
            print(f"  [{label}] WARNING: no extraction found for {fname}, skipping")
            continue
        rows.append((fname, score_image(truth, predicted)))

    n = len(rows)
    doc_type_correct = sum(1 for _, r in rows if r["doc_type_correct"])
    total_fields = sum(r["field_total"] or 0 for _, r in rows)
    total_matches = sum(r["field_matches"] or 0 for _, r in rows)

    print(f"\n=== {label} ===")
    print(f"doc_type accuracy: {doc_type_correct}/{n} ({100*doc_type_correct/n:.1f}%)" if n else "no images")
    if total_fields:
        print(f"field-level accuracy (on correctly-typed docs): {total_matches}/{total_fields} "
              f"({100*total_matches/total_fields:.1f}%)")
    for fname, r in rows:
        status = "OK " if r["doc_type_correct"] else "MISCLASSIFIED"
        detail = f"{r['field_matches']}/{r['field_total']} fields" if r["field_total"] is not None else ""
        print(f"  {fname}: {status} {detail}")

    return {
        "n_images": n,
        "doc_type_accuracy": doc_type_correct / n if n else None,
        "field_accuracy": total_matches / total_fields if total_fields else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default="eval/ground_truth.json")
    parser.add_argument("--baseline", default="results/baseline_extractions.json")
    parser.add_argument("--optimized", default="results/optimized_extractions.json")
    args = parser.parse_args()

    with open(args.ground_truth) as f:
        truth = json.load(f)

    with open(args.baseline) as f:
        baseline = json.load(f)
    with open(args.optimized) as f:
        optimized = json.load(f)

    b = evaluate(truth, baseline, "BASELINE")
    o = evaluate(truth, optimized, "OPTIMIZED")

    print("\n=== COMPARISON ===")
    print(json.dumps({"baseline": b, "optimized": o}, indent=2))


if __name__ == "__main__":
    main()
