# Optera Document Pipeline

Turns heterogeneous phone-photo documents (handwritten mechanic logs, printed
vendor bills, meter/dashboard readings) into structured JSON, and rejects
non-documents (batteries, tyres, plates) instead of hallucinating structure.

Built for the Optera 24-hour challenge. Runs a naive one-call-per-image
**baseline** and a routed, cost-optimized pipeline side by side, and logs
real token counts and cost for both.

## Architecture

```
                     ┌─────────────────────┐
  image  ───────────▶│  Ollama (free,      │
                     │  local) classifies  │
                     └──────────┬──────────┘
                                │
              confident              confident real type      uncertain
              "not a document"       (log/bill/meter)          or low confidence
                     │                     │                        │
                     ▼                     ▼                        ▼
              reject, $0 cost      resize + targeted        resize + escalate to
                                    prompt on cheap tier      mid-tier model with
                                    (Gemini Flash-Lite)       full universal prompt
                                                               (Gemini Flash)
```

The **baseline** skips all of this: every image, full resolution, one
universal classify-and-extract prompt, sent straight to the biggest model
(Gemini 2.5 Pro). That's the naive 1x this pipeline is measured against.

Why this shape, not something else:
- **Rejection has to be free.** ~15-20% of a real inbox is non-documents.
  Paying a multimodal model to look at a battery photo and confirm "not a
  document" is wasted spend a local model can catch for $0.
- **Routing needs a confidence floor, not just a label.** A model that's only
  55% sure it's looking at a vendor bill shouldn't get the cheap, narrow
  prompt — it should escalate. Cheaper-but-wrong doesn't count (this was
  explicit in the brief), so escalation is a deliberate safety net.
- **Resizing is a real cost lever here, not a trick.** Several sample images
  are 3000-4000px phone photos. Multimodal APIs bill image input by
  resolution/tile count, so downscaling to ~1280px before any paid call cuts
  cost with no real legibility loss for text extraction.
- **Caching guards against duplicate resubmissions**, which is realistic for
  a WhatsApp-based intake pipeline.

## Setup

```bash
# 1. Get a free Gemini API key (no billing required for free tier):
#    https://aistudio.google.com -> "Get API key"
cp .env.example .env
# edit .env, paste your key in place of "your_key_here"

# 2. Make sure Ollama is running with a vision-capable model:
ollama pull llava
ollama serve   # if not already running

# 3. Put your images in images/ (already populated with the 47 sample images)
```

## Run

```bash
./run.sh
```

This creates a venv, installs dependencies, and runs both pipelines over
`images/`. Or run directly:

```bash
python main.py --images images/ --mode both
```

Flags: `--mode baseline|optimized|both`, `--images <dir>`, `--out <dir>` (default `results/`).

## Outputs (in `results/`)

- `baseline_extractions.json`, `optimized_extractions.json` — the structured data
- `cost_log.jsonl` / `cost_log.csv` — every single API call, with real token
  counts and computed cost (see `src/cost_logger.py`)
- `summary.json` — aggregated cost-per-doc, savings %, rejection counts

## Accuracy evaluation

There's no ground truth provided (deliberately — it's real, sometimes
illegible handwritten data). `eval/ground_truth_template.json` shows the
format; hand-label ~10-15 sample images across all 4 types by actually
reading them, save as `eval/ground_truth.json`, then run:

```bash
python eval/accuracy.py
```

This is intentionally a small, honest, human-verified sample rather than an
automated scorer — see DESIGN.md for why.

## Repo layout

```
optera-pipeline/
├── main.py                     # CLI entry point
├── DESIGN.md                   # Architecture, design decisions, limitations, roadmap
├── README.md                   # Project overview and usage
├── requirements.txt            # Python dependencies
├── .gitignore
├── LICENSE
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Model configuration, pricing, thresholds
│   ├── schema.py               # Canonical document schemas and validation
│   ├── prompts.py              # Baseline and targeted prompts
│   ├── image_prep.py           # Image resize/compression utilities
│   ├── ollama_router.py        # Local document classifier
│   ├── gemini_client.py        # Gemini REST API wrapper
│   ├── cache.py                # Content-hash extraction cache
│   ├── cost_logger.py          # API cost tracking and summaries
│   └── pipeline.py             # End-to-end extraction pipeline
│
├── eval/
│   ├── accuracy.py             # Field-level evaluation metrics
│   └── ground_truth_template.json
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline_logic.py  # Mocked end-to-end tests (no API calls)
│
├── data/
│   ├── input/                  # Sample documents
│   ├── output/                 # Extraction results
│   └── cache/                  # Cached responses (optional)
│
├── docs/
│   ├── architecture.md
│   ├── pipeline.md
│   └── api.md
│
└── examples/
    ├── sample_invoice.jpg
    ├── sample_receipt.jpg
    └── sample_run.py
```
