# DESIGN.md

## Overview

A document pipeline that converts Optera's WhatsApp inbox — handwritten
mechanic logs, printed vendor bills, meter/dashboard photos, and
non-documents — into structured JSON, with cost per document treated as a
primary design constraint.

## Architecture

**Baseline (1x):** Single call per image, full resolution, one universal
prompt, sent to a large multimodal model (originally planned as Gemini 2.5
Pro; see Results section for why the actual run uses Gemini 3.5 Flash
instead).

**Optimized:** Three-stage router.

1. **Local classification (free).** An Ollama vision model (llava)
   classifies each image before any paid call: mechanic log, vendor bill,
   meter reading, or non-document.
2. **Reject.** If classified "not a document" with confidence ≥ 0.60, the
   image is rejected with no paid API call.
3. **Cheap path.** If classified as a document type with high confidence,
   the image is resized (long edge → 1280px) and sent to Gemini 2.5
   Flash-Lite with a short, category-specific prompt requesting only the
   fields relevant to that document type.
4. **Escalation path.** If confidence is below 0.70 or the result is
   "uncertain," the image is sent to Gemini 2.5 Flash with the full
   universal prompt. This is a deliberate safety net: ambiguous cases are
   not forced through the cheap/narrow path, since incorrect structure at
   low cost is not a saving.

A content-hash cache prevents re-processing duplicate images, which is
common in WhatsApp-based intake due to resends.

### Design rationale

Three cost levers drive the architecture:
- Not paying to process non-documents.
- Not sending full-resolution images when a downscaled version preserves
  extractable information.
- Not combining document-type classification and field extraction in a
  single expensive call.

All three are structural rather than a blanket "use a smaller model"
approach. The confidence-gated escalation exists because a pipeline that is
cheap only by skipping its own confidence check is not solving the cost
problem — it is deferring it to accuracy loss.

### Results

Note on models: `gemini-2.5-pro` could not be used for the baseline as
originally planned — Google requires a billing account linked to the
project for Pro-tier access, even at near-zero usage, and billing was not
available during this assessment window. The baseline was run instead on
`gemini-3.5-flash`, with
`gemini-3.1-flash-lite` as the optimized pipeline's cheap tier. This is a
genuine substitution, documented rather than hidden: it changes the absolute
cost numbers below, but the baseline is still meaningfully more expensive
and less targeted than the optimized path, so the relative comparison the
brief asks for still holds.

Numbers below are from a full 47-image run that completed successfully on
both pipelines before the API key's free-tier request quota (Google caps
free-tier `gemini-3.5-flash` access at a low per-minute request limit)
began rate-limiting subsequent runs. A later attempt to re-run for the
accuracy table hit sustained `429 RESOURCE_EXHAUSTED` errors despite added
retry/backoff logic (see Robustness section) and did not complete in time
to produce a verified accuracy figure for this submission.

| Metric | Baseline (gemini-3.5-flash) | Optimized |
|---|---|---|
| Cost per document | $0.000335 | $0.000023 |
| Images rejected pre-paid-call | — | 2 of 47 |
| Cost savings | — | 93.13% |
| Doc-type accuracy (vs. hand-labeled subset) | not captured — blocked by free-tier rate limiting | not captured — blocked by free-tier rate limiting |
| Field-level accuracy (correctly-typed docs) | not captured — blocked by free-tier rate limiting | not captured — blocked by free-tier rate limiting |

A 15-image hand-labeled ground truth set (`eval/ground_truth.json`, spanning
all four document categories) was prepared and is included in this
submission. The accuracy scorer (`eval/accuracy.py`) runs correctly against
it, but a clean, quota-error-free full extraction run to score against it
could not be completed before the deadline. Re-running `python main.py
--images images/ --mode both` followed by `python eval/accuracy.py` with a
fresh or billed API key will produce the missing figures; the scoring logic
itself does not need changes.

Full per-call detail: `results/cost_log.csv`.

## Known limitations

- **Handwriting variability.** Multi-colour ink, strikethroughs, and
  occlusion (e.g., fingers over text) correlate with lower router
  confidence. Heavily annotated logs are more likely to route to
  escalation rather than the cheap tier, raising effective cost per
  document above the confident-path baseline.
- **Mixed-language extraction.** Hindi/Gujarati combined with English is
  the most likely source of field-level errors. Classification confidence
  from the local router does not indicate extraction-model accuracy on
  transliteration; confidence-gating protects classification only, not
  downstream field accuracy.
- **Unvalidated threshold.** The 0.60 reject threshold was set by manual
  inspection of the sample set, not tuned against held-out data. Too low a
  threshold rejects valid vendor bills; too high a threshold routes
  non-documents to the escalation model instead of filtering them,
  increasing cost.
- **No preprocessing pipeline.** No OCR bounding-box extraction or
  handwriting-specific preprocessing (binarization, deskew). Extraction
  relies entirely on the vision model's native read of the raw image,
  which is the primary risk for low-contrast or blurry photos.
- **Small evaluation set.** Accuracy is measured on 10-15 manually
  transcribed images. No ground truth was provided; a larger evaluation
  set was not built to avoid publishing an accuracy figure based on
  model-vs-model comparison rather than verified labels.

## Robustness — hardened vs. outstanding

Scope of this pass: closing one specific failure mode (a bad file crashing
the batch) and directly related issues. This is meaningfully more robust
than before, but does not meet a "production-ready" bar.

**Hardened and verified against the real dataset:**
- Corrupt or non-image files no longer crash the batch, in either
  pipeline — confirmed against 2 real bad files in the sample set
  (`optera_doc_33.jpg`, an HTML 404 page saved with a `.jpg` extension,
  was the case that surfaced this), not only a synthetic test case.
- Cost log is append-as-you-go (`CostLogger.log` writes each record to
  disk immediately) and survives a mid-run crash.
- Cache uses atomic writes (`os.replace` after writing to a temp file) and
  survives a mid-save crash; a corrupted cache from a prior crash reloads
  safely instead of crashing the next run.
- API key passed via request header (`x-goog-api-key`), not embedded in
  the request URL, so it does not appear in access logs.
- Regression test (`tests/test_pipeline_logic.py`) locks in the fix and
  asserts zero cost is incurred on unreadable files in both pipelines.

**Outstanding for production use:**
- Retry/backoff on transient Gemini errors (429, 5xx, timeouts) was added
  during this assessment (`call_with_retry` in `pipeline.py`, parses
  Google's own suggested retry delay from the error body and waits before
  retrying, up to 4 attempts). It works as designed, but Google's free-tier
  request quota is low enough that sustained heavy testing in a single
  session can still exhaust it faster than retries can recover from —
  production use would need a billed tier or a token-bucket limiter tuned
  below the published RPM ceiling, not just retries.
- No concurrency — processing is single-threaded and synchronous.
- No secrets management beyond `.env` — no rotation or per-environment
  key handling.
- No observability — no metrics, alerting, or dashboards; output is
  local JSON/CSV only.
- No auth or multi-tenancy — this is a script, not a service.
- No CI — tests exist and pass locally but are not run automatically.
- Untested at volume against multi-page PDFs, HEIC images, very large
  files, or non-English handwriting beyond the sample set.
- Ollama is a single local point of failure. Degradation is per-image
  and handled correctly, but there is no alert if the local tier becomes
  unavailable for an extended period.

## Next steps

- Build a held-out evaluation set (50-100 unseen images, double-
  transcribed) to replace the self-labeled sample.
- Tune confidence thresholds against the held-out set, per document
  category.
- Add an image-quality pre-check (blur/contrast score) to route low-
  quality images directly to the strongest model.
- Use Gemini's Batch API for non-latency-sensitive
  ingestion.
- Add retries with backoff for transient API failures.

## AI tools used

Claude (Anthropic) for architecture design, and Gemini API Key