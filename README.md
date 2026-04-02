# Visual QA Compliance MVP

This repository provides a local-first MVP for single-image visual QA/compliance checks using CLIP-like image-text embeddings.

## Features
- 13 prompt-set classifiers (positive + negative prompts)
- Conservative FP-sensitive calibration profile
- Per-classifier score + pass/fail/ambiguous decision
- Top prompt matches and confusion signals for debugging
- FastAPI endpoint for one-image inference
- OpenCLIP backend support with deterministic fallback embeddings

## Project Structure

```text
app/
  main.py
  schemas.py
  store.py
  scoring.py
  embedding.py
configs/
  classifier_sets/
    visual_qa_compliance_v1.json
  calibration_profiles/
    fp_sensitive_v1.json
data/images/
requirements.txt
README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Test API

```bash
curl -X POST "http://localhost:8000/v1/runs" \
  -H "Content-Type: application/json" \
  -d '{
    "image_uri": "file:///absolute/path/to/sample.jpg",
    "model_id": "openclip:ViT-B-32:laion2b_s34b_b79k",
    "classifier_set_name": "visual_qa_compliance_v1",
    "classifier_set_version": "1.0.0",
    "calibration_profile_name": "fp_sensitive_v1",
    "calibration_profile_version": "1.0.0",
    "options": {
      "return_top_k_matches": 5,
      "include_prompt_text": true
    }
  }'
```

## Model Notes

- `app/embedding.py` attempts to use OpenCLIP (`open_clip_torch` + `torch`) if available.
- If unavailable, it falls back to deterministic mock embeddings so the API still works for local integration and UI testing.
- For production behavior, install OpenCLIP dependencies and use a supported model id such as:
  - `openclip:ViT-B-32:laion2b_s34b_b79k`

## Endpoints
- `GET /health`
- `POST /v1/runs`
