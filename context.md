System with 14 visual classifiers (checklist items).
Each classifier should detect a specific visual condition in an image (e.g., “person holding a hose with visible water discharge in a field”).
Core Approach
Use CLIP-like embeddings (text + image).
Each classifier uses prompt set design (Option B):
Positive prompts (required)
Negative prompts (recommended)
Analyze one input image at a time.
Compute similarity-based score per classifier.
Return:
per-classifier score
pass/fail (and optionally ambiguous)
top matches / confusion signals
Priorities
Audience: internal QA operators
False-positive sensitivity: false positives are worse than false negatives
Full checklist expectation: ideally all 14 pass
Iteration loop: if classifiers confuse each other, revise prompt text, re-embed, and retest
UI should support switching models (different CLIP variants)
Desired Outputs
Please provide:
Extensive product + technical spec
Local setup plan (MVP first)
Postgres schema suggestion (pgvector-friendly)
Scoring logic with thresholds/margins
API contract draft
UI workflow for QA operators
Calibration/testing plan without labeled data
Constraints
Start simple and practical for local testing
Emphasize clarity, debuggability, and iterative improvement
Keep naming generic (no dependency on any existing repository or prior project context)
