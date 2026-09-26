# Resume-to-Job-Description Matcher

**AIVI Intelligence — AI Engineer Intern Challenge — Deliverable 3**
Author: Dhruv Sharma

## What it does

Takes raw resume text and a job description, sends both to Gemini with a
strict system prompt, and returns clean JSON:

```json
{
  "match_score": 58,
  "top_strengths": ["..."],
  "missing_skills": ["..."],
  "summary": "..."
}
```

## Why it's built this way

This script isn't a bare API wrapper — its design directly fixes 3 failures
I found while adversarially testing AIVI Campus OS in Deliverable 1:

| Problem found in Deliverable 1 | Fix applied here |
|---|---|
| Garbage input ("Meow") produced a full fabricated evaluation | Input is validated **before** any API call — empty/too-short input is rejected with a clear error, never sent to the model |
| Negated skills ("I don't know Python") were read as strengths | System prompt explicitly instructs the model to detect negation and exclude those skills from `top_strengths` |
| No error handling — a bad response or API failure would break silently | Every failure path (missing key, bad JSON, schema mismatch, rate limit, timeout, model overload) returns a structured JSON error instead of crashing |

Full reasoning for the prompt design is in Deliverable 2
(`Deliverable2_System_Prompt_Architecture.docx`).

## Setup

```bash
pip install google-genai --break-system-packages
export GEMINI_API_KEY="your-key-from-aistudio.google.com"
```

## Run

```bash
python resume_matcher.py
```

Runs on a built-in sample resume/JD pair. To use your own text, import
and call `analyze()` directly:

```python
from resume_matcher import analyze
result = analyze(resume_text, job_description)
```

## Error handling covered

- Empty or too-short resume/JD → `invalid_input`
- Missing `google-genai` package → `missing_dependency`
- Missing API key → `missing_api_key`
- Malformed JSON from the model → `invalid_json` (with raw response attached for debugging)
- Response missing required fields / wrong types → `schema_mismatch`
- HTTP 429 (rate limit) → automatic retry with exponential backoff (2s → 4s → 8s), then structured error
- Timeouts → same retry/backoff, then structured error
- HTTP 503 (model overloaded) → same retry/backoff, then structured error — confirmed working against a live 503 during testing

Every failure mode was tested (including a real 503 from Gemini during
development, and a mocked 503-then-success case to verify the retry logic).
The script never crashes and never silently returns a wrong default score.
