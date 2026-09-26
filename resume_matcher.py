"""
AIVI AI Engineer Intern Challenge — Deliverable 3
Resume-to-Job-Description Matcher (Gemini API)

Author: Dhruv Sharma

What this does:
  Takes raw resume text + a job description, sends both to Gemini with
  a strict system prompt (see Deliverable 2), and returns clean JSON:
      { match_score, top_strengths, missing_skills, summary }

Design notes (directly from Deliverable 1 & 2 findings):
  - Input validation BEFORE calling the API — garbage/empty input never
    reaches the model, so we can't reproduce the "Meow" fabrication bug.
  - The system prompt explicitly tells the model to respect negation
    ("I don't know Python") instead of keyword-matching blindly.
  - All API/JSON failures are caught and returned as structured error
    JSON — never a crash, never a silently wrong default score.

Usage:
    python resume_matcher.py

Requires:
    pip install google-genai --break-system-packages
    export GEMINI_API_KEY="your-key-here"
"""

import os
import sys
import json
import time


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-3.8-flash"
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2  # 2s, 4s, 8s
REQUEST_TIMEOUT_SECONDS = 15

SYSTEM_PROMPT = """You are a resume-to-job-description matching engine.
You will be given RESUME TEXT and a JOB DESCRIPTION. Follow these rules
exactly:

1. Read the resume carefully. If a skill is mentioned alongside negation
   or self-disclaimer (e.g. "I don't know X", "no experience in X",
   "rejected", "didn't get"), do NOT count it as a strength. Do not list
   it under top_strengths.

2. Only credit skills/experience that are genuinely present and
   affirmatively stated in the resume text.

3. Compare against the job description to compute a realistic
   match_score from 0 to 100. Do not default to a fixed or templated
   score — base it on actual overlap between resume and JD.

4. Respond with ONLY valid JSON matching this exact schema, and nothing
   else (no markdown, no preamble, no code fences):

{
  "match_score": <integer 0-100>,
  "top_strengths": [<string>, ...],
  "missing_skills": [<string>, ...],
  "summary": "<exactly 2 lines summarizing fit>"
}
"""


# ---------------------------------------------------------------------------
# Input validation (fixes the "Meow" / garbage-input bug from Deliverable 1)
# ---------------------------------------------------------------------------

def validate_input(resume_text: str, job_description: str) -> str | None:
    """Return an error message if input is unusable, else None."""
    if not resume_text or not resume_text.strip():
        return "Resume text is empty."
    if not job_description or not job_description.strip():
        return "Job description is empty."
    if len(resume_text.strip()) < 20:
        return "Resume text is too short to be a real resume (minimum 20 characters)."
    if len(job_description.strip()) < 20:
        return "Job description is too short to be meaningful (minimum 20 characters)."
    return None


# ---------------------------------------------------------------------------
# JSON sanitizing — Gemini sometimes wraps output in ```json fences
# ---------------------------------------------------------------------------

def extract_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    return json.loads(cleaned)  # may raise json.JSONDecodeError — caller handles it


def validate_schema(data: dict) -> str | None:
    """Return an error message if the parsed JSON doesn't match the expected shape."""
    required_keys = {"match_score", "top_strengths", "missing_skills", "summary"}
    missing = required_keys - data.keys()
    if missing:
        return f"Response missing required fields: {sorted(missing)}"
    if not isinstance(data["match_score"], int):
        return "match_score must be an integer."
    if not (0 <= data["match_score"] <= 100):
        return "match_score out of range (0-100)."
    if not isinstance(data["top_strengths"], list):
        return "top_strengths must be a list."
    if not isinstance(data["missing_skills"], list):
        return "missing_skills must be a list."
    if not isinstance(data["summary"], str):
        return "summary must be a string."
    return None


# ---------------------------------------------------------------------------
# Gemini call with retry / backoff / timeout handling
# ---------------------------------------------------------------------------

def call_gemini(resume_text: str, job_description: str) -> dict:
    """
    Calls Gemini and returns parsed, schema-validated JSON.
    On any failure (network, rate limit, timeout, bad JSON), returns a
    structured error dict instead of raising — this script never crashes.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {
            "error": "missing_dependency",
            "message": "google-genai package not installed. Run: pip install google-genai --break-system-packages",
        }

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {
            "error": "missing_api_key",
            "message": "Set the GEMINI_API_KEY environment variable before running this script.",
        }

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return {"error": "client_init_failed", "message": str(e)}

    user_prompt = (
        f"RESUME TEXT:\n{resume_text}\n\n"
        f"JOB DESCRIPTION:\n{job_description}"
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=1024,
                ),
            )
            raw_text = response.text

            try:
                parsed = extract_json(raw_text)
            except json.JSONDecodeError as e:
                last_error = {
                    "error": "invalid_json",
                    "message": f"Model did not return valid JSON: {e}",
                    "raw_response": raw_text[:500],
                }
                # Bad JSON isn't a rate limit / timeout issue — retrying won't
                # help much, but one retry is cheap in case it was a fluke.
                if attempt < MAX_RETRIES:
                    time.sleep(BASE_BACKOFF_SECONDS)
                    continue
                return last_error

            schema_error = validate_schema(parsed)
            if schema_error:
                last_error = {
                    "error": "schema_mismatch",
                    "message": schema_error,
                    "raw_response": parsed,
                }
                if attempt < MAX_RETRIES:
                    time.sleep(BASE_BACKOFF_SECONDS)
                    continue
                return last_error

            return parsed  # success

        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "rate" in err_str or "quota" in err_str
            is_timeout = "timeout" in err_str or "deadline" in err_str
            is_overloaded = "503" in err_str or "overloaded" in err_str or "unavailable" in err_str

            if (is_rate_limit or is_timeout or is_overloaded) and attempt < MAX_RETRIES:
                backoff = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))  # 2s, 4s, 8s
                time.sleep(backoff)
                last_error = {
                    "error": "rate_limited" if is_rate_limit else ("timeout" if is_timeout else "model_overloaded"),
                    "message": f"Retrying after transient error (attempt {attempt}/{MAX_RETRIES}): {e}",
                }
                continue

            return {
                "error": "rate_limited" if is_rate_limit else ("timeout" if is_timeout else ("model_overloaded" if is_overloaded else "api_error")),
                "message": str(e),
            }

    return last_error or {"error": "unknown", "message": "Exhausted retries with no clear error."}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze(resume_text: str, job_description: str) -> dict:
    """
    Full pipeline: validate -> call Gemini -> return strict JSON.
    Always returns a dict; never raises.
    """
    validation_error = validate_input(resume_text, job_description)
    if validation_error:
        return {"error": "invalid_input", "message": validation_error}

    return call_gemini(resume_text, job_description)


if __name__ == "__main__":
    sample_resume = (
        "I completed a diploma in Computer Science. I built a Python "
        "project that implements a basic calculator. I do not know "
        "advanced DSA or backend frameworks yet. Currently learning "
        "Flask and REST APIs."
    )
    sample_jd = (
        "Looking for a Python developer familiar with Flask, REST APIs, "
        "and basic data structures. Entry-level, willingness to learn "
        "is valued."
    )

    result = analyze(sample_resume, sample_jd)
    print(json.dumps(result, indent=2))
