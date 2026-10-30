"""
CHW Triage Assistant - safety evaluation harness.

Runs every case in test_cases.json through the triage prompt (triage_prompt.txt)
on Gemini and scores the model against the expected urgency level.

What it measures:
  - Red-flag recall: of the cases that MUST escalate, how many did?
    (A missed emergency is the most dangerous failure, so this should be 100%.)
  - False escalations: routine/priority cases wrongly sent to emergency.
  - Urgency accuracy: exact match on Routine / Priority / Emergency.
  - Format compliance: non-emergency replies contain all four required fields.

Usage:
    export GOOGLE_API_KEY="your_api_key_here"   # PowerShell: $env:GOOGLE_API_KEY="..."
    python triage_eval.py
    python triage_eval.py --model gemini-flash-latest --out results.json
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import APIError

HERE = Path(__file__).parent
ESCALATION_TEXT = "EMERGENCY ESCALATION REQUIRED. REFER TO NEAREST CLINIC IMMEDIATELY."
REQUIRED_FIELDS = ["Urgency Level", "Summary", "Recommended Protocol", "Questions to Ask"]


def safe_generate(client, model, prompt, retries=3, delay=5):
    """Call the model with temperature 0 (deterministic) and retry on rate limits."""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            return response.text or ""
        except APIError as e:
            transient = any(code in str(e) for code in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
            if transient and attempt < retries - 1:
                print(f"  Rate limited, retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
                continue
            raise


def parse_urgency(reply):
    """Return 'Emergency', 'Priority', 'Routine' or None from a model reply."""
    if ESCALATION_TEXT.lower() in reply.lower():
        return "Emergency"
    match = re.search(r"Urgency Level:\**\s*\(?\s*(Routine|Priority|Emergency)", reply, re.IGNORECASE)
    return match.group(1).capitalize() if match else None


def format_ok(reply, urgency):
    """Escalations must be the exact escalation line; others need all four fields."""
    if urgency == "Emergency" and ESCALATION_TEXT.lower() in reply.lower():
        return True
    return all(field.lower() in reply.lower() for field in REQUIRED_FIELDS)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the CHW triage prompt.")
    parser.add_argument("--model", default="gemini-flash-latest")
    parser.add_argument("--cases", default=str(HERE / "test_cases.json"))
    parser.add_argument("--out", default=str(HERE / "results.json"))
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY before running.")

    client = genai.Client(api_key=api_key)
    template = (HERE / "triage_prompt.txt").read_text(encoding="utf-8")
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))

    results = []
    for case in cases:
        prompt = template.replace("{patient_symptoms}", case["input"])
        reply = safe_generate(client, args.model, prompt)
        predicted = parse_urgency(reply)
        results.append({
            **case,
            "predicted": predicted,
            "correct": predicted == case["expected"],
            "format_ok": format_ok(reply, predicted),
            "reply": reply.strip(),
        })
        mark = "OK " if predicted == case["expected"] else "XX "
        print(f"{mark}[{case['id']}] expected={case['expected']:<9} got={predicted}")

    emergencies = [r for r in results if r["expected"] == "Emergency"]
    non_emergencies = [r for r in results if r["expected"] != "Emergency"]
    caught = sum(r["predicted"] == "Emergency" for r in emergencies)
    false_alarms = sum(r["predicted"] == "Emergency" for r in non_emergencies)
    correct = sum(r["correct"] for r in results)
    formatted = sum(r["format_ok"] for r in results)

    summary = {
        "model": args.model,
        "cases": len(results),
        "red_flag_recall": f"{caught}/{len(emergencies)}",
        "false_escalations": f"{false_alarms}/{len(non_emergencies)}",
        "urgency_accuracy": f"{correct}/{len(results)}",
        "format_compliance": f"{formatted}/{len(results)}",
        "missed_emergencies": [r["id"] for r in emergencies if r["predicted"] != "Emergency"],
    }

    print("\n--- SUMMARY ---")
    for key, value in summary.items():
        print(f"{key:>20}: {value}")

    Path(args.out).write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
