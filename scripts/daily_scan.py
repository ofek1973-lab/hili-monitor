"""
Daily automated scan for Hili Hodaya Shmuel — clinical trials and DMG developments.
Runs via GitHub Actions every day at 07:00 Israel time.
"""

import json
import os
import sys
import time
from datetime import date
import anthropic
import httpx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIALS_PATH = os.path.join(REPO_ROOT, "data", "trials.json")
DEVELOPMENTS_PATH = os.path.join(REPO_ROOT, "data", "developments.json")
STATE_PATH = os.path.join(REPO_ROOT, "data", "research-state.json")

PATIENT_CONTEXT = """
PATIENT PROFILE (confidential — do NOT include identifying details in output):
- Display name: הילי הודיה (Hili Hodaya) — no family name in dashboard
- Diagnosis: Diffuse Midline Glioma H3K27-altered, EGFR exon 20 insertion, TP53 ×3, WHO Grade 4
- Location: Right thalamus + brainstem
- DOB: 2022-05-06 (age ~4.5)
- RT completed: ~August 8, 2026
- CRITICAL: EGFR exon 20 insertion reduces ONC201/dordaviprone efficacy (SNO 2025 abstract DMG-63)
"""

RESEARCH_PROMPT = """You are a medical research agent performing a daily scan for pediatric DMG clinical trials.

{patient_context}

Today's date: {today}

CURRENT DATABASE STATE:
Known trial IDs: {known_trial_ids}
Known development IDs: {known_development_ids}
Last scan: {last_scan}

YOUR TASK:
1. Search for NEW clinical trials and developments NOT already in the database above.
2. Check for STATUS CHANGES in existing trials (recruiting → closed, new sites, etc.)

SEARCH SCOPE — check all of these:
- ClinicalTrials.gov: H3K27M DMG pediatric, diffuse midline glioma children recruiting
- PubMed: H3K27M diffuse midline glioma treatment 2025 2026
- EUCTR (Europe): diffuse midline glioma pediatric
- PMDA Japan: dordaviprone H3K27M, Modeyso
- jRCT / UMIN Japan: H3K27M pediatric
- SNO/ASCO/SIOPE abstracts: new DMG results
- Israeli hospitals: Sheba, Schneider, Hadassah, Rambam, Sourasky, Soroka, Rabin
- BLU-5082 (Blueprint Medicines) — EGFR TKI brain-penetrant (critical for this patient's EGFR exon 20)
- ACT001/CONNECT2110 — SickKids Toronto, European sites
- WHO ICTRP: H3K27M pediatric
- cancer.org.il Israeli trials

OUTPUT FORMAT — respond with valid JSON only, no other text:
{{
  "new_trials": [
    {{
      "nct_id": "NCT...",
      "title_he": "כותרת בעברית",
      "title_en": "English title",
      "sponsor": "Institution name",
      "phase": "Phase 1",
      "match_category": "reasonable|possible|follow",
      "priority": "RED|ORANGE|WATCH",
      "score": 75,
      "age_eligible": true,
      "recruiting_status": "Recruiting",
      "locations": ["City, Country"],
      "summary_he": "סיכום בעברית",
      "key_inclusion": "H3K27M required, age ≥ X months",
      "action_he": "פעולה מומלצת",
      "source_url": "https://clinicaltrials.gov/..."
    }}
  ],
  "updated_trials": [
    {{
      "nct_id": "NCT...",
      "field": "recruiting_status",
      "old_value": "Recruiting",
      "new_value": "Completed"
    }}
  ],
  "new_developments": [
    {{
      "id": "UNIQUE-ID-CAPS",
      "title_he": "כותרת",
      "title_en": "English title",
      "category": "regulatory|research|access|trial",
      "priority": "RED|ORANGE|WATCH",
      "score": 70,
      "summary_he": "סיכום",
      "action_he": "פעולה",
      "source": "Source name",
      "source_url": "https://..."
    }}
  ],
  "key_findings": [
    "Finding 1 (one sentence)",
    "Finding 2"
  ],
  "changes_detected": true,
  "scan_notes": "Brief note about what was checked"
}}

IMPORTANT RULES:
- Return ONLY trials/developments NOT already in the known IDs lists above
- Prioritize anything with EGFR exon 20 relevance (this patient has this mutation)
- Mark any ONC201/dordaviprone trial with a note about EGFR resistance risk
- If nothing new found, return empty arrays and changes_detected: false
- Respond ONLY with valid JSON — no markdown, no explanation
"""


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_scan():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    trials = load_json(TRIALS_PATH)
    developments = load_json(DEVELOPMENTS_PATH)
    state = load_json(STATE_PATH)

    known_trial_ids = state.get("known_trial_ids", [])
    known_dev_ids = state.get("known_development_ids", [])
    last_scan = state.get("_meta", {}).get("last_full_scan", "unknown")
    today = str(date.today())

    prompt = RESEARCH_PROMPT.format(
        patient_context=PATIENT_CONTEXT,
        today=today,
        known_trial_ids=json.dumps(known_trial_ids),
        known_development_ids=json.dumps(known_dev_ids),
        last_scan=last_scan,
    )

    print(f"Starting daily scan for {today}...")
    client = anthropic.Anthropic(
        api_key=api_key,
        max_retries=3,
        timeout=httpx.Timeout(120.0, connect=30.0),
    )

    for attempt in range(3):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            if attempt == 2:
                raise
            print(f"Connection attempt {attempt + 1} failed: {e}. Retrying in 15s...")
            time.sleep(15)

    raw = message.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)

    new_trials = result.get("new_trials", [])
    updated_trials = result.get("updated_trials", [])
    new_developments = result.get("new_developments", [])
    key_findings = result.get("key_findings", [])
    changes_detected = result.get("changes_detected", False)

    print(f"New trials found: {len(new_trials)}")
    print(f"Updated trials: {len(updated_trials)}")
    print(f"New developments: {len(new_developments)}")

    # Apply new trials
    for trial in new_trials:
        nct_id = trial.get("nct_id")
        if nct_id and nct_id not in known_trial_ids:
            trials.append(trial)
            known_trial_ids.append(nct_id)
            state["trial_statuses"][nct_id] = trial.get("recruiting_status", "Unknown")
            print(f"  Added trial: {nct_id}")

    # Apply trial updates
    for update in updated_trials:
        nct_id = update.get("nct_id")
        field = update.get("field")
        new_val = update.get("new_value")
        for t in trials:
            if t.get("nct_id") == nct_id and field:
                t[field] = new_val
                if field == "recruiting_status":
                    state["trial_statuses"][nct_id] = new_val
                print(f"  Updated {nct_id}.{field} -> {new_val}")

    # Apply new developments
    for dev in new_developments:
        dev_id = dev.get("id")
        if dev_id and dev_id not in known_dev_ids:
            developments.append(dev)
            known_dev_ids.append(dev_id)
            print(f"  Added development: {dev_id}")

    # Update state
    state["_meta"]["last_full_scan"] = today
    state["_meta"]["next_scheduled_scan"] = today
    state["known_trial_ids"] = known_trial_ids
    state["known_development_ids"] = known_dev_ids
    state["last_run_summary"] = {
        "date": today,
        "type": "DAILY_AUTOMATED",
        "items_found": len(trials),
        "trials_analyzed": len(trials),
        "changes_detected": changes_detected,
        "new_trials_added": len(new_trials),
        "new_developments_added": len(new_developments),
        "key_findings": key_findings,
        "scan_notes": result.get("scan_notes", ""),
        "sources_checked": [
            "ClinicalTrials.gov API v2",
            "PubMed",
            "EUCTR",
            "PMDA Japan",
            "jRCT / UMIN Japan",
            "SNO/ASCO abstracts",
            "Israeli hospitals",
            "Blueprint Medicines (BLU-5082)",
            "WHO ICTRP",
            "cancer.org.il",
        ],
    }

    save_json(TRIALS_PATH, trials)
    save_json(DEVELOPMENTS_PATH, developments)
    save_json(STATE_PATH, state)

    print(f"Scan complete. Changes detected: {changes_detected}")
    return changes_detected


if __name__ == "__main__":
    run_scan()
