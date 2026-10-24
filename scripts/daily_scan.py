"""
Daily automated scan for Hili Hodaya — REAL-TIME version.
Fetches live data from ClinicalTrials.gov, PubMed, bioRxiv/medRxiv,
then sends to Claude for intelligent analysis and Hebrew summaries.
Runs via GitHub Actions every day at 07:00 Israel time.
"""

import json
import os
import sys
import time
from datetime import date, timedelta
import anthropic
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIALS_PATH = os.path.join(REPO_ROOT, "data", "trials.json")
DEVELOPMENTS_PATH = os.path.join(REPO_ROOT, "data", "developments.json")
STATE_PATH = os.path.join(REPO_ROOT, "data", "research-state.json")

TODAY = str(date.today())
THIRTY_DAYS_AGO = str(date.today() - timedelta(days=30))
SEVEN_DAYS_AGO = str(date.today() - timedelta(days=7))

PATIENT_PROFILE = {
    "display_name": "הילי הודיה",
    "dob": "2022-05-06",
    "age_months": 29,
    "diagnosis": "Diffuse Midline Glioma H3K27-altered",
    "mutations": ["H3K27M", "EGFR exon 20 insertion", "TP53 x3"],
    "grade": "WHO Grade 4",
    "location": "Right thalamus + brainstem",
    "rt_completed": "2026-08-08",
    "critical_note": "EGFR exon 20 insertion REDUCES ONC201/dordaviprone efficacy (SNO 2025 abstract DMG-63). This patient needs non-ONC201 trials or trials that allow EGFR exon 20.",
}

CT_SEARCHES = [
    {"query.cond": "diffuse midline glioma"},
    {"query.cond": "H3K27M glioma pediatric"},
    {"query.cond": "DIPG pediatric"},
    {"query.intr": "ONC201"},
    {"query.intr": "dordaviprone"},
    {"query.intr": "BLU-5082"},
    {"query.intr": "ACT001"},
    {"query.term": "thalamic glioma pediatric"},
    {"query.term": "EGFR glioma pediatric"},
]

PUBMED_QUERIES = [
    "H3K27M diffuse midline glioma",
    "diffuse midline glioma treatment",
    "DIPG treatment 2025 2026",
    "EGFR exon 20 glioma",
    "ONC201 dordaviprone glioma",
    "BLU-5082 brain tumor",
    "pediatric high grade glioma immunotherapy",
    "thalamic glioma pediatric",
    "H3K27M mutation treatment",
    "ACT001 glioma",
    "diffuse intrinsic pontine glioma",
]

PREPRINT_KEYWORDS = [
    "diffuse midline glioma", "H3K27M", "DIPG", "DMG",
    "ONC201", "dordaviprone", "BLU-5082", "ACT001",
    "pediatric glioma", "thalamic glioma", "EGFR glioma",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_get(url, params=None, timeout=30):
    try:
        r = requests.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": "HiliResearch/1.0 (medical research bot)"})
        if r.status_code == 200:
            return r
    except Exception as e:
        print(f"  HTTP error {url}: {e}")
    return None


def fetch_clinical_trials(known_nct_ids):
    """Fetch recruiting/not-yet-recruiting trials from ClinicalTrials.gov API v2."""
    results = []
    seen = set(known_nct_ids)

    for search_params in CT_SEARCHES:
        params = {
            **search_params,
            "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING",
            "format": "json",
            "pageSize": 25,
        }
        resp = safe_get("https://clinicaltrials.gov/api/v2/studies", params=params)
        if not resp:
            continue

        for study in resp.json().get("studies", []):
            proto = study.get("protocolSection", {})
            nct_id = proto.get("identificationModule", {}).get("nctId", "")
            if not nct_id or nct_id in seen:
                continue
            seen.add(nct_id)

            elig = proto.get("eligibilityModule", {})
            min_age = elig.get("minimumAge", "")
            max_age = elig.get("maximumAge", "")
            status_mod = proto.get("statusModule", {})
            id_mod = proto.get("identificationModule", {})
            desc_mod = proto.get("descriptionModule", {})
            locs = proto.get("contactsLocationsModule", {}).get("locations", [])
            countries = list({loc.get("country", "") for loc in locs if loc.get("country")})
            interventions = [
                i.get("interventionName", "")
                for i in proto.get("armsInterventionsModule", {}).get("interventions", [])
            ]

            results.append({
                "nct_id": nct_id,
                "title": id_mod.get("briefTitle", ""),
                "status": status_mod.get("overallStatus", ""),
                "phase": str(proto.get("designModule", {}).get("phases", [])),
                "conditions": proto.get("conditionsModule", {}).get("conditions", []),
                "interventions": interventions,
                "countries": countries,
                "min_age": min_age,
                "max_age": max_age,
                "summary": desc_mod.get("briefSummary", "")[:400],
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
            })
        time.sleep(0.5)

    print(f"ClinicalTrials.gov: {len(results)} new trials")
    return results


def fetch_pubmed_papers(known_ids, days=30):
    """Fetch recent papers from PubMed (NCBI E-utilities)."""
    results = []
    seen = set(known_ids)
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    for query in PUBMED_QUERIES:
        resp = safe_get(f"{base}esearch.fcgi", params={
            "db": "pubmed",
            "term": query,
            "datetype": "pdat",
            "reldate": str(days),
            "retmax": "10",
            "retmode": "json",
            "sort": "pub date",
        })
        if not resp:
            continue

        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        new_pmids = [p for p in pmids if p not in seen]
        if not new_pmids:
            continue
        seen.update(new_pmids)

        resp2 = safe_get(f"{base}esummary.fcgi", params={
            "db": "pubmed",
            "id": ",".join(new_pmids),
            "retmode": "json",
        })
        if not resp2:
            continue

        summaries = resp2.json().get("result", {})
        for pmid in new_pmids:
            s = summaries.get(pmid, {})
            if not s or "title" not in s:
                continue
            results.append({
                "pmid": pmid,
                "title": s.get("title", ""),
                "authors": ", ".join(a.get("name", "") for a in s.get("authors", [])[:4]),
                "journal": s.get("fulljournalname", ""),
                "pub_date": s.get("pubdate", ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        time.sleep(0.3)

    print(f"PubMed: {len(results)} new papers")
    return results


def fetch_preprints(days=14):
    """Fetch recent preprints from medRxiv (clinical) and bioRxiv (basic science)."""
    results = []
    date_from = str(date.today() - timedelta(days=days))
    date_to = TODAY

    for server in ["medrxiv", "biorxiv"]:
        cursor = 0
        while cursor < 200:
            resp = safe_get(
                f"https://api.biorxiv.org/details/{server}/{date_from}/{date_to}/{cursor}/json"
            )
            if not resp:
                break
            data = resp.json()
            collection = data.get("collection", [])
            if not collection:
                break

            for paper in collection:
                title = paper.get("title", "")
                abstract = paper.get("abstract", "")
                text = (title + " " + abstract).lower()
                if any(kw.lower() in text for kw in PREPRINT_KEYWORDS):
                    doi = paper.get("doi", "")
                    results.append({
                        "id": f"{server}:{doi}",
                        "title": title,
                        "abstract": abstract[:350],
                        "date": paper.get("date", ""),
                        "server": server,
                        "url": f"https://doi.org/{doi}",
                    })
            cursor += 100
            if len(collection) < 100:
                break
            time.sleep(0.3)

    print(f"Preprints: {len(results)} new preprints")
    return results


def analyze_with_claude(client, trials_raw, papers_raw, preprints_raw, known_trial_ids, state):
    """Send fetched data to Claude for intelligent analysis and prioritization."""

    # Compact representations for prompt
    trials_summary = json.dumps([{
        "nct": t["nct_id"],
        "title": t["title"][:120],
        "status": t["status"],
        "min_age": t["min_age"],
        "max_age": t["max_age"],
        "conditions": t["conditions"][:3],
        "interventions": t["interventions"][:3],
        "countries": t["countries"][:5],
        "url": t["url"],
    } for t in trials_raw], ensure_ascii=False)

    papers_summary = json.dumps([{
        "pmid": p["pmid"],
        "title": p["title"][:120],
        "journal": p["journal"],
        "date": p["pub_date"],
        "url": p["url"],
    } for p in papers_raw], ensure_ascii=False)

    preprints_summary = json.dumps([{
        "id": p["id"],
        "title": p["title"][:120],
        "abstract": p["abstract"][:250],
        "date": p["date"],
        "url": p["url"],
    } for p in preprints_raw], ensure_ascii=False)

    known_devs = state.get("known_development_ids", [])

    prompt = f"""You are a pediatric neuro-oncology research analyst. Analyze the following REAL-TIME data fetched today ({TODAY}) for a specific patient:

PATIENT PROFILE:
- Age: ~29 months (DOB 2022-05-06)
- Diagnosis: Diffuse Midline Glioma H3K27-altered
- Mutations: H3K27M, EGFR exon 20 insertion, TP53 ×3
- Grade: WHO Grade 4, Right thalamus + brainstem
- RT completed: August 8, 2026
- CRITICAL: EGFR exon 20 insertion REDUCES ONC201/dordaviprone efficacy
- Display name for dashboard: הילי הודיה

ALREADY KNOWN TRIAL IDs (skip these): {json.dumps(known_trial_ids)}
ALREADY KNOWN DEVELOPMENT IDs (skip these): {json.dumps(known_devs)}

=== NEWLY FETCHED CLINICAL TRIALS (ClinicalTrials.gov, recruiting today) ===
{trials_summary[:3000]}

=== NEWLY FETCHED RESEARCH PAPERS (PubMed, last 30 days) ===
{papers_summary[:2000]}

=== NEWLY FETCHED PREPRINTS (medRxiv/bioRxiv, last 14 days) ===
{preprints_summary[:1500]}

YOUR TASK:
1. From the trials list: identify which are potentially relevant for this specific patient (age-eligible, H3K27M relevant, not excluded by EGFR exon 20)
2. From papers/preprints: identify significant findings about DMG treatment, H3K27M, EGFR exon 20 resistance, or new therapeutic approaches
3. Flag anything urgent (window closing, new drug approval, compassionate use)
4. Provide Hebrew summaries for the dashboard

CRITICAL RULES:
- If a trial uses ONC201/dordaviprone as PRIMARY treatment, note the EGFR exon 20 resistance risk
- Age eligibility: patient is ~29 months old — must accept children under 3 or not specify minimum age
- Only include trials/developments NOT in the "already known" lists above
- If nothing new or relevant found, say so honestly

OUTPUT: Valid JSON only, no other text:
{{
  "new_trials": [
    {{
      "nct_id": "NCT...",
      "title_he": "כותרת בעברית קצרה",
      "title_en": "English title",
      "sponsor": "Institution",
      "phase": "Phase 1",
      "match_category": "reasonable|possible|follow",
      "priority": "RED|ORANGE|WATCH",
      "score": 75,
      "age_eligible": true,
      "recruiting_status": "Recruiting",
      "locations": ["City, Country"],
      "summary_he": "סיכום בעברית",
      "key_inclusion": "H3K27M required, age X+",
      "egfr_note": "relevant note if ONC201 involved, else null",
      "action_he": "פעולה מומלצת",
      "source_url": "https://clinicaltrials.gov/study/NCT..."
    }}
  ],
  "new_developments": [
    {{
      "id": "UNIQUE-ID-CAPS-{TODAY.replace('-','')}",
      "title_he": "כותרת",
      "title_en": "English title",
      "category": "research|regulatory|access|trial",
      "priority": "RED|ORANGE|WATCH",
      "score": 70,
      "summary_he": "סיכום בעברית",
      "action_he": "פעולה",
      "source": "PubMed/bioRxiv/etc",
      "source_url": "https://..."
    }}
  ],
  "key_findings": ["Finding 1", "Finding 2"],
  "changes_detected": true,
  "scan_notes": "Brief summary of what was checked and found"
}}"""

    for attempt in range(3):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            print(f"Claude attempt {attempt + 1} failed: {type(e).__name__}: {e}")
            if attempt == 2:
                raise
            time.sleep(15)


def run_scan():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)
    print(f"Key length: {len(api_key)}, prefix: {api_key[:12]}")
    print(f"Starting real-time scan for {TODAY}...")

    trials = load_json(TRIALS_PATH)
    developments = load_json(DEVELOPMENTS_PATH)
    state = load_json(STATE_PATH)

    known_trial_ids = state.get("known_trial_ids", [t.get("nct_id","") for t in trials])
    known_dev_ids = state.get("known_development_ids", [d.get("id","") for d in developments])
    known_pmids = state.get("known_pmids", [])
    known_preprint_ids = state.get("known_preprint_ids", [])

    # === REAL-TIME FETCHING ===
    print("\n--- Fetching ClinicalTrials.gov ---")
    trials_raw = fetch_clinical_trials(known_trial_ids)

    print("\n--- Fetching PubMed ---")
    papers_raw = fetch_pubmed_papers(known_pmids)

    print("\n--- Fetching preprints ---")
    preprints_raw = fetch_preprints(days=14)

    print(f"\nTotal fetched: {len(trials_raw)} trials, {len(papers_raw)} papers, {len(preprints_raw)} preprints")

    # === CLAUDE ANALYSIS ===
    print("\n--- Analyzing with Claude ---")
    client = anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=90.0)
    result = analyze_with_claude(client, trials_raw, papers_raw, preprints_raw, known_trial_ids, state)

    new_trials = result.get("new_trials", [])
    new_developments = result.get("new_developments", [])
    key_findings = result.get("key_findings", [])
    changes_detected = result.get("changes_detected", False)

    print(f"Claude found: {len(new_trials)} relevant trials, {len(new_developments)} developments")

    # === APPLY UPDATES ===
    for trial in new_trials:
        nct_id = trial.get("nct_id")
        if nct_id and nct_id not in known_trial_ids:
            trials.append(trial)
            known_trial_ids.append(nct_id)
            state.setdefault("trial_statuses", {})[nct_id] = trial.get("recruiting_status", "Unknown")
            print(f"  + Trial: {nct_id}")

    for dev in new_developments:
        dev_id = dev.get("id")
        if dev_id and dev_id not in known_dev_ids:
            developments.append(dev)
            known_dev_ids.append(dev_id)
            print(f"  + Development: {dev_id}")

    # Track all fetched IDs to avoid re-fetching
    for p in papers_raw:
        pmid = p.get("pmid")
        if pmid and pmid not in known_pmids:
            known_pmids.append(pmid)
    for pp in preprints_raw:
        pid = pp.get("id")
        if pid and pid not in known_preprint_ids:
            known_preprint_ids.append(pid)

    # Update state
    state["_meta"]["last_full_scan"] = TODAY
    state["known_trial_ids"] = known_trial_ids
    state["known_development_ids"] = known_dev_ids
    state["known_pmids"] = known_pmids[-500:]   # keep last 500
    state["known_preprint_ids"] = known_preprint_ids[-500:]
    state["last_run_summary"] = {
        "date": TODAY,
        "type": "DAILY_REALTIME",
        "changes_detected": changes_detected,
        "new_trials_added": len(new_trials),
        "new_developments_added": len(new_developments),
        "fetched": {
            "clinical_trials": len(trials_raw),
            "pubmed_papers": len(papers_raw),
            "preprints": len(preprints_raw),
        },
        "key_findings": key_findings,
        "scan_notes": result.get("scan_notes", ""),
        "sources_checked": [
            "ClinicalTrials.gov API v2 (live)",
            "PubMed E-utilities (last 30 days)",
            "medRxiv API (last 14 days)",
            "bioRxiv API (last 14 days)",
        ],
    }

    save_json(TRIALS_PATH, trials)
    save_json(DEVELOPMENTS_PATH, developments)
    save_json(STATE_PATH, state)

    print(f"\nScan complete. Changes detected: {changes_detected}")
    if key_findings:
        for f in key_findings:
            print(f"  → {f}")
    return changes_detected


if __name__ == "__main__":
    run_scan()
