"""
Daily automated scan for Hili Hodaya — REAL-TIME version.
Fetches live data from ClinicalTrials.gov, PubMed, bioRxiv/medRxiv, and RSS feeds.
Runs via GitHub Actions every day at 07:00 Israel time.
"""

import json
import os
import sys
import time
from datetime import date, timedelta
from xml.etree import ElementTree as ET
import anthropic
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIALS_PATH = os.path.join(REPO_ROOT, "data", "trials.json")
DEVELOPMENTS_PATH = os.path.join(REPO_ROOT, "data", "developments.json")
STATE_PATH = os.path.join(REPO_ROOT, "data", "research-state.json")

TODAY = str(date.today())
THIRTY_DAYS_AGO = str(date.today() - timedelta(days=30))

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

KEYWORDS = [
    "diffuse midline glioma", "H3K27M", "H3 K27", "DIPG", "DMG",
    "ONC201", "dordaviprone", "BLU-5082", "ACT001",
    "pediatric glioma", "thalamic glioma", "EGFR glioma",
    "pediatric brain tumor", "pediatric brain cancer",
]

RSS_FEEDS = [
    ("Neuro-Oncology Journal", "https://academic.oup.com/rss/site_5252/advanceAccess.xml"),
    ("Lancet Oncology",        "https://www.thelancet.com/rssfeed/lanonc_online.xml"),
    ("Nature Medicine",        "https://www.nature.com/nm.rss"),
    ("JCO - ASCO",             "https://ascopubs.org/rss/feed/journal/jco"),
    ("Cancer Discovery",       "https://cancerdiscovery.aacrjournals.org/rss/current.xml"),
    ("NCI News",               "https://www.cancer.gov/news-events/press-releases/feed"),
    ("Pediatric Blood Cancer", "https://onlinelibrary.wiley.com/feed/10969810/most-recent"),
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_get(url, params=None, timeout=25):
    try:
        r = requests.get(
            url, params=params, timeout=timeout,
            headers={"User-Agent": "HiliResearch/2.0 (medical research bot)"},
        )
        if r.status_code == 200:
            return r
    except Exception as e:
        print(f"  HTTP error {url[:60]}: {e}")
    return None


def fetch_clinical_trials(known_nct_ids):
    """Fetch recruiting trials from ClinicalTrials.gov API v2."""
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
                "status": proto.get("statusModule", {}).get("overallStatus", ""),
                "phase": str(proto.get("designModule", {}).get("phases", [])),
                "conditions": proto.get("conditionsModule", {}).get("conditions", []),
                "interventions": interventions,
                "countries": countries,
                "min_age": elig.get("minimumAge", ""),
                "max_age": elig.get("maximumAge", ""),
                "summary": desc_mod.get("briefSummary", "")[:400],
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
            })
        time.sleep(0.5)

    print(f"ClinicalTrials.gov: {len(results)} new trials")
    return results


def fetch_pubmed_papers(known_pmids, days=30):
    """Fetch recent papers from PubMed."""
    results = []
    seen = set(known_pmids)
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    for query in PUBMED_QUERIES:
        resp = safe_get(f"{base}esearch.fcgi", params={
            "db": "pubmed", "term": query,
            "datetype": "pdat", "reldate": str(days),
            "retmax": "10", "retmode": "json", "sort": "pub date",
        })
        if not resp:
            continue
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        new_pmids = [p for p in pmids if p not in seen]
        if not new_pmids:
            continue
        seen.update(new_pmids)
        resp2 = safe_get(f"{base}esummary.fcgi", params={
            "db": "pubmed", "id": ",".join(new_pmids), "retmode": "json",
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
    """Fetch relevant preprints from medRxiv and bioRxiv."""
    results = []
    date_from = str(date.today() - timedelta(days=days))

    for server in ["medrxiv", "biorxiv"]:
        cursor = 0
        while cursor < 200:
            resp = safe_get(f"https://api.biorxiv.org/details/{server}/{date_from}/{TODAY}/{cursor}/json")
            if not resp:
                break
            collection = resp.json().get("collection", [])
            if not collection:
                break
            for paper in collection:
                text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
                if any(kw.lower() in text for kw in KEYWORDS):
                    doi = paper.get("doi", "")
                    results.append({
                        "id": f"{server}:{doi}",
                        "title": paper.get("title", ""),
                        "abstract": paper.get("abstract", "")[:350],
                        "date": paper.get("date", ""),
                        "server": server,
                        "url": f"https://doi.org/{doi}",
                    })
            cursor += 100
            if len(collection) < 100:
                break
            time.sleep(0.3)

    print(f"Preprints: {len(results)} relevant")
    return results


def fetch_rss_news(known_rss_ids):
    """Fetch recent articles from medical journal and news RSS feeds."""
    results = []
    seen = set(known_rss_ids)

    for source_name, url in RSS_FEEDS:
        resp = safe_get(url, timeout=20)
        if not resp:
            continue
        try:
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")
            if not items:
                items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

            for item in items[:25]:
                def _get(tag):
                    el = item.find(tag)
                    if el is None:
                        el = item.find("{http://www.w3.org/2005/Atom}" + tag)
                    return (el.text or "").strip() if el is not None and el.text else ""

                title = _get("title")
                link_el = item.find("link") or item.find("{http://www.w3.org/2005/Atom}link")
                link = ""
                if link_el is not None:
                    link = (link_el.text or link_el.get("href", "")).strip()
                desc = _get("description") or _get("summary") or _get("content")

                if not title:
                    continue
                text = (title + " " + desc).lower()
                if not any(kw.lower() in text for kw in KEYWORDS):
                    continue

                item_id = f"rss:{source_name}:{abs(hash(title)) % 1000000}"
                if item_id in seen:
                    continue
                seen.add(item_id)
                results.append({
                    "id": item_id,
                    "title": title,
                    "description": desc[:400] if desc else "",
                    "url": link,
                    "source": source_name,
                })
        except Exception as e:
            print(f"  RSS parse error {source_name}: {e}")
        time.sleep(0.2)

    print(f"RSS news: {len(results)} relevant articles")
    return results


def analyze_with_claude(client, trials_raw, papers_raw, preprints_raw, rss_raw,
                        known_trial_ids, state):
    """Send all fetched real-time data to Claude for intelligent analysis."""
    known_dev_ids = state.get("known_development_ids", [])

    trials_s = json.dumps([{
        "nct": t["nct_id"], "title": t["title"][:110],
        "status": t["status"], "min_age": t["min_age"],
        "interventions": t["interventions"][:3], "countries": t["countries"][:5],
        "url": t["url"],
    } for t in trials_raw], ensure_ascii=False)

    papers_s = json.dumps([{
        "pmid": p["pmid"], "title": p["title"][:110],
        "journal": p["journal"], "date": p["pub_date"], "url": p["url"],
    } for p in papers_raw], ensure_ascii=False)

    preprints_s = json.dumps([{
        "id": p["id"], "title": p["title"][:110],
        "abstract": p["abstract"][:200], "date": p["date"], "url": p["url"],
    } for p in preprints_raw], ensure_ascii=False)

    rss_s = json.dumps([{
        "id": r["id"], "title": r["title"][:110],
        "desc": r["description"][:180], "source": r["source"], "url": r["url"],
    } for r in rss_raw], ensure_ascii=False)

    prompt = f"""Pediatric neuro-oncology research analyst. Today: {TODAY}.

PATIENT:
- Age: ~29 months (DOB 2022-05-06), female
- Diagnosis: Diffuse Midline Glioma H3K27-altered, Right thalamus + brainstem, WHO Grade 4
- Mutations: H3K27M + EGFR exon 20 insertion + TP53 x3
- RT completed: 2026-08-08
- CRITICAL: EGFR exon 20 insertion REDUCES ONC201 efficacy (SNO 2025)
- Display name for dashboard: הילי הודיה

SKIP — already known trial IDs: {json.dumps(known_trial_ids[:35])}
SKIP — already known dev IDs: {json.dumps(known_dev_ids[:35])}

=== LIVE TRIALS (ClinicalTrials.gov, recruiting today) ===
{trials_s[:2500]}

=== PAPERS (PubMed, last 30 days) ===
{papers_s[:1500]}

=== PREPRINTS (medRxiv/bioRxiv, last 14 days) ===
{preprints_s[:800]}

=== MEDICAL NEWS (RSS: journals, NCI, ASCO) ===
{rss_s[:1200]}

ANALYZE: What is relevant to this specific patient? Not in already-known lists.
- Trials: age ≤ 29 months eligible? EGFR exon 20 issue?
- Papers/news: breakthrough? approval? compassionate access?
- RED = urgent now; ORANGE = important; WATCH = monitor
- Be honest: if nothing truly new, say so.

Return ONLY valid JSON:
{{
  "new_trials": [
    {{
      "nct_id": "NCT...",
      "title_he": "כותרת עברית קצרה",
      "title_en": "English title",
      "phase": "Phase X",
      "match_category": "reasonable|possible|follow|not_eligible",
      "priority": "RED|ORANGE|WATCH",
      "score": 70,
      "age_eligible": true,
      "recruiting_status": "Recruiting",
      "location_he": "מיקום",
      "summary_he": "סיכום בעברית",
      "key_inclusion": ["קריטריון"],
      "egfr_note": "הערה על ONC201/EGFR או null",
      "action_he": "פעולה מומלצת",
      "source_url": "https://clinicaltrials.gov/study/NCT..."
    }}
  ],
  "new_developments": [
    {{
      "id": "UNIQUE-ID-{TODAY.replace('-','')}",
      "priority": "RED|ORANGE|WATCH",
      "is_new": true,
      "title_he": "כותרת",
      "title_en": "English",
      "category": "research|regulatory|access|trial|news",
      "what_changed": "מה חדש",
      "why_matters_bullets": ["סיבה 1"],
      "action": "פעולה",
      "source_url": "https://...",
      "date_detected": "{TODAY}"
    }}
  ],
  "key_findings": ["ממצא 1", "ממצא 2"],
  "changes_detected": true,
  "scan_notes": "what was scanned and found"
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

    # Load existing data — JSON files have wrapper objects {_meta, trials/developments}
    trials_file = load_json(TRIALS_PATH)
    trials_list = trials_file.get("trials", [])
    developments_file = load_json(DEVELOPMENTS_PATH)
    developments_list = developments_file.get("developments", [])
    state = load_json(STATE_PATH)

    known_trial_ids = state.get("known_trial_ids",
                                [t.get("nct_id", "") for t in trials_list])
    known_dev_ids = state.get("known_development_ids",
                              [d.get("id", "") for d in developments_list])
    known_pmids = state.get("known_pmids", [])
    known_preprint_ids = state.get("known_preprint_ids", [])
    known_rss_ids = state.get("known_rss_ids", [])

    # === REAL-TIME FETCHING ===
    print("\n--- Fetching ClinicalTrials.gov ---")
    trials_raw = fetch_clinical_trials(known_trial_ids)

    print("\n--- Fetching PubMed ---")
    papers_raw = fetch_pubmed_papers(known_pmids)

    print("\n--- Fetching preprints ---")
    preprints_raw = fetch_preprints(days=14)

    print("\n--- Fetching RSS news feeds ---")
    rss_raw = fetch_rss_news(known_rss_ids)

    total = len(trials_raw) + len(papers_raw) + len(preprints_raw) + len(rss_raw)
    print(f"\nTotal fetched: {len(trials_raw)} trials | {len(papers_raw)} papers | "
          f"{len(preprints_raw)} preprints | {len(rss_raw)} news — {total} items total")

    # === CLAUDE ANALYSIS ===
    print("\n--- Analyzing with Claude ---")
    client = anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=90.0)
    result = analyze_with_claude(client, trials_raw, papers_raw, preprints_raw, rss_raw,
                                 known_trial_ids, state)

    new_trials = result.get("new_trials", [])
    new_developments = result.get("new_developments", [])
    key_findings = result.get("key_findings", [])
    changes_detected = result.get("changes_detected", False)

    print(f"Claude found: {len(new_trials)} relevant trials, {len(new_developments)} developments")

    # === APPLY UPDATES ===
    for trial in new_trials:
        nct_id = trial.get("nct_id")
        if nct_id and nct_id not in known_trial_ids:
            trials_list.append(trial)
            known_trial_ids.append(nct_id)
            print(f"  + Trial: {nct_id}")

    for dev in new_developments:
        dev_id = dev.get("id")
        if dev_id and dev_id not in known_dev_ids:
            developments_list.append(dev)
            known_dev_ids.append(dev_id)
            print(f"  + Development: {dev_id}")

    # Track all seen IDs to avoid re-processing
    for p in papers_raw:
        if p.get("pmid") and p["pmid"] not in known_pmids:
            known_pmids.append(p["pmid"])
    for pp in preprints_raw:
        if pp.get("id") and pp["id"] not in known_preprint_ids:
            known_preprint_ids.append(pp["id"])
    for r in rss_raw:
        if r.get("id") and r["id"] not in known_rss_ids:
            known_rss_ids.append(r["id"])

    # Update state
    state["_meta"]["last_full_scan"] = TODAY
    state["_meta"]["next_scheduled_scan"] = str(date.today() + timedelta(days=1))
    state["known_trial_ids"] = known_trial_ids
    state["known_development_ids"] = known_dev_ids
    state["known_pmids"] = known_pmids[-500:]
    state["known_preprint_ids"] = known_preprint_ids[-500:]
    state["known_rss_ids"] = known_rss_ids[-1000:]
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
            "rss_articles": len(rss_raw),
        },
        "key_findings": key_findings,
        "scan_notes": result.get("scan_notes", ""),
        "sources_checked": [
            "ClinicalTrials.gov API v2 (live)",
            "PubMed E-utilities (last 30 days)",
            "medRxiv API (last 14 days)",
            "bioRxiv API (last 14 days)",
            "Neuro-Oncology Journal RSS",
            "Lancet Oncology RSS",
            "Nature Medicine RSS",
            "JCO / ASCO RSS",
            "Cancer Discovery RSS",
            "NCI News RSS",
            "Pediatric Blood Cancer Journal RSS",
        ],
    }

    # Save — preserve wrapper objects
    trials_file["trials"] = trials_list
    trials_file["_meta"]["last_updated"] = TODAY
    developments_file["developments"] = developments_list
    developments_file["_meta"]["last_updated"] = TODAY

    save_json(TRIALS_PATH, trials_file)
    save_json(DEVELOPMENTS_PATH, developments_file)
    save_json(STATE_PATH, state)

    print(f"\nScan complete. Changes: {changes_detected}")
    for f in key_findings:
        print(f"  → {f}")
    return changes_detected


if __name__ == "__main__":
    run_scan()
