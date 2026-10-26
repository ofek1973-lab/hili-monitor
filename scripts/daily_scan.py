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
HYPOTHESES_PATH = os.path.join(REPO_ROOT, "data", "hypotheses.json")

TODAY = str(date.today())
THIRTY_DAYS_AGO = str(date.today() - timedelta(days=30))

PUBMED_QUERIES = [
    # Core DMG — pediatric and adult
    "H3K27M diffuse midline glioma pediatric",
    "H3K27M diffuse midline glioma adult",
    "H3.3K27M glioma treatment",
    "H3.1K27M glioma treatment",
    "diffuse midline glioma treatment 2025 2026",
    "diffuse intrinsic pontine glioma treatment",
    "thalamic glioma pediatric treatment",
    # Targeted therapy
    "EGFR exon 20 insertion glioma",
    "ONC201 dordaviprone glioma",
    "BLU-5082 glioma EGFR",
    "ACT001 glioma",
    # Immunotherapy
    "GD2 CAR-T glioma",
    "H3K27M peptide vaccine glioma",
    "pediatric high grade glioma immunotherapy",
    "CAR-T brain tumor pediatric",
    # Combination and BBB
    "blood brain barrier drug delivery glioma",
    "convection enhanced delivery glioma pediatric",
    "combination therapy H3K27M glioma",
    "ONC201 combination therapy glioma",
    # Broader breakthroughs
    "pediatric brain tumor breakthrough 2025 2026",
    "glioma oncolytic virus treatment",
    "pediatric high grade glioma overall survival",
]

CT_SEARCHES = [
    # Core DMG
    {"query.cond": "diffuse midline glioma"},
    {"query.cond": "H3K27M glioma"},
    {"query.cond": "diffuse midline glioma", "query.term": "pediatric"},
    {"query.cond": "DIPG"},
    {"query.cond": "thalamic glioma"},
    # Specific drugs
    {"query.intr": "ONC201"},
    {"query.intr": "dordaviprone"},
    {"query.intr": "BLU-5082"},
    {"query.intr": "ACT001"},
    {"query.intr": "GD2 CAR-T"},
    # Broader pediatric brain tumor
    {"query.cond": "pediatric high grade glioma"},
    {"query.cond": "H3K27 altered glioma"},
    {"query.term": "EGFR exon 20 glioma"},
]

KEYWORDS = [
    "diffuse midline glioma", "H3K27M", "H3 K27", "DIPG", "DMG",
    "ONC201", "dordaviprone", "BLU-5082", "ACT001",
    "pediatric glioma", "thalamic glioma", "EGFR glioma",
    "pediatric brain tumor", "pediatric brain cancer",
    "glioma", "brain tumor", "brain cancer", "neuro-oncology", "neurooncology",
    "pediatric oncology", "high grade glioma", "midline glioma",
]

RSS_FEEDS = [
    ("Neuro-Oncology Journal", "https://academic.oup.com/rss/site_5252/advanceAccess.xml"),
    ("Lancet Oncology",        "https://www.thelancet.com/rssfeed/lanonc_online.xml"),
    ("Nature Medicine",        "https://www.nature.com/nm.rss"),
    ("JCO - ASCO",             "https://ascopubs.org/rss/feed/journal/jco"),
    ("Cancer Discovery",       "https://cancerdiscovery.aacrjournals.org/rss/current.xml"),
    ("NCI News",               "https://www.cancer.gov/news-events/press-releases/feed"),
    ("Pediatric Blood Cancer", "https://onlinelibrary.wiley.com/feed/10969810/most-recent"),
    ("NEJM",                   "https://www.nejm.org/action/showFeed?jc=nejm&type=etoc&feed=rss"),
    ("FDA Drug Approvals",     "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drug-approvals-and-databases/rss.xml"),
    ("SNO News",               "https://www.soc-neuro-onc.org/SNO/RSS/News_RSS.aspx"),
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


def fetch_who_ictrp(known_nct_ids):
    """Fetch international trials from WHO ICTRP (covers EU, Israel, Asia, Australia)."""
    results = []
    seen = set(known_nct_ids)
    queries = ["diffuse midline glioma", "H3K27M glioma", "DIPG pediatric", "ONC201"]
    for q in queries:
        resp = safe_get(
            "https://trialsearch.who.int/Trial2.aspx",
            params={"TrialID": "", "Title": q, "Condition": "", "Phase": "", "Country": "",
                    "RecDate": "", "TrialSponsor": "", "GenderAge": "", "StudyDesign": "",
                    "Status": "Recruiting", "Submit": "Search"},
            timeout=20,
        )
        if not resp:
            continue
        try:
            text = resp.text
            import re
            ids = re.findall(r'(EUCTR\d+-\d+-\d+|ISRCTN\d+|ACTRN\d+|ChiCTR\d+|NL\d+|DRKS\d+)', text)
            titles = re.findall(r'class="[^"]*title[^"]*">([^<]{10,120})<', text)
            urls = re.findall(r'href="(Trial2\.aspx\?TrialID=[^"]+)"', text)
            for i, tid in enumerate(ids[:8]):
                if tid in seen:
                    continue
                seen.add(tid)
                title = titles[i] if i < len(titles) else q + " trial"
                url = f"https://trialsearch.who.int/{urls[i]}" if i < len(urls) else "https://trialsearch.who.int"
                results.append({"nct_id": tid, "title": title.strip(), "status": "Recruiting",
                                 "min_age": "", "interventions": [], "countries": ["International"],
                                 "url": url, "source": "WHO-ICTRP"})
        except Exception as e:
            print(f"  WHO ICTRP parse error: {e}")
        time.sleep(0.5)
    print(f"WHO ICTRP: {len(results)} new international trials")
    return results


def analyze_with_claude(client, trials_raw, papers_raw, preprints_raw, rss_raw,
                        known_trial_ids, state):
    """Send all fetched real-time data to Claude for intelligent analysis."""
    known_dev_ids = state.get("known_development_ids", [])

    trials_s = json.dumps([{
        "nct": t["nct_id"], "title": t["title"][:90],
        "status": t["status"], "min_age": t["min_age"],
        "interventions": t["interventions"][:2], "countries": t["countries"][:3],
        "url": t["url"],
    } for t in trials_raw[:15]], ensure_ascii=False)

    papers_s = json.dumps([{
        "pmid": p["pmid"], "title": p["title"][:90],
        "journal": p["journal"][:50], "date": p["pub_date"], "url": p["url"],
    } for p in papers_raw[:15]], ensure_ascii=False)

    preprints_s = json.dumps([{
        "id": p["id"], "title": p["title"][:90],
        "abstract": p["abstract"][:150], "url": p["url"],
    } for p in preprints_raw[:10]], ensure_ascii=False)

    rss_s = json.dumps([{
        "id": r["id"], "title": r["title"][:90],
        "source": r["source"], "url": r["url"],
    } for r in rss_raw[:10]], ensure_ascii=False)

    prompt = f"""You are a pediatric neuro-oncology research analyst. Your role is a personal medical intelligence assistant — NOT to generate alerts, but to identify information that has genuine clinical relevance to this specific patient.
Today: {TODAY}.

═══ PATIENT PROFILE (verified) ═══
- Age: ~29 months (DOB 2022-05-06), female
- Diagnosis: Diffuse Midline Glioma H3K27-altered, Right thalamus + brainstem, WHO Grade 4
- Mutations: H3K27M + EGFR exon 20 insertion + TP53 ×3
- RT completed: 2026-08-08
- ACTIVE TREATMENT: Zegfrovy (Sunvozertinib) — EGFR exon 20 inhibitor, gen-3, currently receiving
- PENDING TREATMENT: ONC201 (Dordaviprone) — FDA-approved Aug 2025, ordered, en route to Israel, NOT YET STARTED
- CRITICAL: EGFR exon 20 insertion REDUCES ONC201 efficacy (SNO 2025, ICR London data)
- BBB CRITERION: Blood-brain barrier penetration is a PRIMARY selection criterion. Zegfrovy was chosen over Amivantamab partly for superior BBB penetration.
- Display name: הילי הודיה

═══ KEY EVALUATION CRITERIA ═══
For every treatment finding, assess:
1. BBB penetration evidence level: human PK data > animal CNS data > in-vitro > mechanistic hypothesis
2. CNS activity evidence: intracranial response data > CNS disease-specific data > general activity
3. Relevance to mutations: H3K27M | EGFR exon 20 insertion | TP53 | DMG thalamic
4. Interaction with Zegfrovy (EGFR inhibitor) or ONC201 (DRD2/ClpP agonist)
5. Evidence quality: human_clinical > human_case > animal > preclinical > hypothesis > preprint
IMPORTANT: BBB penetration ≠ therapeutic CNS concentration ≠ clinical efficacy. Distinguish clearly.

═══ ALREADY KNOWN (SKIP THESE) ═══
Trial IDs to skip: {json.dumps(known_trial_ids[:35])}
Dev IDs to skip: {json.dumps(known_dev_ids[:35])}

═══ LIVE TRIALS (ClinicalTrials.gov) ═══
{trials_s}

═══ PAPERS (PubMed, last 30 days) ═══
{papers_s}

═══ PREPRINTS (medRxiv/bioRxiv, last 14 days) ═══
{preprints_s}

═══ MEDICAL NEWS (RSS) ═══
{rss_s}

═══ INSTRUCTIONS ═══
Analyze ONLY items not already known. Apply strict relevance filter — only include if genuinely relevant to this patient.
- Trials: Is patient age ≤29 months eligible? Any EGFR ex20 note? Any thalamic location inclusion?
- Developments: Does this change the understanding of Hili's situation, Zegfrovy, ONC201, or treatment options?
- Do NOT create alerts just to seem active. If nothing new is relevant, say so clearly.
- Unify related findings into one development rather than creating multiple.
- Priority:
  URGENT = requires immediate medical attention (new contraindication, safety signal, time-sensitive trial)
  DISCUSS = important for next team meeting (new relevant data, trial opening, treatment option)
  MONITOR = worth watching but no immediate action needed

Return ONLY valid JSON (no markdown, no preamble):
{{
  "new_trials": [
    {{
      "nct_id": "NCT...",
      "title_he": "כותרת עברית ברורה — המשמעות הרפואית, לא שם המחקר",
      "title_en": "Short English title",
      "phase": "Phase X",
      "match_category": "reasonable|possible|follow|not_eligible",
      "priority": "URGENT|DISCUSS|MONITOR",
      "score": 70,
      "age_eligible": true,
      "recruiting_status": "Recruiting",
      "location_he": "מיקום",
      "summary_he": "סיכום 2-3 שורות: מה זה, למה רלוונטי, מה לא ידוע",
      "key_inclusion": ["קריטריון הכללה מרכזי"],
      "egfr_note": "הערה על ONC201/EGFR exon 20 אם רלוונטי, אחרת null",
      "bbb_note": "נתוני חדירת BBB זמינים? מה הם?",
      "action_he": "פעולה אחת ברורה",
      "source_url": "https://clinicaltrials.gov/study/NCT..."
    }}
  ],
  "new_developments": [
    {{
      "id": "KEYWORD-YYYYMMDD",
      "priority": "URGENT|DISCUSS|MONITOR",
      "title_he": "כותרת ברורה בעברית — משמעות רפואית, לא שם מדעי",
      "title_en": "Clear English title",
      "category": "research|regulatory|access|trial|news",
      "what_changed": "מה חדש — 2-3 שורות ברורות. לא לחזור על השם המדעי.",
      "why_matters_bullets": ["סיבה ספציפית 1 — עם קשר לפרופיל הילי", "סיבה 2"],
      "relevance_to_hili": {{
        "level": "confirmed|possible|needs_more_info|no_match",
        "explanation": "הסבר קצר מדוע — על בסיס נתוני המוטציות והטיפול המאומתים"
      }},
      "evidence_quality": {{
        "level": "human_clinical|human_case|animal|preclinical|hypothesis|preprint",
        "details": "מה המקור: כמה חולים? מה שלב המחקר?"
      }},
      "bbb_assessment": {{
        "evidence": "yes|partial|no|unknown",
        "details": "מה הראיות לחדירת BBB, אם קיימות"
      }},
      "expandable_details": {{
        "scientific_explanation": "הסבר מדעי מפורט יותר",
        "study_limitations": "מגבלות: גודל מדגם, population, follow-up",
        "connection_to_current_treatment": "קשר ל-Zegfrovy ו/או ONC201 — האם משלים? מתחרה? לא ידוע?"
      }},
      "action": "פעולה אחת ברורה: להעלות בפגישה / לפנות לצוות / להמשיך לעקוב / אין צורך בפעולה",
      "source_url": "https://...",
      "date_detected": "{TODAY}"
    }}
  ],
  "key_findings": ["ממצא מהותי 1", "ממצא מהותי 2"],
  "changes_detected": true,
  "scan_notes": "brief: what sources were scanned, how many items, what was found/not found",
  "research_hypotheses": [
    {{
      "id": "HYP-KEYWORD",
      "title_he": "כיוון שילוב לבחינה — ניסוח כשאלה",
      "title_en": "Research direction as a question",
      "urgency": "HIGH|MEDIUM|SPECULATIVE",
      "rationale_he": "למה זה רלוונטי לפרופיל הספציפי של הילי — H3K27M + EGFR ex20 + Zegfrovy",
      "supporting_evidence": ["ראיה מבוססת 1", "ראיה 2"],
      "next_step": "פעולה ספציפית: שאל רופא / חפש מחקר / בקש גישה",
      "relevant_trials": ["NCT..."],
      "date_generated": "{TODAY}"
    }}
  ]
}}"""

    for attempt in range(3):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=8192,
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

    print("\n--- Fetching WHO ICTRP (international trials) ---")
    who_raw = fetch_who_ictrp(known_trial_ids)
    trials_raw = trials_raw + who_raw

    total = len(trials_raw) + len(papers_raw) + len(preprints_raw) + len(rss_raw)
    print(f"\nTotal fetched: {len(trials_raw)} trials (incl. WHO) | {len(papers_raw)} papers | "
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
            "WHO ICTRP (international trials)",
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
            "NEJM RSS",
            "FDA Drug Approvals RSS",
            "SNO News RSS",
        ],
    }

    # Save — preserve wrapper objects
    trials_file["trials"] = trials_list
    trials_file["_meta"]["last_updated"] = TODAY
    developments_file["developments"] = developments_list
    developments_file["_meta"]["last_updated"] = TODAY

    # Save hypotheses (merge new with existing, deduplicate by id)
    new_hyps = result.get("research_hypotheses", [])
    if new_hyps:
        hyp_file = load_json(HYPOTHESES_PATH)
        existing_hyps = hyp_file.get("hypotheses", [])
        existing_ids = {h.get("id") for h in existing_hyps}
        added = 0
        for h in new_hyps:
            if h.get("id") and h["id"] not in existing_ids:
                h["date_generated"] = TODAY
                existing_hyps.insert(0, h)
                existing_ids.add(h["id"])
                added += 1
        hyp_file["hypotheses"] = existing_hyps[:50]
        hyp_file["_meta"]["last_updated"] = TODAY
        save_json(HYPOTHESES_PATH, hyp_file)
        print(f"Hypotheses: {added} new directions saved")

    save_json(TRIALS_PATH, trials_file)
    save_json(DEVELOPMENTS_PATH, developments_file)
    save_json(STATE_PATH, state)

    print(f"\nScan complete. Changes: {changes_detected}")
    for f in key_findings:
        print(f"  → {f}")
    return changes_detected


if __name__ == "__main__":
    run_scan()
