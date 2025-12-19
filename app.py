import streamlit as st
import pandas as pd
import asyncio
import aiohttp
import psycopg2
import re
import time
import json
from difflib import SequenceMatcher
from typing import Dict, Any

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Affiliation Validation POC", layout="wide")
st.title("🏥 Affiliation Validation – Ordered & Stable POC")

# =====================================================
# POSTGRES CONFIG
# =====================================================
PG_CONFIG = {
    "host": st.secrets["postgres"]["host"],
    "port": st.secrets["postgres"]["port"],
    "dbname": st.secrets["postgres"]["dbname"],
    "user": st.secrets["postgres"]["user"],
    "password": st.secrets["postgres"]["password"],
}

def get_pg_conn():
    return psycopg2.connect(**PG_CONFIG)

# =====================================================
# GLOBAL CONSTANTS
# =====================================================
DUCKDUCKGO_HTML = "https://duckduckgo.com/html/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

EXPLICIT_PHRASES = [
    "part of", "member of", "owned by", "division of",
    "subsidiary of", "a part of", "an affiliate of"
]

LEGAL_SUFFIXES = [
    "inc", "llc", "llp", "pllc", "pc", "pa", "ltd",
    "corp", "corporation", "co", "company"
]

SYSTEM_STOP_WORDS = [
    "admin", "administration", "administrative",
    "office", "offices", "corporate", "headquarters",
    "hq", "management", "services", "group",
    "health", "healthcare", "medical", "center", "centre",
    "clinic", "clinics", "hospital", "hospitals",
    "system", "systems", "network", "networks",
    "associates", "association", "associations",
    "practice", "practices"
]

ADDRESS_TOKENS = [
    "ste", "suite", "plaza", "building",
    "bldg", "floor", "fl"
]

MAX_SERP_RESULTS = 5
MAX_PAGE_FETCH = 3

# =====================================================
# SIDEBAR – LLM CONFIG
# =====================================================
st.sidebar.header("🤖 LLM Configuration")

llm_enabled = st.sidebar.checkbox(
    "Enable LLM fallback (verifier only)", value=False
)

llm_model = st.sidebar.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-4o", "o3-mini-2025-01-31", "o4-mini-2025-04-16"],
    disabled=not llm_enabled
)

llm_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    disabled=not llm_enabled
)

# 🔹 NEW: ADDITIONAL NOTES INPUT
st.sidebar.subheader("📝 LLM Additional Notes")

llm_additional_notes = st.sidebar.text_area(
    "Optional instructions for LLM (verifier only)",
    placeholder=(
        "Examples:\n"
        "- Be extra strict for hospitals\n"
        "- Ignore directory listings\n"
        "- Prefer official system websites\n"
    ),
    disabled=not llm_enabled,
    help="These notes are appended to the LLM prompt. Rules still apply."
)

# =====================================================
# SIDEBAR – RUN OPTIONS
# =====================================================
st.sidebar.header("⚙️ Run Options")
use_persisted = st.sidebar.checkbox("Use persisted DB result", value=True)
concurrency = st.sidebar.slider("Concurrency", 1, 5, 3)

# =====================================================
# NORMALIZATION & BRANDING
# =====================================================
def normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    tokens = [
        t for t in name.split()
        if t not in LEGAL_SUFFIXES
        and t not in ADDRESS_TOKENS
    ]
    return " ".join(tokens)

def extract_system_brand(system_name: str) -> str:
    tokens = normalize_name(system_name).split()
    tokens = [t for t in tokens if t not in SYSTEM_STOP_WORDS]
    return " ".join(tokens)

def branding_match(org_name: str, system_name: str) -> bool:
    org_norm = normalize_name(org_name)
    system_brand = extract_system_brand(system_name)

    if not system_brand:
        return False

    if system_brand in org_norm:
        return True

    ratio = SequenceMatcher(None, system_brand, org_norm).ratio()
    return ratio >= 0.88

# =====================================================
# DB HELPERS (PK = org_name)
# =====================================================
def fetch_from_db(org_name: str):
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT org_name, system_name, affiliated,
               decision_source, confidence, elapsed_sec, updated_at
        FROM affiliation_results
        WHERE org_name = %s
    """, (org_name,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

def upsert_db(result: dict):
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO affiliation_results
        (org_name, system_name, affiliated,
         decision_source, confidence, elapsed_sec)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (org_name)
        DO UPDATE SET
          system_name=EXCLUDED.system_name,
          affiliated=EXCLUDED.affiliated,
          decision_source=EXCLUDED.decision_source,
          confidence=EXCLUDED.confidence,
          elapsed_sec=EXCLUDED.elapsed_sec,
          updated_at=CURRENT_TIMESTAMP
    """, (
        result["org_name"],
        result["system_name"],
        result["affiliated"],
        result["decision_source"],
        result["confidence"],
        result["elapsed_sec"]
    ))
    conn.commit()
    cur.close()
    conn.close()

def fetch_all_from_db() -> pd.DataFrame:
    conn = get_pg_conn()
    df = pd.read_sql("""
        SELECT org_name, system_name, affiliated,
               decision_source, confidence, elapsed_sec, updated_at
        FROM affiliation_results
        ORDER BY updated_at DESC
    """, conn)
    conn.close()
    return df

# =====================================================
# DUCKDUCKGO – SERP PARSER
# =====================================================
async def duckduckgo_serp(query: str, session: aiohttp.ClientSession):
    try:
        async with session.post(
            DUCKDUCKGO_HTML,
            data={"q": query},
            timeout=aiohttp.ClientTimeout(total=20)
        ) as r:
            html = await r.text()
    except Exception:
        return []

    results = []
    blocks = re.findall(r'<div class="result__body">(.*?)</div>', html, re.S)

    for b in blocks[:MAX_SERP_RESULTS]:
        title = re.search(r'result__a.*?>(.*?)</a>', b, re.S)
        snippet = re.search(r'result__snippet.*?>(.*?)</a>', b, re.S)
        url = re.search(r'href="(https?://[^"]+)"', b)

        results.append({
            "title": re.sub("<.*?>", "", title.group(1)) if title else "",
            "snippet": re.sub("<.*?>", "", snippet.group(1)) if snippet else "",
            "url": url.group(1) if url else None
        })
    return results

def explicit_found(text: str) -> bool:
    return any(p in text.lower() for p in EXPLICIT_PHRASES)

# =====================================================
# PAGE FETCH
# =====================================================
async def fetch_page(url: str, session: aiohttp.ClientSession) -> str:
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=20)
        ) as r:
            html = await r.text()
            return re.sub(r"\s+", " ", re.sub(r"<.*?>", " ", html))
    except Exception:
        return ""

# =====================================================
# LLM (PATCHED WITH ADDITIONAL NOTES)
# =====================================================
def get_llm_client():
    if llm_enabled and llm_api_key:
        from openai import OpenAI
        return OpenAI(api_key=llm_api_key)
    return None

async def llm_verify(client, org, system, serp, pages, additional_notes: str):
    if not client:
        return False, 0.0

    notes_block = ""
    if additional_notes and additional_notes.strip():
        notes_block = f"""
Additional Analyst Notes (FOLLOW IF COMPATIBLE WITH RULES):
{additional_notes}
"""

    prompt = f"""
You are an affiliation verification analyst.

Organization: {org}
Proposed System: {system}

Evidence:
{json.dumps({"serp": serp, "pages": pages[:2]}, indent=2)}

STRICT RULES (NON-NEGOTIABLE):
- Do NOT infer affiliation
- Name similarity alone is NOT sufficient
- Return TRUE only if explicit ownership or network membership is stated
- If unclear or implicit, return FALSE

{notes_block}

Return STRICT JSON only:
{{"explicit": true|false, "confidence": number}}
"""

    r = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}]
        
    )

    out = json.loads(r.choices[0].message.content)
    return out["explicit"], out["confidence"]

# =====================================================
# CORE VALIDATION (ORDER GUARANTEED)
# =====================================================
async def validate_affiliation(row, llm_client, session):
    start = time.time()
    org = row["org_name"]
    system = row["system_name"]

    if use_persisted:
        cached = fetch_from_db(org)
        if cached:
            return dict(zip(
                ["org_name","system_name","affiliated",
                 "decision_source","confidence","elapsed_sec","updated_at"],
                cached
            ))

    if branding_match(org, system):
        result = {
            "org_name": org,
            "system_name": system,
            "affiliated": True,
            "decision_source": "branding_match",
            "confidence": 0.95,
            "elapsed_sec": round(time.time() - start, 2)
        }
        upsert_db(result)
        return result

    serp = await duckduckgo_serp(f'"{org}" "{system}"', session)

    for r in serp:
        if explicit_found(f"{r['title']} {r['snippet']}"):
            result = {
                "org_name": org,
                "system_name": system,
                "affiliated": True,
                "decision_source": "duckduckgo_serp",
                "confidence": 0.85,
                "elapsed_sec": round(time.time() - start, 2)
            }
            upsert_db(result)
            return result

    pages = []
    for r in serp[:MAX_PAGE_FETCH]:
        if not r["url"]:
            continue
        page = await fetch_page(r["url"], session)
        pages.append(page)
        if explicit_found(page):
            result = {
                "org_name": org,
                "system_name": system,
                "affiliated": True,
                "decision_source": "duckduckgo_page",
                "confidence": 0.8,
                "elapsed_sec": round(time.time() - start, 2)
            }
            upsert_db(result)
            return result

    explicit, conf = await llm_verify(
        llm_client,
        org,
        system,
        serp,
        pages,
        llm_additional_notes
    )

    result = {
        "org_name": org,
        "system_name": system,
        "affiliated": explicit,
        "decision_source": "llm_verified" if explicit else "no_explicit_evidence",
        "confidence": conf if explicit else 0.6,
        "elapsed_sec": round(time.time() - start, 2)
    }
    upsert_db(result)
    return result

# =====================================================
# ASYNC BATCH RUNNER
# =====================================================
async def run_batch(rows):
    sem = asyncio.Semaphore(concurrency)
    client = get_llm_client()
    progress = st.progress(0.0)
    out = []

    connector = aiohttp.TCPConnector(limit=5, ssl=False)

    async with aiohttp.ClientSession(
        connector=connector,
        headers=HEADERS
    ) as session:

        async def task(r, i):
            async with sem:
                await asyncio.sleep(0.5)
                res = await validate_affiliation(r, client, session)
                progress.progress((i + 1) / len(rows))
                return res

        tasks = [task(r, i) for i, r in enumerate(rows)]
        for c in asyncio.as_completed(tasks):
            out.append(await c)

    return out

# =====================================================
# UI
# =====================================================
uploaded = st.file_uploader("Upload CSV (org_name, system_name)", type=["csv"])

st.divider()
st.subheader("📥 Download Persisted Results (DB)")

if st.button("⬇ Download All DB Results"):
    db_df = fetch_all_from_db()
    if db_df.empty:
        st.warning("No records found in database.")
    else:
        st.success(f"Fetched {len(db_df)} records from DB")
        st.dataframe(db_df)
        st.download_button(
            label="Download DB Results as CSV",
            data=db_df.to_csv(index=False),
            file_name="affiliation_results_db.csv",
            mime="text/csv"
        )

if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    rows = df[["org_name", "system_name"]].to_dict("records")

    if st.button("▶ Run Validation"):
        results = asyncio.run(run_batch(rows))
        st.dataframe(pd.DataFrame(results))
