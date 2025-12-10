import os
import json
import re
import wikipediaapi
import streamlit as st
from typing import Optional, List
from pydantic import BaseModel
from groq import Groq
import pandas as pd
from difflib import SequenceMatcher
from dotenv import load_dotenv
from openai import OpenAI
import time
from contextlib import contextmanager

@contextmanager
def timed(label: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[TIME] {label}: {end - start:.2f}s")


load_dotenv()
# -------------------------
# Groq Client
# -------------------------
##client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"

# -------------------------
# Wikipedia
# -------------------------
wiki_api = wikipediaapi.Wikipedia(
    user_agent="OrgAffiliationBot/1.0",
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

def fetch_wikipedia_text(org_name: str) -> Optional[str]:
    page = wiki_api.page(org_name)
    if not page.exists():
        return None
    return page.text

# -------------------------
# Prompts
# -------------------------
WIKI_PROMPT = """
You are extracting organizational affiliation information.

Rules:
- Use ONLY the provided text.
- If information is not explicitly stated, return null.
- Do NOT hallucinate affiliations.
- Return JSON only.

Schema:
{{
  "legal_name": string | null,
  "parent_organization": string | null,
  "health_system": string | null,
  "affiliations": string[] | null,
  "religious_or_academic_affiliation": string | null,
  "ownership_type": string | null,
  "confidence": number
}}

Text:
{wiki_text}
"""


NPI_PROMPT = """
You are extracting organizational affiliation information from NPI Registry data.

Rules:
- Use ONLY the provided data.
- Do NOT guess.
- Confidence must be <= 0.7.
- Return JSON only.

Schema:
{{
  "legal_name": string | null,
  "parent_organization": string | null,
  "health_system": string | null,
  "affiliations": string[] | null,
  "religious_or_academic_affiliation": string | null,
  "ownership_type": string | null,
  "confidence": number
}}

NPI Registry Data:
{npi_text}
"""
DDG_PROMPT = """
You are extracting organizational affiliation information from web search snippets.

Rules:
- Use ONLY the provided search text.
- Be conservative.
- If affiliation is not explicit, return null.
- Confidence must be <= 0.6.
- Return JSON only.

Schema:
{
  "legal_name": string | null,
  "parent_organization": string | null,
  "health_system": string | null,
  "affiliations": string[] | null,
  "religious_or_academic_affiliation": string | null,
  "ownership_type": string | null,
  "confidence": number
}

Search Text:
{search_text}
"""


FALLBACK_PROMPT = FALLBACK_PROMPT = """
Given the organization name below, infer possible affiliations.

Rules:
- Return JSON only.

Schema:
{{
  "legal_name": string | null,
  "parent_organization": string | null,
  "health_system": string | null,
  "affiliations": string[] | null,
  "religious_or_academic_affiliation": string | null,
  "ownership_type": string | null,
  "confidence": number
}}

Organization:
{org_name}
"""

def get_groq_client():
    api_key = st.secrets["GROQ"]["API_KEY"]
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment.")
    return Groq(api_key=api_key)

class AffiliationResult(BaseModel):
    legal_name: Optional[str]
    parent_organization: Optional[str]
    health_system: Optional[str]
    affiliations: Optional[List[str]]
    religious_or_academic_affiliation: Optional[str]
    ownership_type: Optional[str]
    confidence: float
    source: Optional[str] = None

def llm_extract(prompt: str, variables: dict, client, model):
    formatted = prompt
    for k, v in variables.items():
        formatted = formatted.replace(f"{{{k}}}", str(v))

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": formatted},
        ],
        
    )

    raw = resp.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(match.group())


def find_affiliation(
    org_name: str,
    client,
    model,
    city: str = None,
    state: str = None,
):
    # -------------------------
    # Level 1: Wikipedia
    # -------------------------
    with timed(f"{org_name} – wikipedia"):
        wiki_text = fetch_wikipedia_text(org_name)
        if wiki_text:
            data = llm_extract(
                WIKI_PROMPT,
                {"wiki_text": wiki_text},
                client,
                model,
            )
            return AffiliationResult(**data, source="wikipedia").model_dump()
    with timed(f"{org_name} – npi"):
        # -------------------------
        # Level 2: NPI Registry
        # -------------------------
        npi = query_npi_registry(org_name, city, state)
        if npi:
            npi_text = json.dumps(npi, indent=2)
            data = llm_extract(
                NPI_PROMPT,
                {"npi_text": npi_text},
                client,
                model,
            )
            data["confidence"] = min(data.get("confidence", 0.0), 0.7)
            return AffiliationResult(**data, source="npi_registry").model_dump()

    # -------------------------
    # Level 3: DuckDuckGo
    # -------------------------
    with timed(f"{org_name} – duckduckgo"):
        search_text = duckduckgo_search(org_name)
        if search_text:
            data = llm_extract(
                DDG_PROMPT,
                {"search_text": search_text},
                client,
                model,
            )
            data["confidence"] = min(data.get("confidence", 0.0), 0.6)
            return AffiliationResult(**data, source="duckduckgo").model_dump()

    # -------------------------
    # Level 4: LLM fallback
    # -------------------------
    with timed(f"{org_name} – llm_fallback"):
        data = llm_extract(
            FALLBACK_PROMPT,
            {"org_name": org_name},
            client,
            model,
        )
        data["confidence"] = min(data.get("confidence", 0.0), 0.4)
        return AffiliationResult(**data, source="llm_fallback").model_dump()

# -------------------------
# Normalization + Matching
# -------------------------
def normalize_name(name):
    if not name:
        return None
    name = name.lower()
    name = re.sub(r'[^a-z0-9 ]', '', name)
    name = re.sub(r'\b(inc|llc|ltd|corp|hospital|healthcare|system)\b', '', name)
    return " ".join(name.split())

def verify_system(parent, system):
    if not parent or not system:
        return "UNKNOWN", 0.0, "Missing"

    p = normalize_name(parent)
    s = normalize_name(system)

    if p in s or s in p:
        return "MATCH", 0.95, "Exact"

    score = SequenceMatcher(None, p, s).ratio()
    if score >= 0.85:
        return "MATCH", score, "High fuzzy"
    elif score >= 0.60:
        return "PARTIAL", score, "Medium fuzzy"
    else:
        return "MISMATCH", score, "Low similarity"


# -------------------------
# Main Runner
# -------------------------
def run_org_system_verification(df: pd.DataFrame, client, model):
    results = []

    for _, row in df.iterrows():
        affiliation = find_affiliation(
            row["org_name"],
            client,
            model,
            city=row.get("CITY"),
            state=row.get("STATE"),
        )

        parent = affiliation["parent_organization"]
        conf = affiliation["confidence"]
        source = affiliation["source"]

        status, score, reason = verify_system(parent, row["system_name"])

        results.append({
            "ORG_VID": row["ORG_VID"],
            "org_name": row["org_name"],
            "system_name": row["system_name"],
            "parent_org_derived": parent,
            "derivation_confidence": conf,
            "derivation_source": source,
            "match_status": status,
            "match_score": round(score, 2),
            "verification_reason": reason
        })

    return pd.DataFrame(results)


import requests

def normalize_simple(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return re.sub(r"[^a-z0-9]", "", text.lower())


def query_npi_registry(
    org_name: str,
    city: Optional[str] = None,
    state: Optional[str] = None,
    limit: int = 10,
) -> Optional[dict]:
    """
    Level-2: NPI Registry lookup (organization only).
    """
    url = "https://npiregistry.cms.hhs.gov/api/"
    params = {
        "version": "2.1",
        "organization_name": org_name,
        "enumeration_type": "NPI-2",
        "limit": limit,
    }

    if city:
        params["city"] = city
    if state:
        params["state"] = state

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    if data.get("result_count", 0) == 0:
        return None

    org_norm = normalize_simple(org_name)
    best, best_score = None, 0.0

    for res in data.get("results", []):
        name = res.get("basic", {}).get("organization_name", "")
        n = normalize_simple(name)

        score = 0.0
        if n == org_norm:
            score += 1.0
        elif org_norm and org_norm in n:
            score += 0.8

        if score > best_score:
            best_score = score
            best = res

    if not best or best_score < 0.6:
        return None

    return {
        "npi": best["number"],
        "legal_name": best["basic"]["organization_name"],
        "confidence": round(best_score, 2),
    }

def duckduckgo_search(org_name: str) -> Optional[str]:
    """
    Level-3: DuckDuckGo search (best effort).
    Returns aggregated snippet text or None if blocked.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        url = "https://duckduckgo.com/lite/"
        params = {"q": f"{org_name} parent organization health system"}
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0 Safari/537.36"
            )
        }

        r = requests.get(url, params=params, headers=headers, timeout=5)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        snippets = []
        for a in soup.select("a.result-link"):
            text = a.get_text(strip=True)
            if text:
                snippets.append(text)

        return "\n".join(snippets[:5]) if snippets else None

    except Exception:
        return None
