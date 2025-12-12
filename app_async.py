import streamlit as st
import pandas as pd
import json
import re
import time
import asyncio
from typing import Dict, Any

# =========================================================
# Utility: Build verification question
# =========================================================
def build_org_question(row) -> str:
    return (
        f"Is the organization '{row['org_name']}' located at "
        f"{row['ADDRESS_LINE_1']}, {row['CITY']}, {row['STATE']} "
        f"affiliated with '{row['system_name']}'?"
    )

# =========================================================
# LLM Prompt (IMPORTANT: escaped braces {{ }})
# =========================================================
LLM_PROMPT = """
Answer the verification question below by checking official organization sources
(such as the organization website, health system website, or reliable healthcare directories).

Rules:
- Return STRICT JSON only
- Do not include explanations, markdown, or comments
- Use null if information cannot be verified

Output JSON format (example):
{{
  "legal_name": null,
  "parent_organization": null,
  "reference_url": null,
  "True_False": null
}}

Verification Question:
{org_question}
"""

# =========================================================
# Safe JSON Parser
# =========================================================
def safe_parse_llm_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError(f"No JSON found in response:\n{text}")
        return json.loads(match.group())

# =========================================================
# Async LLM Call (Groq / OpenAI)
# =========================================================
async def call_llm_async(provider, model, api_key, prompt):
    if provider == "groq":
        from groq import AsyncGroq
        client = AsyncGroq(api_key=api_key)

        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

    elif provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)

        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1
        )

    else:
        raise ValueError("Unsupported LLM provider")

    raw = resp.choices[0].message.content.strip()
    return safe_parse_llm_json(raw)

# =========================================================
# Async Row Processor (with semaphore)
# =========================================================
async def process_row(row, provider, model, api_key, semaphore):
    async with semaphore:
        org_question = build_org_question(row)
        prompt = LLM_PROMPT.format(org_question=org_question)

        try:
            return await call_llm_async(provider, model, api_key, prompt)
        except Exception as e:
            return {
                "legal_name": None,
                "parent_organization": None,
                "reference_url": None,
                "True_False": "ERROR",
                "confidence": 0.0,
                "error": str(e)
            }

# =========================================================
# Async Batch Runner
# =========================================================
async def run_async_batch(df, provider, model, api_key, max_concurrency):
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        process_row(row, provider, model, api_key, semaphore)
        for _, row in df.iterrows()
    ]

    return await asyncio.gather(*tasks)

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="Organization Affiliation Validator", layout="wide")
st.title("🏥 Organization Affiliation Validation (Async LLM)")

with st.sidebar:
    st.header("🔐 LLM Configuration")
    provider = st.selectbox("LLM Provider", ["groq", "openai"])
    model_name = st.text_input(
        "Model name",
        value="openai/gpt-oss-120b" if provider == "groq" else "o3-2025-04-16"
    )
    api_key = st.text_input("API Key", type="password")
    max_concurrency = st.slider("Max Concurrency", 1, 20, 10)

uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    total_loaded = len(df)
    preview_rows = min(5, total_loaded)

    st.subheader("📊 File Load Summary")
    c1, c2 = st.columns(2)
    c1.metric("Total Records Loaded", total_loaded)
    c2.metric("Preview Rows", preview_rows)

    st.subheader("📄 Input Preview")
    st.dataframe(df.head(preview_rows))

    if not api_key:
        st.warning("Please enter an API key to proceed.")

    if api_key and st.button("▶️ Run Affiliation Validation"):
        start_time = time.perf_counter()

        with st.spinner("Running async LLM validation..."):
            results = asyncio.run(
                run_async_batch(
                    df=df,
                    provider=provider,
                    model=model_name,
                    api_key=api_key,
                    max_concurrency=max_concurrency
                )
            )

        elapsed = round(time.perf_counter() - start_time, 2)

        result_df = pd.concat(
            [df.reset_index(drop=True), pd.DataFrame(results)],
            axis=1
        )

        success_count = sum(1 for r in results if r.get("True_False") != "ERROR")
        error_count = total_loaded - success_count

        st.subheader("📊 Run Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", total_loaded)
        m2.metric("Successful", success_count)
        m3.metric("Errors", error_count)
        m4.metric("Time Taken (sec)", elapsed)

        st.caption(f"⚡ Avg time per record: {round(elapsed / total_loaded, 2)} sec")

        st.subheader("✅ Results")
        st.dataframe(result_df)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Result CSV",
            data=csv_bytes,
            file_name="affiliation_validation_output.csv",
            mime="text/csv"
        )
