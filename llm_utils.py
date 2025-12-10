from groq import Groq
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

def llm_summarize(df: pd.DataFrame,client,model) -> str:
    sample = df.to_dict(orient="records")

    prompt = f"""
You are analyzing a dataset of healthcare organization affiliations.
Write a concise 150–200 word summary including:

- how many matched vs mismatched,
- how many low-confidence derivations,
- any patterns in parent organizations,
- any commons sources of mismatch.

Sample data:
{sample}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert healthcare analyst."},
            {"role": "user", "content": prompt},
        ]
    )

    return resp.choices[0].message.content
