import streamlit as st
import pandas as pd
import io
from groq import Groq
import os
from groq import Groq
from openai import OpenAI

client = Groq(
    api_key=st.secrets["GROQ"]["API_KEY"]
)
model="openai/gpt-oss-120b"
# -------------------------
# Import your backend logic
# -------------------------
from backend import (
    run_org_system_verification,
    find_affiliation,
    verify_system,
)

from llm_utils import llm_summarize


# -------------------------
# Page Title
# -------------------------
st.title("🏥 Healthcare Affiliation & System Verification Tool")

st.write("""
Upload a CSV/Excel file with columns:

- **ORG_VID**
- **org_name**
- **system_name**

The app will:
1. Derive parent organization using Wikipedia + LLM fallback  
2. Compare derived parent vs system_name  
3. Produce match score + explanation  
4. Give LLM summary  
5. Allow CSV download  
""")

# -------------------------
# File Upload
# -------------------------
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded:
    # Read file
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.success(f"File uploaded successfully. Rows = {len(df)}")
    st.dataframe(df.head())

    # Validate required columns
    required = {"ORG_VID", "org_name", "system_name"}
    if not required.issubset(df.columns):
        st.error(f"Missing required columns: {required - set(df.columns)}")
        st.stop()

    # -------------------------
    # Run verification
    # -------------------------
    if st.button("🚀 Run Affiliation + System Verification"):
        with st.spinner("Running LLM + Wikipedia extraction…"):
            result_df = run_org_system_verification(df,client,model)

        st.success("Processing complete.")
        st.dataframe(result_df)

        # Download button
        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name="verification_output.csv",
            mime="text/csv"
        )

        # -------------------------
        # LLM Summary Section
        # -------------------------
        st.subheader("🧠 Summary")
        summary_text = llm_summarize(result_df,client,model)
        st.write(summary_text)
