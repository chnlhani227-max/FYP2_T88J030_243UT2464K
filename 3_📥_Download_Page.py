import streamlit as st
import pandas as pd
import os
from core.packager import create_zip
from core.config_user import COPYRIGHT_TEXT, PROJECT_TITLE

st.set_page_config(page_title="Download", layout="wide")
st.title(f"📥 Download: {PROJECT_TITLE}")
st.sidebar.markdown("---")
st.sidebar.info(COPYRIGHT_TEXT)

if not st.session_state.results:
    st.warning("Please complete training first.")
    st.stop()

df = pd.DataFrame(st.session_state.results)

# Updated to match the new column name: 'F1-Score (%)'
best = df.loc[df['F1-Score (%)'].idxmax()]
st.success(f"🏆 Best Model based on F1-Score: **{best['Algorithm']}**")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{best['Accuracy (%)']:.2f}%")
col2.metric("Detection Rate", f"{best['Recall/Detection Rate (%)']:.2f}%")
col3.metric("FPR", f"{best['False Positive Rate (%)']:.4f}%")

st.markdown("---")
st.markdown("### 📦 Generate Deployment Firmware")
zip_path = create_zip(best['Algorithm'])
with open(zip_path, "rb") as f:
    st.download_button(
        label=f"⬇️ Download {best['Algorithm']} for Raspberry Pi",
        data=f,
        file_name=os.path.basename(zip_path),
        mime="application/zip"
    )
