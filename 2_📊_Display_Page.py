import streamlit as st
import pandas as pd
import plotly.express as px
from core.config_user import COPYRIGHT_TEXT, PROJECT_TITLE

st.set_page_config(page_title="Display", layout="wide")
st.title(f"📊 Hypothesis Validation: {PROJECT_TITLE}")
st.sidebar.markdown("---")
st.sidebar.info(COPYRIGHT_TEXT)

if not st.session_state.results:
    st.warning("No results found. Train models first.")
    st.stop()

df = pd.DataFrame(st.session_state.results)

# HYPOTHESIS 2: Latency Threshold
st.subheader("🚀 Hypothesis 2: Edge AI Latency Analysis")
threshold = 25.0 
df['Meets WiFi 7 Requirement'] = df['Latency (ms)'] < threshold

fig_lat = px.bar(df, x="Algorithm", y="Latency (ms)", color="Meets WiFi 7 Requirement",
                 color_discrete_map={True: "green", False: "red"},
                 text="Latency (ms)")
fig_lat.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="WiFi 7 Max Limit (25ms)")
st.plotly_chart(fig_lat, use_container_width=True)

# HYPOTHESIS 3: Detection Rate vs False Positive Rate
st.subheader("🛡️ Hypothesis 3: False Positive Analysis")
fig_perf = px.scatter(df, x="False Positive Rate (%)", y="Recall/Detection Rate (%)", 
                      size="Accuracy (%)", color="Algorithm", 
                      title="Goal: High Detection Rate (Top) and Low FPR (Left)")
st.plotly_chart(fig_perf, use_container_width=True)

st.subheader("📋 Performance Summary Table")
st.dataframe(df)
