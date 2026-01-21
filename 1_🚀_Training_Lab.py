import streamlit as st
import pandas as pd
import os
from core.registry import CATEGORIES
from core.trainer import StudentTrainer
from core.profiler import EdgeProfiler
from core.config_user import COPYRIGHT_TEXT, PROJECT_TITLE

if 'results' not in st.session_state:
    st.session_state.results = []

st.set_page_config(page_title="Training", layout="wide")
st.title(f"🚀 Training Lab: {PROJECT_TITLE}")
st.sidebar.markdown("---")
st.sidebar.info(COPYRIGHT_TEXT)

profiler = EdgeProfiler()

with st.sidebar:
    st.header("1. Target Device")
    device = st.selectbox("Hardware", profiler.get_device_list())
    
    st.header("2. Dataset")
    dataset = st.selectbox("Data", [f for f in os.listdir("data") if f.endswith(".csv")])
    
    st.header("3. Algorithms")
    all_models = []
    # Prioritize Custom and Core
    if "My Selected Algorithms" in CATEGORIES:
        all_models += CATEGORIES["My Selected Algorithms"]
    all_models += CATEGORIES["Core Requirements"]
    selected = st.multiselect("Select Algorithms", list(dict.fromkeys(all_models)))
    
    st.header("4. Tuning")
    hp_epochs = st.slider("Epochs (DL)", 1, 20, 5)
    hp_batch = st.select_slider("Batch Size", [8, 16, 32, 64], value=32)
    hp_est = st.slider("Estimators (RF)", 10, 100, 20)

if st.button("🚀 START TRAINING"):
    if not selected:
        st.error("Select algorithm.")
    else:
        trainer = StudentTrainer()
        st.session_state.results = []
        bar = st.progress(0)
        for i, algo in enumerate(selected):
            try:
                res = trainer.train_model(algo, f"data/{dataset}", device, hp_epochs, hp_batch, hp_est)
                st.session_state.results.append(res)
            except Exception as e:
                st.error(f"Error ({algo}): {e}")
            bar.progress((i+1)/len(selected))
        st.success("Training Complete!")

if st.session_state.results:
    st.write(f"## Results for {device}")
    st.dataframe(pd.DataFrame(st.session_state.results))
