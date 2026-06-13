import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="WiFi 7 IoT IDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ WiFi 7 IoT Intrusion Detection Dashboard")
st.caption("Edge AI-based IDS using Raspberry Pi traffic and ns-3 simulation")

# =========================
# Sidebar
# =========================
st.sidebar.title("IDS Control Panel")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["XGBoost", "Random Forest", "SVM", "DNN", "LSTM"]
)





# =========================
# Label mapping
# =========================
label_names = {
    0: "Normal",
    1: "Flooding / Port Scan",
    2: "Blackhole",
    3: "Grayhole"
}

required_features = ["packet_size", "protocol", "time_diff", "packet_rate"]

# =========================
# Helper functions
# =========================
def clean_features(df):
    X = df[required_features].copy()
    X = X.replace([float("inf"), float("-inf")], 0)
    X = X.fillna(0)
    return X


def load_selected_model(choice):
    if choice == "XGBoost":
        return joblib.load("xgboost_multiclass_ids_model.pkl"), None, "sklearn"

    if choice == "Random Forest":
        return joblib.load("rf_ids_model.pkl"), None, "sklearn"

    if choice == "SVM":
        return joblib.load("svm_ids_model.pkl"), None, "sklearn"

    if choice == "DNN":
        model = tf.keras.models.load_model("dnn_ids_model.keras")
        scaler = joblib.load("dnn_scaler.pkl")
        return model, scaler, "dnn"

    if choice == "LSTM":
        model = tf.keras.models.load_model("lstm_ids_model.keras")
        scaler = joblib.load("lstm_scaler.pkl")
        return model, scaler, "lstm"

    return None, None, None


def predict_attack(model, scaler, model_type, X):
    if model_type == "sklearn":
        return model.predict(X)

    if model_type == "dnn":
        X_scaled = scaler.transform(X)
        pred_prob = model.predict(X_scaled)
        return np.argmax(pred_prob, axis=1)

    if model_type == "lstm":
        X_scaled = scaler.transform(X)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        pred_prob = model.predict(X_scaled)
        return np.argmax(pred_prob, axis=1)

    return None

# =========================
# Sidebar Navigation
# =========================
page = st.sidebar.radio(
    "Select Page",
    [
        "Dataset Preview",
        "IDS Testing",
        "Model Comparison",
        "Methodology"
    ]
)

# =========================
# Dataset Preview Page
# =========================
if page == "Dataset Preview":

    st.subheader("Dataset Preview")

    preview_file = st.file_uploader(
        "Upload CSV file for preview",
        type=["csv"],
        key="preview"
    )

    if preview_file is not None:
        preview_df = pd.read_csv(preview_file)

        st.dataframe(preview_df.head(20), use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Total Rows", len(preview_df))
        col2.metric("Total Columns", len(preview_df.columns))

    else:
        st.info("Upload a dataset to preview it.")

# =========================
# IDS Testing Page
# =========================
if page == "IDS Testing":
    st.subheader("Upload IDS Dataset")

    uploaded_file = st.file_uploader(
        "Upload final_dataset.csv or any CSV with the required features",
        type=["csv"]
    )

    run_detection = st.button("Run Detection")

    if uploaded_file is not None and run_detection:
        df = pd.read_csv(uploaded_file)

        st.write("### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        total_rows = len(df)
        total_cols = len(df.columns)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", total_rows)
        col2.metric("Total Columns", total_cols)
        col3.metric("Selected Model", model_choice)

        missing_features = [col for col in required_features if col not in df.columns]

        if missing_features:
            st.error(f"Missing required feature columns: {missing_features}")
            st.stop()

        X = clean_features(df)

        try:
            model, scaler, model_type = load_selected_model(model_choice)
            predictions = predict_attack(model, scaler, model_type, X)

            prediction_labels = [label_names.get(int(pred), str(pred)) for pred in predictions]

            result_df = df.copy()
            result_df["Predicted_Label"] = predictions
            result_df["Predicted_Attack"] = prediction_labels

            st.write("### Prediction Results")
            st.dataframe(result_df.head(20), use_container_width=True)

            st.write("### Attack Prediction Summary")
            summary = result_df["Predicted_Attack"].value_counts().reset_index()
            summary.columns = ["Attack Type", "Count"]
            st.dataframe(summary, use_container_width=True)
            st.bar_chart(summary.set_index("Attack Type"))

            if "label" in df.columns:
                st.write("### Evaluation Result")

                y_true = df["label"].copy()
                y_true = y_true.replace({3: 2, 4: 3})

                accuracy = accuracy_score(y_true, predictions)
                st.metric("Accuracy", f"{accuracy:.4f}")

                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_true, predictions)
                cm_df = pd.DataFrame(cm)
                st.dataframe(cm_df, use_container_width=True)

                st.write("### Classification Report")
                report = classification_report(y_true, predictions, zero_division=0, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)

            st.success("Prediction completed successfully.")

        except FileNotFoundError as e:
            st.error(f"Model file not found: {e}")
            st.info("Make sure all trained model files are in the same folder as app.py.")

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.info("Upload a CSV file to start prediction.")
        st.write("Required columns:")
        st.code("packet_size, protocol, time_diff, packet_rate")

# =========================
# Model Comparison Page
# =========================
if page == "Model Comparison":

    st.subheader("Model Comparison Testing")

    comparison_file = st.file_uploader(
        "Upload Dataset for Algorithm Comparison",
        type=["csv"],
        key="comparison"
    )

    run_comparison = st.button("Run Algorithm Comparison")

    if comparison_file is not None and run_comparison:

        compare_df = pd.read_csv(comparison_file)

        missing_features = [col for col in required_features if col not in compare_df.columns]

        if missing_features:
            st.error(f"Missing required columns: {missing_features}")
            st.stop()

        if "label" not in compare_df.columns:
            st.error("Dataset must contain label column for comparison.")
            st.stop()

        X_compare = clean_features(compare_df)

        y_compare = compare_df["label"].copy()
        y_compare = y_compare.replace({3: 2, 4: 3})

        algorithms = {
            "XGBoost": ("xgboost_multiclass_ids_model.pkl", None, "sklearn"),
            "Random Forest": ("rf_ids_model.pkl", None, "sklearn"),
            "SVM": ("svm_ids_model.pkl", None, "sklearn"),
            "DNN": ("dnn_ids_model.keras", "dnn_scaler.pkl", "dnn"),
            "LSTM": ("lstm_ids_model.keras", "lstm_scaler.pkl", "lstm")
        }

        comparison_results = []

        progress = st.progress(0)

        for idx, (algo_name, model_info) in enumerate(algorithms.items()):

            model_path, scaler_path, model_type = model_info

            if model_type == "sklearn":
                model = joblib.load(model_path)
                predictions = model.predict(X_compare)

            elif model_type == "dnn":
                model = tf.keras.models.load_model(model_path)
                scaler = joblib.load(scaler_path)

                X_scaled = scaler.transform(X_compare)
                pred_probs = model.predict(X_scaled)
                predictions = np.argmax(pred_probs, axis=1)

            elif model_type == "lstm":
                model = tf.keras.models.load_model(model_path)
                scaler = joblib.load(scaler_path)

                X_scaled = scaler.transform(X_compare)
                X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

                pred_probs = model.predict(X_scaled)
                predictions = np.argmax(pred_probs, axis=1)

            accuracy = accuracy_score(y_compare, predictions)

            report = classification_report(
                y_compare,
                predictions,
                output_dict=True,
                zero_division=0
            )

            comparison_results.append({
                "Algorithm": algo_name,
                "Accuracy": round(accuracy, 4),
                "Macro Precision": round(report["macro avg"]["precision"], 4),
                "Macro Recall": round(report["macro avg"]["recall"], 4),
                "Macro F1-score": round(report["macro avg"]["f1-score"], 4)
            })

            progress.progress((idx + 1) / len(algorithms))

        result_df = pd.DataFrame(comparison_results)
        result_df = result_df.sort_values(by="Macro F1-score", ascending=False)

        st.write("### Algorithm Comparison Result")
        st.dataframe(result_df, use_container_width=True)

        st.write("### Macro F1-score Comparison")
        st.bar_chart(result_df.set_index("Algorithm")["Macro F1-score"])

        best_model = result_df.iloc[0]["Algorithm"]
        st.success(f"Best Current Model: {best_model}")

    else:
        st.info("Upload dataset and click 'Run Algorithm Comparison' to begin testing.")

# =========================
# Methodology Page
# =========================
if page == "Methodology":
    st.subheader("Project Methodology")

    st.write("### 1. Raspberry Pi Traffic Capture")
    st.write("Raspberry Pi is used as the edge device to collect real network traffic using tcpdump.")

    st.write("### 2. Attack Generation")
    st.write("Flooding/DoS and port scanning traffic are generated using ping flood and Nmap.")

    st.write("### 3. ns-3 Simulation")
    st.write("Blackhole and grayhole attacks are simulated in ns-3 using packet dropping behaviour.")

    st.write("### 4. Feature Extraction")
    st.write("PCAP files are converted into CSV format using packet-level features.")

    st.code("packet_size, protocol, time_diff, packet_rate")

    st.write("### 5. Model Training")
    st.write("Five algorithms are trained and compared: RF, XGBoost, SVM, DNN and LSTM.")

    st.write("### 6. Edge AI Deployment")
    st.write("The best model will be deployed on Raspberry Pi for lightweight IDS testing.")

    
