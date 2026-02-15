# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# ------------------------------------------------------------
# Page Config + Simple CSS for nicer UI
# ------------------------------------------------------------
st.set_page_config(
    page_title="Telcom Customer Churn Prediction",
    page_icon="📞",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        padding: 18px 20px;
        border-radius: 14px;
        color: white;
        font-weight: 700;
        font-size: 28px;
        margin-bottom: 14px;
    }
    .subtext {
        color: #334155;
        font-size: 15px;
        margin-top: -6px;
        margin-bottom: 18px;
    }
    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 1px 8px rgba(15, 23, 42, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">📞 Telco Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtext">Upload a test CSV, select a model, and get predictions + evaluation metrics (if <b>Churn</b> column exists).</div>',
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Model directory + mapping (joblib)
# ------------------------------------------------------------
MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes_gaussian.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib"
}

SAMPLE_CSV_PATH = os.path.join("data", "telco_test_data.csv")

@st.cache_resource

def sparse_to_dense(x):
    return x.toarray() if sp.issparse(x) else x
    
def load_pipeline(model_path: str):
    return joblib.load(model_path)

def safe_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning consistent with training assumptions."""
    df = df.copy()

    # Handle TotalCharges like training (spaces/blanks -> numeric)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")

    # Drop customerID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df

def get_probabilities(pipeline, X_data):
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X_data)[:, 1]
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X_data)
        return 1 / (1 + np.exp(-scores))
    return pipeline.predict(X_data).astype(float)

def plot_confusion_matrix(cm, labels=("No", "Yes")):
    """
    Styled 2x2 confusion matrix:
    - Fixed size: figsize=(5,4)
    - 4 colored blocks
    - TP / FN / FP / TN labels
    - Axis labels: Actual value vs Predicted value
    labels: (negative_label, positive_label)
    """
    neg_label, pos_label = labels

    # Requested size
    fig, ax = plt.subplots(figsize=(5, 4))

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([
        "#6dd38b",  # TP
        "#f4a6a6",  # FN
        "#f7b2b2",  # FP
        "#7bd88f"   # TN
    ])

    # paint each cell with a fixed category index
    ax.imshow([[0, 1],
               [2, 3]], cmap=cmap)

    # labels with counts
    text_labels = np.array([
        [f"TP\n{cm[0,0]}", f"FN\n{cm[0,1]}"],
        [f"FP\n{cm[1,0]}", f"TN\n{cm[1,1]}"]
    ])

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, text_labels[i, j],
                ha="center", va="center",
                fontsize=12, fontweight="bold", color="black"
            )

    # axis ticks (Positive=Yes, Negative=No)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([pos_label, neg_label], fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([pos_label, neg_label], fontsize=10)

    ax.set_xlabel("Predicted value", fontsize=10)
    ax.set_ylabel("Actual value", fontsize=10)

    # remove borders & ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    plt.tight_layout()
    return fig

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

selected_model = st.sidebar.selectbox("Select ML Model", list(MODEL_FILES.keys()), index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Sample Test CSV")
if os.path.exists(SAMPLE_CSV_PATH):
    with open(SAMPLE_CSV_PATH, "rb") as f:
        st.sidebar.download_button(
            label="Download sample test CSV",
            data=f.read(),
            file_name="telco_test_data.csv",
            mime="text/csv"
        )
else:
    st.sidebar.info("Add sample CSV at: data/telco_test_data.csv")

st.sidebar.markdown("---")
st.sidebar.caption("BITS WILP ML Assignment-2 • 6 Models • Telco Churn")

# ------------------------------------------------------------
# Main: Upload + Predict
# ------------------------------------------------------------
left, right = st.columns([1.15, 0.85])

with left:
    st.markdown("#### 📂 Upload Test CSV")
    st.info(
        "Upload **test data only**. If your CSV includes `Churn` (Yes/No), the app will show metrics + confusion matrix."
    )
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

with right:
    st.markdown("#### ✅ Selected Model")
    st.markdown(
        f'<div class="card"><b>{selected_model}</b><br/>Pipeline includes preprocessing + model.</div>',
        unsafe_allow_html=True
    )

if uploaded is None:
    st.warning("👆 Upload a CSV to begin. You can also download the sample test CSV from the sidebar.")
    st.stop()

# Read + preview
try:
    df_upload = pd.read_csv(uploaded)
except Exception as e:
    st.error("Could not read the uploaded CSV. Please upload a valid .csv file.")
    st.exception(e)
    st.stop()

st.markdown("---")
st.markdown("#### 🔎 Preview of Uploaded Data")
st.dataframe(df_upload.head(20), use_container_width=True)

# Clean
df_upload = safe_prepare_dataframe(df_upload)

# Separate y if present
y_available = "Churn" in df_upload.columns
if y_available:
    y_true = df_upload["Churn"].map({"No": 0, "Yes": 1})
    if y_true.isna().any():
        st.warning("`Churn` column has values other than Yes/No. Metrics will be skipped.")
        y_available = False
    else:
        y_true = y_true.astype(int)
        X_infer = df_upload.drop(columns=["Churn"])
else:
    X_infer = df_upload

# Load model
model_path = os.path.join(MODEL_DIR, MODEL_FILES[selected_model])
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}. Ensure joblib models exist in the /model folder.")
    st.stop()

try:
    pipeline = load_pipeline(model_path)
except Exception as e:
    st.error("Failed to load the selected model pipeline. This is usually a version mismatch issue.")
    st.exception(e)
    st.stop()

# Predict button
st.markdown("---")
if st.button("🚀 Predict", use_container_width=True):
    try:
        proba = get_probabilities(pipeline, X_infer)
        y_pred = (proba >= 0.5).astype(int)
    except Exception as e:
        st.error("Prediction failed. Please ensure uploaded columns match training dataset structure.")
        st.exception(e)
        st.stop()

    pred_labels = np.where(y_pred == 1, "Yes", "No")

    out = X_infer.copy()
    out["PredictedChurn"] = pred_labels
    out["ChurnProbability(Yes)"] = np.round(proba, 4)

    # ============================================================
    # SINGLE PAGE LAYOUT (No tabs)
    # Order required:
    # Predictions Summary -> Metrics -> Reports -> Download -> Predictions Output
    # ============================================================

    st.markdown("---")
    st.header("📌 Predictions")

    st.subheader("📈 Prediction Summary")
    churn_yes_pct = y_pred.mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Churn % (Yes)", f"{churn_yes_pct:.2f}%")
    c2.metric("Total Rows", f"{len(out)}")
    c3.metric("Model", selected_model)

    st.write(out["PredictedChurn"].value_counts())

    # ============================================================
    st.markdown("---")
    st.header("📊 Metrics")

    st.subheader("✅ Evaluation Metrics")
    if not y_available:
        st.info("No valid `Churn` column found in uploaded CSV → metrics not available.")
    else:
        auc = roc_auc_score(y_true, proba)
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": auc,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred)
        }

        m1, m2, m3 = st.columns(3)
        m1.metric("🎯 Accuracy", f"{metrics['Accuracy']:.4f}")
        m2.metric("📈 AUC", f"{metrics['AUC']:.4f}")
        m3.metric("🔍 Precision", f"{metrics['Precision']:.4f}")

        m4, m5, m6 = st.columns(3)
        m4.metric("📣 Recall", f"{metrics['Recall']:.4f}")
        m5.metric("⚖️ F1", f"{metrics['F1']:.4f}")
        m6.metric("📐 MCC", f"{metrics['MCC']:.4f}")

        st.subheader("🧩 Confusion Matrix")
        # Keep Yes as positive class first to match TP/FN/FP/TN layout
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # Yes then No
        fig = plot_confusion_matrix(cm, labels=("No", "Yes"))
        st.pyplot(fig, use_container_width=False)

    # ============================================================
    st.markdown("---")
    st.header("🧾 Reports")

    st.subheader("📄 Classification Report")
    if not y_available:
        st.info("Upload CSV with `Churn` column (Yes/No) to view the classification report.")
    else:
        report_dict = classification_report(
            y_true, y_pred,
            target_names=["No", "Yes"],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).T

        if "support" in report_df.columns:
            report_df["support"] = report_df["support"].astype(int)

        for col in ["precision", "recall", "f1-score"]:
            if col in report_df.columns:
                report_df[col] = report_df[col].round(4)

        st.dataframe(report_df, use_container_width=True)

    # ============================================================
    st.markdown("---")
    st.subheader("⬇️ Download predictions")

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions as CSV",
        data=csv_bytes,
        file_name="predictions_output.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ============================================================
    st.markdown("---")
    st.header("📌 Predictions Output (Detailed)")
    st.dataframe(out.head(100), use_container_width=True)
