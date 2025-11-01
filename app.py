import streamlit as st
import pandas as pd
import joblib

# ─── App Setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pipeline Condition Predictor", layout="wide")
st.title("🔧 Pipeline Condition Predictor")

# ─── Load the Full Pipeline ───────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load("best_pipeline_rf_model.pkl")

pipeline = load_pipeline()

# ─── Constants ────────────────────────────────────────────────────────────────
EXPECTED_COLS = [
    "corrosion rate (Mpy)", "Diameter Size (inch)","Probe Type",
    "Line Length (KM)", "Pigging Frequency (Days)",
    "Surfactant Dosing (Litres)"
]
INV_LABEL_MAP = {0: "Healthy", 1: "Monitor", 2: "Repair Required"}

# ─── File Upload ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a CSV with these columns:\n" + ", ".join(EXPECTED_COLS),
    type="csv"
)
if not uploaded:
    st.info("Awaiting CSV upload.")
    st.stop()

# ─── Read & Validate ───────────────────────────────────────────────────────────
df = pd.read_csv(uploaded)
missing = set(EXPECTED_COLS) - set(df.columns)
if missing:
    st.error(f"❌ Missing columns: {sorted(missing)}")
    st.stop()

# ─── Slice to Required Features ───────────────────────────────────────────────
X_raw = df[EXPECTED_COLS].copy()
st.write("### Columns used for prediction", X_raw.columns.tolist())

# ─── Predict ──────────────────────────────────────────────────────────────────
preds = pipeline.predict(X_raw)
probs = pipeline.predict_proba(X_raw).max(axis=1)

# ─── Attach & Display ─────────────────────────────────────────────────────────
df["PredictedCondition"] = [INV_LABEL_MAP[p] for p in preds]
df["Confidence"] = probs.round(2)
st.success("✅ Predictions complete")
st.dataframe(df, use_container_width=True)

# ─── Download ─────────────────────────────────────────────────────────────────
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download predictions CSV",
    data=csv_bytes,
    file_name="predictions.csv",
    mime="text/csv"
)
