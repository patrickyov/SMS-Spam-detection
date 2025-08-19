import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

st.set_page_config(page_title="SMS Spam Detector (SVM)", layout="centered")

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "spam_svm_pipeline.joblib"
META_PATH = ARTIFACTS_DIR / "meta.json"

@st.cache_resource(show_spinner=True)
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Make sure you've uploaded artifacts/ to your app.")
    model = joblib.load(MODEL_PATH)
    classes = ["ham", "spam"]
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text())
            if isinstance(meta, dict) and "classes" in meta:
                classes = list(map(str, meta["classes"]))
        except Exception:
            pass
    return model, classes

model, class_names = load_artifacts()

st.title("SMS Spam Detector — SVM")
st.write("Enter an SMS message and I will predict whether it is ham (not spam) or spam.")

with st.sidebar:
    st.header("About this app")
    st.write("This demo uses a scikit-learn Pipeline (vectorizer -> SVM). It supports single-message predictions and batch CSV uploads.")
    st.caption("Tip: If your classifier does not expose predict_proba, the app shows decision scores instead.")

def predict_messages(texts):
    preds = model.predict(texts)
    scores = None
    probas = None
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(texts)
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(texts)
        if isinstance(raw, list):
            raw = np.asarray(raw)
        scores = raw
    return preds, probas, scores

tab1, tab2 = st.tabs(["Single message", "CSV batch"])

with tab1:
    default_text = "Congratulations! You have won a $1000 gift card. Reply WIN to claim."
    sms = st.text_area("Paste an SMS message:", value=default_text, height=140)
    if st.button("Predict", use_container_width=True):
        if not sms.strip():
            st.warning("Please enter a message.")
        else:
            labels, probas, scores = predict_messages([sms])
            label = str(labels[0])
            st.subheader(f"Prediction: {label}")
            if probas is not None:
                prob_df = pd.DataFrame([probas[0]], columns=class_names)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
            elif scores is not None:
                st.caption("Classifier does not expose predict_proba; showing decision score instead (margin from the boundary).")
                st.write(f"Decision score: {float(scores[0]):.4f}")

with tab2:
    st.write("Upload a CSV containing a text column (e.g., message or text).")
    file = st.file_uploader("CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write("Preview:", df.head())
        candidate_cols = [c for c in df.columns if c.lower() in {"message","text","sms","content","body"}]
        if not candidate_cols:
            candidate_cols = [c for c in df.columns if df[c].dtype == object]
        if not candidate_cols:
            st.error("No text column found. Please include a column named message or text.")
        else:
            text_col = candidate_cols[0]
            st.info(f"Using text column: {text_col}")
            labels, probas, scores = predict_messages(df[text_col].astype(str).tolist())
            out = df.copy()
            out["prediction"] = labels
            if probas is not None:
                try:
                    max_proba = probas.max(axis=1)
                except Exception:
                    max_proba = [None]*len(labels)
                out["pred_proba"] = max_proba
            elif scores is not None:
                out["decision_score"] = scores
            st.write("Results preview:", out.head())
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv", use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit + scikit-learn")