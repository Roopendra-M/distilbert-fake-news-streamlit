import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.markdown("Model: **Pulk17/Fake-News-Detection** (BERT fine-tuned, ~99.6% accuracy)")

@st.cache_resource
def load_model():
    model_name = "Pulk17/Fake-News-Detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

text = st.text_area("Enter news text here:", height=200, placeholder="Paste article or headline...")

if st.button("🔍 Detect"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("🧠 Analyzing..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                logits = model(**inputs).logits
                pred = torch.argmax(logits, dim=1).item()
        label = "❌ FAKE" if pred == 1 else "✅ REAL"
        st.success(f"**Prediction:** {label}")
