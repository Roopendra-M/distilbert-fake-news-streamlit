import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# UI setup
st.set_page_config(page_title="Text Classifier", page_icon="🧠")
st.title("🧠 Text Emotion Classifier (DistilBERT Demo)")
st.markdown("Model: `bhadresh-savani/distilbert-base-uncased-emotion` (for demo)")

# Load model
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
    model = DistilBertForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
    return tokenizer, model

tokenizer, model = load_model()

# Input text
text = st.text_area("✍️ Enter text (e.g., news content, tweet, etc.)", height=200)

if st.button("🔍 Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()

        # You can customize label mapping if known
        st.success(f"Predicted Class ID: `{pred}`")
