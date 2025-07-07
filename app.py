import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Set page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.markdown("""
Built with **DistilBERT** fine-tuned on a fake news dataset  
Model: [`mrm8488/distilbert-base-uncased-finetuned-fake-news`](https://huggingface.co/mrm8488/distilbert-base-uncased-finetuned-fake-news)
""")

# Load model and tokenizer with spinner
@st.cache_resource
def load_model():
    with st.spinner("🔄 Loading model and tokenizer... please wait"):
        tokenizer = DistilBertTokenizer.from_pretrained("mrm8488/distilbert-base-uncased-finetuned-fake-news")
        model = DistilBertForSequenceClassification.from_pretrained("mrm8488/distilbert-base-uncased-finetuned-fake-news")
    return tokenizer, model

tokenizer, model = load_model()

# Input section
st.markdown("#### ✍️ Enter the news article text:")
text = st.text_area("", height=200, placeholder="Type or paste news content here...")

if st.button("🔍 Detect"):
    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🧠 Analyzing with DistilBERT..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()

        label = "❌ FAKE NEWS" if prediction == 1 else "✅ REAL NEWS"
        st.success(f"**Prediction:** {label}")
