import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Setup
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🧠")
st.title("📰 Fake News Detector")
st.markdown("""
Model: **[Pulk17/Fake-News-Detection](https://huggingface.co/Pulk17/Fake-News-Detection)**  
Fine-tuned BERT model (~99.6% accuracy)  
Paste news text or upload a file to detect fake content.
""")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "Pulk17/Fake-News-Detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Sidebar: File upload and text stats
with st.sidebar:
    st.header("📎 Upload a .txt File")
    uploaded_file = st.file_uploader("Choose a file", type=["txt"])
    example_fake = st.button("💡 Load Example FAKE")
    example_real = st.button("💡 Load Example REAL")

# Input handling
text = ""

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
elif example_fake:
    text = "NASA confirms Earth will go dark for six days in November due to planetary alignment."
elif example_real:
    text = "The Reserve Bank of India raised interest rates by 0.25% in its latest monetary policy meeting."
else:
    text = st.text_area("✍️ Enter or paste news article content:", height=200)

# Text stats
with st.sidebar:
    if text:
        st.header("📊 Text Stats")
        st.write("📝 Words:", len(text.split()))
        st.write("🔡 Characters:", len(text))

# Detect button
if st.button("🔍 Detect"):
    if not text.strip():
        st.warning("Please enter or upload some text.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                logits = model(**inputs).logits
                pred = torch.argmax(logits, dim=1).item()
                probs = F.softmax(logits, dim=1)
                confidence = probs[0][pred].item()

        label = "❌ FAKE NEWS" if pred == 1 else "✅ REAL NEWS"
        st.success(f"**Prediction:** {label}")
        st.info(f"🧮 Confidence Score: `{confidence * 100:.2f}%`")

# Footer
st.markdown("---")
st.caption("Built with 🤗 Transformers and Streamlit by Roopendra R")
