import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Page config
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🧠")
st.title("📰 Fake News Detector")
st.markdown("""
Model: **[Pulk17/Fake-News-Detection](https://huggingface.co/Pulk17/Fake-News-Detection)**  
Fine-tuned BERT model (~99.6% accuracy)  
Paste news text or upload a file to detect fake content.
""")

# Load model + tokenizer
@st.cache_resource
def load_model():
    model_name = "Pulk17/Fake-News-Detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Sidebar controls
with st.sidebar:
    st.header("📎 Upload or Load Example")
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    example_fake = st.button("💡 Load Example FAKE")
    example_real = st.button("💡 Load Example REAL")

# Session state for input
if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""

# Load from file or buttons
if uploaded_file:
    st.session_state["text_input"] = uploaded_file.read().decode("utf-8")
elif example_fake:
    st.session_state["text_input"] = "NASA confirms Earth will go dark for six days in November due to planetary alignment."
elif example_real:
    st.session_state["text_input"] = "The Reserve Bank of India raised interest rates by 0.25% in its latest monetary policy meeting."

# Text input area
text = st.text_area("✍️ Enter or paste news article content:", value=st.session_state["text_input"], height=200)

# Sidebar text stats
with st.sidebar:
    if text:
        st.header("📊 Text Stats")
        st.write("📝 Words:", len(text.split()))
        st.write("🔡 Characters:", len(text))

# Predict button
if st.button("🔍 Detect"):
    if not text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("🧠 Analyzing..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                logits = model(**inputs).logits
                pred = torch.argmax(logits, dim=1).item()
                probs = F.softmax(logits, dim=1)
                confidence = probs[0][pred].item()

        label = "❌ FAKE NEWS" if pred == 1 else "✅ REAL NEWS"
        st.success(f"**Prediction:** {label}")
        st.info(f"🔢 Confidence: `{confidence * 100:.2f}%`")

# Footer
st.markdown("---")
st.caption("Built by Roopendra R using 🤗 Transformers and Streamlit")

