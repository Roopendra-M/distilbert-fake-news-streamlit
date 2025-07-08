import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🧠")
st.title("📰 Fake News Detector")
st.markdown("""
Model: **[Pulk17/Fake-News-Detection](https://huggingface.co/Pulk17/Fake-News-Detection)**  
Fine-tuned BERT model for detecting fake news.  
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

    st.subheader("🧾 Load Example Texts")
    example_type = st.selectbox("Choose Example Type", ["None", "Real News", "Fake News"])

    real_examples = [
        "The Reserve Bank of India raised interest rates by 0.25% in its latest monetary policy meeting.",
        "India launched its third lunar mission, Chandrayaan-3, successfully in 2023.",
        "The World Health Organization announced the official end of the COVID-19 global health emergency."
    ]

    fake_examples = [
        "NASA confirms Earth will go dark for six days in November due to planetary alignment.",
        "Scientists clone dinosaurs from DNA trapped in amber and open theme park in Brazil.",
        "Government to give free iPhones to all citizens under new technology policy."
    ]

    example_choice = None
    if example_type == "Real News":
        example_choice = st.selectbox("Select Real News Example", real_examples)
    elif example_type == "Fake News":
        example_choice = st.selectbox("Select Fake News Example", fake_examples)

# Session state for input
if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""

# Load selected example
if uploaded_file:
    st.session_state["text_input"] = uploaded_file.read().decode("utf-8")
elif example_choice:
    st.session_state["text_input"] = example_choice

# Text area
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
                probs = F.softmax(logits, dim=1)[0].tolist()
                pred = int(torch.argmax(torch.tensor(probs)))

        label_map = ["✅ REAL NEWS", "❌ FAKE NEWS"]
        st.success(f"**Prediction:** {label_map[pred]}")
        st.info(f"🔢 Confidence: `{probs[pred] * 100:.2f}%`")

        # Confidence chart
        fig, ax = plt.subplots()
        ax.bar(["REAL", "FAKE"], probs, color=["green", "red"])
        ax.set_ylim([0, 1])
        ax.set_ylabel("Confidence")
        ax.set_title("Model Confidence Scores")
        st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Built by Roopendra R using 🤗 Transformers, PyTorch, and Streamlit")
