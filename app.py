import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import plotly.graph_objects as go

# Page settings
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🧠")
st.title("📰 Fake News Detector")
st.markdown("""
Model: **[distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)**  
General-purpose binary classifier (used here to detect real/fake news).  
✔️ High accuracy & reliability  
📎 Upload a file or select a sample news headline to test it.
""")

# Load tokenizer & model
@st.cache_resource
def load_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Examples
with st.sidebar:
    st.header("📝 Load Examples or Upload")
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    example_type = st.radio("Choose Example Type", ["None", "Real News", "Fake News"])

    real_examples = [
        "India successfully launched its third lunar mission, Chandrayaan-3.",
        "The WHO declared the end of the COVID-19 global health emergency.",
        "The RBI raised the repo rate by 0.25% to control inflation."
    ]

    fake_examples = [
        "NASA confirms Earth will go dark for six days due to planetary alignment.",
        "Government to distribute free iPhones to all citizens this Diwali.",
        "Aliens landed in the Sahara desert and signed a peace treaty."
    ]

    example_text = ""
    if example_type == "Real News":
        example_text = st.selectbox("Choose Real News", real_examples)
    elif example_type == "Fake News":
        example_text = st.selectbox("Choose Fake News", fake_examples)

# Store in session
if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""

if uploaded_file:
    st.session_state["text_input"] = uploaded_file.read().decode("utf-8")
elif example_text:
    st.session_state["text_input"] = example_text

# Input area
text = st.text_area("✍️ Enter or paste news article:", value=st.session_state["text_input"], height=200)

# Sidebar stats
with st.sidebar:
    if text:
        st.header("📊 Text Stats")
        st.write("Words:", len(text.split()))
        st.write("Characters:", len(text))
        if len(text.split()) > 100:
            st.warning("⚠️ Text too long. Only the first 512 tokens will be considered.")

# Detect
if st.button("🔍 Detect"):
    if not text.strip():
        st.warning("Please enter or paste some content.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=1)[0].tolist()
                pred = int(torch.argmax(torch.tensor(probs)))

        label_map = {0: "❌ FAKE NEWS", 1: "✅ REAL NEWS"}
        emoji_map = {0: "🔴", 1: "🟢"}
        colors = ["crimson", "seagreen"]

        st.markdown(f"### {emoji_map[pred]} **Prediction:** {label_map[pred]}")
        st.markdown(f"**Confidence:** `{probs[pred]*100:.2f}%`")

        # Plotly confidence bar
        fig = go.Figure(go.Bar(
            x=["Fake", "Real"],
            y=probs,
            marker_color=colors,
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="auto"
        ))
        fig.update_layout(title="Confidence Scores", yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Built by Roopendra R using 🤗 Transformers, Plotly, and Streamlit")
