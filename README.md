# 📰 Fake News Detection using DistilBERT – Streamlit App

A web app to classify news content as **real** or **fake**, using a pretrained transformer model. Built with **Streamlit**, this app lets anyone paste a news article and get an instant prediction powered by **DistilBERT**.

> 🔗 Live Demo: _[Add your Streamlit Cloud link here after deployment]_

---

## 🎯 Objective

This project demonstrates how pre-trained transformer models like DistilBERT can be fine-tuned and used for downstream NLP tasks like **fake news detection**. The goal is to provide:

- A quick, accessible way for users to verify the credibility of news content.
- A showcase of using Hugging Face transformers in production via Streamlit.

---

## ⚙️ Features

- ✅ Real-time text classification (Fake vs Real)
- ✅ Uses pretrained model from Hugging Face – no training required
- ✅ Clean and user-friendly Streamlit UI
- ✅ Deployable on [Streamlit Cloud](https://streamlit.io/cloud)
- ✅ Fast inference using lightweight DistilBERT

---

## 🧠 Model Info

This app uses the model:

📦 [`mrm8488/distilbert-base-uncased-finetuned-fake-news`](https://huggingface.co/mrm8488/distilbert-base-uncased-finetuned-fake-news)

- Model type: DistilBERT
- Fine-tuned on a fake news classification dataset
- Output labels: `0 = Real`, `1 = Fake`

---

## 🛠️ Tech Stack

| Component   | Tech Used                             |
|-------------|----------------------------------------|
| Language    | Python                                 |
| Model       | DistilBERT via Hugging Face Transformers |
| UI          | Streamlit                              |
| Hosting     | Streamlit Cloud                        |
| Backend     | PyTorch                                |

---

## 🚀 How to Use

### 🔧 Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/fake-news-detection-app.git
cd fake-news-detection-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
