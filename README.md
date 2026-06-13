# 📊 Reddit-Pulse – AI Sentiment Analyzer

A powerful **Streamlit-based AI application** that fetches live Reddit posts using the Reddit API and performs **advanced sentiment analysis + Named Entity Recognition (NER)** using both **VADER and Hugging Face BERT models**.

It provides real-time insights with an interactive dashboard and downloadable reports.

---

## 🚀 Features

* 🔥 Fetch live posts from multiple subreddits using Reddit API (PRAW)
* 🧠 Dual sentiment analysis:
  * VADER (rule-based NLP model)
  * BERT (Hugging Face Transformer model)
* 🏷️ Named Entity Recognition (NER) using Hugging Face BERT
* 🧹 Advanced text preprocessing pipeline (cleaning, normalization, stopwords removal)
* 📊 Interactive Streamlit dashboard
* 🔗 Clickable Reddit post links
* 📁 CSV export for analysis reports
* ⚡ Model switching (VADER / BERT)
* 🔐 Secure API credentials using `.env`

---

## 🧠 How It Works

1. User enters subreddit names and selects post limit
2. Reddit data is fetched using **PRAW API**
3. Text is cleaned and preprocessed
4. Sentiment is analyzed using:
   * VADER OR
   * BERT (Hugging Face Transformer)
5. Named entities (people, orgs, locations) are extracted using NER
6. Results are displayed in a Streamlit dashboard
7. Data can be downloaded as CSV

---

## 🛠️ Tech Stack

* Python 🐍
* Streamlit 🎈
* PRAW (Reddit API)
* Pandas 📊
* NLTK (VADER Sentiment Analysis)
* Hugging Face Transformers 🤗
* PyTorch 🔥
* python-dotenv 🔐

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```
REDDIT_CLIENT_ID=YOUR_REDDIT_CLIENT_ID
REDDIT_CLIENT_SECRET=YOUR_REDDIT_CLIENT_SECRET
REDDIT_USER_AGENT=YOUR_REDDIT_USER_AGENT
```

---

## 📦 Installation

### Clone Repository

```bash
git clone https://github.com/abdullahwaseem404/Reddit-Pulse.git
cd Reddit-Pulse
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Run Application

```bash
streamlit run app.py
```

---

## 📊 Output Features

* Sentiment Label: Positive / Neutral / Negative
* Named Entities (NER): People, Organizations, Locations
* Author, Upvotes, Comments
* Timestamp
* Clickable Reddit links
* CSV export for further analysis

---
