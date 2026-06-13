import os
import re
import pandas as pd
import streamlit as st
import nltk
import praw
from datetime import datetime
from dotenv import load_dotenv

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

from transformers import pipeline

load_dotenv()

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

if not CLIENT_ID or not CLIENT_SECRET or not USER_AGENT:
    st.error("❌ Reddit API credentials missing in .env")
    st.stop()

nltk.download("stopwords", quiet=True)
nltk.download("vader_lexicon", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
SIA = SentimentIntensityAnalyzer()

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

ner_model = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple"
)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)

def vader_sentiment(text):
    score = SIA.polarity_scores(text)["compound"]
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    return "Neutral"

def bert_sentiment(text):
    try:
        result = sentiment_model(text[:512])[0]
        label = result["label"]
        return "Positive" if label == "POSITIVE" else "Negative"
    except:
        return "Neutral"

def extract_entities(text):
    try:
        entities = ner_model(text[:512])
        return ", ".join(set([e["word"] for e in entities]))
    except:
        return ""

def fetch_reddit_data(subreddits, limit, model_choice):
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )

    rows = []

    for sub in subreddits:
        subreddit = reddit.subreddit(sub.strip())

        for post in subreddit.hot(limit=limit):
            raw_text = f"{post.title} {post.selftext}"
            cleaned = clean_text(raw_text)

            post_link = f"https://www.reddit.com{post.permalink}"
            title_html = f'<a href="{post_link}" target="_blank">{post.title}</a>'

            if model_choice == "VADER":
                sentiment = vader_sentiment(cleaned)
            else:
                sentiment = bert_sentiment(cleaned)

            entities = extract_entities(raw_text)

            rows.append({
                "Subreddit": sub,
                "Title": title_html,
                "Cleaned Text": cleaned,
                "Sentiment": sentiment,
                "Entities (NER)": entities,
                "Author": str(post.author),
                "Upvotes": post.score,
                "Comments": post.num_comments,
                "Timestamp": datetime.utcfromtimestamp(post.created_utc),
                "Link": post_link
            })

    return pd.DataFrame(rows)

st.set_page_config(page_title="Reddit-Pulse AI Analyzer", layout="wide")

st.title("📊 Reddit-Pulse – AI Sentiment Analyzer")
st.markdown("Real-time Reddit sentiment analysis using **VADER + BERT + NER (Hugging Face)**")

subs_input = st.text_input("Subreddits (comma-separated):", "technology,AskReddit")
post_limit = st.slider("Posts per subreddit:", 5, 50, 20)

model_choice = st.radio(
    "Select Sentiment Model:",
    ["VADER", "BERT (Hugging Face)"]
)

if st.button("🚀 Run Analysis"):

    subreddits = [s.strip() for s in subs_input.split(",") if s.strip()]

    if not subreddits:
        st.warning("Please enter at least one subreddit.")
        st.stop()

    with st.spinner("Fetching Reddit data + running AI models..."):
        df = fetch_reddit_data(subreddits, post_limit, model_choice)

    st.success(f"✅ Done! Fetched {len(df)} posts")

    st.write(
        df[[
            "Subreddit",
            "Title",
            "Sentiment",
            "Entities (NER)",
            "Author",
            "Upvotes",
            "Comments",
            "Timestamp"
        ]].to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

    csv_df = df.copy()
    csv_df["Title"] = csv_df["Title"].str.replace(r"<.*?>", "", regex=True)

    csv = csv_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download CSV Report",
        data=csv,
        file_name="reddit_pulse_ai_sentiment.csv",
        mime="text/csv"
    )