import os
import re
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import praw
from datetime import datetime

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

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)

def classify_sentiment(text):
    score = SIA.polarity_scores(text)["compound"]
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    return "Neutral"

def color_sentiment(val):
    return {
        "Positive": "color:green",
        "Negative": "color:red",
        "Neutral": "color:orange"
    }.get(val, "")

def fetch_reddit_data(subreddits, limit):
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )

    rows = []
    for sub in subreddits:
        subreddit = reddit.subreddit(sub.strip())
        for post in subreddit.hot(limit=limit):
            cleaned = clean_text(post.title)
            post_link = f"https://www.reddit.com{post.permalink}"
            title_html = f'<a href="{post_link}" target="_blank">{post.title}</a>'
            rows.append({
                "Subreddit": sub,
                "Title": title_html,
                "Cleaned Text": cleaned,
                "Sentiment": classify_sentiment(cleaned),
                "Author": str(post.author),
                "Upvotes": post.score,
                "Comments": post.num_comments,
                "Timestamp": datetime.utcfromtimestamp(post.created_utc),
                "Link": post_link  
            })
    return pd.DataFrame(rows)

st.set_page_config(page_title="Reddit Sentiment Analyzer", layout="wide")
st.title("📊 Reddit Sentiment Analyzer")

st.markdown("""
Enter subreddit names (comma-separated) and select how many posts to fetch per subreddit.
""")

subs_input = st.text_input("Subreddits:", "technology,AskReddit")
post_limit = st.slider("Posts per subreddit:", 5, 50, 20)

if st.button("🚀 Run Analysis"):
    subreddits = [s.strip() for s in subs_input.split(",")]
    
    if not subreddits:
        st.warning("Please enter at least one subreddit.")
    else:
        with st.spinner("Fetching and analyzing Reddit posts..."):
            df = fetch_reddit_data(subreddits, post_limit)

        st.success(f"✅ Analysis complete! Fetched {len(df)} posts.")

        st.write(
            df[['Subreddit','Title','Cleaned Text','Sentiment','Author','Upvotes','Comments','Timestamp']].to_html(escape=False, index=False),
            unsafe_allow_html=True
        )

        csv_df = df.copy()
        csv_df['Title'] = csv_df['Title'].str.replace(r"<.*?>", "", regex=True)
        csv = csv_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name="reddit_sentiment.csv",
            mime="text/csv"
        )