import os
import re
from datetime import datetime, timezone
from pathlib import Path
import nltk
import pandas as pd
import praw
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import logging as hf_logging, pipeline

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
hf_logging.set_verbosity_error()


class RedditPulsePipeline:

  def __init__(self):
    self.client_id = os.getenv("REDDIT_CLIENT_ID")
    self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    self.user_agent = os.getenv("REDDIT_USER_AGENT")

    if not self.client_id or not self.client_secret or not self.user_agent:
      raise ValueError("❌ Reddit API credentials missing from .env file.")

    self._setup_nltk()
    self.stop_words = set(stopwords.words("english"))
    self.sia = SentimentIntensityAnalyzer()
    self.sentiment_model, self.ner_model = self._load_hf_pipelines()

  def _setup_nltk(self):
    nltk.download("stopwords", quiet=True)
    nltk.download("vader_lexicon", quiet=True)

  def _load_hf_pipelines(self):
    sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt",
    )
    ner = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
        framework="pt",
    )
    return sentiment, ner

  def clean_text(self, text):
    if not isinstance(text, str):
      return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in self.stop_words]
    return " ".join(tokens)

  def vader_sentiment(self, text):
    score = self.sia.polarity_scores(text)["compound"]
    if score > 0.05:
      return "Positive"
    elif score < -0.05:
      return "Negative"
    return "Neutral"

  def bert_sentiment(self, text):
    try:
      result = self.sentiment_model(text[:512])[0]
      return "Positive" if result["label"] == "POSITIVE" else "Negative"
    except Exception:
      return "Neutral"

  def extract_entities(self, text):
    try:
      entities = self.ner_model(text[:512])
      clean_entities = set(
          [
              e["word"].replace("##", "").strip()
              for e in entities
              if not e["word"].startswith("##") or len(e["word"]) > 3
          ]
      )
      return ", ".join(clean_entities)
    except Exception:
      return ""

  def fetch_reddit_data(self, subreddits, limit, model_choice):
    reddit = praw.Reddit(
        client_id=self.client_id,
        client_secret=self.client_secret,
        user_agent=self.user_agent,
    )

    rows = []
    for sub in subreddits:
      subreddit = reddit.subreddit(sub.strip())
      for post in subreddit.hot(limit=limit):
        raw_text = f"{post.title} {post.selftext}"
        cleaned = self.clean_text(raw_text)
        post_link = f"https://www.reddit.com{post.permalink}"
        title_html = f'<a href="{post_link}" target="_blank">{post.title}</a>'

        sentiment = (
            self.vader_sentiment(cleaned)
            if model_choice == "VADER"
            else self.bert_sentiment(cleaned)
        )
        entities = self.extract_entities(raw_text)

        rows.append({
            "Subreddit": sub,
            "Title": title_html,
            "Cleaned Text": cleaned,
            "Sentiment": sentiment,
            "Entities (NER)": entities,
            "Author": str(post.author),
            "Upvotes": post.score,
            "Comments": post.num_comments,
            "Timestamp": datetime.fromtimestamp(
                post.created_utc, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "Link": post_link,
        })
    return pd.DataFrame(rows)