import os
import re
import pandas as pd
import streamlit as st
from data_pipeline import RedditPulsePipeline

st.set_page_config(page_title="Reddit-Pulse AI Analyzer", layout="wide")

@st.cache_resource
def get_pipeline():
  return RedditPulsePipeline()


pipeline = get_pipeline()

if (
    not pipeline.client_id
    or not pipeline.client_secret
    or not pipeline.user_agent
):
  st.error("❌ Reddit API credentials missing in .env")
  st.stop()

st.title("📊 Reddit-Pulse – AI Sentiment Analyzer")
st.markdown(
    "Real-time Reddit sentiment analysis using **VADER + BERT + NER (Hugging"
    " Face)**"
)

subs_input = st.text_input("Subreddits (comma-separated):", "technology,AskReddit")
post_limit = st.slider("Posts per subreddit:", 5, 50, 20)
model_choice = st.radio(
    "Select Sentiment Model:", ["VADER", "BERT (Hugging Face)"]
)

if st.button("🚀 Run Analysis"):
  subreddits = [s.strip() for s in subs_input.split(",") if s.strip()]

  if not subreddits:
    st.warning("Please enter at least one subreddit.")
    st.stop()

  with st.spinner("Fetching Reddit data + running AI models..."):
    df = pipeline.fetch_reddit_data(subreddits, post_limit, model_choice)

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
          "Timestamp",
      ]].to_html(escape=False, index=False),
      unsafe_allow_html=True,
  )

  csv_df = df.copy()
  csv_df["Title"] = csv_df["Title"].str.replace(r"<.*?>", "", regex=True)
  csv = csv_df.to_csv(index=False).encode("utf-8")

  st.download_button(
      label="⬇️ Download CSV Report",
      data=csv,
      file_name="reddit_pulse_ai_sentiment.csv",
      mime="text/csv",
  )