# 📊 Reddit-Pulse
# Reddit Sentiment Analyzer

A Streamlit-based application that fetches live Reddit posts using the Reddit API and performs **sentiment analysis** on post titles using **NLTK’s VADER** sentiment analyzer.

Working Demo : https://youtu.be/TS8x5n8o-Qs

## 🚀 Features
- Fetch live posts from multiple subreddits
- Sentiment classification: **Positive, Neutral, Negative**
- Text cleaning and preprocessing
- Interactive Streamlit interface
- Clickable Reddit post links
- CSV export of analyzed data
- Secure API credentials via `.env` file

## 🧠 How It Works
1. User inputs subreddit names and post limit
2. Reddit data is fetched using **PRAW**
3. Text is cleaned and processed
4. Sentiment is classified using **VADER**
5. Results are displayed and downloadable

## 🛠️ Tech Stack
- Python
- PRAW (Reddit API)
- Pandas
- NLTK (VADER Sentiment Analyzer)
- Streamlit
- python-dotenv

## 🔐 Environment Variables

Create a `.env` file in the project root:

```
REDDIT_CLIENT_ID=YOUR_REDDIT_CLIENT_ID
REDDIT_CLIENT_SECRET=YOUR_REDDIT_CLIENT_SECRET
REDDIT_USER_AGENT=YOUR_REDDIT_USER_AGENT
````

## ▶️ How to Run
1. Clone the repository:
```bash
git clone https://github.com/abdullahwaseem404/Reddit-Pulse.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the app:
```bash
streamlit run app.py
```
