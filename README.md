🚀 Project Overview

This project analyzes 14,000+ tweets about major U.S. airlines to uncover customer sentiment patterns using NLP, Machine Learning, and interactive dashboards.

💡 Key objectives:

-Identify positive, neutral, and negative tweets.

-Compare airlines based on sentiment trends.

-Visualize most common words, hashtags, and emojis.

-Generate actionable insights and recommendations.

🎯 Features
Notebook / Pipeline

1)Text Preprocessing: Clean tweets (remove URLs, mentions, hashtags, special characters, stopwords)

2)VADER Sentiment Analysis: Rule-based NLP labeling (positive, neutral, negative)

3)Machine Learning Model: Logistic Regression (primary, lightweight for Streamlit)

4)Optional Model Comparison: Random Forest (not deployed due to size)

5)EDA:

-Sentiment distribution plots

-Word clouds by sentiment

-Retweet engagement analysis

-Airline-level sentiment comparison

6)Exports:

-Clean dataset: airline_tweets_sentiment_dashboard.csv

-Logistic Regression model: sentiment_model.pkl

-TF-IDF vectorizer: tfidf_vectorizer.pkl

Streamlit Dashboard

1)Sidebar filters: Airline & Sentiment

2)KPIs: Total tweets, sentiment percentages, airline with most negative tweets

3)Sentiment Distribution: VADER histogram

4)Word Clouds: Overall & by airline

5)Retweet Analysis: Distribution, averages, and airline-wise comparison

6)Emoji Sentiment (optional)

7)Dynamic Insights & Recommendations

| Model                    | Purpose                 
| ------------------------ | ----------------------- 
| Logistic Regression      | Predict tweet sentiment 
| Random Forest (Optional) | Accuracy comparison     
| VADER Sentiment          | Rule-based labeling     


🧰 Tech Stack

1)Python 3.9+

2)Libraries: Pandas, NumPy, Matplotlib, Seaborn, Plotly, WordCloud, NLTK, Scikit-learn, Streamlit, Emoji

3)ML Models: Logistic Regression (primary), Random Forest (optional)

4)Deployment: Streamlit Dashboard

## 🚀 Demo
[View the live Streamlit app here](https://airlinesentimentanalysis-wccoeappaudxq86guygehw4.streamlit.app/)
