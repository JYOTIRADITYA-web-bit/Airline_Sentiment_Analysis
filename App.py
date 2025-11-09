# ===============================
# Airline Sentiment Analysis Dashboard
# ===============================

# Step 0: Import Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
import plotly.express as px
import pickle

# Optional: Emoji support
try:
    import emoji
    EMOJI_AVAILABLE = True
except ModuleNotFoundError:
    EMOJI_AVAILABLE = False

# Streamlit page config
st.set_page_config(page_title="Airline Sentiment Dashboard", layout="wide")

# ===============================
# Step 1: Load Dataset
# ===============================
@st.cache_data
def load_data(path):
    """Load CSV data into a pandas DataFrame."""
    return pd.read_csv(path)

df = load_data("airline_tweets_sentiment_dashboard.csv")

# ===============================
# Step 2: Load ML Model & Vectorizer
# ===============================
# Logistic Regression model trained on TF-IDF features
with open("sentiment_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# ===============================
# Step 3: Apply Logistic Regression Model
# ===============================
# Create a new column for ML-predicted sentiment
# 'rf_predicted' is for ML predictions
df['ml_predicted'] = sentiment_model.predict(tfidf_vectorizer.transform(df['clean_text'].astype(str)))

# ===============================
# Step 4: Sidebar Filters
# ===============================
st.sidebar.header("Filters")
airlines = ['All'] + df['airline'].unique().tolist()
selected_airline = st.sidebar.selectbox("Select Airline", airlines)

sentiments = ['All', 'positive', 'neutral', 'negative']
selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiments)

# Filter dataset based on sidebar selection
filtered_df = df.copy()
if selected_airline != 'All':
    filtered_df = filtered_df[filtered_df['airline'] == selected_airline]
if selected_sentiment != 'All':
    filtered_df = filtered_df[filtered_df['vader_sentiment'] == selected_sentiment]

# ===============================
# Step 5: KPIs
# ===============================
total_tweets = len(filtered_df)
positive_pct = round((filtered_df['vader_sentiment'] == 'positive').mean() * 100, 2)
neutral_pct = round((filtered_df['vader_sentiment'] == 'neutral').mean() * 100, 2)
negative_pct = round((filtered_df['vader_sentiment'] == 'negative').mean() * 100, 2)

# Airline with most negative tweets (based on full dataset)
airline_most_negative = df[df['vader_sentiment']=='negative']['airline'].value_counts().idxmax() \
                        if len(df[df['vader_sentiment']=='negative'])>0 else "N/A"

# Display KPIs
st.title("✈️ Airline Tweet Sentiment Dashboard")
st.subheader("📊 Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tweets", total_tweets)
col2.metric("Positive (%)", f"{positive_pct}%")
col3.metric("Neutral (%)", f"{neutral_pct}%")
col4.metric("Negative (%)", f"{negative_pct}%")
st.metric("Airline with Most Negative Tweets", airline_most_negative)

# ===============================
# Step 6: Sentiment Distribution (VADER)
# ===============================
palette = {'positive':'#2ECC71','neutral':'#F1C40F','negative':'#E74C3C'}
fig = px.histogram(filtered_df, x='vader_sentiment', color='vader_sentiment',
                   category_orders={'vader_sentiment':['positive','neutral','negative']},
                   color_discrete_map=palette, title="VADER Sentiment Counts")
st.plotly_chart(fig)

# ===============================
# Step 7: Word Clouds
# ===============================
@st.cache_data
def generate_wordcloud(text, width=800, height=400, bg_color='white', cmap='viridis'):
    """Generate a word cloud from text."""
    return WordCloud(width=width, height=height, background_color=bg_color, colormap=cmap).generate(text)

# Overall Word Cloud
st.subheader("☁️ Word Cloud (Overall)")
text_wc = ' '.join(filtered_df['clean_text'].dropna().astype(str))
wc = generate_wordcloud(text_wc)
fig_wc, ax = plt.subplots(figsize=(10,4))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig_wc)

# Word Cloud by Airline
st.subheader("☁️ Word Cloud by Airline")
airline_options = ['All'] + df['airline'].unique().tolist()
selected_wc_airline = st.selectbox("Select Airline for Word Cloud", airline_options)

if selected_wc_airline == 'All':
    text_airline_wc = ' '.join(df['clean_text'].dropna().astype(str))
else:
    text_airline_wc = ' '.join(df[df['airline']==selected_wc_airline]['clean_text'].dropna().astype(str))

wc_airline = generate_wordcloud(text_airline_wc, cmap='plasma')
fig_wc_airline, ax = plt.subplots(figsize=(10,4))
ax.imshow(wc_airline, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig_wc_airline)

# ===============================
# Step 8: Retweet Analysis
# ===============================
if 'retweet_count' in df.columns:
    st.subheader("🔁 Retweets by Sentiment")
    fig_retweet = px.box(filtered_df, x='vader_sentiment', y='retweet_count',
                         category_orders={'vader_sentiment':['positive','neutral','negative']},
                         color='vader_sentiment', color_discrete_map=palette,
                         title="Retweets Distribution per Sentiment")
    st.plotly_chart(fig_retweet)

# Average retweets per sentiment
if 'retweet_count' in df.columns:
    st.subheader("🔁 Average Retweet Engagement by Sentiment")
    avg_retweets = filtered_df.groupby('vader_sentiment')['retweet_count'].mean().reindex(['positive','neutral','negative'])
    col1, col2, col3 = st.columns(3)
    col1.metric("Positive Avg Retweets", f"{round(avg_retweets['positive'], 2)}")
    col2.metric("Neutral Avg Retweets", f"{round(avg_retweets['neutral'], 2)}")
    col3.metric("Negative Avg Retweets", f"{round(avg_retweets['negative'], 2)}")

    fig_retweet_avg = px.bar(
        avg_retweets.reset_index(),
        x='vader_sentiment', y='retweet_count',
        color='vader_sentiment',
        color_discrete_map=palette,
        title="Average Retweets per Sentiment"
    )
    st.plotly_chart(fig_retweet_avg)

# Retweet engagement by airline and sentiment
if 'retweet_count' in df.columns:
    st.subheader("📈 Retweet Engagement by Airline and Sentiment")
    airline_retweet = filtered_df.groupby(['airline', 'vader_sentiment'])['retweet_count'].mean().reset_index()
    fig_airline_retweet = px.bar(
        airline_retweet,
        x='airline',
        y='retweet_count',
        color='vader_sentiment',
        barmode='group',
        color_discrete_map=palette,
        title="Average Retweets by Airline and Sentiment",
        labels={'retweet_count': 'Average Retweet Count'}
    )
    st.plotly_chart(fig_airline_retweet)

# ===============================
# Step 9: Emoji Sentiment (Optional)
# ===============================
if EMOJI_AVAILABLE:
    def extract_emoji(text):
        return [c for c in text if c in emoji.EMOJI_DATA]

    emoji_sentiment = {
        "😡":"negative","😠":"negative","😢":"negative","😭":"negative",
        "😍":"positive","😊":"positive","😃":"positive","😁":"positive",
        "👍":"positive","😔":"negative","😎":"positive"
    }

    def emoji_score(text):
        emojis = extract_emoji(text)
        if not emojis:
            return "neutral"
        scores = [emoji_sentiment.get(e,"neutral") for e in emojis]
        return max(set(scores), key=scores.count)

    df['emoji_sentiment'] = df['text'].astype(str).apply(emoji_score)
    filtered_df = filtered_df.copy()
    filtered_df['emoji_sentiment'] = df.loc[filtered_df.index, 'emoji_sentiment']

# ===============================
# Step 10: Dynamic Insights & Recommendations
# ===============================
with st.expander("🔍 Insights"):
    insights_list = []

    if len(filtered_df[filtered_df['vader_sentiment']=='negative']) > 0:
        top_neg_airline = filtered_df[filtered_df['vader_sentiment']=='negative']['airline'].value_counts().idxmax()
        insights_list.append(f"Negative sentiment dominates {top_neg_airline} tweets")

    positive_text = ' '.join(filtered_df[filtered_df['vader_sentiment']=='positive']['clean_text'].dropna())
    top_words = [w for w, c in Counter(positive_text.split()).most_common(3)]
    if top_words:
        insights_list.append(f"Positive tweets often mention {', '.join(top_words)}")

    if 'retweet_count' in filtered_df.columns:
        avg_retweet_neg = filtered_df[filtered_df['vader_sentiment']=='negative']['retweet_count'].mean()
        avg_retweet_pos = filtered_df[filtered_df['vader_sentiment']=='positive']['retweet_count'].mean()
        if avg_retweet_neg > avg_retweet_pos:
            insights_list.append("Negative tweets have higher average retweets than positive tweets")

    neutral_text = ' '.join(filtered_df[filtered_df['vader_sentiment']=='neutral']['clean_text'].dropna())
    if len(neutral_text) > 0:
        insights_list.append("Neutral tweets mostly contain general travel info")

    st.table(pd.DataFrame({"Insight": insights_list}))

with st.expander("✅ Recommendations"):
    recommendations_list = []
    if len(filtered_df[filtered_df['vader_sentiment']=='negative']) > 0:
        recommendations_list.append("Improve customer service to reduce negative sentiment")
    if len(filtered_df[filtered_df['vader_sentiment']=='positive']) > 0:
        recommendations_list.append("Highlight positive customer experiences in campaigns")
    if 'retweet_count' in filtered_df.columns:
        recommendations_list.append("Monitor highly retweeted negative tweets for quick response")
    if len(filtered_df[filtered_df['vader_sentiment']=='neutral']) > 0:
        recommendations_list.append("Engage with neutral tweets to convert them into positive experiences")

    st.table(pd.DataFrame({"Recommendation": recommendations_list}))


