import sys
import os
import yfinance as yf
from flask import Flask, request, jsonify, render_template
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from dotenv import load_dotenv  # Import dotenv to load .env file

# Load environment variables from .env file
load_dotenv()

# API key for NewsAPI (now sourced from environment variable)
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

# Add the main project directory to sys.path for importing utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the cleanText function from utils
from src.utils.text_cleaning import cleanText

# Initialize Flask app
app = Flask(__name__)

# Load the saved models and vectorizer
model = joblib.load(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'best_model.pkl'))
vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'vectorizer.pkl'))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'scaler.pkl'))

# Keywords to indicate investment importance
important_keywords = [
    'merger', 'acquisition', 'earnings', 'IPO', 'bankruptcy', 'growth', 'expansion', 'dividend',
    'revenue', 'profit', 'loss', 'layoff', 'restructuring', 'recovery', 'stock split', 'buyback',
    'partnership', 'joint venture', 'strategic alliance', 'valuation', 'capital raise', 'shareholders',
    'funding', 'investor sentiment', 'regulation', 'lawsuit', 'court ruling', 'tax reform', 'interest rates',
    'inflation', 'market share', 'competition', 'technology', 'disruption', 'innovation', 'market trend',
    'global economy', 'geopolitical', 'tariff', 'sanction', 'trade agreement', 'stimulus package', 'recession',
    'stocks', 'bonds', 'commodities', 'forex', 'trading', 'investment', 'portfolio', 'asset management',
    'risk management', 'capital gains', 'dividend yield', 'bull market', 'bear market', 'market volatility',
    'equity', 'debt', 'credit rating', 'inflation rate', 'interest rate cut', 'interest rate hike', 'credit default',
    'banking', 'financial crisis', 'liquidity', 'stock price', 'earnings report', 'quarterly report', 'annual report',
    'private equity', 'public offering', 'investment bank', 'brokerage', 'market cap', 'asset allocation',
    'margin trading', 'short selling', 'arbitrage', 'underwriting', 'securities', 'derivatives', 'hedge fund',
    'exchange rate', 'mutual funds', 'ETF', 'index fund', 'real estate', 'commodities market', 'oil prices',
    'gold prices', 'silver prices', 'cryptocurrency', 'bitcoin', 'ethereum', 'blockchain', 'crowdfunding',
    'angel investor', 'venture capital', 'private placement', 'public offering', 'share buyback', 'stock buyback',
    'capital structure', 'futures market', 'economic outlook', 'bank rate', 'inflationary pressure', 'deflation',
    'trade deficit', 'foreign exchange', 'market sentiment', 'institutional investor', 'retail investor',
    'investment strategy', 'portfolio diversification', 'hedging', 'capital preservation', 'asset bubble', 'default risk',
    'credit default swap', 'interest rate policy', 'policy change', 'regulatory change', 'geopolitical risks', 'trade war',
    'Apple', 'macOS', 'iOS', 'Mozilla', 'iPad', 'AI'
]

# Function to calculate an "importance score" based on keywords
def calculate_importance_score(text):
    score = 0
    for keyword in important_keywords:
        if keyword.lower() in text.lower():
            score += 1
    return score

# Improved rule-based prediction (adjusted based on sentiment and importance score)
def predict_stock_movement(sentiment_label, importance_score):
    print(f"Sentiment: {sentiment_label}, Importance Score: {importance_score}")

    # Improved prediction logic
    if sentiment_label == 'POSITIVE' and importance_score > 2:
        return "Stock likely to go up"
    elif sentiment_label == 'NEGATIVE' and importance_score > 1:
        return "Stock likely to go down"
    elif sentiment_label == 'POSITIVE' and importance_score <= 2:
        return "Stock may remain neutral (Positive sentiment, low importance)"
    elif sentiment_label == 'NEGATIVE' and importance_score <= 1:
        return "Stock may remain neutral (Negative sentiment, low importance)"
    else:
        return "Stock may remain neutral (Sentiment not detected or invalid)"

# Fetch stock data before and after news articles
def fetch_stock_data_for_news(ticker, news_date, days_before=3, days_after=3):
    stock = yf.Ticker(ticker)
    start_date = pd.to_datetime(news_date) - pd.Timedelta(days=days_before)
    end_date = pd.to_datetime(news_date) + pd.Timedelta(days=days_after)
    stock_data = stock.history(start=start_date, end=end_date)
    return stock_data

# Convert stock data to a list of dictionaries for easier display
def fetch_stock_data_for_display(stock_data):
    stock_data_list = []
    for date, row in stock_data.iterrows():
        stock_data_list.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": row["Open"],
            "high": row["High"],
            "low": row["Low"],
            "close": row["Close"],
            "volume": row["Volume"]
        })
    return stock_data_list

# Fetch historical stock data using yfinance
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date)
    return stock_data

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_news", methods=["POST"])
def predict_news():
    try:
        news = request.form["news"]
        # Clean the news text before analysis
        cleaned_news = cleanText(news)

        if cleaned_news is None:
            return jsonify({"error": "Failed to clean the text."}), 500

        # Vectorize the cleaned news text
        news_vectorized = vectorizer.transform([cleaned_news])

        # Predict sentiment using the trained model
        sentiment = model.predict(news_vectorized)
        sentiment_label = "POSITIVE" if sentiment[0] == 1 else "NEGATIVE"

        # Calculate importance score
        importance_score = calculate_importance_score(news)

        # Predict stock movement based on sentiment and importance score
        stock_movement = predict_stock_movement(sentiment_label, importance_score)

        print(f"Sentiment: {sentiment_label}, Stock Movement: {stock_movement}")  # Debugging line

        # Ensure values are passed to the template
        return render_template("index.html", sentiment=f"Sentiment: {sentiment_label}", stock_movement=stock_movement, news=news)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_date_range", methods=["POST"])
def predict_date_range():
    try:
        # Retrieve form data
        from_date = request.form["from_date"]
        to_date = request.form["to_date"]
        sort_order = request.form["sort_order"]
        stock_ticker = request.form["stock_ticker"]

        # Fetch stock data before and after news articles
        stock_data = fetch_stock_data_for_news(stock_ticker, from_date)

        # Convert stock data to list of dictionaries
        stock_data_list = fetch_stock_data_for_display(stock_data)

        # Form the URL for the NewsAPI request with the provided dates and stock ticker
        url = f"https://newsapi.org/v2/everything?q={stock_ticker}&from={from_date}&to={to_date}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"

        # Make the GET request to NewsAPI
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            articles_data = []

            for article in data.get("articles", []):
                title = article.get("title", "")
                description = article.get("description", "")
                text = title + (" " + description if description else "")

                # Clean the article text before analysis
                cleaned_text = cleanText(text)

                if cleaned_text is None:
                    continue

                # Vectorize the cleaned article text
                news_vectorized = vectorizer.transform([cleaned_text])

                # Predict sentiment using the trained model (scaled compound score)
                sentiment_scaled = model.predict(news_vectorized)

                # Inverse the scaling of predictions
                sentiment_compound = scaler.inverse_transform(sentiment_scaled.reshape(-1, 1))[0][0]

                # Map the sentiment prediction to 'positive', 'negative', or 'neutral' based on compound score
                sentiment_label = 'POSITIVE' if sentiment_compound > 0 else 'NEGATIVE' if sentiment_compound < 0 else 'NEUTRAL'

                # Calculate importance score based on keywords
                importance_score = calculate_importance_score(title + " " + description)

                # Calculate the combined score: Importance + Sentiment
                combined_score = importance_score + sentiment_compound

                # Predict stock movement based on sentiment and importance score
                stock_movement = predict_stock_movement(sentiment_label, importance_score)

                # Store article data
                articles_data.append({
                    "title": title,
                    "importance_score": importance_score,
                    "combined_score": combined_score,
                    "model_prediction": sentiment_label,
                    "sentiment_compound": sentiment_compound,
                    "stock_movement": stock_movement
                })

            # Sort articles based on the sentiment_compound value
            if sort_order == "asc":
                # Sort ascending by sentiment_compound
                articles_data.sort(key=lambda x: x["sentiment_compound"], reverse=False)
            elif sort_order == "desc":
                # Sort descending by sentiment_compound
                articles_data.sort(key=lambda x: x["sentiment_compound"], reverse=True)

            return render_template("index.html", articles=articles_data, stock_data=stock_data_list)

        else:
            return jsonify({"error": "Failed to fetch news."}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
