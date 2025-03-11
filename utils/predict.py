@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the request
        data = request.get_json()

        # Debugging: Print the incoming data to ensure it's being received correctly
        print("Received data:", data)

        # Check if 'news' field is provided
        if 'news' not in data:
            return jsonify({'error': 'No news text provided in the request.'}), 400

        # Get the list of news articles
        news_articles = data['news']
        
        # Debugging: Print the incoming news articles
        print("Incoming news articles:", news_articles)

        # Process and rank the news based on sentiment and other factors
        ranked_news = process_and_rank_news(news_articles, model, vectorizer, stock_keywords)
        
        # Debugging: Print the ranked news
        print("Ranked news:", ranked_news)

        # Get top 5 ranked news
        top_ranked_news = ranked_news[:5]

        # Return the top-ranked news
        return jsonify({'top_ranked_news': top_ranked_news})

    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging: Print the error if any
        return jsonify({'error': str(e)}), 500
