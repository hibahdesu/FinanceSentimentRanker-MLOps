import os
import pandas as pd

def load_data():
    # Check if the directory exists, if not, create it
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
    os.makedirs(data_dir, exist_ok=True)

    url = "https://github.com/hibahdesu/News-Sentiment-Analysis/raw/main/data/news.csv"
    df = pd.read_csv(url)  # Load CSV data directly from the URL
    
    # Save to the correct path
    df.to_csv(os.path.join(data_dir, 'news.csv'), index=False)  # Save to local file
    print("Data downloaded and saved to data/raw/news.csv")
    return df

load_data()
