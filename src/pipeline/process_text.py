import sys
import os

# Add the src directory to the system path

# Now you can import cleanText from utils.text_cleaning
from src.utils.text_cleaning import cleanText


def process_text(df):
    df['news'] = df['news'].apply(cleanText)
    return df
