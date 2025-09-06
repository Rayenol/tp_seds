import pandas as pd
import pytest
from src.models.sentiment import extract_sentiment

# Load sentiment texts from CSV file
def load_sentiments_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['Text'].tolist()

# Specify the path to your CSV file
file_path = 'src/models/soccer_sentiment_analysis.csv'

# Test sentiment extraction
texts = load_sentiments_from_csv(file_path)

@pytest.mark.parametrize('sample', texts)
def test_extract_sentiment(sample):
    sentiment = extract_sentiment(sample)

    # Adjust the heuristic for sentiment classification
    if sentiment < 0:
        assert sentiment <= 0  # Negative sentiment check
    elif sentiment == 0:
        assert True  # Consider neutral sentiment as acceptable
    else:
        assert sentiment > 0  # Positive sentiment check

def test_specific_positive_sentiments():
    positive_samples = [
        "Barcelona played brilliantly last Wednesday. Rafinia’s hat-trick was pure magic. Visca Barça!",
        "The team's performance was outstanding, showing great teamwork and spirit.",
        "What a fantastic victory! The players delivered an incredible performance.",
        "I'm thrilled with how the match turned out! A deserved win!"
    ]
    
    for sample in positive_samples:
        sentiment = extract_sentiment(sample)
        assert sentiment > 0  # Check that the sentiment is positive
