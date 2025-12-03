"""
Configuration file for MH[FG] - Fake News Detector Project
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
DATASETS_DIR = DATA_DIR / "datasets"
SRC_DIR = BASE_DIR / "src"

# API Configuration (Load from .env)
from dotenv import load_dotenv
load_dotenv()

API_CONFIG = {
    'GOOGLE_NEWS_API_KEY': os.getenv('GOOGLE_NEWS_API_KEY', ''),
    'TWITTER_API_KEY': os.getenv('TWITTER_API_KEY', ''),
    'TWITTER_API_SECRET': os.getenv('TWITTER_API_SECRET', ''),
    'TWITTER_ACCESS_TOKEN': os.getenv('TWITTER_ACCESS_TOKEN', ''),
    'TWITTER_ACCESS_SECRET': os.getenv('TWITTER_ACCESS_SECRET', ''),
}

# Model Paths
MODEL_PATHS = {
    'logistic_regression': MODELS_DIR / "logistic_regression.pkl",
    'naive_bayes': MODELS_DIR / "naive_bayes.pkl",
    'bert': MODELS_DIR / "bert",
    'ensemble': MODELS_DIR / "ensemble.pkl",
    'tfidf_vectorizer': MODELS_DIR / "tfidf_vectorizer.pkl",
    'label_encoder': MODELS_DIR / "label_encoder.pkl"
}

# Dataset Paths
DATASET_PATHS = {
    'liar': DATASETS_DIR / "liar_dataset.csv",
    'fakenewsnet': DATASETS_DIR / "fakenewsnet.csv",
    'kaggle': DATASETS_DIR / "kaggle_fake_news.csv",
    'combined': DATASETS_DIR / "combined_dataset.csv"
}

# Model Parameters
MODEL_PARAMS = {
    'max_features': 5000,
    'test_size': 0.2,
    'random_state': 42,
    'bert_max_length': 128,
    'bert_batch_size': 16,
    'bert_epochs': 3
}

# Categories for news classification
CATEGORIES = [
    'politics',
    'health',
    'technology',
    'entertainment',
    'sports',
    'business',
    'science',
    'other'
]

# Sentiment Labels
SENTIMENT_LABELS = {
    'positive': '🟢 Positive',
    'negative': '🔴 Negative',
    'neutral': '⚪ Neutral'
}

# Fake News Indicators
FAKE_INDICATORS = [
    "exclamation marks",
    "emotional language",
    "lack of sources",
    "sensational headlines",
    "urgent calls to action",
    "conspiracy theories"
]