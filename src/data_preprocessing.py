import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import os

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self, fake_path, true_path):
        """Load fake and true news datasets"""
        print("Loading datasets...")
        
        # Load fake news
        fake_df = pd.read_csv(fake_path)
        fake_df['label'] = 0  # 0 for fake
        fake_df['text'] = fake_df['text'].astype(str)
        
        # Load true news
        true_df = pd.read_csv(true_path)
        true_df['label'] = 1  # 1 for true
        true_df['text'] = true_df['text'].astype(str)
        
        # Combine datasets
        df = pd.concat([fake_df, true_df], ignore_index=True)
        
        print(f"Total samples: {len(df)}")
        print(f"Fake news: {len(fake_df)}")
        print(f"True news: {len(true_df)}")
        
        return df
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits (keep only letters)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df):
        """Apply preprocessing to entire dataframe"""
        print("Cleaning text data...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Remove empty texts after cleaning
        df = df[df['cleaned_text'].str.len() > 10]
        
        print(f"Samples after cleaning: {len(df)}")
        return df
    
    def prepare_features(self, df, test_size=0.2, random_state=42):
        """Split data and prepare for training"""
        print("Splitting data...")
        
        # Split features and labels
        X = df['cleaned_text'].values
        y = df['label'].values
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test