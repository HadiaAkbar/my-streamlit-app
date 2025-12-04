from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
import numpy as np

class FeatureEngineer:
    def __init__(self, max_features=5000, use_tfidf=True):
        self.max_features = max_features
        self.use_tfidf = use_tfidf
        self.vectorizer = None
        
    def create_vectorizer(self):
        """Create TF-IDF or Count vectorizer"""
        if self.use_tfidf:
            return TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2),  # Use uni and bi-grams
                min_df=5,
                max_df=0.7
            )
        else:
            return CountVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
    
    def extract_features(self, X_train, X_test=None):
        """Extract features from text"""
        print("Extracting features...")
        
        if self.vectorizer is None:
            self.vectorizer = self.create_vectorizer()
            X_train_features = self.vectorizer.fit_transform(X_train)
        else:
            X_train_features = self.vectorizer.transform(X_train)
        
        if X_test is not None:
            X_test_features = self.vectorizer.transform(X_test)
            return X_train_features, X_test_features
        
        return X_train_features
    
    def reduce_dimensions(self, X, n_components=100):
        """Reduce feature dimensions using SVD"""
        print(f"Reducing dimensions to {n_components}...")
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_reduced = svd.fit_transform(X)
        return X_reduced, svd
    
    def save_vectorizer(self, path='models/tfidf_vectorizer.pkl'):
        """Save vectorizer to disk"""
        if self.vectorizer:
            joblib.dump(self.vectorizer, path)
            print(f"Vectorizer saved to {path}")
    
    def load_vectorizer(self, path='models/tfidf_vectorizer.pkl'):
        """Load vectorizer from disk"""
        self.vectorizer = joblib.load(path)
        print(f"Vectorizer loaded from {path}")
        return self.vectorizer