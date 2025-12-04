from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        print("Training Logistic Regression...")
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        return model
    
    def train_naive_bayes(self, X_train, y_train):
        """Train Naive Bayes model"""
        print("Training Naive Bayes...")
        model = MultinomialNB(alpha=0.1)
        model.fit(X_train, y_train)
        self.models['naive_bayes'] = model
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        print("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_svm(self, X_train, y_train):
        """Train SVM model"""
        print("Training SVM...")
        model = SVC(kernel='linear', probability=True, random_state=42)
        calibrated_svc = CalibratedClassifierCV(model, cv=3)
        calibrated_svc.fit(X_train, y_train)
        self.models['svm'] = calibrated_svc
        return calibrated_svc
    
    def train_neural_network(self, X_train, y_train):
        """Train Neural Network"""
        print("Training Neural Network...")
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        model.fit(X_train, y_train)
        self.models['neural_network'] = model
        return model
    
    def train_ensemble(self, X_train, y_train):
        """Train Ensemble model"""
        print("Training Ensemble Model...")
        
        # Make sure we have individual models
        if 'logistic_regression' not in self.models:
            self.train_logistic_regression(X_train, y_train)
        if 'naive_bayes' not in self.models:
            self.train_naive_bayes(X_train, y_train)
        if 'random_forest' not in self.models:
            self.train_random_forest(X_train, y_train)
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=[
                ('lr', self.models['logistic_regression']),
                ('nb', self.models['naive_bayes']),
                ('rf', self.models['random_forest'])
            ],
            voting='soft',  # Use probability for averaging
            weights=[1, 1, 1]
        )
        
        ensemble.fit(X_train, y_train)
        self.models['ensemble'] = ensemble
        self.best_model = ensemble
        return ensemble
    
    def train_all_models(self, X_train, y_train):
        """Train all models"""
        print("\n" + "="*50)
        print("TRAINING ALL MODELS")
        print("="*50)
        
        self.train_logistic_regression(X_train, y_train)
        self.train_naive_bayes(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_ensemble(X_train, y_train)
        
        print("\nAll models trained successfully!")
        return self.models
    
    def save_models(self, model_dir='models'):
        """Save all trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        for name, model in self.models.items():
            path = f"{model_dir}/{name}.pkl"
            joblib.dump(model, path)
            print(f"Saved {name} to {path}")
    
    def load_models(self, model_dir='models'):
        """Load trained models"""
        import glob
        
        model_files = glob.glob(f"{model_dir}/*.pkl")
        
        for file in model_files:
            name = file.split('/')[-1].replace('.pkl', '')
            self.models[name] = joblib.load(file)
            print(f"Loaded {name} from {file}")
        
        # Set ensemble as best model if available
        if 'ensemble' in self.models:
            self.best_model = self.models['ensemble']
        
        return self.models