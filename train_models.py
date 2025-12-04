import sys
import os
sys.path.append('.')

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
import pandas as pd
import numpy as np

def main():
    print("="*60)
    print("FAKE NEWS DETECTOR - MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\n1. LOADING DATA")
    print("-"*40)
    
    preprocessor = DataPreprocessor()
    
    # Update these paths to your actual dataset locations
    fake_path = "data/Fake.csv"  # Update this path
    true_path = "data/True.csv"  # Update this path
    
    # If you don't have the dataset, use sample data
    if not os.path.exists(fake_path):
        print("Dataset not found. Using sample data...")
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        fake_samples = [' '.join(['fake'] * 20) for _ in range(n_samples//2)]
        true_samples = [' '.join(['true'] * 20) for _ in range(n_samples//2)]
        
        df = pd.DataFrame({
            'text': fake_samples + true_samples,
            'label': [0]*(n_samples//2) + [1]*(n_samples//2)
        })
    else:
        df = preprocessor.load_data(fake_path, true_path)
    
    # Preprocess data
    df = preprocessor.preprocess_dataframe(df)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.prepare_features(df)
    
    # Step 2: Feature engineering
    print("\n2. FEATURE ENGINEERING")
    print("-"*40)
    
    feature_engineer = FeatureEngineer(max_features=3000)
    X_train_features, X_test_features = feature_engineer.extract_features(X_train, X_test)
    
    print(f"Feature matrix shape - Train: {X_train_features.shape}")
    print(f"Feature matrix shape - Test: {X_test_features.shape}")
    
    # Save vectorizer
    feature_engineer.save_vectorizer()
    
    # Step 3: Train models
    print("\n3. MODEL TRAINING")
    print("-"*40)
    
    model_trainer = ModelTrainer()
    models = model_trainer.train_all_models(X_train_features, y_train)
    
    # Save models
    model_trainer.save_models()
    
    # Step 4: Evaluate models
    print("\n4. MODEL EVALUATION")
    print("-"*40)
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(models, X_test_features, y_test)
    
    # Plot ROC curves
    evaluator.plot_roc_curves(models, X_test_features, y_test)
    
    # Save metrics
    evaluator.save_metrics()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nModels saved to: models/")
    print("Assets saved to: assets/")
    
    # Show best model
    best_model_name = results['accuracy'].idxmax()
    print(f"\nBest model by accuracy: {best_model_name}")
    print(f"Best accuracy: {results.loc[best_model_name, 'accuracy']:.4f}")

if __name__ == "__main__":
    main()