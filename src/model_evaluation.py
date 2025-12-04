from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Store metrics
        self.metrics[model_name] = metrics
        
        # Print results
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, model_name)
        
        return metrics
    
    def evaluate_all_models(self, models, X_test, y_test):
        """Evaluate all models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        results = {}
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):  # Only evaluate if it can predict probabilities
                results[name] = self.evaluate_model(model, X_test, y_test, name)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(results).T
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        print(comparison_df.round(4))
        
        # Plot comparison
        self.plot_model_comparison(comparison_df)
        
        return comparison_df
    
    def plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Fake', 'Real'], 
                   yticklabels=['Fake', 'Real'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plt.savefig(f'assets/cm_{model_name}.png', bbox_inches='tight', dpi=100)
        plt.close()
    
    def plot_model_comparison(self, df):
        """Plot model comparison"""
        plt.figure(figsize=(12, 6))
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        df[metrics_to_plot].plot(kind='bar', figsize=(12, 6))
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.savefig('assets/model_comparison.png', bbox_inches='tight', dpi=100)
        plt.close()
    
    def plot_roc_curves(self, models, X_test, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.savefig('assets/roc_curves.png', bbox_inches='tight', dpi=100)
        plt.close()
    
    def save_metrics(self, path='models/model_metrics.pkl'):
        """Save metrics to disk"""
        joblib.dump(self.metrics, path)
        print(f"Metrics saved to {path}")