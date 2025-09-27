# Iris Classification Utilities
# Additional helper functions for the enhanced iris project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import pickle

class IrisAnalyzer:
    """
    Enhanced Iris analysis utility class
    """
    
    def __init__(self, model_path='model.pkl', metadata_path='model_metadata.pkl'):
        """Initialize the analyzer with saved model and metadata"""
        self.model = self.load_model(model_path)
        self.metadata = self.load_metadata(metadata_path)
        
    def load_model(self, path):
        """Load the trained model"""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Model file {path} not found!")
            return None
    
    def load_metadata(self, path):
        """Load model metadata"""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Metadata file {path} not found!")
            return {}
    
    def predict_single(self, sepal_length, sepal_width, petal_length, petal_width):
        """
        Predict single flower classification
        
        Args:
            sepal_length, sepal_width, petal_length, petal_width: float values
            
        Returns:
            dict: prediction results including class and probabilities
        """
        if self.model is None:
            return None
            
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = self.model.predict(features)[0]
        
        result = {
            'prediction': prediction,
            'species': ['setosa', 'versicolor', 'virginica'][prediction],
            'input_features': {
                'sepal_length': sepal_length,
                'sepal_width': sepal_width, 
                'petal_length': petal_length,
                'petal_width': petal_width
            }
        }
        
        # Add probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            result['probabilities'] = {
                'setosa': probabilities[0],
                'versicolor': probabilities[1],
                'virginica': probabilities[2]
            }
        
        return result
    
    def predict_batch(self, data):
        """
        Predict batch of flowers
        
        Args:
            data: DataFrame or array-like with 4 features
            
        Returns:
            DataFrame: predictions with probabilities
        """
        if self.model is None:
            return None
            
        if isinstance(data, pd.DataFrame):
            features = data.values
        else:
            features = np.array(data)
            
        predictions = self.model.predict(features)
        species_names = ['setosa', 'versicolor', 'virginica']
        
        results = pd.DataFrame(features, columns=[
            'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
        ])
        
        results['predicted_class'] = predictions
        results['predicted_species'] = [species_names[p] for p in predictions]
        
        # Add probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
            for i, species in enumerate(species_names):
                results[f'prob_{species}'] = probabilities[:, i]
        
        return results
    
    def generate_report(self):
        """Generate a comprehensive model report"""
        if not self.metadata:
            return "No metadata available"
            
        report = f"""
        🌸 IRIS CLASSIFICATION MODEL REPORT
        =====================================
        
        Model Information:
        - Type: {self.metadata.get('model_name', 'Unknown')}
        - Accuracy: {self.metadata.get('accuracy', 0):.4f} ({self.metadata.get('accuracy', 0)*100:.2f}%)
        - Cross-validation Score: {self.metadata.get('cv_score', 'N/A')}
        
        Dataset Information:
        - Features: {len(self.metadata.get('features', []))}
        - Classes: {len(self.metadata.get('target_names', []))}
        - Target Classes: {', '.join(self.metadata.get('target_names', []))}
        
        Features Used:
        {chr(10).join([f'- {feature}' for feature in self.metadata.get('features', [])])}
        """
        
        if 'all_model_results' in self.metadata:
            report += "\n\nModel Comparison Results:\n"
            for model_name, results in self.metadata['all_model_results'].items():
                report += f"- {model_name}: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n"
        
        return report
    
    def plot_feature_importance(self):
        """Plot feature importance if available"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            print("Feature importance not available for this model")
            return
            
        features = self.metadata.get('features', [f'Feature_{i}' for i in range(len(self.model.feature_importances_))])
        importances = self.model.feature_importances_
        
        plt.figure(figsize=(10, 6))
        plt.bar(features, importances, color='skyblue')
        plt.title(f'Feature Importance - {self.metadata.get("model_name", "Model")}')
        plt.ylabel('Importance')
        plt.xlabel('Features')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def create_sample_data(n_samples=10):
    """
    Create sample data for testing
    
    Args:
        n_samples: number of samples to generate
        
    Returns:
        DataFrame: sample data with realistic iris measurements
    """
    np.random.seed(42)
    
    # Realistic ranges for iris measurements
    sepal_length = np.random.uniform(4.3, 7.9, n_samples)
    sepal_width = np.random.uniform(2.0, 4.4, n_samples)  
    petal_length = np.random.uniform(1.0, 6.9, n_samples)
    petal_width = np.random.uniform(0.1, 2.5, n_samples)
    
    return pd.DataFrame({
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length, 
        'petal_width': petal_width
    })

def export_predictions_to_csv(predictions_df, filename=None):
    """
    Export predictions to CSV file
    
    Args:
        predictions_df: DataFrame with predictions
        filename: output filename (optional)
        
    Returns:
        str: filename of saved file
    """
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'iris_predictions_{timestamp}.csv'
    
    predictions_df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")
    return filename

def load_and_compare_models(model_paths):
    """
    Load and compare multiple models
    
    Args:
        model_paths: list of model file paths
        
    Returns:
        dict: comparison results
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    # Load test data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=42
    )
    
    results = {}
    
    for i, path in enumerate(model_paths):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            
            # Evaluate model
            accuracy = model.score(X_test, y_test)
            predictions = model.predict(X_test)
            
            results[f'Model_{i+1}'] = {
                'path': path,
                'accuracy': accuracy,
                'predictions': predictions
            }
            
        except Exception as e:
            print(f"Error loading model {path}: {e}")
    
    return results

# Example usage functions
def demo_single_prediction():
    """Demonstrate single prediction"""
    analyzer = IrisAnalyzer()
    
    # Example flower measurements
    result = analyzer.predict_single(5.1, 3.5, 1.4, 0.2)
    
    if result:
        print("🌸 Single Prediction Demo:")
        print(f"Predicted Species: {result['species']}")
        if 'probabilities' in result:
            print("Probabilities:")
            for species, prob in result['probabilities'].items():
                print(f"  {species}: {prob:.3f}")

def demo_batch_prediction():
    """Demonstrate batch prediction"""
    analyzer = IrisAnalyzer()
    
    # Create sample data
    sample_data = create_sample_data(5)
    
    # Make predictions
    results = analyzer.predict_batch(sample_data)
    
    if results is not None:
        print("\n🌸 Batch Prediction Demo:")
        print(results)
        
        # Export results
        filename = export_predictions_to_csv(results)
        return filename

if __name__ == "__main__":
    # Run demos
    demo_single_prediction()
    demo_batch_prediction()
    
    # Generate report
    analyzer = IrisAnalyzer()
    print(analyzer.generate_report())