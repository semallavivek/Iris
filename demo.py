#!/usr/bin/env python3
"""
Enhanced Iris Classification Project - Demo Script

This script demonstrates the new features added to the Iris classification project.
Run this script to see the enhanced capabilities in action.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'plotly', 'streamlit'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("💡 Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def demo_model_training():
    """Demonstrate the enhanced model training"""
    print("\n🤖 ENHANCED MODEL TRAINING DEMO")
    print("=" * 50)
    
    try:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, cross_val_score
        import pandas as pd
        
        # Load data
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        print(f"📊 Dataset: {len(X)} samples, {len(X.columns)} features, {len(iris.target_names)} classes")
        print(f"🎯 Training set: {len(X_train)}, Test set: {len(X_test)}")
        print("\n🏆 Model Performance:")
        
        best_accuracy = 0
        best_model_name = ""
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            accuracy = model.score(X_test, y_test)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            print(f"   • {name}: {accuracy:.4f} (CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
        
        print(f"\n🥇 Best Model: {best_model_name} with accuracy: {best_accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model training demo: {e}")
        return False

def demo_utilities():
    """Demonstrate the utility functions"""
    print("\n🔧 UTILITY FUNCTIONS DEMO") 
    print("=" * 50)
    
    try:
        # Check if model exists
        if not os.path.exists('model.pkl'):
            print("📝 No trained model found. Please run the Jupyter notebook first.")
            print("   Or run: jupyter notebook model_training.ipynb")
            return False
        
        from iris_utils import IrisAnalyzer, create_sample_data
        
        # Initialize analyzer
        analyzer = IrisAnalyzer()
        
        # Single prediction demo
        print("🌸 Single Prediction:")
        result = analyzer.predict_single(5.1, 3.5, 1.4, 0.2)
        if result:
            print(f"   Input: [5.1, 3.5, 1.4, 0.2]")
            print(f"   Predicted Species: {result['species']}")
            if 'probabilities' in result:
                print("   Probabilities:")
                for species, prob in result['probabilities'].items():
                    print(f"     {species}: {prob:.3f}")
        
        # Batch prediction demo
        print("\n📊 Batch Prediction:")
        sample_data = create_sample_data(3)
        results = analyzer.predict_batch(sample_data)
        if results is not None:
            print("   Sample Data & Predictions:")
            print(results[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'predicted_species']])
        
        # Generate report
        print("\n📋 Model Report:")
        report = analyzer.generate_report()
        print(report)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in utilities demo: {e}")
        return False

def demo_visualizations():
    """Demonstrate visualization capabilities"""
    print("\n📈 VISUALIZATION DEMO")
    print("=" * 50)
    
    try:
        import matplotlib.pyplot as plt
        from sklearn.datasets import load_iris
        import pandas as pd
        import numpy as np
        
        # Load data
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        
        print("📊 Creating visualizations...")
        
        # Simple correlation heatmap
        plt.figure(figsize=(8, 6))
        correlation = df.iloc[:, :-1].corr()
        
        # Create a simple heatmap without seaborn dependency
        plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Feature Correlation Matrix')
        
        # Add labels
        features = correlation.columns
        plt.xticks(range(len(features)), features, rotation=45)
        plt.yticks(range(len(features)), features)
        
        # Add correlation values
        for i in range(len(features)):
            for j in range(len(features)):
                plt.text(j, i, f'{correlation.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='black')
        
        plt.tight_layout()
        plt.savefig('correlation_demo.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Correlation heatmap saved as 'correlation_demo.png'")
        
        # Feature distribution plot
        plt.figure(figsize=(12, 8))
        
        for i, feature in enumerate(features):
            plt.subplot(2, 2, i+1)
            
            for species in iris.target_names:
                species_data = df[df['species'] == species][feature]
                plt.hist(species_data, alpha=0.7, label=species, bins=10)
            
            plt.title(f'{feature} Distribution')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('distributions_demo.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Feature distributions saved as 'distributions_demo.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in visualization demo: {e}")
        return False

def demo_streamlit_info():
    """Show information about Streamlit apps"""
    print("\n🌐 STREAMLIT WEB APPLICATIONS")
    print("=" * 50)
    
    print("🚀 Enhanced Application (Recommended):")
    print("   Command: streamlit run enhanced_streamlit_app.py")
    print("   Features:")
    print("   • 🏠 Interactive prediction with probabilities")
    print("   • 📊 Data exploration with visualizations") 
    print("   • 🔍 Model analysis and comparison")
    print("   • 📈 Batch prediction and CSV export")
    print("   • ℹ️ Comprehensive documentation")
    
    print("\n📱 Original Application:")
    print("   Command: streamlit run streamlit-app.py")  
    print("   Features:")
    print("   • 🌸 Simple flower classification")
    print("   • 🎚️ Interactive sliders for input")
    print("   • 🖼️ Flower species images")
    
    print("\n💡 To start either application:")
    print("   1. Make sure all requirements are installed")
    print("   2. Run the training notebook (optional)")
    print("   3. Execute one of the streamlit commands above")
    print("   4. Open your browser to the displayed URL")

def main():
    """Main demo function"""
    print("🌸 ENHANCED IRIS CLASSIFICATION PROJECT DEMO")
    print("=" * 60)
    print("This demo showcases the new features added to the project!")
    
    # Check requirements
    if not check_requirements():
        return
    
    # Run demos
    success_count = 0
    
    if demo_model_training():
        success_count += 1
    
    if demo_utilities():
        success_count += 1
        
    if demo_visualizations():
        success_count += 1
    
    # Always show Streamlit info
    demo_streamlit_info()
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Demo completed successfully! ({success_count}/3 sections passed)")
    
    if success_count == 3:
        print("🎉 All features are working correctly!")
        print("\n🚀 Next Steps:")
        print("   1. Run: jupyter notebook model_training.ipynb")
        print("   2. Train models and explore the enhanced features")
        print("   3. Launch the web app: streamlit run enhanced_streamlit_app.py")
        print("   4. Explore all the new capabilities!")
    else:
        print("\n⚠️ Some features may not be fully functional.")
        print("   Please check the error messages above and install missing dependencies.")
    
    print("\n📚 For more information, see the updated README.md file!")

if __name__ == "__main__":
    main()