# Enhanced Iris Classification App with Advanced Features

from sklearn.datasets import load_iris
import pandas as pd
import streamlit as st 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="🌸 Enhanced Iris Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and metadata
@st.cache_data
def load_model_and_metadata():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return model, metadata
    except FileNotFoundError:
        # Fallback to original model if enhanced version not available
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        metadata = {
            'model_name': 'Decision Tree',
            'accuracy': 0.97,
            'features': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
            'target_names': ['setosa', 'versicolor', 'virginica']
        }
        return model, metadata

# Load data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df

def main():
    # Load model and data
    model, metadata = load_model_and_metadata()
    iris_df = load_data()
    
    # Main title
    st.title("🌸 Enhanced Iris Flower Classification")
    st.markdown("### Advanced Machine Learning Dashboard")
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Navigation
    page = st.sidebar.selectbox("Choose a page:", [
        "🏠 Home & Prediction",
        "📊 Data Exploration", 
        "🔍 Model Analysis",
        "📈 Batch Prediction",
        "ℹ️ About"
    ])
    
    if page == "🏠 Home & Prediction":
        home_and_prediction(model, metadata, iris_df)
    elif page == "📊 Data Exploration":
        data_exploration(iris_df)
    elif page == "🔍 Model Analysis":
        model_analysis(model, metadata, iris_df)
    elif page == "📈 Batch Prediction":
        batch_prediction(model, metadata)
    else:
        about_page(metadata)

def home_and_prediction(model, metadata, iris_df):
    """Home page with single prediction functionality"""
    
    # Two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🌺 Make a Prediction")
        
        # Input sliders
        st.markdown("#### Adjust the flower measurements:")
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.1, 0.1)
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.8, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.0, 3.0, 1.2, 0.1)
        
        # Create input array
        input_features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_features)[0]
            prediction = model.predict(input_features)[0]
        else:
            prediction = model.predict(input_features)[0]
            probabilities = None
        
        species_names = ['setosa', 'versicolor', 'virginica']
        predicted_species = species_names[prediction]
        
        # Display prediction
        st.markdown("#### 🎯 Prediction Result:")
        st.success(f"**Predicted Species: {predicted_species.title()}**")
        
        # Show input summary
        st.markdown("#### 📋 Input Summary:")
        input_df = pd.DataFrame({
            'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
        })
        st.dataframe(input_df, use_container_width=True)
    
    with col2:
        # Show flower image
        image_files = {'setosa': 'setosa.jpg', 'versicolor': 'versicolor.jpg', 'virginica': 'virginica.jpg'}
        try:
            st.image(image_files[predicted_species], 
                    caption=f"{predicted_species.title()} Iris", 
                    use_column_width=True)
        except:
            st.info("Flower image not available")
        
        # Show probabilities if available
        if probabilities is not None:
            st.markdown("#### 📊 Prediction Probabilities:")
            prob_df = pd.DataFrame({
                'Species': species_names,
                'Probability': probabilities
            })
            
            # Create probability chart
            fig = px.bar(prob_df, x='Species', y='Probability', 
                        color='Probability', 
                        color_continuous_scale='viridis',
                        title="Classification Probabilities")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show probability table
            prob_df['Probability %'] = (prob_df['Probability'] * 100).round(1)
            st.dataframe(prob_df[['Species', 'Probability %']], use_container_width=True)
    
    # Model info
    st.markdown("---")
    st.markdown("#### 🤖 Model Information:")
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.metric("Model Type", metadata.get('model_name', 'Unknown'))
    with info_col2:
        st.metric("Accuracy", f"{metadata.get('accuracy', 0):.1%}")
    with info_col3:
        st.metric("Features Used", len(metadata.get('features', [])))

def data_exploration(iris_df):
    """Data exploration page"""
    st.subheader("📊 Dataset Exploration")
    
    # Dataset overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Dataset Overview")
        st.dataframe(iris_df.head(10))
        
    with col2:
        st.markdown("#### Statistical Summary")
        st.dataframe(iris_df.describe())
    
    # Visualizations
    st.markdown("#### 📈 Data Visualizations")
    
    viz_option = st.selectbox("Choose visualization:", [
        "Pairplot Matrix",
        "Feature Distributions", 
        "Correlation Heatmap",
        "3D Scatter Plot"
    ])
    
    if viz_option == "Pairplot Matrix":
        fig = px.scatter_matrix(iris_df, 
                               dimensions=iris_df.columns[:-1],
                               color='species',
                               title="Feature Relationships")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_option == "Feature Distributions":
        feature = st.selectbox("Select feature:", iris_df.columns[:-1])
        fig = px.histogram(iris_df, x=feature, color='species', 
                          title=f"{feature} Distribution by Species")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_option == "Correlation Heatmap":
        corr_matrix = iris_df.iloc[:, :-1].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # 3D Scatter Plot
        fig = px.scatter_3d(iris_df, 
                           x='sepal length (cm)', 
                           y='sepal width (cm)', 
                           z='petal length (cm)',
                           color='species',
                           size='petal width (cm)',
                           title="3D Feature Visualization")
        st.plotly_chart(fig, use_container_width=True)

def model_analysis(model, metadata, iris_df):
    """Model analysis page"""
    st.subheader("🔍 Model Performance Analysis")
    
    # Model metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", metadata.get('model_name', 'Unknown'))
    with col2:
        st.metric("Accuracy", f"{metadata.get('accuracy', 0):.1%}")
    with col3:
        if 'cv_score' in metadata:
            st.metric("CV Score", f"{metadata['cv_score']:.3f}")
        else:
            st.metric("Features", len(metadata.get('features', [])))
    with col4:
        st.metric("Classes", len(metadata.get('target_names', [])))
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.markdown("#### 🎯 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': metadata.get('features', []),
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Feature', y='Importance',
                    title="Feature Importance Analysis")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison (if available)
    if 'all_model_results' in metadata:
        st.markdown("#### 🏆 Model Comparison")
        results_df = pd.DataFrame(metadata['all_model_results']).T
        results_df = results_df[['accuracy']].sort_values('accuracy', ascending=False)
        
        fig = px.bar(results_df, y=results_df.index, x='accuracy', 
                    orientation='h', title="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)

def batch_prediction(model, metadata):
    """Batch prediction page"""
    st.subheader("📈 Batch Prediction")
    
    st.markdown("Upload a CSV file or enter multiple measurements for batch prediction.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("#### Uploaded Data:")
        st.dataframe(df.head())
        
        if st.button("Predict All"):
            try:
                predictions = model.predict(df.values)
                species_names = ['setosa', 'versicolor', 'virginica']
                df['Predicted Species'] = [species_names[pred] for pred in predictions]
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(df.iloc[:, :-1].values)
                    for i, species in enumerate(species_names):
                        df[f'{species} Probability'] = probabilities[:, i]
                
                st.markdown("#### Prediction Results:")
                st.dataframe(df)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"iris_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
    
    else:
        # Manual batch input
        st.markdown("#### Manual Batch Input:")
        
        num_samples = st.number_input("Number of samples", min_value=1, max_value=20, value=3)
        
        samples_data = []
        for i in range(num_samples):
            st.markdown(f"**Sample {i+1}:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sepal_len = st.number_input(f"Sepal Length", key=f"sl_{i}", value=5.0)
            with col2:
                sepal_wid = st.number_input(f"Sepal Width", key=f"sw_{i}", value=3.0)
            with col3:
                petal_len = st.number_input(f"Petal Length", key=f"pl_{i}", value=4.0)
            with col4:
                petal_wid = st.number_input(f"Petal Width", key=f"pw_{i}", value=1.0)
            
            samples_data.append([sepal_len, sepal_wid, petal_len, petal_wid])
        
        if st.button("Predict Batch"):
            predictions = model.predict(samples_data)
            species_names = ['setosa', 'versicolor', 'virginica']
            
            results_df = pd.DataFrame(samples_data, 
                                    columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
            results_df['Predicted Species'] = [species_names[pred] for pred in predictions]
            
            st.markdown("#### Batch Prediction Results:")
            st.dataframe(results_df)

def about_page(metadata):
    """About page"""
    st.subheader("ℹ️ About This Application")
    
    st.markdown("""
    ### 🌸 Enhanced Iris Classification Dashboard
    
    This is an advanced machine learning application for classifying Iris flowers based on their physical measurements.
    
    #### 🚀 Features:
    - **Single Prediction**: Classify individual flowers with probability scores
    - **Data Exploration**: Interactive visualizations and statistical analysis
    - **Model Analysis**: Performance metrics and feature importance
    - **Batch Prediction**: Process multiple samples at once
    - **Export Functionality**: Download prediction results
    
    #### 🔬 Model Information:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Model Type:** {metadata.get('model_name', 'Unknown')}  
        **Accuracy:** {metadata.get('accuracy', 0):.1%}  
        **Features:** {len(metadata.get('features', []))}  
        **Classes:** {len(metadata.get('target_names', []))}
        """)
    
    with col2:
        st.success("""
        **Dataset:** Iris Flower Dataset  
        **Samples:** 150  
        **Classes:** Setosa, Versicolor, Virginica  
        **Source:** UCI ML Repository
        """)
    
    st.markdown("""
    #### 🎯 How to Use:
    1. **Home & Prediction**: Use sliders to input flower measurements and get instant predictions
    2. **Data Exploration**: Explore the dataset with interactive charts and statistics
    3. **Model Analysis**: View model performance and feature importance
    4. **Batch Prediction**: Upload CSV files or enter multiple samples for bulk processing
    
    #### 📊 Features Measured:
    - **Sepal Length**: Length of the sepal in centimeters
    - **Sepal Width**: Width of the sepal in centimeters  
    - **Petal Length**: Length of the petal in centimeters
    - **Petal Width**: Width of the petal in centimeters
    
    ---
    *Built with Streamlit, Scikit-learn, and Plotly*
    """)

if __name__ == "__main__":
    main()