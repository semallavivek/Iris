# Configuration file for Enhanced Iris Classification Project
# This file contains customizable settings for the application

# Model Configuration
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.25,
    'cv_folds': 5,
    
    # Model parameters
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42
    },
    
    'decision_tree': {
        'random_state': 42
    },
    
    'logistic_regression': {
        'random_state': 42,
        'max_iter': 1000
    },
    
    'svm': {
        'probability': True,
        'random_state': 42
    }
}

# Streamlit App Configuration
STREAMLIT_CONFIG = {
    'page_title': '🌸 Enhanced Iris Classifier',
    'page_icon': '🌸',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Feature Configuration
FEATURE_CONFIG = {
    'names': [
        'sepal length (cm)',
        'sepal width (cm)', 
        'petal length (cm)',
        'petal width (cm)'
    ],
    
    'ranges': {
        'sepal_length': {'min': 4.0, 'max': 8.0, 'default': 5.8},
        'sepal_width': {'min': 2.0, 'max': 5.0, 'default': 3.1},
        'petal_length': {'min': 1.0, 'max': 7.0, 'default': 3.8},
        'petal_width': {'min': 0.0, 'max': 3.0, 'default': 1.2}
    }
}

# Species Configuration
SPECIES_CONFIG = {
    'names': ['setosa', 'versicolor', 'virginica'],
    'images': {
        'setosa': 'setosa.jpg',
        'versicolor': 'versicolor.jpg', 
        'virginica': 'virginica.jpg'
    },
    'colors': {
        'setosa': '#FF6B6B',
        'versicolor': '#4ECDC4',
        'virginica': '#45B7D1'
    }
}

# Visualization Configuration  
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'color_palette': 'viridis',
    'style': 'whitegrid',
    
    'plotly': {
        'height': 400,
        'color_scale': 'viridis',
        'template': 'plotly_white'
    }
}

# File Paths
FILE_PATHS = {
    'model': 'model.pkl',
    'metadata': 'model_metadata.pkl',
    'data_export': 'predictions',
    'logs': 'logs'
}

# Performance Thresholds
PERFORMANCE_CONFIG = {
    'min_accuracy': 0.85,
    'min_cv_score': 0.80,
    'warning_threshold': 0.90
}

# Export Configuration
EXPORT_CONFIG = {
    'csv_separator': ',',
    'date_format': '%Y%m%d_%H%M%S',
    'include_probabilities': True,
    'decimal_places': 4
}