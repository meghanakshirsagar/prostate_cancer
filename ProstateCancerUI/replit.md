# Machine Learning Analysis Tool

## Overview

This is a Streamlit-based web application designed for comprehensive machine learning analysis, particularly focused on medical data analysis (such as prostate cancer datasets). The application provides an end-to-end workflow from data upload and preprocessing to model training, evaluation, and interpretation. It features an intuitive interface with advanced statistical analysis capabilities, multiple imputation methods, and sophisticated visualization tools.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with wide layout configuration
- **User Interface**: Clean, medical-themed interface with custom CSS styling and emoji icons
- **Session Management**: Persistent state management across user interactions using Streamlit's session state
- **Interactive Components**: File uploaders, selectboxes, sliders, multi-column layouts, and dynamic content updates
- **Responsive Design**: Wide layout optimized for data analysis workflows

### Backend Architecture
- **Modular Design**: Three-tier architecture with separated concerns:
  - `app.py`: Main application logic and UI orchestration
  - `models.py`: Machine learning model implementations and training
  - `utils.py`: Data processing utilities and analysis functions
- **Data Pipeline**: Sequential processing from upload to results visualization
- **State Management**: Session-based persistence for trained models and processed data

## Key Components

### Data Processing Engine (`utils.py`)
- **Multiple Imputation Methods**: 
  - Simple strategies (mean, median, mode)
  - Advanced methods (KNN, MICE, Bayesian Ridge)
  - Configurable parameters (e.g., KNN neighbors)
- **Feature Engineering**:
  - Polynomial feature creation
  - Multiple scaling methods (Standard, MinMax, Robust)
  - Power transformations and quantile transformations
- **Statistical Analysis**:
  - Correlation analysis with heatmap visualization
  - Distribution analysis and visualization
  - Automatic feature selection algorithms
- **Model Interpretation**:
  - SHAP (SHapley Additive exPlanations) analysis for feature importance
  - Pre-training and post-training interpretability

### Machine Learning Models (`models.py`)
- **Classification**: 
  - Logistic Regression with cross-validation
  - SMOTE integration for handling imbalanced datasets
  - Stratified K-fold validation
- **Regression**:
  - LASSO Regression with automatic regularization parameter selection
  - Cross-validated feature selection
- **Data Preprocessing**:
  - Categorical encoding (Label Encoder for ordinal, One-Hot for nominal)
  - Robust handling of unseen categories in test sets
  - Automatic data type detection and conversion

### Visualization and Analytics
- **Performance Visualization**:
  - ROC curves with AUC calculation
  - Confusion matrices with color-coded heatmaps
  - Regularization path visualization
  - Calibration curves and reliability diagrams
- **Data Exploration**:
  - Missing value pattern analysis
  - Interactive correlation matrices
  - Distribution plots and statistical summaries
- **Model Interpretation**:
  - Feature importance rankings
  - SHAP value plots and explanations
  - Demographic parity analysis for fairness assessment

## Data Flow

1. **Data Ingestion**: CSV file upload with automatic validation and structure analysis
2. **Exploratory Data Analysis**: Missing value analysis, data type detection, and statistical summaries
3. **Data Preprocessing**: User-guided imputation strategy selection and execution
4. **Feature Engineering**: Optional transformations including scaling, polynomial features, and automatic selection
5. **Model Training**: Algorithm selection, hyperparameter tuning, and cross-validation
6. **Model Evaluation**: Comprehensive metrics calculation and performance visualization
7. **Model Interpretation**: SHAP analysis, feature importance, and fairness assessment
8. **Results Export**: Downloadable reports and model artifacts

## External Dependencies

### Core ML Libraries
- **scikit-learn**: Primary machine learning framework for models, preprocessing, and evaluation
- **imbalanced-learn**: SMOTE implementation for handling class imbalance
- **numpy/pandas**: Data manipulation and numerical computing

### Visualization Stack
- **Streamlit**: Web application framework and UI components
- **Plotly**: Interactive visualizations (ROC curves, correlation heatmaps, 3D plots)
- **Matplotlib/Seaborn**: Statistical plots and publication-ready figures

### Advanced Analytics
- **SHAP**: Model interpretability and feature importance analysis (conditionally imported)
- **scipy**: Statistical functions and distributions

### Data Processing
- **sklearn.experimental**: Access to experimental features like IterativeImputer
- **Feature engineering libraries**: Polynomial features, various scalers, and transformers

## Deployment Strategy

### Local Development
- **Environment**: Python-based with pip package management
- **Dependencies**: Specified in imports across modules
- **Data Persistence**: Session-state based, suitable for single-user sessions

### Production Considerations
- **Scalability**: Designed for single-user sessions, suitable for small to medium datasets
- **Memory Management**: Efficient data handling with copy-on-write operations
- **Error Handling**: Comprehensive exception handling with user-friendly error messages

### Security and Privacy
- **Data Handling**: Local processing without external data transmission
- **File Upload**: Restricted to CSV format with validation
- **Session Isolation**: Each user session maintains independent state

## User Preferences

Preferred communication style: Simple, everyday language.

## Changelog

Changelog:
- June 28, 2025. Initial setup