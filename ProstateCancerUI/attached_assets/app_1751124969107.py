import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils import (
    perform_imputation, calculate_metrics, plot_roc_curve, plot_regularization_path, 
    apply_feature_transformation, create_polynomial_features,
    plot_confusion_matrix, create_correlation_heatmap, automatic_feature_selection,
    shap_analysis
)
from models import train_logistic_regression, train_lasso_regression, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Machine Learning Analysis Tool",
    page_icon="⚕️",
    layout="wide"
)

# Custom header with icon
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 1rem;">
    <h1 style="margin: 0;">Machine Learning Analysis Tool</h1>
</div>
""", unsafe_allow_html=True)

# Initialize session state for data persistence
if 'df_imputed' not in st.session_state:
    st.session_state['df_imputed'] = None
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
if 'selected_features' not in st.session_state:
    st.session_state['selected_features'] = []
if 'target_variable' not in st.session_state:
    st.session_state['target_variable'] = None

# File upload section
st.header("1. Data Upload and Preprocessing")
uploaded_file = st.file_uploader("Upload your dataset (CSV format only)", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        # Only display simple data info
        st.write(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        df_encoded, _ = preprocess_data(df, df)
        
        # Missing values analysis
        missing_values = df.isnull().sum()
        st.subheader("Missing Values Analysis")
        
        if missing_values.sum() > 0:
            # Create a DataFrame to display missing values stats
            missing_stats = pd.DataFrame({
                'Column': missing_values.index,
                'Missing Values': missing_values.values,
                'Percentage (%)': (missing_values.values / len(df) * 100).round(2)
            })
            missing_stats = missing_stats[missing_stats['Missing Values'] > 0].sort_values(
                by='Missing Values', ascending=False
            ).reset_index(drop=True)
            
            st.write(missing_stats)
                
            # Column selection for imputation
            columns_with_missing = missing_values[missing_values > 0].index.tolist()
            
            st.subheader("Select Columns for Imputation")
            # Dropdown for categorical imputation
            cat_impute_strategy = st.selectbox(
                "Select categorical imputation method:",
                options=["mean", "median", "mode"],
                index=2 # default to 'mode'
            )

            # Dropdown for numerical imputation
            num_impute_strategy = st.selectbox(
                "Select numerical imputation method:",
                options=["mean", "median", "mode", "knn", "bayesian_ridge", "mice"],
                index=0  # default to 'mean'
            )
            if num_impute_strategy == "knn":
                knn_neighbors = st.slider("KNN neighbors (if KNN selected):", 1, 10, 5)
            else:
                knn_neighbors = 5

            if st.button("Perform Imputation"):
                with st.spinner('Performing imputation...'):
                    # 1. Impute categorical columns
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    cat_cols_with_na = [col for col in categorical_cols if df[col].isnull().any()]

                    if cat_cols_with_na:
                        if cat_impute_strategy == 'mean':
                            cat_imputer = SimpleImputer(strategy='mean')
                        elif cat_impute_strategy == 'median':
                            cat_imputer = SimpleImputer(strategy='median')
                        else:  # 'mode'
                            cat_imputer = SimpleImputer(strategy='most_frequent')
                        df[cat_cols_with_na] = cat_imputer.fit_transform(df[cat_cols_with_na])
                        st.write(f"Imputed categorical columns {cat_cols_with_na} using strategy: {cat_impute_strategy}")
                    else:
                        st.write("No missing values in categorical columns.")

                    # 2. Encode categorical columns automatically
                    if categorical_cols:
                        for col in categorical_cols:
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col].astype(str))
                        st.write(f"Encoded categorical columns: {categorical_cols}")
                    else:
                        st.write("No categorical columns to encode.")

                    # 3. Impute numerical columns
                    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
                    num_cols_with_na = [col for col in numerical_cols if df[col].isnull().any()]

                    if num_cols_with_na:
                        if num_impute_strategy in ['mean', 'median', 'mode']:
                            strategy = num_impute_strategy if num_impute_strategy != 'mode' else 'most_frequent'
                            num_imputer = SimpleImputer(strategy=strategy)
                            df[num_cols_with_na] = num_imputer.fit_transform(df[num_cols_with_na])
                        elif num_impute_strategy == 'knn':
                            num_imputer = KNNImputer(n_neighbors=knn_neighbors)
                            df[num_cols_with_na] = num_imputer.fit_transform(df[num_cols_with_na])
                        elif num_impute_strategy == 'bayesian_ridge':
                            num_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)
                            df[num_cols_with_na] = num_imputer.fit_transform(df[num_cols_with_na])
                        elif num_impute_strategy == 'mice':
                            num_imputer = IterativeImputer(random_state=42)
                            df[num_cols_with_na] = num_imputer.fit_transform(df[num_cols_with_na])
                        else:
                            st.error("Invalid numerical imputation strategy.")
                        st.write(f"Imputed numerical columns {num_cols_with_na} using strategy: {num_impute_strategy}")
                    else:
                        st.write("No missing values in numerical columns.")

                    df_imputed = perform_imputation(df, cat_cols_with_na, cat_impute_strategy)
                    df_imputed = perform_imputation(df, num_cols_with_na, num_impute_strategy)
                    st.session_state['df_imputed'] = df_imputed
                    st.success("Imputation completed successfully!")

                    # Show comparison
                    st.subheader("Imputation Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Original Data Sample")
                        st.write(df.head())
                    with col2:
                        st.write("Imputed Data Sample")
                        st.write(df_imputed.head())
                        
                    # Display statistics after imputation
                    st.subheader("Statistics After Imputation")
                    st.write(df_imputed.describe())
                    
                    # Add option to download imputed data
                    st.subheader("Download Imputed Data")
                    
                    # Generate CSV for download
                    csv = df_imputed.to_csv(index=False)
                    
                    # Create download button
                    st.download_button(
                        label="Download Imputed Data as CSV",
                        data=csv,
                        file_name="imputed_data.csv",
                        mime="text/csv",
                        help="Click to download the full dataset with imputed values"
                    )
        else:
            st.info("No missing values found in the dataset.")
            st.session_state['df_imputed'] = df
            # Add option to download data even if no imputation was needed
            st.subheader("Download Data")
            
            # Generate CSV for download
            csv = df.to_csv(index=False)
            
            # Create download button
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv",
                help="Click to download the dataset"
            )

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted.")

# Continue with the rest of the analysis if data is available
if st.session_state['df_imputed'] is not None:
    df_imputed = st.session_state['df_imputed']
    
    # Correlation Analysis and Feature Selection
    st.header("2. Correlation Analysis and Automatic Feature Selection")
    
    # Select target variable
    st.subheader("Select Target Variable")
    target_variable = st.selectbox(
        "Choose target variable:",
        options=df_imputed.columns.tolist(),
        help="Select the column you want to predict"
    )
    
    if target_variable:
        st.session_state['target_variable'] = target_variable
        
        # Create correlation heatmap
        st.subheader("Correlation Heatmap")
        correlation_fig = create_correlation_heatmap(df_imputed)
        st.plotly_chart(correlation_fig, use_container_width=True)
        
        # Automatic feature selection based on correlation
        st.subheader("Automatic Feature Selection")
        correlation_threshold = st.slider(
            "Correlation threshold for feature selection:",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05,
            help="Features with correlation above this threshold with the target will be selected"
        )
        
        selected_features = automatic_feature_selection(df_imputed, target_variable, correlation_threshold)
        st.session_state['selected_features'] = selected_features
        
        if len(selected_features) > 0:
            st.write(f"**Automatically selected {len(selected_features)} features:**")
            st.write(selected_features)
            
            # Show correlation values with target
            correlations = df_imputed[selected_features + [target_variable]].corr()[target_variable].abs().sort_values(ascending=False)
            correlations_df = pd.DataFrame({
                'Feature': correlations.index[1:],  # Exclude target itself
                'Correlation with Target': correlations.values[1:]
            })
            st.write("**Feature correlations with target:**")
            st.dataframe(correlations_df)
        else:
            st.warning(f"No features found with correlation above {correlation_threshold} with the target variable. Please lower the threshold.")
    

    # Model Training and Evaluation
    if len(st.session_state.get('selected_features', [])) > 0 and st.session_state.get('target_variable'):
        st.header("3. Model Training and Evaluation")
        
        # Prepare training data
        X = df_imputed[st.session_state['selected_features']]
        y = df_imputed[st.session_state['target_variable']]
        
        # Feature transformation options
        st.subheader("Feature Transformation (Optional)")
        transform_type = st.selectbox(
            "Select feature transformation:",
            options=["none", "standard", "minmax", "robust", "yeo-johnson", "quantile"],
            help="Choose a transformation to apply to features before training"
        )
        
        if transform_type != "none":
            X_transformed, transformer = apply_feature_transformation(X, transform_type)
            if transformer is not None:
                st.success(f"Applied {transform_type} transformation to features")
                X = X_transformed
        
        # Polynomial features option
        create_poly = st.checkbox("Create polynomial features", help="Generate polynomial and interaction features")
        if create_poly:
            poly_degree = st.slider("Polynomial degree:", 2, 4, 2)
            interaction_only = st.checkbox("Interaction terms only")
            X_poly, poly_transformer = create_polynomial_features(X, degree=poly_degree, interaction_only=interaction_only)
            if poly_transformer is not None:
                st.success(f"Created polynomial features with degree {poly_degree}")
                X = X_poly
        
        # Train-test split
        test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        st.write(f"Training set: {len(X_train)} samples")
        st.write(f"Test set: {len(X_test)} samples")
        
        # Model selection and training
        st.subheader("Model Selection and Training")
        
        # Model selection
        model_type = st.selectbox(
            "Select regression model:",
            ["Logistic Regression", "LASSO Regression"],
            help="Choose the type of regression model to train"
        )
        
        # Common parameters
        cv_folds = st.slider("Cross-validation folds:", 3, 10, 5, help="Number of folds for cross-validation")
        
        # Model-specific parameters
        alpha = None  # Initialize alpha variable
        if model_type == "LASSO Regression":
            alpha_option = st.radio(
                "Alpha selection:",
                ["Auto (Cross-validation)", "Manual"],
                help="Choose how to set the regularization parameter"
            )
            
            if alpha_option == "Manual":
                alpha = st.slider("Alpha (regularization strength):", 0.001, 1.0, 0.1, 0.001)
            else:
                from models import get_optimal_alpha
                alpha = get_optimal_alpha(X_train, y_train)
                st.write(f"Optimal alpha (CV): {alpha:.4f}")
        
        # Single train button
        if st.button(f"Train {model_type}"):
            with st.spinner(f'Training {model_type}...'):
                if model_type == "Logistic Regression":
                    model, metrics = train_logistic_regression(X_train, X_test, y_train, y_test, cv_folds)
                    model_key = "Logistic Regression"
                    analysis_type = "logistic_regression"
                else:  # LASSO Regression
                    if alpha is None:  # Fallback if alpha not set
                        from models import get_optimal_alpha
                        alpha = get_optimal_alpha(X_train, y_train)
                    model, metrics = train_lasso_regression(X_train, X_test, y_train, y_test, alpha, cv_folds)
                    model_key = "LASSO Regression"
                    analysis_type = "lasso_regression"
                
                # Store trained model in session state
                if 'trained_models' not in st.session_state:
                    st.session_state['trained_models'] = {}
                st.session_state['trained_models'][model_key] = {
                    'model': model,
                    'metrics': metrics,
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test
                }
                
                st.success(f"✅ {model_type} Model Trained Successfully!")
                st.write("**Performance Metrics:**")
                for metric, value in metrics.items():
                    st.write(f"- {metric}: {value}")
                
                # Model-specific visualizations
                if model_type == "Logistic Regression":
                    # ROC Curve
                    if hasattr(model, 'predict_proba'):
                        st.write("**ROC Curve:**")
                        X_test_processed, _ = preprocess_data(X_test, X_test)
                        roc_fig = plot_roc_curve(model, X_test_processed, y_test)
                        st.plotly_chart(roc_fig, use_container_width=True)
                    
                    # Confusion Matrix
                    st.write("**Confusion Matrix:**")
                    X_test_processed, _ = preprocess_data(X_test, X_test)
                    y_pred = model.predict(X_test_processed)
                    cm_fig = plot_confusion_matrix(y_test, y_pred)
                    st.plotly_chart(cm_fig, use_container_width=True)
                
                else:  # LASSO Regression
                    # Regularization Path
                    st.write("**Regularization Path:**")
                    reg_path_fig = plot_regularization_path(X_train, y_train)
                    st.plotly_chart(reg_path_fig, use_container_width=True)
                
                # Advanced SHAP Analysis (for both models)
                st.write("**🔍 Advanced Feature Analysis (SHAP):**")
                with st.spinner('Performing advanced SHAP analysis...'):
                    try:
                        analysis_results = shap_analysis(X_train, X_test, y_train, y_test, analysis_type)
                        
                        if analysis_results and 'error' not in analysis_results:
                            # Display coefficient plot
                            if 'coefficient_plot' in analysis_results:
                                st.write("**Feature Coefficients:**")
                                st.plotly_chart(analysis_results['coefficient_plot'], use_container_width=True)
                            
                            # Display detailed feature importance ranking
                            if 'feature_importance' in analysis_results:
                                col1, col2 = st.columns([1, 1])
                                
                                with col1:
                                    st.write("**Feature Importance Ranking:**")
                                    st.dataframe(analysis_results['feature_importance'], use_container_width=True)
                                
                                with col2:
                                    # Show top 5 most important features
                                    top_features = analysis_results['feature_importance'].head(5)
                                    st.write("**Top 5 Most Important Features:**")
                                    for idx, row in top_features.iterrows():
                                        st.write(f"• **{row['Feature']}**: {row['Importance']:.4f}")
                            
                            # Show SHAP summary plot if available
                            if 'shap_summary_plot' in analysis_results:
                                st.write("**SHAP Summary Plot:**")
                                st.plotly_chart(analysis_results['shap_summary_plot'], use_container_width=True)
                            
                            # Show SHAP status
                            if analysis_results.get('has_shap', False):
                                st.success("✅ Advanced SHAP analysis completed successfully")
                            else:
                                st.info("📊 Using coefficient-based feature importance analysis")
                        else:
                            if 'error' in analysis_results:
                                st.info(f"Feature analysis: Using coefficient-based importance")
                            else:
                                st.info("Advanced feature analysis not available")
                    except Exception as e:
                        st.info(f"Using standard feature analysis methods")
        

        # Model Comparison
        st.header("4. Model Comparison")
        st.info("Train both models above to see a comparison of their performance metrics.")

    else:
        if not st.session_state.get('target_variable'):
            st.info("Please select a target variable to proceed with model training.")
        else:
            st.info("No features were automatically selected. Please adjust the correlation threshold.")

else:
    st.info("Please upload a dataset to begin the analysis.")
