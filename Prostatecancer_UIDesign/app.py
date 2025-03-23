import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import (
    perform_imputation, calculate_metrics, plot_roc_curve, plot_regularization_path, 
    plot_coefficient_boxplot, apply_feature_transformation, create_polynomial_features,
    get_feature_importance, plot_feature_importance, plot_confusion_matrix,
    get_advanced_feature_importance
)
from models import train_logistic_regression, train_lasso_regression
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Machine Learning Analysis Tool",
    page_icon="📊",
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
            selected_columns = st.multiselect(
                "Choose columns to impute",
                columns_with_missing,
                default=columns_with_missing
            )

            imputation_method = st.selectbox(
                "Select Imputation Method",
                ["mean", "median", "most_frequent", "knn", "mice", "bayesian_ridge"],
                help="""
                mean: Simple mean imputation
                median: Simple median imputation
                most_frequent: Most frequent value imputation
                knn: K-Nearest Neighbors imputation
                mice: Multivariate Imputation by Chained Equations
                bayesian_ridge: Bayesian Ridge Regression imputation
                """
            )

            if st.button("Perform Imputation"):
                with st.spinner('Performing imputation...'):
                    df_imputed = perform_imputation(df, selected_columns, imputation_method)
                    st.session_state['df_imputed'] = df_imputed
                    st.success("Imputation completed successfully!")

                    # Show comparison
                    st.subheader("Imputation Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Original Data Sample")
                        st.write(df[selected_columns].head())
                    with col2:
                        st.write("Imputed Data Sample")
                        st.write(df_imputed[selected_columns].head())
                        
                    # Display statistics after imputation
                    st.subheader("Statistics After Imputation")
                    st.write(df_imputed[selected_columns].describe())
                    
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
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="data.csv",
                mime="text/csv",
                help="Click to download the full dataset"
            )

        # Model Selection and Training
        if 'df_imputed' in st.session_state and st.session_state['df_imputed'] is not None:
            st.header("2. Model Configuration")

            # Get all available features from the dataset
            available_features = st.session_state['df_imputed'].columns.tolist()
            
            # Try to select numeric columns as default features
            numeric_cols = st.session_state['df_imputed'].select_dtypes(include=['number']).columns.tolist()
            default_features = numeric_cols[:4] if len(numeric_cols) > 0 else []

            features = st.multiselect(
                "Select features for training",
                available_features,
                default=[col for col in default_features if col in available_features]
            )

            # All columns are potential target variables
            target = st.selectbox(
                "Select target variable",
                available_features,
                index=0 if available_features else 0
            )

            model_type = st.radio(
                "Select Model Type",
                ["Logistic Regression", "LASSO Regression"]
            )

            # Feature Engineering Options
            st.subheader("Feature Engineering")
            
            feature_engineering = st.checkbox("Apply Feature Engineering", value=False)
            if feature_engineering:
                # Feature transformation
                col1, col2 = st.columns(2)
                with col1:
                    transform_type = st.selectbox(
                        "Feature Scaling/Transformation",
                        ["None", "standard", "minmax", "robust", "yeo-johnson", "quantile"],
                        help="""
                        standard: StandardScaler (zero mean, unit variance)
                        minmax: MinMaxScaler (scale to range [0, 1])
                        robust: RobustScaler (uses median and IQR)
                        yeo-johnson: PowerTransformer (Yeo-Johnson method)
                        quantile: QuantileTransformer (maps to normal distribution)
                        """
                    )
                
                with col2:
                    create_poly = st.checkbox("Create Polynomial Features", value=False)
                    if create_poly:
                        col_poly1, col_poly2 = st.columns(2)
                        with col_poly1:
                            poly_degree = st.slider("Polynomial Degree", 2, 5, 2)
                        with col_poly2:
                            interaction_only = st.checkbox("Interaction Terms Only", value=False)
            
            # Model parameters
            st.subheader("Model Parameters")
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
                random_state = st.number_input("Random State", 0, 100, 42)
            with col2:
                cv_folds = st.slider("Number of Cross-validation Folds", 2, 10, 5)
                if model_type == "LASSO Regression":
                    alpha = st.slider("LASSO Alpha (λ)", 0.0001, 1.0, 0.01, 0.0001)

            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    # Get basic features from dataframe
                    X_base = st.session_state['df_imputed'][features]
                    y = st.session_state['df_imputed'][target]
                    
                    # Apply feature engineering if checked
                    if feature_engineering:
                        st.subheader("Feature Engineering Results")
                        
                        # Apply transformations if selected
                        if transform_type != "None":
                            X_transformed, transformer = apply_feature_transformation(X_base, transform_type)
                            st.write(f"Applied {transform_type} transformation to features")
                        else:
                            X_transformed = X_base.copy()
                        
                        # Create polynomial features if selected
                        if create_poly:
                            X_poly, poly = create_polynomial_features(
                                X_transformed, 
                                degree=poly_degree, 
                                interaction_only=interaction_only
                            )
                            st.write(f"Created polynomial features (degree={poly_degree}, interaction_only={interaction_only})")
                            st.write(f"Number of features increased from {X_transformed.shape[1]} to {X_poly.shape[1]}")
                            
                            # Preview of generated features
                            if st.checkbox("Show generated polynomial features", value=False):
                                st.dataframe(X_poly.head())
                                
                            # Use the polynomial features
                            X = X_poly
                        else:
                            X = X_transformed
                    else:
                        # Use original features
                        X = X_base

                    # If target is not numeric, convert it to numeric
                    if not np.issubdtype(y.dtype, np.number):
                        # If binary categorical (yes/no, true/false, etc), convert to 0/1
                        if len(y.unique()) == 2:
                            try:
                                # Try to convert to 0/1 based on YES/NO, True/False, etc.
                                y = y.map(lambda x: 1 if str(x).lower() in ['yes', 'true', '1', 't', 'y'] else 0)
                            except:
                                st.error(f"Could not convert categorical target variable '{target}' to numeric. Please select a numeric target.")
                                st.stop()
                        else:
                            st.error(f"Target variable '{target}' is categorical with {len(y.unique())} unique values. For non-binary classification, please convert to numeric values first.")
                            st.stop()

                    # Ensure y is numeric for LASSO
                    if model_type == "LASSO Regression" and not np.issubdtype(y.dtype, np.number):
                        st.error("LASSO regression requires a numeric target variable. Please select a numeric target or convert your target to numeric values.")
                        st.stop()

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, 
                        stratify=y if len(np.unique(y)) > 1 and len(np.unique(y)) < len(y)/2 else None
                    )

                    st.header("3. Model Results")

                    if model_type == "Logistic Regression":
                        model, metrics = train_logistic_regression(
                            X_train, X_test, y_train, y_test, cv_folds
                        )

                        # Display metrics in a clean format
                        st.subheader("Performance Metrics")
                        metrics_df = pd.DataFrame({
                            'Metric': list(metrics.keys()),
                            'Value': list(metrics.values())
                        })
                        st.table(metrics_df)

                        # ROC Curve
                        st.subheader("ROC Curve")
                        fig_roc = plot_roc_curve(model, X_test, y_test)
                        st.plotly_chart(fig_roc, use_container_width=True)

                        # Confusion Matrix
                        st.subheader("Confusion Matrix")
                        y_pred = model.predict(X_test)
                        fig_cm = plot_confusion_matrix(y_test, y_pred)
                        st.plotly_chart(fig_cm, use_container_width=True)

                        # Feature Importance
                        st.subheader("Feature Importance")
                        importance_df = get_feature_importance(model, X_test, y_test)
                        fig_importance = plot_feature_importance(importance_df)
                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # Advanced Feature Importance (SHAP)
                        try:
                            st.subheader("Advanced Feature Importance (SHAP)")
                            shap_values, shap_fig = get_advanced_feature_importance(model, X_train, X_test)
                            st.pyplot(shap_fig)
                        except Exception as e:
                            st.info(f"SHAP visualization could not be generated: {str(e)}")

                        # Coefficient Analysis
                        if hasattr(model, 'coef_'):
                            st.subheader("Coefficient Analysis")
                            coefs = pd.DataFrame({
                                'Feature': X.columns,
                                'Coefficient': model.coef_[0]
                            }).sort_values('Coefficient', ascending=False)
                            
                            fig_coef = px.bar(
                                coefs, 
                                x='Coefficient', 
                                y='Feature', 
                                orientation='h',
                                title='Logistic Regression Coefficients'
                            )
                            st.plotly_chart(fig_coef, use_container_width=True)

                    else:  # LASSO Regression
                        model, metrics = train_lasso_regression(
                            X_train, X_test, y_train, y_test, alpha, cv_folds
                        )

                        # Display metrics
                        st.subheader("Performance Metrics")
                        
                        # Separate traditional metrics from probabilistic ones
                        traditional_metrics = {k: v for k, v in metrics.items() 
                                              if k not in ["Avg Prediction Std", "Avg Interval Width", "Interval Coverage (%)"]}
                        probabilistic_metrics = {k: v for k, v in metrics.items()
                                              if k in ["Avg Prediction Std", "Avg Interval Width", "Interval Coverage (%)"]}
                        
                        # Display traditional metrics
                        metrics_df = pd.DataFrame({
                            'Metric': list(traditional_metrics.keys()),
                            'Value': list(traditional_metrics.values())
                        })
                        st.table(metrics_df)
                        
                        # Display probabilistic metrics separately
                        if probabilistic_metrics:
                            st.subheader("Probabilistic Scores")
                            prob_metrics_df = pd.DataFrame({
                                'Metric': list(probabilistic_metrics.keys()),
                                'Value': list(probabilistic_metrics.values())
                            })
                            st.table(prob_metrics_df)
                            
                            st.info("""
                            **Understanding Probabilistic Scores:**
                            * **Avg Prediction Std**: Average standard deviation of predictions - lower values indicate more certain predictions
                            * **Avg Interval Width**: Average width of the 95% prediction intervals
                            * **Interval Coverage (%)**: Percentage of actual values that fall within the prediction intervals
                            """)

                        # Regularization path
                        st.subheader("LASSO Regularization Path")
                        fig_path = plot_regularization_path(X, y)
                        st.plotly_chart(fig_path, use_container_width=True)

                        # Selected Features (non-zero coefficients)
                        st.subheader("Selected Features (non-zero coefficients)")
                        
                        coefs = pd.DataFrame({
                            'Feature': X.columns,
                            'Coefficient': model.coef_
                        })
                        coefs = coefs.loc[coefs['Coefficient'] != 0].sort_values(
                            by='Coefficient', ascending=False
                        )
                        
                        if not coefs.empty:
                            fig_coef = go.Figure()
                            fig_coef.add_trace(go.Bar(
                                x=coefs['Coefficient'],
                                y=coefs['Feature'],
                                orientation='h'
                            ))
                            fig_coef.update_layout(
                                title='LASSO Selected Features',
                                xaxis_title='Coefficient Value',
                                yaxis_title='Feature',
                                height=max(500, len(coefs) * 30)
                            )
                            st.plotly_chart(fig_coef, use_container_width=True)
                            
                            st.write(f"LASSO selected {len(coefs)} features out of {len(X.columns)} original features.")
                        else:
                            st.info("LASSO set all coefficients to zero. Try reducing the alpha parameter.")

                    # Set model trained flag to True
                    st.session_state['model_trained'] = True

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
else:
    st.info("Please upload a CSV file to begin analysis.")
    
    # Add some information about the tool
    st.markdown("""
    ## About This Tool
    
    This machine learning analysis tool helps you:
    
    - Explore your dataset
    - Handle missing values with various imputation methods
    - Train classification (Logistic Regression) or regression (LASSO) models
    - Visualize model performance
    - Identify important features 
    - Apply feature engineering techniques
    
    Upload a CSV file to get started!
    """)
