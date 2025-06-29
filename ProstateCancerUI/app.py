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
    shap_analysis, pre_training_shap_analysis, calculate_predictive_parity,
    plot_calibration_curve, calculate_calibration_metrics
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
if 'pre_training_shap_done' not in st.session_state:
    st.session_state['pre_training_shap_done'] = False

# File upload section
st.header("1. Data Upload and Preprocessing")
uploaded_file = st.file_uploader("Upload your dataset (CSV format only)", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head(), hide_index=True)

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
                    st.session_state['pre_training_shap_done'] = False  # Reset SHAP analysis flag
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
            st.session_state['pre_training_shap_done'] = False  # Reset SHAP analysis flag
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
        st.write("Select the correlation range for feature selection:")
        
        col1, col2 = st.columns(2)
        with col1:
            min_threshold = st.slider(
                "Minimum correlation threshold:",
                min_value=0.0,
                max_value=0.8,
                value=0.1,
                step=0.05,
                help="Features with correlation above this minimum will be considered"
            )
        with col2:
            max_threshold = st.slider(
                "Maximum correlation threshold:",
                min_value=0.2,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Features with correlation below this maximum will be considered"
            )
        
        # Ensure min is less than max
        if min_threshold >= max_threshold:
            st.error("Minimum threshold must be less than maximum threshold")
            min_threshold = max_threshold - 0.1
        
        selected_features = automatic_feature_selection(df_imputed, target_variable, min_threshold, max_threshold)
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
            st.dataframe(correlations_df, hide_index=True)
        else:
            st.warning(f"No features found with correlation between {min_threshold} and {max_threshold} with the target variable. Please adjust the threshold range.")

    # Pre-training SHAP Analysis Section
    if (len(st.session_state.get('selected_features', [])) > 0 and 
        st.session_state.get('target_variable')):
        
        st.header("3. Pre-Training Feature Importance Analysis (SHAP)")
        st.write("This analysis uses a simple baseline model to understand feature importance in your imputed data before actual model training.")
        
        # Show SHAP results if already generated
        if st.session_state.get('pre_training_shap_results'):
            st.subheader("SHAP Analysis Results")
            shap_figs = st.session_state['pre_training_shap_results']
            
            # Display SHAP plots
            for title, fig in shap_figs.items():
                st.write(f"**{title}**")
                st.plotly_chart(fig, use_container_width=True)
        
        # Button to generate or regenerate SHAP analysis
        button_text = "Regenerate SHAP Analysis" if st.session_state.get('pre_training_shap_results') else "Generate Pre-Training SHAP Analysis"
        
        if st.button(button_text):
            with st.spinner('Generating SHAP analysis for imputed data...'):
                try:
                    X = df_imputed[st.session_state['selected_features']]
                    y = df_imputed[st.session_state['target_variable']]
                    
                    # Generate pre-training SHAP analysis
                    shap_figs = pre_training_shap_analysis(X, y)
                    
                    if shap_figs:
                        # Store results in session state
                        st.session_state['pre_training_shap_results'] = shap_figs
                        st.success("Pre-training SHAP analysis completed!")
                        st.rerun()  # Refresh to show results
                    else:
                        st.error("Could not generate SHAP analysis. Please check your data.")
                        
                except Exception as e:
                    st.error(f"Error in pre-training SHAP analysis: {str(e)}")

    # Model Training and Evaluation
    if len(st.session_state.get('selected_features', [])) > 0 and st.session_state.get('target_variable'):
        st.header("4. Model Training and Evaluation")
        
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
        st.subheader("Polynomial Features (Optional)")
        create_poly = st.checkbox("Create polynomial features", help="Generate polynomial and interaction features")
        
        if create_poly:
            poly_degree = st.slider("Polynomial degree:", 2, 4, 2)
            interaction_only = st.checkbox("Interaction features only", value=True)
            
            X_poly, poly_transformer = create_polynomial_features(X, poly_degree, interaction_only)
            if poly_transformer is not None:
                st.success(f"Created polynomial features with degree {poly_degree}")
                X = X_poly
        
        # Train-test split
        st.subheader("Train-Test Split Configuration")
        test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state:", value=42, min_value=0)
        
        # Determine if this is a classification or regression problem
        is_classification = len(y.unique()) <= 20 and y.dtype in ['object', 'int64', 'bool']
        
        if is_classification:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            st.info(f"Classification problem detected. Target has {len(y.unique())} unique values.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            st.info("Regression problem detected.")
        
        st.write(f"Training set size: {X_train.shape[0]} samples")
        st.write(f"Test set size: {X_test.shape[0]} samples")
        
        # Model selection and training
        st.subheader("Model Selection and Training")
        
        # Choose model type - Allow both models regardless of problem type
        model_options = ["Logistic Regression", "LASSO Regression"]
        
        # Provide guidance based on problem type
        if is_classification:
            st.info("💡 **Recommendation**: Logistic Regression is typically better for classification problems.")
        else:
            st.info("💡 **Recommendation**: LASSO Regression is typically better for regression problems.")
        
        selected_model = st.selectbox(
            "Select model type:",
            options=model_options,
            help="Choose the machine learning algorithm. You can select either model regardless of problem type."
        )
        
        # Cross-validation folds
        cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)
        
        # Model-specific parameters
        if selected_model == "LASSO Regression":
            auto_alpha = st.checkbox("Auto-select alpha (recommended)", value=True)
            if not auto_alpha:
                alpha = st.slider("Alpha (regularization strength):", 0.001, 10.0, 1.0, 0.001)
            else:
                alpha = None
        
        # Single train button
        if st.button(f"Train {selected_model}"):
            with st.spinner(f'Training {selected_model}...'):
                try:
                    if selected_model == "Logistic Regression":
                        model, metrics = train_logistic_regression(X_train, X_test, y_train, y_test, cv_folds)
                        model_type = "logistic"
                    else:  # LASSO Regression
                        if alpha is None:
                            from models import get_optimal_alpha
                            alpha = get_optimal_alpha(X_train, y_train, cv_folds)
                            st.info(f"Optimal alpha selected: {alpha:.6f}")
                        
                        model, metrics = train_lasso_regression(X_train, X_test, y_train, y_test, alpha, cv_folds)
                        model_type = "lasso"
                    
                    st.session_state['model_trained'] = True
                    st.session_state['trained_model'] = model
                    st.session_state['model_type'] = model_type
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    
                    st.success(f"{selected_model} training completed!")
                    
                    # Display metrics
                    st.subheader("Model Performance Metrics")
                    
                    # Add SMOTE indicator for classification models
                    if selected_model == "Logistic Regression":
                        st.info("ℹ️ SMOTE (Synthetic Minority Oversampling Technique) was automatically applied to balance the dataset during training.")
                    
                    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"Error training {selected_model}: {str(e)}")

        # Results visualization and analysis
        if st.session_state.get('model_trained', False):
            st.header("5. Model Results and Analysis")
            
            model = st.session_state['trained_model']
            model_type = st.session_state['model_type']
            X_train = st.session_state['X_train']
            X_test = st.session_state['X_test']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs([
                "Performance Plots", 
                "Feature Importance (SHAP)", 
                "Predictive Parity Analysis", 
                "Calibration Analysis"
            ])
            
            with tab1:
                st.subheader("Model Performance Visualizations")
                
                if model_type == "logistic":
                    # ROC Curve
                    st.write("**ROC Curve**")
                    try:
                        roc_fig = plot_roc_curve(model, X_test, y_test)
                        st.plotly_chart(roc_fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting ROC curve: {str(e)}")
                    
                    # Confusion Matrix
                    st.write("**Confusion Matrix**")
                    try:
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        X_test_scaled = scaler.fit_transform(X_test)
                        y_pred = model.predict(X_test_scaled)
                        
                        cm_fig = plot_confusion_matrix(y_test, y_pred)
                        st.plotly_chart(cm_fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting confusion matrix: {str(e)}")
                
                else:  # LASSO
                    # Regularization Path
                    st.write("**LASSO Regularization Path**")
                    try:
                        reg_fig = plot_regularization_path(X_train, y_train)
                        st.plotly_chart(reg_fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting regularization path: {str(e)}")
            
            with tab2:
                st.subheader("SHAP Feature Importance Analysis")
                
                try:
                    # Generate SHAP analysis
                    shap_figs = shap_analysis(model, X_train, X_test, y_test, model_type)
                    
                    if shap_figs:
                        for title, fig in shap_figs.items():
                            st.write(f"**{title}**")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Could not generate SHAP analysis")
                        
                except Exception as e:
                    st.error(f"Error in SHAP analysis: {str(e)}")
            
            with tab3:
                st.subheader("Predictive Parity Analysis")
                st.write("**What is Predictive Parity?**")
                st.info("""
                Predictive Parity ensures that the accuracy of positive predictions (precision) is equal across different demographic groups. 
                This fairness metric helps identify if your model is biased in its predictions across different patient groups.
                """)
                
                if model_type == "logistic":
                    # Look for potential sensitive attributes
                    potential_sensitive_attrs = []
                    
                    # Check for age-related columns
                    age_columns = [col for col in df_imputed.columns if 'age' in col.lower()]
                    potential_sensitive_attrs.extend(age_columns)
                    
                    # Check for gender-related columns
                    gender_columns = [col for col in df_imputed.columns if any(gender_term in col.lower() for gender_term in ['gender', 'sex', 'male', 'female'])]
                    potential_sensitive_attrs.extend(gender_columns)
                    
                    # Check for race/ethnicity columns
                    race_columns = [col for col in df_imputed.columns if any(race_term in col.lower() for race_term in ['race', 'ethnic', 'black', 'white', 'hispanic', 'asian'])]
                    potential_sensitive_attrs.extend(race_columns)
                    
                    # Check for income/socioeconomic columns
                    income_columns = [col for col in df_imputed.columns if any(income_term in col.lower() for income_term in ['income', 'salary', 'socioeconomic', 'education'])]
                    potential_sensitive_attrs.extend(income_columns)
                    
                    # Remove duplicates and exclude target variable
                    potential_sensitive_attrs = list(set(potential_sensitive_attrs))
                    if st.session_state.get('target_variable') in potential_sensitive_attrs:
                        potential_sensitive_attrs.remove(st.session_state['target_variable'])
                    
                    if potential_sensitive_attrs:
                        st.write("**Select Sensitive Attribute for Analysis:**")
                        selected_attr = st.selectbox(
                            "Choose the demographic attribute to analyze:",
                            options=potential_sensitive_attrs,
                            help="Select the column containing demographic information for fairness analysis"
                        )
                        
                        if selected_attr:
                            # Get the sensitive attribute data
                            sensitive_data = df_imputed[selected_attr]
                            
                            # Determine if it's categorical or numerical
                            if sensitive_data.dtype in ['object', 'category'] or len(sensitive_data.unique()) <= 10:
                                # Categorical attribute
                                st.write(f"**{selected_attr} Distribution:**")
                                attr_dist = sensitive_data.value_counts().sort_index()
                                st.write(attr_dist)
                                
                                # Use the categorical values directly
                                sensitive_attr = sensitive_data
                                
                            else:
                                # Numerical attribute - create bins
                                st.write(f"**{selected_attr} Distribution:**")
                                st.write(sensitive_data.describe())
                                
                                # Create bins for numerical data
                                if 'age' in selected_attr.lower():
                                    # Age-specific bins
                                    bins = [0, 30, 50, 70, 100]
                                    labels = ['Under 30', '30-50', '50-70', 'Over 70']
                                else:
                                    # Generic bins based on quantiles
                                    bins = [sensitive_data.min()] + list(sensitive_data.quantile([0.25, 0.5, 0.75])) + [sensitive_data.max()]
                                    labels = ['Q1', 'Q2', 'Q3', 'Q4']
                                
                                sensitive_attr = pd.cut(sensitive_data, bins=bins, labels=labels, include_lowest=True)
                                
                                st.write(f"**{selected_attr} Groups:**")
                                group_dist = pd.Series(sensitive_attr).value_counts().sort_index()
                                st.write(group_dist)
                            
                            # Perform predictive parity analysis
                            if st.button("Calculate Predictive Parity"):
                                with st.spinner('Calculating predictive parity metrics...'):
                                    try:
                                        # Get predictions
                                        from sklearn.preprocessing import StandardScaler
                                        scaler = StandardScaler()
                                        X_test_scaled = scaler.fit_transform(X_test)
                                        y_pred = model.predict(X_test_scaled)
                                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                                        
                                        # Get sensitive attribute values for test set
                                        sensitive_test = sensitive_attr.loc[X_test.index]
                                        
                                        # Calculate predictive parity
                                        parity_results = calculate_predictive_parity(
                                            y_test, y_pred, y_pred_proba, sensitive_test
                                        )
                                        
                                        if parity_results and 'error' not in parity_results:
                                            st.success("Predictive parity analysis completed!")
                                            
                                            # Display explanation
                                            st.write("**Analysis Results:**")
                                            st.info("""
                                            **What this analysis shows:**
                                            - **Precision (Blue bars)**: Accuracy of positive predictions for each group
                                            - **Actual Positive Rate (Orange bars)**: True rate of positive outcomes in each group
                                            - **Fairness**: For fair predictions, precision should be similar across all groups
                                            """)
                                            
                                            # Display group-wise metrics
                                            st.write("**Group-wise Performance Metrics:**")
                                            metrics_df = pd.DataFrame(parity_results['group_metrics']).T
                                            st.dataframe(metrics_df, use_container_width=True, hide_index=False)
                                            
                                            # Display fairness metrics
                                            st.write("**Predictive Parity Fairness Metrics:**")
                                            fairness_metrics = {
                                                'Predictive Parity Difference': f"{parity_results['predictive_parity_difference']:.4f}",
                                                'Maximum Precision': f"{parity_results['fairness_metrics']['max_precision']:.4f}",
                                                'Minimum Precision': f"{parity_results['fairness_metrics']['min_precision']:.4f}",
                                                'Precision Standard Deviation': f"{parity_results['fairness_metrics']['precision_std']:.4f}",
                                                'Number of Groups': parity_results['fairness_metrics']['num_groups']
                                            }
                                            fairness_df = pd.DataFrame(
                                                list(fairness_metrics.items()),
                                                columns=['Metric', 'Value']
                                            )
                                            st.dataframe(fairness_df, hide_index=True)
                                            
                                            # Interpret results
                                            pp_diff = parity_results['predictive_parity_difference']
                                            if pp_diff < 0.05:
                                                st.success("✅ **Good Predictive Parity**: Precision differences across groups are minimal (< 0.05)")
                                            elif pp_diff < 0.1:
                                                st.warning("⚠️ **Moderate Predictive Parity Issues**: Some precision differences across groups (0.05-0.1)")
                                            else:
                                                st.error("❌ **Poor Predictive Parity**: Significant precision differences across groups (> 0.1)")
                                            
                                            # Plot visualization
                                            if 'visualization' in parity_results:
                                                st.write("**Predictive Parity Visualization:**")
                                                st.plotly_chart(parity_results['visualization'], use_container_width=True)
                                        
                                        else:
                                            st.error("Could not calculate predictive parity. Please check your data.")
                                            
                                    except Exception as e:
                                        st.error(f"Error in predictive parity analysis: {str(e)}")
                    else:
                        st.info("""
                        **No demographic attributes found for analysis.**
                        
                        To perform predictive parity analysis, your dataset should include columns with demographic information such as:
                        - Age-related columns (containing 'age' in the name)
                        - Gender-related columns (containing 'gender', 'sex', 'male', 'female')
                        - Race/ethnicity columns (containing 'race', 'ethnic', 'black', 'white', 'hispanic', 'asian')
                        - Income/socioeconomic columns (containing 'income', 'salary', 'education')
                        
                        Please add such columns to your dataset and re-upload it.
                        """)
                else:
                    st.info("Predictive parity analysis is currently only available for classification models (Logistic Regression).")
            
            with tab4:
                st.subheader("Calibration Analysis")
                
                if model_type == "logistic":
                    try:
                        # Get predictions
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        X_test_scaled = scaler.fit_transform(X_test)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                        
                        # Calculate calibration metrics
                        calibration_results = calculate_calibration_metrics(y_test, y_pred_proba)
                        
                        # Display calibration metrics
                        st.write("**Calibration Metrics:**")
                        metrics_data = []
                        for key, value in calibration_results.items():
                            # Only display numerical metrics, skip arrays
                            if isinstance(value, (int, float, np.number)) and key not in ['fraction_of_positives', 'mean_predicted_value']:
                                metrics_data.append([key, float(value)])
                        
                        if metrics_data:
                            cal_metrics_df = pd.DataFrame(
                                metrics_data,
                                columns=['Metric', 'Value']
                            )
                            st.dataframe(cal_metrics_df, hide_index=True)
                        
                        # Plot calibration curve
                        st.write("**Calibration Plot (Reliability Diagram):**")
                        cal_fig = plot_calibration_curve(y_test, y_pred_proba)
                        st.plotly_chart(cal_fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error in calibration analysis: {str(e)}")
                else:
                    st.info("Calibration analysis is currently only available for classification models")
            


else:
    st.info("Please upload a dataset to begin analysis.")
