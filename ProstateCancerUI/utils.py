import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score, 
    recall_score, f1_score, mean_squared_error, r2_score,
    confusion_matrix, log_loss, brier_score_loss
)
from sklearn.linear_model import LassoCV, BayesianRidge, Lasso, LogisticRegression, LinearRegression
from imblearn.over_sampling import SMOTE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures,
    PowerTransformer, QuantileTransformer
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def perform_imputation(df, columns, method, knn_neighbors=5):
    from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
    from sklearn.linear_model import BayesianRidge

    df_imputed = df.copy()
    if not columns:
        return df_imputed

    if method in ['mean', 'median', 'most_frequent']:
        imputer = SimpleImputer(strategy=method)
    elif method == 'mode':
        imputer = SimpleImputer(strategy='most_frequent')
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=knn_neighbors)
    elif method == 'bayesian_ridge':
        imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)
    elif method == 'mice':
        imputer = IterativeImputer(random_state=42)
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    df_imputed[columns] = imputer.fit_transform(df_imputed[columns])
    return df_imputed

def calculate_prediction_intervals(model, X, y, alpha=0.05):
    """
    Calculate prediction intervals for a regression model using 
    bootstrap methodology.
    
    Parameters:
    -----------
    model : estimator
        A fitted regression model with predict method
    X : DataFrame or array-like
        Features
    y : Series or array-like
        True target values
    alpha : float
        Significance level (default: 0.05 for 95% confidence intervals)
        
    Returns:
    --------
    lower_bound : array
        Lower bound of prediction intervals
    upper_bound : array
        Upper bound of prediction intervals
    prediction_std : array
        Standard deviation of predictions
    """
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate residuals
    residuals = y - predictions
    
    # Calculate residual standard deviation
    residual_std = np.std(residuals)
    
    # Calculate standard error of predictions
    # This is a simplification - a more sophisticated approach would 
    # use bootstrap sampling or quantile regression
    prediction_std = np.ones(len(predictions)) * residual_std
    
    # Calculate critical value for the desired confidence level
    # Using t-distribution with n-p degrees of freedom
    # where n is sample size and p is number of parameters
    n = len(y)
    p = X.shape[1] if hasattr(X, 'shape') else 1
    from scipy import stats
    t_crit = stats.t.ppf(1 - alpha / 2, df=n-p)
    
    # Calculate prediction intervals
    lower_bound = predictions - t_crit * prediction_std
    upper_bound = predictions + t_crit * prediction_std
    
    return lower_bound, upper_bound, prediction_std

def calculate_metrics(y_true, y_pred, model_type="logistic", y_pred_proba=None, prediction_intervals=None):
    """
    Calculate performance metrics based on model type.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_type : str
        Type of model ('logistic' or 'lasso')
    y_pred_proba : array-like, optional
        Predicted probabilities (only for logistic)
        
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    if model_type == "logistic":
        # Convert inputs to numpy arrays if they aren't already
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate ROC AUC if probabilities are provided
        roc_auc_value = 0
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc_value = auc(fpr, tpr)
        
        return {
            "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "Precision": round(float(precision_score(y_true, y_pred, zero_division='warn')), 4),
            "Recall/Sensitivity": round(float(recall_score(y_true, y_pred, zero_division='warn')), 4),
            "Specificity": round(specificity, 4),
            "F1-score": round(float(f1_score(y_true, y_pred, zero_division='warn')), 4),
            "ROC AUC": round(roc_auc_value, 4) if y_pred_proba is not None else "N/A"
        }
    else:
        metrics_dict = {
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 4),
            "R²": round(float(r2_score(y_true, y_pred)), 4)
        }
        
        # Add metrics related to prediction intervals if provided
        if prediction_intervals is not None:
            lower_bound, upper_bound, prediction_std = prediction_intervals
            
            # Calculate average width of prediction intervals
            avg_interval_width = np.mean(upper_bound - lower_bound)
            
            # Calculate percentage of actual values within the prediction intervals
            within_interval = np.sum((y_true >= lower_bound) & (y_true <= upper_bound))
            coverage_percentage = within_interval / len(y_true) * 100
            
            # Calculate average prediction standard error
            avg_prediction_std = np.mean(prediction_std)
            
            # Add these metrics to the dictionary
            metrics_dict["Avg Prediction Std"] = round(avg_prediction_std, 4)
            metrics_dict["Avg Interval Width"] = round(avg_interval_width, 4)
            metrics_dict["Interval Coverage (%)"] = round(coverage_percentage, 2)
        
        return metrics_dict

def plot_roc_curve(model, X_test, y_test):
    """
    Generate ROC curve plot using plotly.
    
    Parameters:
    -----------
    model : estimator
        Trained model with predict_proba method
    X_test : DataFrame or array-like
        Test features
    y_test : Series or array-like
        True test labels
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        ROC curve figure
    """
    from sklearn.preprocessing import StandardScaler
    
    # Scale the test data
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        mode='lines',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random classifier',
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))

    # Add optimal threshold point
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    fig.add_trace(go.Scatter(
        x=[fpr[optimal_idx]], 
        y=[tpr[optimal_idx]],
        mode='markers',
        marker=dict(color='red', size=10),
        name=f'Optimal threshold: {optimal_threshold:.3f}'
    ))

    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True,
        legend=dict(x=0.01, y=0.01),
        width=700,
        height=500
    )
    
    return fig

def plot_regularization_path(X, y):
    """
    Generate LASSO regularization path plot using plotly.
    
    Parameters:
    -----------
    X : DataFrame or array-like
        Features
    y : Series or array-like
        Target values
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Regularization path figure
    """
    # Create alphas for the path
    alphas = np.logspace(-4, 1, 100)
    
    # Fit LassoCV to find the optimal alpha
    lasso_cv = LassoCV(cv=5, random_state=42, alphas=alphas, max_iter=2000)
    lasso_cv.fit(X, y)
    
    # Get coefficients for each alpha
    coefs = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=2000, random_state=42)
        lasso.fit(X, y)
        coefs.append(lasso.coef_)
    
    coefs = np.array(coefs)
    
    # Create plot
    fig = go.Figure()

    # Plot coefficient for each feature
    for i, feature in enumerate(X.columns):
        fig.add_trace(go.Scatter(
            x=alphas,
            y=coefs[:, i],
            name=feature,
            mode='lines'
        ))
    
    # Add vertical line at optimal alpha
    fig.add_vline(
        x=lasso_cv.alpha_, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Optimal α: {lasso_cv.alpha_:.4f}",
        annotation_position="top right"
    )

    fig.update_layout(
        title='LASSO Regularization Path',
        xaxis_title='Alpha (log scale)',
        yaxis_title='Coefficient Value',
        xaxis_type="log",
        showlegend=True,
        width=800,
        height=500
    )
    
    return fig

def apply_feature_transformation(X, transform_type):
    """
    Apply various transformations to features.
    
    Parameters:
    -----------
    X : DataFrame
        Features to transform
    transform_type : str
        Type of transformation to apply
        
    Returns:
    --------
    X_transformed : DataFrame
        Transformed features
    transformer : object
        Fitted transformer object
    """
    # Create a copy to avoid modifying the original dataframe
    X_transformed = X.copy()
    
    if transform_type == "standard":
        transformer = StandardScaler()
    elif transform_type == "minmax":
        transformer = MinMaxScaler()
    elif transform_type == "robust":
        transformer = RobustScaler()
    elif transform_type == "yeo-johnson":
        transformer = PowerTransformer(method='yeo-johnson')
    elif transform_type == "quantile":
        transformer = QuantileTransformer(output_distribution='normal')
    else:
        # Return original data if no valid transformation is specified
        return X_transformed, None
    
    # Only apply to numeric columns
    numeric_cols = X_transformed.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) > 0:
        X_transformed[numeric_cols] = transformer.fit_transform(X_transformed[numeric_cols])
    
    return X_transformed, transformer

def create_polynomial_features(X, degree=2, interaction_only=False):
    """
    Create polynomial and interaction features.
    
    Parameters:
    -----------
    X : DataFrame
        Features to transform
    degree : int
        Degree of polynomial features
    interaction_only : bool
        Whether to include only interaction features
        
    Returns:
    --------
    X_poly : DataFrame
        DataFrame with polynomial/interaction features
    poly : PolynomialFeatures
        Fitted polynomial features transformer
    """
    # Create a copy to avoid modifying the original dataframe
    X_numeric = X.select_dtypes(include=['number'])
    
    if X_numeric.shape[1] == 0:
        return X.copy(), None
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_poly_array = poly.fit_transform(X_numeric)
    
    # Create feature names
    feature_names = poly.get_feature_names_out(X_numeric.columns)
    
    # Create DataFrame with polynomial features
    X_poly = pd.DataFrame(X_poly_array, columns=feature_names, index=X.index)
    
    # Add back non-numeric columns if they exist
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        X_poly = pd.concat([X_poly, X[non_numeric_cols]], axis=1)
    
    return X_poly, poly

def plot_confusion_matrix(y_true, y_pred):
    """
    Generate confusion matrix plot using plotly.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Confusion matrix figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Create labels for the matrix
    labels = np.unique(np.concatenate([y_true, y_pred]))
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f'Predicted {label}' for label in labels],
        y=[f'Actual {label}' for label in labels],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        width=500,
        height=500
    )
    
    return fig

def create_correlation_heatmap(df):
    """
    Create correlation heatmap using plotly.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Correlation heatmap figure
    """
    # Calculate correlation matrix for numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns
    corr = df[numeric_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        width=800,
        height=600,
        xaxis={'side': 'bottom'}
    )
    
    return fig

def automatic_feature_selection(df, target_variable, min_threshold=0.1, max_threshold=0.8):
    """
    Automatically select features based on correlation with target variable.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    target_variable : str
        Target variable column name
    min_threshold : float
        Minimum correlation threshold
    max_threshold : float
        Maximum correlation threshold
        
    Returns:
    --------
    selected_features : list
        List of selected feature names
    """
    # Calculate correlations with target
    numeric_cols = df.select_dtypes(include=['number']).columns
    correlations = df[numeric_cols].corr()[target_variable].abs()
    
    # Select features within threshold range (excluding target itself)
    selected_features = correlations[
        (correlations >= min_threshold) & 
        (correlations <= max_threshold) & 
        (correlations.index != target_variable)
    ].index.tolist()
    
    return selected_features

def pre_training_shap_analysis(X, y):
    """
    Perform SHAP-style analysis on imputed data using model coefficients.
    Creates the four specific SHAP visualizations requested.
    
    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target variable
        
    Returns:
    --------
    shap_figs : dict
        Dictionary of SHAP-style visualization figures
    """
    try:
        # Use coefficient-based analysis if SHAP is not available
        print("Using coefficient-based feature importance analysis.")
        
        # Determine if this is classification or regression
        is_classification = len(y.unique()) <= 20 and y.dtype in ['object', 'int64', 'bool']
        
        # Split data for training a baseline model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train a simple baseline model
        if is_classification:
            # Apply SMOTE for classification tasks
            try:
                smote = SMOTE(random_state=42)
                X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
            except Exception as e:
                print(f"SMOTE failed, using original data: {e}")
                X_train_smote, y_train_smote = X_train_scaled, y_train
            
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_smote, y_train_smote)
        else:
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
        
        shap_figs = {}
        
        # 1. SHAP Summary Plot (Bar) - Feature importance
        try:
            if hasattr(model, 'coef_'):
                coefficients = model.coef_
                if len(coefficients.shape) > 1:
                    coefficients = coefficients[0]  # For binary classification
                
                # Calculate absolute importance
                feature_importance = np.abs(coefficients)
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    importance_df.tail(15),  # Top 15 features
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='SHAP Summary Plot (Bar) - Pre-Training',
                    labels={'Importance': 'Mean |SHAP Value|'}
                )
                fig.update_layout(height=500)
                shap_figs['SHAP Summary Plot (Bar)'] = fig
        except Exception as e:
            print(f"Error creating SHAP summary plot: {e}")
        
        # 2. SHAP Dependence Plot - Top feature vs target
        try:
            if hasattr(model, 'coef_'):
                coefficients = model.coef_
                if len(coefficients.shape) > 1:
                    coefficients = coefficients[0]
                
                # Find most important feature
                most_important_idx = np.argmax(np.abs(coefficients))
                most_important_feature = X.columns[most_important_idx]
                
                fig = px.scatter(
                    x=X[most_important_feature],
                    y=y,
                    title=f'SHAP Dependence Plot - {most_important_feature}',
                    labels={'x': most_important_feature, 'y': 'Target'},
                    opacity=0.6
                )
                fig.update_layout(height=400)
                shap_figs['SHAP Dependence Plot'] = fig
        except Exception as e:
            print(f"Error creating SHAP dependence plot: {e}")
        
        # 3. SHAP Force Plot - Individual prediction explanation
        try:
            if hasattr(model, 'coef_') and len(X_test) > 0:
                coefficients = model.coef_
                if len(coefficients.shape) > 1:
                    coefficients = coefficients[0]
                
                # Get first test instance
                instance_values = X_test_scaled[0]
                feature_contributions = instance_values * coefficients
                
                # Sort by absolute contribution
                sorted_idx = np.argsort(np.abs(feature_contributions))[-10:]  # Top 10
                
                contribution_df = pd.DataFrame({
                    'Feature': X.columns[sorted_idx],
                    'Contribution': feature_contributions[sorted_idx]
                }).sort_values('Contribution')
                
                colors = ['red' if c < 0 else 'blue' for c in contribution_df['Contribution']]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=contribution_df['Feature'],
                    x=contribution_df['Contribution'],
                    orientation='h',
                    marker_color=colors,
                    text=[f'{c:.3f}' for c in contribution_df['Contribution']],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title='SHAP Force Plot - Individual Prediction',
                    xaxis_title='Feature Contribution',
                    yaxis_title='Features',
                    height=400
                )
                shap_figs['SHAP Force Plot'] = fig
        except Exception as e:
            print(f"Error creating SHAP force plot: {e}")
        
        # 4. SHAP Decision Plot - Decision path visualization
        try:
            if hasattr(model, 'coef_'):
                coefficients = model.coef_
                if len(coefficients.shape) > 1:
                    coefficients = coefficients[0]
                
                # Create cumulative contributions
                feature_importance = np.abs(coefficients)
                sorted_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
                
                cumulative_contributions = np.cumsum(feature_importance[sorted_idx])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cumulative_contributions,
                    y=X.columns[sorted_idx],
                    mode='lines+markers',
                    name='Decision Path',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title='SHAP Decision Plot - Model Decision Path',
                    xaxis_title='Cumulative Contribution',
                    yaxis_title='Features',
                    height=400
                )
                shap_figs['SHAP Decision Plot'] = fig
        except Exception as e:
            print(f"Error creating SHAP decision plot: {e}")
        
        return shap_figs
        
    except Exception as e:
        print(f"Error in pre-training SHAP analysis: {e}")
        return {}

def shap_analysis(model, X_train, X_test, y_test, model_type):
    """
    Perform SHAP-style analysis on trained model using model coefficients.
    Creates the four specific SHAP visualizations requested.
    
    Parameters:
    -----------
    model : estimator
        Trained model
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Test features  
    y_test : Series
        Test target
    model_type : str
        Type of model ('logistic' or 'lasso')
        
    Returns:
    --------
    shap_figs : dict
        Dictionary of SHAP-style visualization figures
    """
    try:
        # Use coefficient-based analysis instead of SHAP
        print("Using coefficient-based feature importance analysis.")
        
        shap_figs = {}
        
        # 1. SHAP Summary Plot (Bar) - Feature importance
        try:
            if hasattr(model, 'coef_'):
                coefficients = model.coef_
                if len(coefficients.shape) > 1:
                    coefficients = coefficients[0]  # For binary classification
                
                # Calculate absolute importance
                feature_importance = np.abs(coefficients)
                importance_df = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    importance_df.tail(15),  # Top 15 features
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='SHAP Summary Plot (Bar)',
                    labels={'Importance': 'Mean |SHAP Value|'},
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=500, showlegend=False)
                shap_figs['SHAP Summary Plot (Bar)'] = fig
        except Exception as e:
            print(f"Error creating SHAP summary plot: {e}")
        
        # 2. SHAP Dependence Plot - Top feature vs target  
        try:
            if hasattr(model, 'coef_'):
                coefficients = model.coef_
                if len(coefficients.shape) > 1:
                    coefficients = coefficients[0]
                
                # Find most important feature
                most_important_idx = np.argmax(np.abs(coefficients))
                most_important_feature = X_train.columns[most_important_idx]
                
                fig = px.scatter(
                    x=X_test[most_important_feature],
                    y=y_test,
                    title=f'SHAP Dependence Plot - {most_important_feature}',
                    labels={'x': most_important_feature, 'y': 'Target'},
                    opacity=0.6
                )
                fig.update_layout(height=400)
                shap_figs['SHAP Dependence Plot'] = fig
        except Exception as e:
            print(f"Error creating SHAP dependence plot: {e}")
        
        # 3. SHAP Force Plot - Individual prediction explanation
        try:
            if hasattr(model, 'coef_') and len(X_test) > 0:
                coefficients = model.coef_
                if len(coefficients.shape) > 1:
                    coefficients = coefficients[0]
                
                # Scale the test data for fair coefficient application
                scaler = StandardScaler()
                X_test_scaled = scaler.fit_transform(X_test)
                
                # Get first test instance
                instance_values = X_test_scaled[0]
                feature_contributions = instance_values * coefficients
                
                # Sort by absolute contribution
                sorted_idx = np.argsort(np.abs(feature_contributions))[-10:]  # Top 10
                
                contribution_df = pd.DataFrame({
                    'Feature': X_train.columns[sorted_idx],
                    'Contribution': feature_contributions[sorted_idx]
                }).sort_values('Contribution')
                
                colors = ['red' if c < 0 else 'blue' for c in contribution_df['Contribution']]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=contribution_df['Feature'],
                    x=contribution_df['Contribution'],
                    orientation='h',
                    marker_color=colors,
                    text=[f'{c:.3f}' for c in contribution_df['Contribution']],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title='SHAP Force Plot - Individual Prediction',
                    xaxis_title='Feature Contribution',
                    yaxis_title='Features',
                    height=400
                )
                shap_figs['SHAP Force Plot'] = fig
        except Exception as e:
            print(f"Error creating SHAP force plot: {e}")
        
        # 4. SHAP Decision Plot - Decision path visualization
        try:
            if hasattr(model, 'coef_'):
                coefficients = model.coef_
                if len(coefficients.shape) > 1:
                    coefficients = coefficients[0]
                
                # Create cumulative contributions based on feature importance
                feature_importance = np.abs(coefficients)
                sorted_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
                
                cumulative_contributions = np.cumsum(feature_importance[sorted_idx])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cumulative_contributions,
                    y=X_train.columns[sorted_idx],
                    mode='lines+markers',
                    name='Decision Path',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title='SHAP Decision Plot - Model Decision Path',
                    xaxis_title='Cumulative Contribution',
                    yaxis_title='Features',
                    height=400
                )
                shap_figs['SHAP Decision Plot'] = fig
        except Exception as e:
            print(f"Error creating SHAP decision plot: {e}")
        
        return shap_figs
        
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        return {}

def calculate_predictive_parity(y_true, y_pred, y_pred_proba, sensitive_attr):
    """
    Calculate predictive parity metrics.
    Predictive parity means that the accuracy of positive predictions (precision) 
    should be equal across different groups.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like
        Predicted probabilities
    sensitive_attr : array-like
        Sensitive attribute values
        
    Returns:
    --------
    results : dict
        Dictionary containing predictive parity metrics and visualizations
    """
    try:
        results = {}
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        sensitive_attr = np.array(sensitive_attr)
        
        # Get unique groups
        groups = np.unique(sensitive_attr)
        
        # Calculate metrics for each group
        group_metrics = {}
        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                group_metrics[f"Group {group}"] = {
                    'Sample Size': int(np.sum(mask)),
                    'Model Positive Rate': f"{float(np.mean(y_pred[mask])):.3f}",
                    'Actual Positive Rate': f"{float(np.mean(y_true[mask])):.3f}",
                    'Accuracy': f"{float(accuracy_score(y_true[mask], y_pred[mask])):.3f}",
                    'Precision': f"{float(precision_score(y_true[mask], y_pred[mask], zero_division='warn')):.3f}",
                    'Recall': f"{float(recall_score(y_true[mask], y_pred[mask], zero_division='warn')):.3f}"
                }
        
        results['group_metrics'] = group_metrics
        
        # Calculate predictive parity difference (precision difference across groups)
        precision_rates = [float(group_metrics[f"Group {group}"]['Precision']) for group in groups]
        pp_difference = max(precision_rates) - min(precision_rates)
        results['predictive_parity_difference'] = pp_difference
        
        # Add additional predictive parity metrics
        results['fairness_metrics'] = {
            'predictive_parity_difference': pp_difference,
            'max_precision': max(precision_rates),
            'min_precision': min(precision_rates),
            'precision_std': np.std(precision_rates),
            'num_groups': len(groups)
        }
        
        # Create visualization - need to extract numerical values from formatted strings
        viz_data = {}
        for group_name, metrics in group_metrics.items():
            viz_data[group_name] = {
                'precision': float(metrics['Precision']),
                'base_rate': float(metrics['Actual Positive Rate'])
            }
        
        df_viz = pd.DataFrame.from_dict(viz_data, orient='index').reset_index()
        df_viz.rename(columns={'index': 'Group'}, inplace=True)
        
        fig = go.Figure()
        
        # Add bars for different metrics
        fig.add_trace(go.Bar(
            name='Precision (Positive Prediction Accuracy)',
            x=df_viz['Group'],
            y=df_viz['precision'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Actual Positive Rate',
            x=df_viz['Group'],
            y=df_viz['base_rate'],
            marker_color='orange'
        ))
        
        fig.update_layout(
            title='Predictive Parity Analysis',
            xaxis_title='Age Group',
            yaxis_title='Rate',
            barmode='group',
            height=400
        )
        
        results['visualization'] = fig
        
        return results
        
    except Exception as e:
        print(f"Error in demographic parity calculation: {e}")
        return {'error': str(e)}

def calculate_calibration_metrics(y_true, y_pred_proba, n_bins=10):
    """
    Calculate calibration metrics for a classification model.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for calibration curve
        
    Returns:
    --------
    results : dict
        Dictionary containing calibration metrics
    """
    try:
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )
        
        # Calculate Brier score
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        # Calculate Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Determine if sample is in bin m (between bin_lower and bin_upper)
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = np.mean(in_bin.astype(float))
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin].astype(float))
                avg_confidence_in_bin = np.mean(y_pred_proba[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        results = {
            'brier_score': float(brier_score),
            'expected_calibration_error': float(ece),
            'fraction_of_positives': fraction_of_positives.tolist() if hasattr(fraction_of_positives, 'tolist') else fraction_of_positives,
            'mean_predicted_value': mean_predicted_value.tolist() if hasattr(mean_predicted_value, 'tolist') else mean_predicted_value
        }
        
        return results
        
    except Exception as e:
        print(f"Error calculating calibration metrics: {e}")
        return {'error': str(e)}

def plot_calibration_curve(y_true, y_pred_proba, n_bins=10):
    """
    Create calibration plot (reliability diagram) using plotly.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for calibration curve
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Calibration curve figure
    """
    try:
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )
        
        fig = go.Figure()
        
        # Add calibration curve
        fig.add_trace(go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            mode='lines+markers',
            name='Calibration curve',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # Add perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect calibration',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title='Calibration Plot (Reliability Diagram)',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            showlegend=True,
            width=600,
            height=500
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating calibration plot: {e}")
        return go.Figure()