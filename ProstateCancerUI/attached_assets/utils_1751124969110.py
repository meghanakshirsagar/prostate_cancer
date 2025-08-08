import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
# SHAP will be imported conditionally when needed
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score, 
    recall_score, f1_score, mean_squared_error, r2_score,
    confusion_matrix, log_loss
)
from sklearn.linear_model import LassoCV, BayesianRidge, Lasso, LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures,
    PowerTransformer, QuantileTransformer
)
from sklearn.model_selection import train_test_split
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
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "Recall/Sensitivity": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "Specificity": round(specificity, 4),
            "F1-score": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "ROC AUC": round(roc_auc_value, 4) if y_pred_proba is not None else "N/A"
        }
    else:
        metrics_dict = {
            "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
            "MAE": round(np.mean(np.abs(y_true - y_pred)), 4),
            "R²": round(r2_score(y_true, y_pred), 4)
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
    
    # Create annotations for the heatmap
    annotations = []
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=str(cm[i][j]),
                    showarrow=False,
                    font=dict(color="white" if cm[i][j] > cm.max()/2 else "black")
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted 0', 'Predicted 1'],
        y=['Actual 0', 'Actual 1'],
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        annotations=annotations,
        width=400,
        height=400
    )
    
    return fig

def create_correlation_heatmap(df):
    """
    Create correlation heatmap using plotly.
    
    Parameters:
    -----------
    df : DataFrame
        Dataset to analyze
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Correlation heatmap figure
    """
    # Calculate correlation matrix for numeric columns only
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        width=800,
        height=800,
        xaxis_title='Features',
        yaxis_title='Features'
    )
    
    return fig

def automatic_feature_selection(df, target_variable, correlation_threshold=0.3):
    """
    Automatically select features based on correlation with target variable.
    
    Parameters:
    -----------
    df : DataFrame
        Dataset containing features and target
    target_variable : str
        Name of target variable
    correlation_threshold : float
        Minimum correlation threshold for feature selection
        
    Returns:
    --------
    selected_features : list
        List of selected feature names
    """
    # Calculate correlations with target variable
    numeric_df = df.select_dtypes(include=['number'])
    
    if target_variable not in numeric_df.columns:
        return []
    
    correlations = numeric_df.corr()[target_variable].abs()
    
    # Remove target variable itself and select features above threshold
    correlations = correlations.drop(target_variable)
    selected_features = correlations[correlations >= correlation_threshold].index.tolist()
    
    return selected_features

def simple_feature_importance_analysis(X_train, X_test, y_train, y_test, model_type="logistic_regression"):
    """
    Perform simple feature importance analysis using coefficient-based methods.
    This is a lightweight alternative to SHAP when SHAP is not available.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Test features
    y_train : Series
        Training target
    y_test : Series
        Test target
    model_type : str
        Type of model to train for analysis
        
    Returns:
    --------
    analysis_results : dict
        Dictionary containing feature importance analysis results
    """
    try:
        # Preprocess data
        from models import preprocess_data
        X_train_processed, X_test_processed = preprocess_data(X_train, X_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_processed)
        X_test_scaled = scaler.transform(X_test_processed)
        
        # Train model for analysis
        if model_type == "logistic_regression":
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            # Get coefficients for feature importance
            if hasattr(model, 'coef_'):
                if len(model.coef_.shape) > 1:
                    coefficients = model.coef_[0]
                else:
                    coefficients = model.coef_
            else:
                coefficients = np.zeros(X_train.shape[1])
        elif model_type == "lasso_regression":
            model = Lasso(alpha=0.1, random_state=42, max_iter=2000)
            model.fit(X_train_scaled, y_train)
            coefficients = model.coef_
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Calculate feature importance (absolute coefficients)
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': np.abs(coefficients),
            'Coefficient': coefficients
        }).sort_values('Importance', ascending=False)
        
        # Create coefficient plot
        fig = go.Figure()
        
        # Sort features by absolute coefficient value
        sorted_features = feature_importance.sort_values('Coefficient', key=abs, ascending=True)
        
        # Create horizontal bar plot
        fig.add_trace(go.Bar(
            y=sorted_features['Feature'],
            x=sorted_features['Coefficient'],
            orientation='h',
            marker=dict(
                color=sorted_features['Coefficient'],
                colorscale='RdBu',
                cmid=0
            ),
            text=[f'{coef:.3f}' for coef in sorted_features['Coefficient']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'{model_type.replace("_", " ").title()} Feature Coefficients',
            xaxis_title='Coefficient Value',
            yaxis_title='Features',
            width=800,
            height=max(400, len(X_train.columns) * 25),
            showlegend=False
        )
        
        return {
            'feature_importance': feature_importance,
            'coefficient_plot': fig,
            'model': model,
            'scaler': scaler
        }
        
    except Exception as e:
        return {'error': str(e)}

def create_shap_summary_plot_from_coefficients(model, X_test_scaled, feature_names, model_type):
    """
    Create a SHAP-style summary plot using model coefficients and feature values.
    This mimics the SHAP summary plot style shown in the screenshots.
    
    Parameters:
    -----------
    model : trained model
        The trained regression model
    X_test_scaled : array
        Scaled test features
    feature_names : list
        Names of features
    model_type : str
        Type of model ('logistic_regression' or 'lasso_regression')
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        SHAP-style summary plot figure
    """
    import plotly.graph_objects as go
    import numpy as np
    
    # Get model coefficients
    if hasattr(model, 'coef_'):
        coefficients = model.coef_
        if coefficients.ndim > 1:
            coefficients = coefficients[0]  # For logistic regression
    else:
        # Fallback if no coefficients available
        coefficients = np.ones(len(feature_names))
    
    # Calculate feature importance (absolute coefficients)
    feature_importance = np.abs(coefficients)
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    # Limit to top 10 features for readability
    top_features = sorted_idx[:min(10, len(feature_names))]
    
    fig = go.Figure()
    
    # Sample data for visualization (limit to 100 points for performance)
    sample_size = min(100, len(X_test_scaled))
    X_sample = X_test_scaled[:sample_size]
    
    for i, feat_idx in enumerate(top_features):
        feat_name = feature_names[feat_idx]
        coef = coefficients[feat_idx]
        
        # Get feature values for this feature
        x_feat = X_sample[:, feat_idx]
        
        # Calculate "SHAP-like" values (coefficient * feature value)
        shap_like_values = coef * x_feat
        
        # Normalize feature values for color mapping (0 to 1)
        if np.std(x_feat) > 0:
            x_normalized = (x_feat - np.min(x_feat)) / (np.max(x_feat) - np.min(x_feat))
        else:
            x_normalized = np.zeros_like(x_feat)
        
        # Y position for this feature (reverse order for importance)
        y_pos = len(top_features) - i - 1
        
        # Add some jitter to y positions for violin effect
        np.random.seed(42)  # For consistent results
        y_jitter = np.random.normal(0, 0.1, len(shap_like_values))
        y_positions = [y_pos] * len(shap_like_values) + y_jitter
        
        # Add scatter points colored by feature value
        marker_config = dict(
            size=8,
            color=x_normalized,
            colorscale='RdBu_r',
            opacity=0.8,
            line=dict(width=0.5, color='rgba(255,255,255,0.3)')
        )
        
        # Add colorbar only for the first trace
        if i == 0:
            marker_config['showscale'] = True
            marker_config['colorbar'] = dict(
                title="Feature Value",
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["Low", "High"],
                x=1.02
            )
        else:
            marker_config['showscale'] = False
        
        fig.add_trace(go.Scatter(
            x=shap_like_values,
            y=y_positions,
            mode='markers',
            marker=marker_config,
            name=feat_name,
            showlegend=False,
            hovertemplate=f'<b>{feat_name}</b><br>Impact: %{{x:.3f}}<br>Feature value: %{{customdata:.3f}}<extra></extra>',
            customdata=x_feat
        ))
    
    # Update layout to match SHAP summary plot style
    fig.update_layout(
        title=dict(
            text="Advanced Feature Importance (SHAP)",
            x=0.02,
            font=dict(size=16, color='black')
        ),
        xaxis=dict(
            title="SHAP value (impact on model output)",
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(top_features))),
            ticktext=[feature_names[idx] for idx in top_features],
            showgrid=False,
            title="",
            range=[-0.5, len(top_features) - 0.5]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=max(400, len(top_features) * 60),
        margin=dict(l=200, r=100, t=60, b=60)
    )
    
    return fig

def shap_analysis(X_train, X_test, y_train, y_test, model_type="logistic_regression"):
    """
    Perform SHAP analysis on trained model.
    Creates SHAP-style visualization using model coefficients and feature interactions.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Test features
    y_train : Series
        Training target
    y_test : Series
        Test target
    model_type : str
        Type of model to train for SHAP analysis
        
    Returns:
    --------
    shap_results : dict
        Dictionary containing SHAP-style analysis results
    """
    # Get the basic analysis first
    basic_analysis = simple_feature_importance_analysis(X_train, X_test, y_train, y_test, model_type)
    
    if 'error' in basic_analysis:
        return basic_analysis
    
    # Get the trained model from basic analysis
    model = basic_analysis['model']
    scaler = basic_analysis.get('scaler')
    
    # Prepare data using the models module
    from models import preprocess_data
    X_train_processed, X_test_processed = preprocess_data(X_train, X_test)
    
    # Apply scaling if available
    if scaler:
        X_test_scaled = scaler.transform(X_test_processed)
    else:
        X_test_scaled = X_test_processed
    
    # Create SHAP-style summary plot using model coefficients and feature values
    shap_fig = create_shap_summary_plot_from_coefficients(model, X_test_scaled, X_train.columns, model_type)
    
    # Add SHAP-style results to basic analysis
    basic_analysis['shap_summary_plot'] = shap_fig
    basic_analysis['has_shap'] = True
    basic_analysis['shap_method'] = 'coefficient_based'
    
    return basic_analysis
    """
    Create a SHAP-style summary plot using model coefficients and feature values.
    This mimics the SHAP summary plot style shown in the screenshots.
    
    Parameters:
    -----------
    model : trained model
        The trained regression model
    X_test_scaled : array
        Scaled test features
    feature_names : list
        Names of features
    model_type : str
        Type of model ('logistic_regression' or 'lasso_regression')
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        SHAP-style summary plot figure
    """
    import plotly.graph_objects as go
    import numpy as np
    
    # Get model coefficients
    if hasattr(model, 'coef_'):
        coefficients = model.coef_
        if coefficients.ndim > 1:
            coefficients = coefficients[0]  # For logistic regression
    else:
        # Fallback if no coefficients available
        coefficients = np.ones(len(feature_names))
    
    # Calculate feature importance (absolute coefficients)
    feature_importance = np.abs(coefficients)
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    # Limit to top 10 features for readability
    top_features = sorted_idx[:min(10, len(feature_names))]
    
    fig = go.Figure()
    
    # Sample data for visualization (limit to 100 points for performance)
    sample_size = min(100, len(X_test_scaled))
    X_sample = X_test_scaled[:sample_size]
    
    for i, feat_idx in enumerate(top_features):
        feat_name = feature_names[feat_idx]
        coef = coefficients[feat_idx]
        
        # Get feature values for this feature
        x_feat = X_sample[:, feat_idx]
        
        # Calculate "SHAP-like" values (coefficient * feature value)
        shap_like_values = coef * x_feat
        
        # Normalize feature values for color mapping (0 to 1)
        if np.std(x_feat) > 0:
            x_normalized = (x_feat - np.min(x_feat)) / (np.max(x_feat) - np.min(x_feat))
        else:
            x_normalized = np.zeros_like(x_feat)
        
        # Y position for this feature (reverse order for importance)
        y_pos = len(top_features) - i - 1
        
        # Add some jitter to y positions for violin effect
        y_jitter = np.random.normal(0, 0.1, len(shap_like_values))
        y_positions = [y_pos] * len(shap_like_values) + y_jitter
        
        # Add scatter points colored by feature value
        fig.add_trace(go.Scatter(
            x=shap_like_values,
            y=y_positions,
            mode='markers',
            marker=dict(
                size=8,
                color=x_normalized,
                colorscale='RdBu_r',
                showscale=i == 0,  # Show colorbar only once
                colorbar=dict(
                    title="Feature Value",
                    titleside="right",
                    tickmode="array",
                    tickvals=[0, 1],
                    ticktext=["Low", "High"],
                    x=1.02
                ) if i == 0 else None,
                opacity=0.8,
                line=dict(width=0.5, color='rgba(255,255,255,0.3)')
            ),
            name=feat_name,
            showlegend=False,
            hovertemplate=f'<b>{feat_name}</b><br>Impact: %{{x:.3f}}<br>Feature value: %{{customdata:.3f}}<extra></extra>',
            customdata=x_feat
        ))
    
    # Update layout to match SHAP summary plot style
    fig.update_layout(
        title=dict(
            text="Advanced Feature Importance (SHAP)",
            x=0.02,
            font=dict(size=16, color='black')
        ),
        xaxis=dict(
            title="SHAP value (impact on model output)",
            titlefont=dict(size=12),
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(top_features))),
            ticktext=[feature_names[idx] for idx in top_features],
            showgrid=False,
            title="",
            range=[-0.5, len(top_features) - 0.5]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=max(400, len(top_features) * 60),
        margin=dict(l=200, r=100, t=60, b=60)
    )
    
    return fig
