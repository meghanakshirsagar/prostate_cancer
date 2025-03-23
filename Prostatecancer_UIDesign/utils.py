import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score, 
    recall_score, f1_score, mean_squared_error, r2_score,
    confusion_matrix, log_loss
)
from sklearn.linear_model import LassoCV, BayesianRidge, Lasso
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures,
    PowerTransformer, QuantileTransformer
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def perform_imputation(df, columns, method):
    """
    Perform imputation on selected columns using various methods.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with missing values
    columns : list
        List of column names to impute
    method : str
        Imputation method to use
        
    Returns:
    --------
    df_copy : DataFrame
        Dataframe with imputed values
    """
    df_copy = df.copy()

    if method == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif method == "median":
        imputer = SimpleImputer(strategy="median")
    elif method == "most_frequent":
        imputer = SimpleImputer(strategy="most_frequent")
    elif method == "knn":
        imputer = KNNImputer(n_neighbors=5)
    elif method == "mice":
        imputer = IterativeImputer(random_state=42, max_iter=50)
    elif method == "bayesian_ridge":
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            random_state=42,
            max_iter=300
        )

    # Only perform imputation on numeric columns
    numeric_cols = df_copy[columns].select_dtypes(include=['number']).columns.tolist()
    
    if numeric_cols:
        df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])
    
    # Handle categorical columns separately using most_frequent strategy
    categorical_cols = [col for col in columns if col not in numeric_cols]
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df_copy[categorical_cols] = cat_imputer.fit_transform(df_copy[categorical_cols])
    
    return df_copy

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
    prediction_intervals : tuple, optional
        Tuple containing (lower_bound, upper_bound, prediction_std) for regression models
        
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
    y_pred_proba = model.predict_proba(X_test)[:, 1]
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

def plot_coefficient_boxplot(model, feature_names):
    """
    Generate boxplot of coefficients for each feature.
    
    Parameters:
    -----------
    model : estimator
        Trained model with coef_ attribute
    feature_names : list
        List of feature names
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Coefficient boxplot figure
    """
    coefs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
    })
    
    # Sort by coefficient magnitude
    coefs = coefs.reindex(coefs.Coefficient.abs().sort_values(ascending=False).index)
    
    fig = px.bar(
        coefs, 
        x='Coefficient', 
        y='Feature',
        orientation='h',
        title='Feature Coefficients'
    )
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        width=700,
        height=max(500, len(feature_names) * 25)
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
    poly = PolynomialFeatures(
        degree=degree, 
        interaction_only=interaction_only, 
        include_bias=False
    )
    
    # Transform the data
    X_poly_array = poly.fit_transform(X_numeric)
    
    # Create feature names
    feature_names = poly.get_feature_names_out(X_numeric.columns)
    
    # Convert back to DataFrame
    X_poly = pd.DataFrame(
        X_poly_array,
        columns=feature_names,
        index=X.index
    )
    
    # Add any non-numeric columns from the original DataFrame
    for col in X.columns:
        if col not in X_numeric.columns:
            X_poly[col] = X[col]
    
    return X_poly, poly

def get_feature_importance(model, X, y):
    """
    Get feature importances using sklearn's permutation_importance.
    
    Parameters:
    -----------
    model : estimator
        Trained model
    X : DataFrame
        Features
    y : Series
        Target values
        
    Returns:
    --------
    importance_df : DataFrame
        DataFrame with feature importances
    """
    # Calculate permutation importance
    result = permutation_importance(
        model, X, y, 
        n_repeats=10, 
        random_state=42,
        n_jobs=-1
    )
    
    # Create a DataFrame with the results
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    })
    
    # Sort by importance and return
    return importance_df.sort_values('Importance', ascending=False)

def plot_feature_importance(importance_df):
    """
    Generate feature importance plot using plotly.
    
    Parameters:
    -----------
    importance_df : DataFrame
        DataFrame with feature importance information
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Feature importance figure
    """
    # Sort DataFrame by importance for better visualization
    df = importance_df.sort_values('Importance', ascending=True)
    
    # Create bar chart with error bars
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Importance'],
        y=df['Feature'],
        orientation='h',
        error_x=dict(
            type='data',
            array=df['Std'],
            color='rgba(100, 100, 100, 0.5)'
        ),
        marker=dict(
            color='rgba(58, 71, 80, 0.7)',
            line=dict(color='rgba(58, 71, 80, 1.0)', width=2)
        )
    ))
    
    fig.update_layout(
        title='Feature Importance (Permutation Method)',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        width=700,
        height=max(500, len(df) * 25)  # Adjust height based on number of features
    )
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Generate an enhanced confusion matrix visualization.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        Names for the classes (default: ['0', '1'])
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Confusion matrix figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = ['0', '1'] if cm.shape[0] == 2 else [str(i) for i in range(cm.shape[0])]
    
    # Calculate percentages for annotations
    total = np.sum(cm)
    percentage = cm / total * 100
    
    # Create text annotations for the heatmap cells
    annotations = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations.append({
                'x': j,
                'y': i,
                'text': f"{cm[i, j]}<br>({percentage[i, j]:.1f}%)",
                'font': {'color': 'white' if cm[i, j] > cm.max() / 2 else 'black'},
                'showarrow': False
            })
    
    # Create heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        showscale=True
    ))
    
    # Add text annotations to cells
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        annotations=annotations,
        width=600,
        height=600
    )
    
    return fig

def get_advanced_feature_importance(model, X_train, X_test, sample_size=100):
    """
    Generate advanced feature importance visualization using SHAP.
    
    Parameters:
    -----------
    model : estimator
        Trained model
    X_train : DataFrame
        Training features used to fit model
    X_test : DataFrame
        Test features for explanation
    sample_size : int, optional
        Number of samples to use for SHAP explanation
        
    Returns:
    --------
    shap_values : array
        SHAP values
    fig : matplotlib.figure.Figure
        SHAP summary plot figure
    """
    try:
        import shap
        
        # Limit sample size for faster computation
        if len(X_test) > sample_size:
            X_sample = X_test.sample(sample_size, random_state=42)
        else:
            X_sample = X_test
            
        # Create explainer based on model type
        if hasattr(model, 'predict_proba'):
            # For classification models
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_sample)
            
            # Create a plot
            fig = plt.figure(figsize=(10, 8))
            plt.title("SHAP Feature Importance")
            shap.summary_plot(shap_values, X_sample, show=False)
            
            return shap_values, fig
        else:
            # For regression models
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_sample)
            
            # Create a plot
            fig = plt.figure(figsize=(10, 8))
            plt.title("SHAP Feature Importance")
            shap.summary_plot(shap_values, X_sample, show=False)
            
            return shap_values, fig
    except ImportError:
        # Use permutation importance as a fallback when SHAP is not available
        result = permutation_importance(
            model, X_test, model.predict(X_test), 
            n_repeats=10, 
            random_state=42
        )
        
        # Create figure
        importance_values = result.importances_mean
        
        # Create a matplotlib figure for consistency with the expected return
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort features by importance
        indices = np.argsort(importance_values)[::-1]  # reverse to get descending order
        features = X_test.columns[indices]
        
        # Plot bar chart
        ax.barh(range(len(indices)), importance_values[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(features)
        ax.set_title('Feature Importance (Permutation Method)')
        ax.set_xlabel('Relative Importance')
        
        return importance_values, fig
    except Exception as e:
        raise Exception(f"Error generating feature importance values: {str(e)}")
