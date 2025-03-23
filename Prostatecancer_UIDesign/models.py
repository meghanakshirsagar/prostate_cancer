from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from utils import calculate_metrics
import numpy as np
import pandas as pd

def train_logistic_regression(X_train, X_test, y_train, y_test, cv_folds):
    """
    Train logistic regression model with specific configurations from the notebook.
    
    Parameters:
    -----------
    X_train : DataFrame or array-like
        Features for training the model
    X_test : DataFrame or array-like
        Features for testing the model
    y_train : Series or array-like
        Target values for training
    y_test : Series or array-like
        Target values for testing
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    model : LogisticRegression
        Trained logistic regression model
    metrics : dict
        Performance metrics of the model
    """
    # Handle categorical features with label encoding if any exist
    if isinstance(X_train, pd.DataFrame):
        # Create a copy to avoid modifying the original dataframe
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        # Label encode categorical columns
        encoder = LabelEncoder()
        for column in X_train_processed.select_dtypes(include=['object']).columns:
            encoder.fit(X_train_processed[column])
            X_train_processed[column] = encoder.transform(X_train_processed[column])
            # Handle potential unknown categories in test set
            X_test_processed[column] = X_test_processed[column].map(
                lambda x: 0 if x not in encoder.classes_ else encoder.transform([x])[0]
            )
    else:
        X_train_processed = X_train
        X_test_processed = X_test
    
    # Apply SMOTE for imbalanced data (exactly as in notebook)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
        
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test_processed)

    # Initialize model with parameters matching the notebook
    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    # Train model
    model.fit(X_train_scaled, y_train_smote)

    # Cross-validation with stratified sampling
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train_smote, cv=cv, scoring='roc_auc')

    # Predictions and metrics
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, "logistic", y_pred_proba)
    metrics["Cross-validation ROC-AUC"] = round(cv_scores.mean(), 4)
    metrics["Cross-validation Std"] = round(cv_scores.std(), 4)

    return model, metrics

def train_lasso_regression(X_train, X_test, y_train, y_test, alpha, cv_folds):
    """
    Train LASSO regression model for numeric predictions.
    
    Parameters:
    -----------
    X_train : DataFrame or array-like
        Features for training the model
    X_test : DataFrame or array-like
        Features for testing the model
    y_train : Series or array-like
        Target values for training
    y_test : Series or array-like
        Target values for testing
    alpha : float
        Regularization strength parameter
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    model : Lasso
        Trained LASSO regression model
    metrics : dict
        Performance metrics of the model
    """
    # Handle categorical features with label encoding if any exist
    if isinstance(X_train, pd.DataFrame):
        # Create a copy to avoid modifying the original dataframe
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        # Label encode categorical columns
        encoder = LabelEncoder()
        for column in X_train_processed.select_dtypes(include=['object']).columns:
            encoder.fit(X_train_processed[column])
            X_train_processed[column] = encoder.transform(X_train_processed[column])
            # Handle potential unknown categories in test set
            X_test_processed[column] = X_test_processed[column].map(
                lambda x: 0 if x not in encoder.classes_ else encoder.transform([x])[0]
            )
    else:
        X_train_processed = X_train
        X_test_processed = X_test
        
    # Scale features as in the notebook
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)

    # Initialize model with parameters matching notebook
    model = Lasso(
        alpha=alpha,
        random_state=42,
        max_iter=2000
    )

    # Train model
    model.fit(X_train_scaled, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')

    # Predictions and metrics
    y_pred = model.predict(X_test_scaled)
    
    # Calculate prediction intervals for probabilistic scoring
    from utils import calculate_prediction_intervals
    prediction_intervals = calculate_prediction_intervals(model, X_test_scaled, y_test)
    
    # Calculate metrics including probabilistic scores
    metrics = calculate_metrics(y_test, y_pred, "lasso", prediction_intervals=prediction_intervals)
    metrics["Cross-validation R²"] = round(cv_scores.mean(), 4)
    metrics["Cross-validation Std"] = round(cv_scores.std(), 4)

    return model, metrics

def get_optimal_alpha(X, y, cv_folds=5):
    """
    Find the optimal alpha parameter for LASSO regression using cross-validation.
    
    Parameters:
    -----------
    X : DataFrame or array-like
        Features
    y : Series or array-like
        Target values
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    optimal_alpha : float
        Optimal alpha value for regularization
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use LassoCV to find optimal alpha
    alphas = np.logspace(-4, 0, 100)
    lasso_cv = LassoCV(
        alphas=alphas,
        cv=cv_folds,
        random_state=42,
        max_iter=2000
    )
    
    lasso_cv.fit(X_scaled, y)
    
    return lasso_cv.alpha_
