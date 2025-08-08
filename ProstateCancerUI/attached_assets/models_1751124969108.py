from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from utils import calculate_metrics, calculate_prediction_intervals
import numpy as np
import pandas as pd

def encode_categorical_features(df):
    """
    Encode categorical features using One-Hot Encoding.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        encoded_array = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))
        df = df.drop(columns=categorical_cols).reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)
    return df

def preprocess_data(X_train, X_test):
    """
    Encode categorical features and return processed data without imputation.
    """
    if isinstance(X_train, pd.DataFrame):
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        # Encode categorical features
        encoders = {}
        for column in X_train_processed.select_dtypes(include=['object']).columns:
            encoder = LabelEncoder()
            X_train_processed[column] = encoder.fit_transform(X_train_processed[column].astype(str))
            
            # Handle unseen categories in test set by assigning a new category (-1)
            X_test_processed[column] = X_test_processed[column].apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )
            encoders[column] = encoder
        
        return X_train_processed, X_test_processed
    
    return X_train, X_test

def train_logistic_regression(X_train, X_test, y_train, y_test, cv_folds):
    X_train_processed, X_test_processed = preprocess_data(X_train, X_test)
    
    # Apply SMOTE for imbalanced data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
        
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test_processed)

    # Initialize model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train_smote)

    # Cross-validation
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
    X_train_processed, X_test_processed = preprocess_data(X_train, X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)

    # Initialize model
    model = Lasso(alpha=alpha, random_state=42, max_iter=2000)
    model.fit(X_train_scaled, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
    
    # Predictions and metrics
    y_pred = model.predict(X_test_scaled)
    
    # Compute probabilistic scores
    prediction_intervals = calculate_prediction_intervals(model, X_test_scaled, y_test)
    metrics = calculate_metrics(y_test, y_pred, "lasso")
    metrics["Cross-validation R²"] = round(cv_scores.mean(), 4)
    metrics["Cross-validation Std"] = round(cv_scores.std(), 4)
    
    return model, metrics

def get_optimal_alpha(X, y, cv_folds=5):
    X_processed, _ = preprocess_data(X, X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    
    alphas = np.logspace(-4, 0, 100)
    lasso_cv = LassoCV(alphas=alphas, cv=cv_folds, random_state=42, max_iter=2000)
    lasso_cv.fit(X_scaled, y)
    
    return lasso_cv.alpha_
