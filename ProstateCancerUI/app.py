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
from pathlib import Path


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Prostate Cancer Risk Stratification",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Global theme and typography
# ---------------------------------------------------------

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
    :root {
        --brand: #1f4e79;
        --brand-dark: #163a5a;
        --ink: #1e2430;
        --muted: #5b6472;
        --surface: #f5f7fa;
        --border: #e3e8ef;
    }
    html, body, [class*="css"], .stApp, .stMarkdown, p, span, div,
    label, button, input, select, textarea,
    h1, h2, h3, h4, h5, h6 {
        font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }
    .stApp { background-color: #ffffff; color: var(--ink); }
    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3, h4 {
        color: var(--ink) !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
    h2 {
        margin-top: 2.2rem !important;
    }
    p, li { color: var(--muted); line-height: 1.65; }
    .info-card {
        background: #ffffff !important;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 10px rgba(30,36,48,0.04);
        min-height: 230px;
        height: 100%;
    }
    .info-card h4 {
        margin: 0 0 0.7rem 0 !important;
        color: var(--brand) !important;
        font-size: 1.05rem;
        font-weight: 700;
    }
    .info-card p, .info-card li { color: #26303c !important; font-size: 0.95rem; margin-bottom: 0.4rem; }
    .info-card ul { padding-left: 1.1rem; margin: 0; }
    .accent-bar { border-left: 4px solid var(--brand); }
    .hero-eyebrow { font-size: 0.8rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #cfe0ee !important; margin-bottom: 0.6rem; }
    .hero-title { margin: 0 0 0.6rem 0 !important; color: #ffffff !important; font-size: 2.1rem; font-weight: 700; }
    .hero-sub { font-size: 1.12rem; color: #eaf1f7 !important; margin: 0; max-width: 780px; line-height: 1.6; }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricLabel"] p { color: var(--muted) !important; font-weight: 500; }
    [data-testid="stMetricValue"] { color: var(--brand) !important; font-weight: 700; }
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.1rem;
        border: 1px solid var(--border);
        transition: all 0.15s ease;
    }
    .stButton > button[kind="primary"] { background: var(--brand); border-color: var(--brand); color: #ffffff !important; }
    .stButton > button[kind="primary"] p, .stButton > button[kind="primary"] div, .stButton > button[kind="primary"] span { color: #ffffff !important; }
    .stButton > button[kind="primary"]:hover { background: var(--brand-dark); border-color: var(--brand-dark); color: #ffffff !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] { font-weight: 500; padding: 0.5rem 1rem; }
    [data-testid="stAlert"] { border-radius: 10px; }
    [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Landing page
# ---------------------------------------------------------

st.markdown(
    """
    <div style="padding: 2.4rem 2.4rem; border-radius: 16px; background: linear-gradient(135deg, #1f4e79 0%, #2e7d9a 100%); box-shadow: 0 8px 24px rgba(31,78,121,0.18); margin-bottom: 1.8rem;">
        <div class="hero-eyebrow">Machine Learning Research Platform</div>
        <h1 class="hero-title">Prostate Cancer Risk Stratification</h1>
        <p class="hero-sub">A web-based machine-learning platform for explainable and fairness-aware assessment of clinically significant prostate cancer (csPCa) using the PI-CAI dataset.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Resource links
DATASET_URL = "https://pi-cai.grand-challenge.org/"
PUBLICATION_URL = "https://doi.org/10.1007/978-3-032-25035-3_5"
GITHUB_URL = "https://github.com/meghanakshirsagar/prostate_cancer"

lc1, lc2, lc3 = st.columns(3)
with lc1:
    st.link_button("PI-CAI dataset (Click to open)", DATASET_URL, use_container_width=True)
with lc2:
    st.link_button("Publication (DOI) (Click to open)", PUBLICATION_URL, use_container_width=True)
with lc3:
    st.link_button("GitHub repository (Click to open)", GITHUB_URL, use_container_width=True)

st.header("About the platform")

ac1, ac2, ac3 = st.columns(3)
with ac1:
    st.markdown(
        """
        <div class="info-card accent-bar">
        <h4>What it does</h4>
        <p>Supports prostate cancer risk stratification using structured and imaging data drawn from the publicly available PI-CAI challenge dataset.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with ac2:
    st.markdown(
        """
        <div class="info-card accent-bar">
        <h4>End-to-end workflow</h4>
        <p>Covers preprocessing, missing-value imputation, feature exploration, model training, evaluation, explainability, fairness, and calibration.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with ac3:
    st.markdown(
        """
        <div class="info-card accent-bar">
        <h4>Research focus</h4>
        <p>Investigates algorithmic bias and responsible AI in predicting the presence or absence of clinically significant prostate cancer (csPCa) from routinely available patient characteristics.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Patients", "1,422")
col2.metric("Clinical centres", "3")
col3.metric("Collection period", "2011–2021")
col4.metric("Prediction task", "csPCa risk")

st.header("Dataset")

dc1, dc2, dc3 = st.columns(3)
with dc1:
    st.markdown(
        """
        <div class="info-card">
        <h4>Source</h4>
        <p>Publicly available data from the <a href="https://pi-cai.grand-challenge.org/" target="_blank" style="color:#1f4e79;font-weight:600;">Prostate Imaging: Cancer AI (PI-CAI) Challenge</a>.</p>
        <p>Anonymised records from 3 centres across 11 sites, collected 2011–2021, patients aged 35–92 years.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with dc2:
    st.markdown(
        """
        <div class="info-card">
        <h4>Cohort composition</h4>
        <ul>
        <li>1,014 patients with benign tissue or indolent cancer</li>
        <li>408 patients with clinically significant cancer</li>
        </ul>
        <p style="margin-top:0.6rem;">csPCa was confirmed using histopathology.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with dc3:
    st.markdown(
        """
        <div class="info-card">
        <h4>Structured variables</h4>
        <ul>
        <li>Patient age</li>
        <li>PSA level and PSA density</li>
        <li>Prostate volume</li>
        <li>Gleason score and ISUP grade</li>
        <li>Histopathology variables</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

st.caption(
    "The prediction task is a binary classification problem: predicting the "
    "presence or absence of clinically significant prostate cancer (csPCa)."
)

st.header("Analytical workflow")

workflow_col1, workflow_col2 = st.columns(2)

with workflow_col1:
    st.markdown(
        """
        #### Data preparation

        The analytical pipeline provides a structured preprocessing workflow for clinical tabular data. It supports systematic evaluation of missingness, multiple imputation strategies, categorical encoding, feature transformation, nonlinear feature engineering, and optional class balancing before model development.

        Available procedures include:

        - missing-data assessment;
        - mean, median, mode, k-nearest neighbour (kNN), Bayesian Ridge, and MICE imputation;
        - categorical-variable encoding;
        - feature scaling and distributional transformation;
        - polynomial and interaction-feature generation; and
        - optional SMOTE-based class balancing.
        """
    )

with workflow_col2:
    st.markdown(
        """
        #### Model development and evaluation

        The platform implements a complete supervised machine-learning workflow encompassing model training, internal validation, explainability, fairness assessment, and calibration analysis. These components support transparent evaluation of predictive performance and responsible use of machine-learning models in healthcare research.

        Implemented analyses include:

        - regularised logistic regression;
        - hold-out evaluation and k-fold cross-validation;
        - ROC-AUC and confusion-matrix analysis;
        - SHAP-based model interpretation;
        - predictive-parity assessment across age groups; and
        - probability-calibration analysis.
        """
    )

# ---------------------------------------------------------
# Associated publications  — Study 1 + Study 2
# ---------------------------------------------------------

st.header("Associated publications")

st.markdown("This platform is the companion tool to two related studies.")

pub_col1, pub_col2 = st.columns(2)

with pub_col1:
    st.markdown(
        """
        ##### Study 1 — Responsible AI & Bias Mitigation

        **Mitigating Algorithmic Bias in Prostate Cancer Risk Stratification
        with Responsible Artificial Intelligence and Machine Learning**

        **Authors:**  
        Meghana Kshirsagar, Mihir Sontakke, Gauri Vaidya, Ahmad Alkhan,
        Aideen Killeen, and Conor Ryan

        **Venue:**  
        17th International Conference on Agents and Artificial Intelligence
        (ICAART 2025), Volume 3, pages 1085–1092

        **DOI:**  
        [10.5220/0013262600003890](https://doi.org/10.5220/0013262600003890)
        """
    )

with pub_col2:
    st.markdown(
        """
        ##### Study 2 — Web-Based Tool with Explainability and Fairness

        **Machine Learning for Prostate Cancer Risk Stratification:
        A Web-Based Tool with Explainability and Fairness**

        **Authors:**  
        Meghana Kshirsagar, Gauri Vaidya, Yuvraj Srivastava, and Conor Ryan

        **Venue:**  
        Proceedings of the International Conference on Agents and Artificial
        Intelligence, Lecture Notes in Artificial Intelligence, Volume 16518,
        pages 91–102

        **Publisher:** Springer Nature Switzerland

        **DOI:**  
        [10.1007/978-3-032-25035-3_5](https://doi.org/10.1007/978-3-032-25035-3_5)
        """
    )

st.success(
    "In the accompanying studies, the strongest tabular configuration achieved an "
    "AUC of 0.85 using logistic regression with MICE imputation and no oversampling "
    "(Study 2). The image triage approach using ResNet50 achieved 60% test accuracy "
    "for csPCa detection (Study 1)."
)

st.header("Citation")

st.markdown(
    "Please cite both publications when referring to the methodology, platform, or associated results."
)

bibtex_study1 = """@inproceedings{kshirsagar2025mitigating,
  title     = {Mitigating Algorithmic Bias in Prostate Cancer Risk Stratification
               with Responsible Artificial Intelligence and Machine Learning},
  author    = {Kshirsagar, Meghana and Sontakke, Mihir and Vaidya, Gauri and
               Alkhan, Ahmad and Killeen, Aideen and Ryan, Conor},
  booktitle = {Proceedings of the 17th International Conference on
               Agents and Artificial Intelligence (ICAART 2025)},
  volume    = {3},
  pages     = {1085--1092},
  year      = {2025},
  doi       = {10.5220/0013262600003890}
}"""

bibtex_study2 = """@inproceedings{kshirsagar2027prostate,
  title     = {Machine Learning for Prostate Cancer Risk Stratification:
               A Web-Based Tool with Explainability and Fairness},
  author    = {Kshirsagar, Meghana and Vaidya, Gauri and
               Srivastava, Yuvraj and Ryan, Conor},
  booktitle = {Proceedings of the International Conference on
               Agents and Artificial Intelligence},
  series    = {Lecture Notes in Artificial Intelligence},
  volume    = {16518},
  pages     = {91--102},
  publisher = {Springer Nature Switzerland},
  year      = {2027},
  doi       = {10.1007/978-3-032-25035-3_5}
}"""

st.markdown("**Study 1 — BibTeX**")
st.code(bibtex_study1, language="bibtex")
st.download_button(
    label="Download Study 1 BibTeX",
    data=bibtex_study1,
    file_name="kshirsagar2025_mitigating_bias.bib",
    mime="application/x-bibtex"
)

st.markdown("**Study 2 — BibTeX**")
st.code(bibtex_study2, language="bibtex")
st.download_button(
    label="Download Study 2 BibTeX",
    data=bibtex_study2,
    file_name="kshirsagar2027_prostate_risk.bib",
    mime="application/x-bibtex"
)

if "show_ml_platform" not in st.session_state:
    st.session_state["show_ml_platform"] = False

st.divider()

left, centre, right = st.columns([1, 1.4, 1])

with centre:
    if st.button(
        "Open the machine-learning analysis",
        type="primary",
        use_container_width=True
    ):
        st.session_state["show_ml_platform"] = True

if not st.session_state["show_ml_platform"]:
    st.stop()


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

# ---------------------------------------------------------
# Load the study dataset automatically
# ---------------------------------------------------------

st.header("1. Dataset and Preprocessing")

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

DATA_PATH = REPO_ROOT / "ProstateCancerUI" / "pi_cai_dataset.csv"

try:
    df = pd.read_csv(DATA_PATH)

    st.success("The PI-CAI study dataset was loaded successfully.")

    st.subheader("Dataset overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Records", f"{df.shape[0]:,}")
    col2.metric("Variables", f"{df.shape[1]:,}")
    col3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")

    with st.expander("View dataset preview"):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    df_encoded, _ = preprocess_data(df, df)

    # Missing values analysis
    missing_values = df.isnull().sum()
    st.subheader("Missing Values Analysis")

    if missing_values.sum() > 0:
        missing_stats = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Values': missing_values.values,
            'Percentage (%)': (missing_values.values / len(df) * 100).round(2)
        })
        missing_stats = missing_stats[missing_stats['Missing Values'] > 0].sort_values(
            by='Missing Values', ascending=False
        ).reset_index(drop=True)

        st.write(missing_stats)

        columns_with_missing = missing_values[missing_values > 0].index.tolist()

        st.subheader("Select Columns for Imputation")

        # --- Categorical imputation ---
        st.markdown(
            "**Categorical imputation** — missing categorical values are completed using a selected statistical strategy before numerical encoding, thereby maintaining a consistent representation of categorical variables throughout the preprocessing pipeline."
        )
        cat_impute_strategy = st.selectbox(
            "Select categorical imputation method:",
            options=["mean", "median", "mode"],
            index=2
        )

        # --- Numerical imputation ---
        st.markdown(
            "**Numerical imputation** — continuous variables may be imputed using summary-statistic or model-based approaches. The available methods include mean, median, mode, k-nearest neighbours, Bayesian Ridge regression, and Multiple Imputation by Chained Equations (MICE), enabling comparison of increasingly flexible strategies for handling incomplete data."
        )
        num_impute_strategy = st.selectbox(
            "Select numerical imputation method:",
            options=["mean", "median", "mode", "knn", "bayesian_ridge", "mice"],
            index=0
        )
        if num_impute_strategy == "knn":
            st.markdown(
                "**Number of neighbours** — specifies the number of nearest observations used to estimate each missing value. Larger values generally produce smoother estimates, whereas smaller values retain greater local sensitivity."
            )
            knn_neighbors = st.slider("KNN neighbors (if KNN selected):", 1, 10, 5)
        else:
            knn_neighbors = 5

        if st.button("Perform Imputation"):
            with st.spinner('Performing imputation...'):
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                cat_cols_with_na = [col for col in categorical_cols if df[col].isnull().any()]

                if cat_cols_with_na:
                    if cat_impute_strategy == 'mean':
                        cat_imputer = SimpleImputer(strategy='mean')
                    elif cat_impute_strategy == 'median':
                        cat_imputer = SimpleImputer(strategy='median')
                    else:
                        cat_imputer = SimpleImputer(strategy='most_frequent')
                    df[cat_cols_with_na] = cat_imputer.fit_transform(df[cat_cols_with_na])
                    st.write(f"Imputed categorical columns {cat_cols_with_na} using strategy: {cat_impute_strategy}")
                else:
                    st.write("No missing values in categorical columns.")

                if categorical_cols:
                    for col in categorical_cols:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                    st.write(f"Encoded categorical columns: {categorical_cols}")
                else:
                    st.write("No categorical columns to encode.")

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
                st.session_state['pre_training_shap_done'] = False
                st.success("Imputation completed successfully!")

                st.subheader("Imputation Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Original Data Sample")
                    st.write(df.head())
                with col2:
                    st.write("Imputed Data Sample")
                    st.write(df_imputed.head())

                st.subheader("Statistics After Imputation")
                st.write(df_imputed.describe())

                st.subheader("Download Imputed Data")
                csv = df_imputed.to_csv(index=False)
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
        st.session_state['pre_training_shap_done'] = False
        st.subheader("Download Data")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv",
            help="Click to download the dataset"
        )

except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.info("Please ensure the study dataset exists at the expected path and is properly formatted.")

# Continue with the rest of the analysis if data is available
if st.session_state['df_imputed'] is not None:
    df_imputed = st.session_state['df_imputed']

    # ---------------------------------------------------------
    # Correlation Analysis and Feature Selection
    # Target variable is fixed to case_csPCa
    # ---------------------------------------------------------
    st.header("2. Correlation Analysis and Automatic Feature Selection")

    # Fixed target variable — no user selection needed
    DEFAULT_TARGET = "case_csPCa"
    if DEFAULT_TARGET in df_imputed.columns:
        target_variable = DEFAULT_TARGET
    else:
        # Graceful fallback: pick first binary-looking column
        target_variable = df_imputed.columns[0]
        st.warning(
            f"Column '{DEFAULT_TARGET}' was not found. Falling back to '{target_variable}'. "
            "Please ensure the dataset includes a `case_csPCa` column."
        )

    st.session_state['target_variable'] = target_variable
    st.info(f"Target variable fixed to: **{target_variable}** (binary csPCa label).")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    st.markdown(
        "The correlation matrix summarises pairwise Pearson correlation coefficients across the study variables. This analysis supports identification of multicollinearity, strongly associated predictors, and variables with potential linear relationships to the target outcome."
    )
    correlation_fig = create_correlation_heatmap(df_imputed)
    st.plotly_chart(correlation_fig, use_container_width=True)

    # Automatic feature selection
    st.subheader("Automatic Feature Selection")
    st.markdown(
        "Automatic feature selection is performed using the absolute Pearson correlation coefficient with respect to the target variable. User-defined lower and upper thresholds allow weakly associated predictors to be excluded while reducing the risk of retaining variables that may introduce information leakage."
    )
    st.write("Select the correlation range for feature selection:")

    col1, col2 = st.columns(2)
    with col1:
        min_threshold = st.slider(
            "Minimum correlation threshold:",
            min_value=0.0,
            max_value=0.8,
            value=0.1,
            step=0.05,
            help="Features with |correlation| above this minimum will be considered"
        )
    with col2:
        max_threshold = st.slider(
            "Maximum correlation threshold:",
            min_value=0.2,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Features with |correlation| below this maximum will be considered (avoids data leakage)"
        )

    if min_threshold >= max_threshold:
        st.error("Minimum threshold must be less than maximum threshold")
        min_threshold = max_threshold - 0.1

    selected_features = automatic_feature_selection(df_imputed, target_variable, min_threshold, max_threshold)
    st.session_state['selected_features'] = selected_features

    if len(selected_features) > 0:
        st.write(f"**Automatically selected {len(selected_features)} features:**")
        st.write(selected_features)

        correlations = df_imputed[selected_features + [target_variable]].corr()[target_variable].abs().sort_values(ascending=False)
        correlations_df = pd.DataFrame({
            'Feature': correlations.index[1:],
            'Correlation with Target': correlations.values[1:]
        })
        st.write("**Feature correlations with target:**")
        st.dataframe(correlations_df, hide_index=True)
    else:
        st.warning(
            f"No features found with |correlation| between {min_threshold} and {max_threshold} "
            "with the target variable. Please adjust the threshold range."
        )

    # ---------------------------------------------------------
    # Pre-training SHAP Analysis
    # ---------------------------------------------------------
    if (len(st.session_state.get('selected_features', [])) > 0 and
            st.session_state.get('target_variable')):

        st.header("3. Pre-Training Feature Importance Analysis (SHAP)")
        st.markdown(
            "**SHAP (SHapley Additive exPlanations)** is used to quantify feature contributions to model predictions. This preliminary analysis applies a baseline model to the imputed data, providing an early assessment of predictor importance before full model training and evaluation."
        )

        if st.session_state.get('pre_training_shap_results'):
            st.subheader("SHAP Analysis Results")
            shap_figs = st.session_state['pre_training_shap_results']
            for title, fig in shap_figs.items():
                st.write(f"**{title}**")
                st.plotly_chart(fig, use_container_width=True)

        button_text = (
            "Regenerate SHAP Analysis"
            if st.session_state.get('pre_training_shap_results')
            else "Generate Pre-Training SHAP Analysis"
        )

        if st.button(button_text):
            with st.spinner('Generating SHAP analysis for imputed data...'):
                try:
                    X = df_imputed[st.session_state['selected_features']]
                    y = df_imputed[st.session_state['target_variable']]

                    shap_figs = pre_training_shap_analysis(X, y)

                    if shap_figs:
                        st.session_state['pre_training_shap_results'] = shap_figs
                        st.success("Pre-training SHAP analysis completed!")
                        st.rerun()
                    else:
                        st.error("Could not generate SHAP analysis. Please check your data.")

                except Exception as e:
                    st.error(f"Error in pre-training SHAP analysis: {str(e)}")

    # ---------------------------------------------------------
    # Model Training and Evaluation
    # ---------------------------------------------------------
    if len(st.session_state.get('selected_features', [])) > 0 and st.session_state.get('target_variable'):
        st.header("4. Model Training and Evaluation")

        X = df_imputed[st.session_state['selected_features']]
        y = df_imputed[st.session_state['target_variable']]

        # Feature transformation
        st.subheader("Feature Transformation (Optional)")
        st.markdown(
            "Optional feature transformations are available to improve numerical stability, reduce sensitivity to outliers, and accommodate heterogeneous predictor distributions. Supported methods include standardisation, min-max scaling, robust scaling, Yeo-Johnson transformation, and quantile transformation."
        )
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

        # Polynomial features
        st.subheader("Polynomial Features (Optional)")
        st.markdown(
            "Polynomial expansion generates higher-order and interaction terms, enabling linear models to represent nonlinear relationships between predictors. Interaction-only expansion provides a more parsimonious alternative by retaining cross-product terms while excluding polynomial powers."
        )
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
        st.markdown(
            "The dataset is partitioned into independent training and testing subsets to estimate model generalisability. Stratified sampling is used for classification tasks to preserve class proportions across partitions, while the random seed ensures reproducibility of the split."
        )
        test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state:", value=42, min_value=0)

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

        # Model selection
        st.subheader("Model Selection and Training")
        st.markdown(
            "The platform implements regularised Logistic Regression for binary classification and LASSO Regression for continuous outcomes. Logistic Regression provides an interpretable framework for estimating the probability of clinically significant prostate cancer, whereas LASSO applies L1 regularisation to support coefficient shrinkage and embedded feature selection."
        )

        if is_classification:
            model_options = ["Logistic Regression"]
        else:
            model_options = ["LASSO Regression"]

        selected_model = st.selectbox(
            "Select model type:",
            options=model_options,
            help="Choose the machine learning algorithm"
        )

        st.markdown(
            "**Cross-validation folds** — internal validation is performed using k-fold cross-validation. The training data are iteratively partitioned into development and validation folds, providing a more stable estimate of model performance than a single validation split."
        )
        cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)

        if selected_model == "LASSO Regression":
            st.markdown(
                "**Alpha** controls the strength of L1 regularisation. Larger values impose greater coefficient shrinkage, while automatic selection identifies the value that optimises cross-validated model performance."
            )
            auto_alpha = st.checkbox("Auto-select alpha (recommended)", value=True)
            if not auto_alpha:
                alpha = st.slider("Alpha (regularization strength):", 0.001, 10.0, 1.0, 0.001)
            else:
                alpha = None

        if st.button(f"Train {selected_model}"):
            with st.spinner(f'Training {selected_model}...'):
                try:
                    if selected_model == "Logistic Regression":
                        model, metrics = train_logistic_regression(X_train, X_test, y_train, y_test, cv_folds)
                        model_type = "logistic"
                    else:
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

                    st.subheader("Model Performance Metrics")

                    if selected_model == "Logistic Regression":
                        st.info(
                            "ℹ️ SMOTE (Synthetic Minority Oversampling Technique) was automatically applied "
                            "to balance the training classes by generating synthetic minority-class samples — "
                            "mitigating bias toward the more frequent non-csPCa class, consistent with Study 1."
                        )

                    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Error training {selected_model}: {str(e)}")

        # ---------------------------------------------------------
        # Results and analysis tabs
        # ---------------------------------------------------------
        if st.session_state.get('model_trained', False):
            st.header("5. Model Results and Analysis")

            model = st.session_state['trained_model']
            model_type = st.session_state['model_type']
            X_train = st.session_state['X_train']
            X_test = st.session_state['X_test']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']

            tab1, tab2, tab3, tab4 = st.tabs([
                "Performance Plots",
                "Feature Importance (SHAP)",
                "Fairness Analysis",
                "Calibration Analysis"
            ])

            with tab1:
                st.subheader("Model Performance Visualisations")

                if model_type == "logistic":
                    st.markdown(
                        "Receiver operating characteristic (ROC) analysis evaluates discriminatory performance across classification thresholds. The area under the ROC curve (AUC) summarises the model's ability to distinguish between clinically significant and non-significant prostate cancer independently of any single operating threshold."
                    )
                    try:
                        roc_fig = plot_roc_curve(model, X_test, y_test)
                        st.plotly_chart(roc_fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting ROC curve: {str(e)}")

                    st.markdown(
                        "The confusion matrix summarises agreement between predicted and observed class labels, providing a direct overview of correctly classified cases and the distribution of false-positive and false-negative errors."
                    )
                    try:
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        X_test_scaled = scaler.fit_transform(X_test)
                        y_pred = model.predict(X_test_scaled)

                        cm_fig = plot_confusion_matrix(y_test, y_pred)
                        st.plotly_chart(cm_fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting confusion matrix: {str(e)}")

                else:
                    st.markdown(
                        "The LASSO regularisation path illustrates how model coefficients change as the regularisation parameter increases. It provides insight into the order and degree of coefficient shrinkage and the stability of predictor selection."
                    )
                    try:
                        reg_fig = plot_regularization_path(X_train, y_train)
                        st.plotly_chart(reg_fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting regularization path: {str(e)}")

            with tab2:
                st.subheader("SHAP Feature Importance Analysis")
                st.markdown(
                    "Post-training SHAP analysis is applied to the fitted model and evaluation data to characterise the magnitude and direction of individual feature contributions. Mean absolute SHAP values are used to summarise each predictor's overall influence on model output."
                )

                try:
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
                st.markdown(
                    "Predictive parity analysis evaluates whether positive predictive value is consistent across demographic subgroups. Systematic differences between groups may indicate differential model performance and potential algorithmic bias requiring further investigation."
                )

                if model_type == "logistic":
                    age_columns = [col for col in df_imputed.columns if 'age' in col.lower()]

                    if age_columns:
                        st.write("**Patient Age-Based Predictive Parity Analysis**")
                        st.markdown(
                            "Select the variable representing patient age. The analysis stratifies observations into predefined age bands and compares classification performance across groups."
                        )
                        selected_age_col = st.selectbox(
                            "Select age column for fairness analysis:",
                            options=age_columns,
                            help="Choose the column containing patient age information"
                        )

                        if selected_age_col:
                            age_data = df_imputed[selected_age_col]
                            age_groups = pd.cut(
                                age_data,
                                bins=[0, 30, 50, 70, 100],
                                labels=['Under 30', '30-50', '50-70', 'Over 70'],
                                include_lowest=True
                            )

                            st.write("**Age groups distribution:**")
                            age_dist = pd.Series(age_groups).value_counts().sort_index()
                            st.write(age_dist)

                            sensitive_attr = age_groups
                    else:
                        st.info(
                            "No age columns found in the dataset. "
                            "Add a column with 'age' in the name for age-based fairness analysis."
                        )
                        sensitive_attr = None

                    if sensitive_attr is not None:
                        try:
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            X_test_scaled = scaler.fit_transform(X_test)
                            y_pred = model.predict(X_test_scaled)
                            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                            sensitive_test = sensitive_attr.loc[X_test.index]

                            parity_results = calculate_predictive_parity(
                                y_test, y_pred, y_pred_proba, sensitive_test
                            )

                            if parity_results:
                                st.write("**What this graph shows:**")
                                st.info(
                                    "This analysis compares positive predictive value and observed outcome prevalence across age groups. Substantial between-group differences may indicate unequal reliability of positive predictions and should be interpreted alongside subgroup sample sizes and uncertainty."
                                )

                                st.write("**Predictive Parity Results:**")
                                st.write("**Group-wise Performance Metrics:**")
                                st.caption("Each row represents a different age group. Key metrics to compare:")
                                st.caption("• **Model Positive Rate**: How often the model predicts csPCa for this group")
                                st.caption("• **Actual Positive Rate**: True prevalence of csPCa in this group")
                                st.caption("• **Sample Size**: Number of patients in this group")
                                st.caption("• **Accuracy / Precision / Recall**: Standard performance metrics per group")

                                metrics_df = pd.DataFrame(parity_results['group_metrics']).T
                                st.dataframe(metrics_df, use_container_width=True, hide_index=False)

                                st.write("**Overall Predictive Parity Metrics:**")
                                fairness_metrics = {
                                    'Predictive Parity Difference': parity_results['predictive_parity_difference'],
                                    'Maximum Precision': parity_results['fairness_metrics']['max_precision'],
                                    'Minimum Precision': parity_results['fairness_metrics']['min_precision'],
                                    'Precision Standard Deviation': parity_results['fairness_metrics']['precision_std']
                                }
                                fairness_df = pd.DataFrame(
                                    list(fairness_metrics.items()),
                                    columns=['Metric', 'Value']
                                )
                                st.dataframe(fairness_df, hide_index=True)

                                if 'visualization' in parity_results:
                                    st.plotly_chart(parity_results['visualization'], use_container_width=True)

                            else:
                                st.error("Could not calculate demographic parity")

                        except Exception as e:
                            st.error(f"Error in fairness analysis: {str(e)}")
                    else:
                        st.info("No additional columns available for fairness analysis")
                else:
                    st.info("Fairness analysis is currently only available for classification models")

            with tab4:
                st.subheader("Calibration Analysis")
                st.markdown(
                    "Calibration analysis assesses agreement between predicted probabilities and observed event frequencies. Reliability diagrams characterise calibration across the risk range, while the Brier score quantifies overall probabilistic accuracy, with lower values indicating better performance."
                )

                if model_type == "logistic":
                    try:
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        X_test_scaled = scaler.fit_transform(X_test)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                        calibration_results = calculate_calibration_metrics(y_test, y_pred_proba)

                        st.write("**Calibration Metrics:**")
                        metrics_data = []
                        for key, value in calibration_results.items():
                            if isinstance(value, (int, float, np.number)) and key not in ['fraction_of_positives', 'mean_predicted_value']:
                                metrics_data.append([key, float(value)])

                        if metrics_data:
                            cal_metrics_df = pd.DataFrame(metrics_data, columns=['Metric', 'Value'])
                            st.dataframe(cal_metrics_df, hide_index=True)

                        st.write("**Calibration Plot (Reliability Diagram):**")
                        cal_fig = plot_calibration_curve(y_test, y_pred_proba)
                        st.plotly_chart(cal_fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error in calibration analysis: {str(e)}")
                else:
                    st.info("Calibration analysis is currently only available for classification models")
