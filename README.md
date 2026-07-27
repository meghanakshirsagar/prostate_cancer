# Prostate Cancer Risk Stratification: A Web-Based Machine Learning Platform

This repository accompanies the paper:

> **Machine Learning for Prostate Cancer Risk Stratification: A Web-Based Tool with Explainability and Fairness**

The project presents an interactive web-based machine learning platform for prostate cancer risk stratification using structured clinical data from the **PI-CAI (Prostate Imaging: Cancer AI) Challenge**. The platform reproduces the complete analytical workflow described in the accompanying publication, enabling researchers to explore data preprocessing, feature engineering, predictive modelling, explainability, fairness evaluation, and calibration analysis within a unified and interactive environment.

Designed as a reproducible research companion, the application provides an accessible interface for investigating the impact of different preprocessing strategies and modelling decisions while promoting transparent and interpretable machine learning for prostate cancer research.

---

## Features

- Interactive Streamlit-based web application
- Automatic loading of the study dataset
- Missing-value analysis and multiple imputation strategies (Mean, Median, Mode, KNN, Bayesian Ridge, and MICE)
- Correlation-based feature selection
- Feature transformation and polynomial feature generation
- Logistic Regression and LASSO modelling
- SHAP-based explainability
- Predictive parity (fairness) analysis
- Model calibration and reliability assessment
- Interactive visualisation of results
- Export of processed datasets

---

## Dataset

This project uses the publicly available **PI-CAI (Prostate Imaging: Cancer AI) Challenge** clinical dataset.

The study cohort comprises **1,422 patients** collected from **three clinical centres** across **11 sites** between **2011 and 2021**, with the objective of predicting **clinically significant prostate cancer (csPCa)** from routinely available clinical variables.

For access to the original dataset and documentation, please refer to:

- **Clinical Dataset:** https://github.com/DIAGNijmegen/picai_labels/blob/main/clinical_information/marksheet.csv
- **PI-CAI Challenge:** https://pi-cai.grand-challenge.org/
- **Zenodo Repository:** https://zenodo.org/records/6517398

Please ensure that all use of the dataset complies with the licensing and usage terms provided by the PI-CAI Challenge organisers.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/meghanakshirsagar/prostate_cancer.git
cd prostate_cancer/ProstateCancerUI
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

or, if using **uv**:

```bash
uv sync
```

---

## Running the Application

Launch the Streamlit application using:

```bash
streamlit run app.py
```

The application automatically loads the study dataset and provides access to the complete analytical workflow through the web interface.

---

## Citation

If you use this repository in your research, please cite the accompanying publication.

```bibtex
@inproceedings{kshirsagar2027prostate,
  title     = {Machine Learning for Prostate Cancer Risk Stratification:
               A Web-Based Tool with Explainability and Fairness},
  author    = {Kshirsagar, Meghana and Vaidya, Gauri and
               Srivastava, Yuvraj and Ryan, Conor},
  booktitle = {International Conference on Agents and Artificial Intelligence},
  publisher = {Springer},
  year      = {2027},
  doi        = {10.1007/978-3-032-25035-3_5}
}
```
