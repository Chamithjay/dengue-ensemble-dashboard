# Ensemble Learning for Forecasting Weekly Dengue Cases in Sri Lanka

**Live dashboard:** _paste your Streamlit link here_

This project forecasts **next week's dengue cases** in each of Sri Lanka's 26 health regions and
compares four ensemble learning paradigms: **Bagging, Boosting, Stacking and a Hybrid model**.

## Dataset
`srilanka_weekly_data` from the **denguedatahub** R package (v4.1.0) by Thiyanga S. Talagala
(https://cran.r-universe.dev/denguedatahub). It contains weekly reported dengue cases from the
Epidemiology Unit's Weekly Epidemiological Reports, Dec 2006 – Dec 2025, for 25 districts plus
the Kalmunai RDHS division (25,766 rows).

## Models
| Paradigm | Model(s) |
|---|---|
| Bagging | Random Forest |
| Boosting | XGBoost, LightGBM, CatBoost |
| Stacking | Random Forest + LightGBM + Ridge, with a Ridge meta-learner |
| Hybrid | ARIMA (per district) + Random Forest on ARIMA residuals |

All models were tuned with `GridSearchCV` and `TimeSeriesSplit`, trained on 2007–2022 and tested on 2023–2025.

## Results (test set, 2023–2025)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Hybrid (ARIMA + RF)** | 12.074 | **27.714** | **0.840** |
| LightGBM (Boosting) | 11.953 | 27.896 | 0.838 |
| XGBoost (Boosting) | 12.000 | 27.974 | 0.837 |
| Stacking (RF + LGBM + Ridge) | **11.828** | 27.983 | 0.837 |
| Random Forest (Bagging) | 12.097 | 28.021 | 0.837 |
| CatBoost (Boosting) | 11.965 | 28.134 | 0.835 |

MAE and RMSE are in actual cases per district per week. RMSE is the primary metric because it
penalises missed outbreak spikes most heavily.

## Dashboard features
- **Overview:** national and district dengue trends
- **Model Comparison:** metrics table, comparison charts and actual-vs-predicted curves
- **Feature Importance:** importance for each tree-based model
- **Live Prediction:** forecast next week's cases for any district with all six models, with editable
  inputs for what-if scenarios

## Repository structure
```
├── Dengue_Ensemble_Learning.ipynb   # Full analysis notebook
├── app.py                           # Streamlit dashboard
├── requirements.txt                 # Python packages (versions match the training environment)
├── packages.txt                     # System package needed by LightGBM
└── models/                          # Saved models (.pkl) and data used by the dashboard
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## AI Usage Acknowledgment
I declare the following use of AI tools:

- **Tool used:** Claude
- **Where and how it was used:**
  - Generating the Python code for data cleaning, feature engineering, model training
    (Random Forest, XGBoost, LightGBM, CatBoost, Stacking, ARIMA + RF hybrid), hyperparameter
    tuning, evaluation and charts in the notebook
  - Generating the Streamlit dashboard code
  - Debugging errors encountered while running the notebook

