"""
Sri Lanka Dengue Forecasting – Ensemble Learning Dashboard
Loads the models trained in Dengue_Ensemble_Learning.ipynb from the models/ folder.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Sri Lanka Dengue – Ensemble Dashboard",
                   page_icon="🦟", layout="wide")

MODEL_DIR = Path(__file__).parent / "models"

MODEL_FILES = {
    "Random Forest (Bagging)": "random_forest.pkl",
    "XGBoost (Boosting)": "xgboost.pkl",
    "LightGBM (Boosting)": "lightgbm.pkl",
    "CatBoost (Boosting)": "catboost.pkl",
    "Stacking (RF + LGBM + Ridge)": "stacking.pkl",
}
HYBRID = "Hybrid (ARIMA + RF)"
PARADIGM = {
    "Random Forest (Bagging)": "Bagging",
    "XGBoost (Boosting)": "Boosting",
    "LightGBM (Boosting)": "Boosting",
    "CatBoost (Boosting)": "Boosting",
    "Stacking (RF + LGBM + Ridge)": "Stacking",
    HYBRID: "Hybrid",
}

# Colour scheme: one colour per paradigm, one shade per model
PARADIGM_COLORS = {"Bagging": "#16A34A", "Boosting": "#2563EB",
                   "Stacking": "#9333EA", "Hybrid": "#EA580C"}
MODEL_COLORS = {
    "Random Forest (Bagging)": "#16A34A",
    "XGBoost (Boosting)": "#1D4ED8",
    "LightGBM (Boosting)": "#3B82F6",
    "CatBoost (Boosting)": "#93C5FD",
    "Stacking (RF + LGBM + Ridge)": "#9333EA",
    HYBRID: "#EA580C",
}
CASES_COLOR = "#DC2626"
ACTUAL_COLOR = "#0F172A"
HIGHLIGHT = "background-color: #D1FAE5; color: #065F46; font-weight: 600"
px.defaults.template = "plotly_white"

st.markdown("""
<style>
[data-testid="stMetric"] {
    background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px 18px;
}
[data-testid="stSidebar"] { background: #F0FDFA; }
h1, h2, h3 { color: #0F172A; }
</style>
""", unsafe_allow_html=True)


# loading
def load_one(name, filename):
    if name.startswith("XGBoost") and (MODEL_DIR / "xgboost.json").exists():
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(str(MODEL_DIR / "xgboost.json"))
        return model
    return joblib.load(MODEL_DIR / filename)


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    models, failed = {}, {}
    for name, f in MODEL_FILES.items():
        try:
            models[name] = load_one(name, f)
        except Exception as e: 
            failed[name] = f"{type(e).__name__}: {e}"
    hybrid_rf = joblib.load(MODEL_DIR / "hybrid_rf.pkl")
    meta = joblib.load(MODEL_DIR / "metadata.pkl")
    return models, hybrid_rf, meta, failed


@st.cache_data
def load_tables():
    results = pd.read_csv(MODEL_DIR / "results.csv", index_col=0)
    importance = pd.read_csv(MODEL_DIR / "feature_importance.csv", index_col=0)
    data = pd.read_csv(MODEL_DIR / "dashboard_data.csv", parse_dates=["start_date"])
    data = data.sort_values(["district", "start_date"]).reset_index(drop=True)
    preds = pd.read_csv(MODEL_DIR / "test_predictions.csv", parse_dates=["start_date"])
    # Show only the six ensemble/hybrid models (drop any extra rows such as a baseline)
    results = results[results.index.isin(PARADIGM)]
    preds = preds[["district", "start_date", "Actual"] + [c for c in preds.columns if c in PARADIGM]]
    return results, importance, data, preds


@st.cache_resource(show_spinner="Fitting ARIMA for this district...")
def fit_arima(district):
    """Re-fit the district's ARIMA on training years only (same order as the notebook)."""
    meta = load_models()[2]
    data = load_tables()[2]
    s = data[data["district"] == district]
    order = meta["arima_orders"].get(district) or (1, 1, 1)
    return ARIMA(s.loc[s["year"] <= 2022, "log_cases"].values, order=tuple(order)).fit()


# ---------------------------------------------------------------- features
def build_features(history, last_year, national_last_week, forecast_date, district, features):
    """Recreate the notebook's features for one district-week.
    history: last 8 weekly case counts, oldest -> newest."""
    logs = np.log1p(np.asarray(history, dtype=float))
    row = dict.fromkeys(features, 0.0)
    for lag in range(1, 5):
        row[f"lag_{lag}"] = logs[-lag]
    row["lag_52"] = np.log1p(last_year)
    row["roll_mean_4"] = logs[-4:].mean()
    row["roll_mean_8"] = logs.mean()
    row["roll_std_4"] = logs[-4:].std(ddof=1)
    row["trend_1"] = logs[-1] - logs[-2]
    row["national_lag_1"] = np.log1p(national_last_week)
    week = int(forecast_date.isocalendar()[1])
    row["week_sin"] = np.sin(2 * np.pi * week / 52)
    row["week_cos"] = np.cos(2 * np.pi * week / 52)
    if f"d_{district}" in row:
        row[f"d_{district}"] = 1
    return pd.DataFrame([row], columns=features)


def to_cases(log_value):
    return float(np.clip(np.expm1(log_value), 0, None))


# app
models, hybrid_rf, meta, failed = load_models()
results, importance, data, preds = load_tables()
for name, err in failed.items():
    st.error(f"Could not load **{name}** – {err}")
FEATURES = meta["FEATURES"]
districts = sorted(data["district"].unique())
best_model = results["RMSE"].idxmin()

st.sidebar.title("🦟 Dengue Ensemble Dashboard")
page = st.sidebar.radio("Go to", ["🏠 Overview", "📊 Model Comparison",
                                  "🔍 Feature Importance", "🔮 Live Prediction"])

#  overview
if page == "🏠 Overview":
    st.title("Forecasting Weekly Dengue Cases in Sri Lanka")
    st.markdown("Comparing **Bagging, Boosting, Stacking and Hybrid** ensemble models for "
                "one-week-ahead dengue forecasts across 26 Sri Lankan health regions.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Period", f"{data['start_date'].min().year} – {data['start_date'].max().year}")
    c2.metric("Districts / regions", len(districts))
    c3.metric("Best model (RMSE)", best_model.split(" (")[0])
    c4.metric("Best test RMSE", f"{results.loc[best_model, 'RMSE']:.2f} cases")

    st.subheader("Weekly dengue cases")
    choice = st.selectbox("Region", ["All Sri Lanka"] + districts)
    if choice == "All Sri Lanka":
        series = data.groupby("start_date")["cases"].sum().reset_index()
    else:
        series = data[data["district"] == choice][["start_date", "cases"]]
    fig = px.line(series, x="start_date", y="cases",
                  labels={"start_date": "Week", "cases": "Reported cases"})
    fig.update_traces(line_color=CASES_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Total cases by district")
    totals = data.groupby("district")["cases"].sum().sort_values().reset_index()
    fig = px.bar(totals, x="cases", y="district", orientation="h", height=650,
                 labels={"cases": "Total cases", "district": ""})
    fig.update_traces(marker_color="#0F766E")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- comparison
elif page == "📊 Model Comparison":
    st.title("Model Comparison (Test Set: 2023–2025)")

    table = results.copy()
    table.insert(0, "Paradigm", [PARADIGM.get(m, "") for m in table.index])
    st.dataframe(table.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"})
                 .highlight_min(subset=["MAE", "RMSE"], props=HIGHLIGHT)
                 .highlight_max(subset=["R2"], props=HIGHLIGHT),
                 use_container_width=True)
    st.caption("MAE and RMSE are in actual cases per district per week (lower is better). "
               "RMSE is the primary metric because it penalises missed outbreak spikes.")

    metric = st.radio("Metric", ["RMSE", "MAE", "R2"], horizontal=True)
    ordered = results.sort_values(metric, ascending=(metric != "R2")).reset_index(names="Model")
    ordered["Paradigm"] = ordered["Model"].map(PARADIGM)
    lo, hi = ordered[metric].min(), ordered[metric].max()
    pad = max((hi - lo) * 0.5, 1e-3)
    fig = px.bar(ordered, x=metric, y="Model", color="Paradigm", orientation="h",
                 text=ordered[metric].round(3), color_discrete_map=PARADIGM_COLORS)
    fig.update_layout(xaxis_range=[lo - pad, hi + pad], yaxis={"categoryorder": "array",
                      "categoryarray": ordered["Model"][::-1]})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("All models are within about 1.5% of each other on RMSE.")

    st.subheader("Actual vs predicted")
    region = st.selectbox("Region", ["All Sri Lanka"] + districts, key="cmp_region")
    model_cols = [c for c in preds.columns if c not in ("district", "start_date", "Actual")]
    chosen = st.multiselect("Models", model_cols,
                            default=[m for m in [HYBRID, "LightGBM (Boosting)"] if m in model_cols])
    sub = preds if region == "All Sri Lanka" else preds[preds["district"] == region]
    plot_df = sub.groupby("start_date")[["Actual"] + chosen].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["start_date"], y=plot_df["Actual"], name="Actual",
                             line=dict(color=ACTUAL_COLOR, width=3)))
    for m in chosen:
        fig.add_trace(go.Scatter(x=plot_df["start_date"], y=plot_df[m], name=m,
                                 line=dict(color=MODEL_COLORS.get(m), width=1.8)))
    fig.update_layout(xaxis_title="Week", yaxis_title="Cases", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- importance
elif page == "🔍 Feature Importance":
    st.title("Feature Importance")
    imp = importance.copy()
    imp["Average"] = imp.mean(axis=1)
    model = st.selectbox("Model", list(imp.columns)[::-1])
    hide_districts = st.checkbox("Hide district indicator features", value=True)
    top_n = st.slider("Number of features", 5, 20, 12)

    s = imp[model]
    if hide_districts:
        s = s[~s.index.str.startswith("d_")]
    s = s.sort_values(ascending=False).head(top_n).iloc[::-1]
    fig = px.bar(x=s.values, y=s.index, orientation="h",
                 labels={"x": "Share of total importance", "y": ""}, height=500)
    fig.update_traces(marker_color="#0F766E")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("All tree-based models side by side")
    base = imp.drop(columns="Average")
    base = base[~base.index.str.startswith("d_")] if hide_districts else base
    top = base.mean(axis=1).sort_values(ascending=False).head(top_n).index
    long = base.loc[top].reset_index(names="Feature").melt("Feature", var_name="Model",
                                                           value_name="Importance")
    fig = px.bar(long, x="Importance", y="Feature", color="Model", barmode="group",
                 orientation="h", height=600,
                 color_discrete_map={"Random Forest": MODEL_COLORS["Random Forest (Bagging)"],
                                     "XGBoost": MODEL_COLORS["XGBoost (Boosting)"],
                                     "LightGBM": MODEL_COLORS["LightGBM (Boosting)"],
                                     "CatBoost": MODEL_COLORS["CatBoost (Boosting)"]})
    fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": list(top[::-1])})
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- prediction
elif page == "🔮 Live Prediction":
    st.title("Live Prediction: Next Week's Dengue Cases")
    district = st.selectbox("District / region", districts,
                            index=districts.index("Colombo") if "Colombo" in districts else 0)
 
    hist = data[data["district"] == district].reset_index(drop=True)
    last_date = hist["start_date"].iloc[-1]
    forecast_date = last_date + pd.Timedelta(days=7)
    national_default = int(data[data["start_date"] == last_date]["cases"].sum())
    st.markdown("Predicting **next week's** dengue cases from the most recent reported weeks.")
 
    st.markdown("**Reported cases in the last 8 weeks**")
    recent = hist.tail(8)
    history = []
    cols = st.columns(8)
    for i, (col, (_, r)) in enumerate(zip(cols, recent.iterrows())):
        label = "Last week" if i == 7 else f"{8 - i} weeks ago"
        history.append(col.number_input(label, min_value=0,
                                        value=int(r["cases"]), step=1,
                                        key=f"h{i}_{district}"))
 
    c1, c2 = st.columns(2)
    last_year = c1.number_input("Cases in the same week last year", min_value=0,
                                value=int(hist["cases"].iloc[-52]), step=1,
                                key=f"ly_{district}")
    national = c2.number_input("All-island cases last week", min_value=0,
                               value=national_default, step=10, key=f"nat_{district}")
 
    X = build_features(history, last_year, national, forecast_date, district, FEATURES)
    forecasts = {name: to_cases(m.predict(X)[0]) for name, m in models.items()}
 
    # Hybrid: ARIMA one-week-ahead forecast + Random Forest residual correction
    post = hist.loc[hist["year"] >= 2023, "log_cases"].values.copy()
    post[-8:] = np.log1p(np.asarray(history, dtype=float))
    arima_log = float(fit_arima(district).append(post, refit=False).forecast(1)[0])
    forecasts[HYBRID] = to_cases(arima_log + hybrid_rf.predict(X)[0])
 
    out = (pd.DataFrame({"Model": list(forecasts), "Forecast": list(forecasts.values())})
           .assign(Paradigm=lambda d: d["Model"].map(PARADIGM))
           .sort_values("Forecast"))
 
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    best_live = best_model if best_model in forecasts else HYBRID
    m1.metric(f"{best_live.split(' (')[0]} forecast (best model)",
              f"{forecasts[best_live]:.0f} cases",
              delta=f"{forecasts[best_live] - history[-1]:+.0f} vs last week",
              delta_color="inverse")
    m2.metric("Average of all models", f"{out['Forecast'].mean():.0f} cases")
    m3.metric("Range across models",
              f"{out['Forecast'].min():.0f} – {out['Forecast'].max():.0f}")
 
    left, right = st.columns(2)
    with left:
        fig = px.bar(out, x="Forecast", y="Model", color="Paradigm", orientation="h",
                     text=out["Forecast"].round(0), title="Forecast by model",
                     color_discrete_map=PARADIGM_COLORS)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        week_labels = [f"-{8 - i}" for i in range(8)] + ["Next"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=week_labels[:8], y=history, mode="lines+markers",
                                 name="Reported (inputs)", line=dict(color=ACTUAL_COLOR)))
        for name, value in forecasts.items():
            fig.add_trace(go.Scatter(x=[week_labels[7], week_labels[8]], y=[history[-1], value],
                                     mode="lines+markers", name=name,
                                     line=dict(dash="dot", color=MODEL_COLORS.get(name))))
        fig.update_layout(title="Recent weeks and next-week forecasts",
                          xaxis_title="Weeks (relative to now)", yaxis_title="Cases")
        st.plotly_chart(fig, use_container_width=True)