"""Predictor Agent - ML model selection, training, and forecasting."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Tuple, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


def forecast_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    periods: int = 12,
) -> Tuple[pd.DataFrame, go.Figure, Dict[str, Any]]:
    """
    Forecast a time series using Holt-Winters exponential smoothing.

    Returns: (forecast_df, chart, metrics_dict)
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    ts_df = df[[date_col, value_col]].copy()
    ts_df[date_col] = pd.to_datetime(ts_df[date_col])
    ts_df = ts_df.sort_values(date_col).dropna()

    # Detect frequency
    diffs = ts_df[date_col].diff().dropna()
    median_diff = diffs.median()
    if median_diff <= pd.Timedelta(days=2):
        freq = "D"
    elif median_diff <= pd.Timedelta(days=8):
        freq = "W"
    elif median_diff <= pd.Timedelta(days=35):
        freq = "MS"
    else:
        freq = "YS"

    # Set datetime index
    ts_df = ts_df.set_index(date_col)
    ts_df = ts_df.asfreq(freq, method="ffill")
    series = ts_df[value_col]

    # Determine seasonal periods
    seasonal_periods = {"D": 7, "W": 52, "MS": 12, "YS": 1}.get(freq, 12)

    # Fit model
    try:
        if len(series) >= 2 * seasonal_periods and seasonal_periods > 1:
            model = ExponentialSmoothing(
                series, trend="add", seasonal="add",
                seasonal_periods=seasonal_periods,
            ).fit(optimized=True)
        else:
            model = ExponentialSmoothing(
                series, trend="add", seasonal=None,
            ).fit(optimized=True)
    except Exception:
        model = ExponentialSmoothing(
            series, trend="add", seasonal=None,
        ).fit(optimized=True)

    # Generate forecast
    forecast = model.forecast(periods)
    fitted = model.fittedvalues

    # Compute confidence intervals (approximate)
    residuals = series - fitted
    std_resid = residuals.std()
    if std_resid == 0 or pd.isna(std_resid):
        std_resid = series.std() * 0.1 or 1.0
    ci_upper = forecast + 1.96 * std_resid
    ci_lower = forecast - 1.96 * std_resid

    # Metrics
    mape = mean_absolute_percentage_error(series[-min(len(series), periods):], fitted[-min(len(series), periods):])
    rmse = np.sqrt(mean_squared_error(series[-min(len(series), periods):], fitted[-min(len(series), periods):]))

    # Build forecast DataFrame
    forecast_df = pd.DataFrame({
        "Date": forecast.index,
        "Forecast": forecast.values,
        "Lower_CI": ci_lower.values,
        "Upper_CI": ci_upper.values,
    })

    # Build chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        name="Historical", mode="lines", line=dict(color="#2563EB"),
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.values,
        name="Forecast", mode="lines", line=dict(color="#DC2626", dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=list(ci_upper.index) + list(ci_lower.index[::-1]),
        y=list(ci_upper.values) + list(ci_lower.values[::-1]),
        fill="toself", fillcolor="rgba(220,38,38,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Confidence Interval",
    ))
    fig.update_layout(
        title=f"Forecast: {value_col}",
        xaxis_title="Date", yaxis_title=value_col,
        template="plotly_white",
    )

    metrics = {
        "mape": round(mape * 100, 2),
        "rmse": round(rmse, 2),
        "periods_forecast": periods,
        "frequency": freq,
        "method": "Holt-Winters Exponential Smoothing",
    }

    explanation = (
        f"Forecasted {value_col} for the next {periods} periods using "
        f"Holt-Winters Exponential Smoothing. MAPE: {metrics['mape']}%, "
        f"RMSE: {metrics['rmse']}. The shaded area shows the 95% confidence interval."
    )
    metrics["explanation"] = explanation

    return forecast_df, fig, metrics


def cluster_data(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    max_k: int = 8,
) -> Tuple[pd.DataFrame, go.Figure, Dict[str, Any]]:
    """
    Perform K-Means clustering with automatic k selection via elbow method.

    Returns: (clustered_df, chart, cluster_profiles)
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include="number").columns.tolist()

    if len(feature_cols) < 2:
        raise ValueError("Need at least 2 numeric columns for clustering.")

    X = df[feature_cols].dropna()
    if len(X) < 4:
        raise ValueError(f"Need at least 4 rows for clustering, got {len(X)}.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method
    inertias = []
    K_range = range(2, min(max_k + 1, len(X)))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # Find elbow (simple: largest decrease in inertia)
    if len(inertias) > 2:
        diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        diffs2 = [diffs[i] - diffs[i + 1] for i in range(len(diffs) - 1)]
        optimal_k = list(K_range)[np.argmax(diffs2) + 1] if diffs2 else 3
    else:
        optimal_k = 2

    # Final clustering
    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = km_final.fit_predict(X_scaled)

    result_df = df.loc[X.index].copy()
    result_df["Cluster"] = labels

    # Cluster profiles
    profiles = {}
    for c in range(optimal_k):
        mask = labels == c
        profile = {
            "size": int(mask.sum()),
            "percentage": round(mask.sum() / len(labels) * 100, 1),
        }
        for col in feature_cols:
            profile[f"avg_{col}"] = round(float(X.loc[X.index[mask], col].mean()), 2)
        profiles[f"Cluster {c}"] = profile

    # Scatter plot (first 2 features)
    fig = go.Figure()
    for c in range(optimal_k):
        mask = labels == c
        fig.add_trace(go.Scatter(
            x=X_scaled[mask, 0], y=X_scaled[mask, 1],
            mode="markers", name=f"Cluster {c}",
            marker=dict(size=6),
        ))
    fig.update_layout(
        title=f"K-Means Clustering (k={optimal_k})",
        xaxis_title=feature_cols[0] + " (scaled)",
        yaxis_title=feature_cols[1] + " (scaled)",
        template="plotly_white",
    )

    explanation = (
        f"Identified {optimal_k} distinct clusters using K-Means. "
        + " ".join(
            f"Cluster {c}: {profiles[f'Cluster {c}']['size']} records ({profiles[f'Cluster {c}']['percentage']}%)."
            for c in range(optimal_k)
        )
    )

    metrics = {
        "optimal_k": optimal_k,
        "profiles": profiles,
        "features_used": feature_cols,
        "method": "K-Means with Elbow Method",
        "explanation": explanation,
    }

    return result_df, fig, metrics


def detect_anomalies(
    df: pd.DataFrame,
    numeric_cols: Optional[list] = None,
    threshold: float = 1.5,
) -> Tuple[pd.DataFrame, go.Figure, Dict[str, Any]]:
    """
    Detect anomalies using IQR method.

    Returns: (anomaly_df, chart, summary)
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

    anomaly_mask = pd.Series(False, index=df.index)
    anomaly_reasons = pd.Series("", index=df.index)

    for col in numeric_cols:
        if df[col].dropna().empty:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        col_mask = (df[col] < lower) | (df[col] > upper)
        anomaly_mask |= col_mask
        anomaly_reasons = anomaly_reasons.where(
            ~col_mask,
            anomaly_reasons + f"{col} outside [{lower:.2f}, {upper:.2f}]; ",
        )

    anomaly_df = df[anomaly_mask].copy()
    anomaly_df["anomaly_reason"] = anomaly_reasons[anomaly_mask]

    # Visualization: highlight anomalies on the first numeric column
    target_col = numeric_cols[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df[target_col],
        mode="markers", name="Normal",
        marker=dict(color="#2563EB", size=5),
    ))
    if len(anomaly_df) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_df.index, y=anomaly_df[target_col],
            mode="markers", name="Anomaly",
            marker=dict(color="#DC2626", size=10, symbol="x"),
        ))
    fig.update_layout(
        title=f"Anomaly Detection: {target_col}",
        xaxis_title="Index", yaxis_title=target_col,
        template="plotly_white",
    )

    explanation = (
        f"Found {len(anomaly_df)} anomalies out of {len(df)} records "
        f"({round(len(anomaly_df)/max(len(df),1)*100, 1)}%) using the IQR method "
        f"with a threshold of {threshold}x IQR."
    )

    metrics = {
        "total_anomalies": len(anomaly_df),
        "total_records": len(df),
        "anomaly_rate": round(len(anomaly_df) / max(len(df), 1) * 100, 2),
        "method": f"IQR (threshold={threshold}x)",
        "columns_checked": numeric_cols,
        "explanation": explanation,
    }

    return anomaly_df, fig, metrics
