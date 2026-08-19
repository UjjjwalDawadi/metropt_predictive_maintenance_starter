import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="MetroPT Predictive Maintenance Dashboard",
    page_icon="🚆",
    layout="wide"
)

APP_PATH = Path(__file__).resolve()
PROJECT_ROOT = APP_PATH.parent.parent

OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUTPUT_SHAP_DIR = PROJECT_ROOT / "outputs" / "shap"

FAILURE_EVENTS = pd.DataFrame({
    "event_id": ["F1", "F2", "F3", "F4"],
    "failure_type": ["Air leak", "Air leak", "Air leak", "Air leak"],
    "failure_start": [
        "2020-04-18 00:00",
        "2020-05-29 23:30",
        "2020-06-05 10:00",
        "2020-07-15 14:30"
    ],
    "failure_end": [
        "2020-04-18 23:59",
        "2020-05-30 06:00",
        "2020-06-07 14:30",
        "2020-07-15 19:00"
    ]
})

FAILURE_EVENTS["failure_start"] = pd.to_datetime(FAILURE_EVENTS["failure_start"])
FAILURE_EVENTS["failure_end"] = pd.to_datetime(FAILURE_EVENTS["failure_end"])
FAILURE_EVENTS["warning_12h_start"] = FAILURE_EVENTS["failure_start"] - pd.Timedelta(hours=12)
FAILURE_EVENTS["warning_12h_end"] = FAILURE_EVENTS["failure_start"] - pd.Timedelta(seconds=1)


@st.cache_data(show_spinner=False)
def load_csv(path_string, nrows=None, usecols=None):
    path = Path(path_string)
    if not path.exists():
        return None

    try:
        return pd.read_csv(path, nrows=nrows, usecols=usecols)
    except Exception as exc:
        st.warning(f"Could not load {path.name}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_small_table(path_string, max_rows=5000):
    path = Path(path_string)
    if not path.exists():
        return None

    try:
        return pd.read_csv(path, nrows=max_rows)
    except Exception as exc:
        st.warning(f"Could not load {path.name}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def get_prediction_date_range(path_string):
    path = Path(path_string)

    if not path.exists():
        return None, None

    min_ts = None
    max_ts = None

    try:
        for chunk in pd.read_csv(path, usecols=["timestamp"], chunksize=100_000):
            ts = pd.to_datetime(chunk["timestamp"], errors="coerce").dropna()

            if ts.empty:
                continue

            cmin = ts.min()
            cmax = ts.max()

            min_ts = cmin if min_ts is None else min(min_ts, cmin)
            max_ts = cmax if max_ts is None else max(max_ts, cmax)

        return min_ts, max_ts

    except Exception as exc:
        st.warning(f"Could not read prediction date range: {exc}")
        return None, None


@st.cache_data(show_spinner=False)
def load_prediction_date_range(path_string, usecols_tuple, start_date_string, end_date_string):
    """
    Reads only selected columns and selected date range.
    Cached by file path, columns and dates.
    """
    path = Path(path_string)
    usecols = list(usecols_tuple)

    start_ts = pd.Timestamp(start_date_string)
    end_ts = pd.Timestamp(end_date_string) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    pieces = []

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        chunk = chunk.dropna(subset=["timestamp"])

        chunk = chunk[
            (chunk["timestamp"] >= start_ts) &
            (chunk["timestamp"] <= end_ts)
        ].copy()

        if not chunk.empty:
            pieces.append(chunk)

    if not pieces:
        return pd.DataFrame(columns=usecols)

    df = pd.concat(pieces, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if "warning_12h_event_id" in df.columns:
        df["warning_12h_event_id"] = df["warning_12h_event_id"].fillna("None").astype(str)

    if "y_true" in df.columns:
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int)

    for col in df.columns:
        if col not in ["timestamp", "warning_12h_event_id"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def get_prediction_preview(path_string, nrows=5000):
    path = Path(path_string)

    if not path.exists():
        return None

    try:
        preview = pd.read_csv(path, nrows=nrows)

        if "timestamp" in preview.columns:
            preview["timestamp"] = pd.to_datetime(preview["timestamp"], errors="coerce")
            preview = preview.dropna(subset=["timestamp"]).sort_values("timestamp")

        return preview

    except Exception as exc:
        st.warning(f"Could not preview prediction file: {exc}")
        return None


def read_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def file_status(path):
    return "✅ Found" if Path(path).exists() else "❌ Missing"


def format_number(value, decimals=3):
    try:
        if value is None or pd.isna(value):
            return "N/A"

        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{value:,.{decimals}f}"

        return str(value)

    except Exception:
        return str(value)


def first_existing_column(df, possible_columns):
    if df is None:
        return None

    for col in possible_columns:
        if col in df.columns:
            return col

    return None


def normalise_name(value):
    return str(value).replace("_", " ").replace("-", " ").title()




def calculate_detection_rate(row):
    if "failure_detection_rate" in row.index and not pd.isna(row["failure_detection_rate"]):
        return float(row["failure_detection_rate"])

    events_detected = row.get("events_detected", np.nan)
    events_evaluated = row.get("events_evaluated", np.nan)

    if pd.notna(events_detected) and pd.notna(events_evaluated) and events_evaluated != 0:
        return float(events_detected) / float(events_evaluated)

    return np.nan


def get_validation_summary(best_model_selection):
    summary = {
        "model": "N/A",
        "threshold": "N/A"
    }

    if best_model_selection is None or best_model_selection.empty:
        return summary

    model_col = first_existing_column(best_model_selection, ["model", "model_name", "classifier"])
    threshold_col = first_existing_column(best_model_selection, ["threshold", "selected_threshold"])

    if model_col:
        summary["model"] = normalise_name(best_model_selection[model_col].iloc[0])

    if threshold_col:
        summary["threshold"] = format_number(best_model_selection[threshold_col].iloc[0], 2)

    return summary


def get_best_alarm_summary(test_alarm_burden_results):
    summary = {
        "model": "N/A",
        "threshold": "N/A",
        "duration": "N/A",
        "events": "N/A",
        "lead_time": "N/A",
        "false_alarms": "N/A",
        "alarm_time": "N/A"
    }

    if test_alarm_burden_results is None or test_alarm_burden_results.empty:
        return summary, None

    df = test_alarm_burden_results.copy()

    model_col = first_existing_column(df, ["model", "model_name", "classifier"])
    threshold_col = first_existing_column(df, ["threshold", "selected_threshold"])
    duration_col = first_existing_column(df, ["min_duration_minutes", "minimum_duration_minutes", "persistence_minutes"])
    events_col = first_existing_column(df, ["events_detected", "detected_events"])
    evaluated_col = first_existing_column(df, ["events_evaluated", "total_events"])
    lead_col = first_existing_column(df, ["median_lead_time_hours", "lead_time_hours", "median_detection_lead_time_hours"])
    false_col = first_existing_column(df, ["false_alarm_episodes_per_day", "false_alarms_per_day"])
    alarm_pct_col = first_existing_column(df, ["total_alarm_time_percentage", "alarm_time_percentage"])
    alarm_hours_col = first_existing_column(df, ["total_alarm_time_hours", "alarm_time_hours"])
    detection_col = first_existing_column(df, ["failure_detection_rate", "event_recall", "recall"])

    if model_col is None:
        return summary, None

    for col in [threshold_col, duration_col, events_col, evaluated_col, lead_col, false_col, alarm_pct_col, alarm_hours_col, detection_col]:
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if detection_col:
        df["_detection_rate"] = df[detection_col]
    else:
        df["_detection_rate"] = df.apply(calculate_detection_rate, axis=1)

    if events_col is None:
        df["_events_detected"] = df["_detection_rate"]
    else:
        df["_events_detected"] = df[events_col]

    if false_col is None:
        df["_false_alarm_sort"] = 999999
    else:
        df["_false_alarm_sort"] = df[false_col].fillna(999999)

    if alarm_pct_col is None:
        df["_alarm_time_sort"] = 999999
    else:
        df["_alarm_time_sort"] = df[alarm_pct_col].fillna(999999)

    if lead_col is None:
        df["_lead_time_sort"] = 0
    else:
        df["_lead_time_sort"] = df[lead_col].fillna(0)

    df = df.sort_values(
        by=["_detection_rate", "_events_detected", "_false_alarm_sort", "_alarm_time_sort", "_lead_time_sort"],
        ascending=[False, False, True, True, False]
    )

    best_row = df.iloc[0]

    summary["model"] = normalise_name(best_row[model_col])

    if threshold_col:
        summary["threshold"] = format_number(best_row[threshold_col], 2)

    if duration_col:
        summary["duration"] = format_number(best_row[duration_col], 0) + " min"

    if events_col and evaluated_col:
        summary["events"] = f"{format_number(best_row[events_col], 0)}/{format_number(best_row[evaluated_col], 0)}"
    elif detection_col:
        summary["events"] = format_number(best_row[detection_col], 3)

    if lead_col:
        summary["lead_time"] = format_number(best_row[lead_col], 2) + " h"

    if false_col:
        summary["false_alarms"] = format_number(best_row[false_col], 2) + " / day"

    if alarm_pct_col:
        summary["alarm_time"] = format_number(best_row[alarm_pct_col], 2) + "%"
    elif alarm_hours_col:
        summary["alarm_time"] = format_number(best_row[alarm_hours_col], 2) + " h"

    return summary, best_row


def prepare_top_alarm_settings(test_alarm_burden_results, max_rows=10):
    if test_alarm_burden_results is None or test_alarm_burden_results.empty:
        return None

    df = test_alarm_burden_results.copy()

    model_col = first_existing_column(df, ["model", "model_name", "classifier"])
    threshold_col = first_existing_column(df, ["threshold", "selected_threshold"])
    duration_col = first_existing_column(df, ["min_duration_minutes", "minimum_duration_minutes", "persistence_minutes"])
    events_col = first_existing_column(df, ["events_detected", "detected_events"])
    evaluated_col = first_existing_column(df, ["events_evaluated", "total_events"])
    lead_col = first_existing_column(df, ["median_lead_time_hours", "lead_time_hours", "median_detection_lead_time_hours"])
    false_col = first_existing_column(df, ["false_alarm_episodes_per_day", "false_alarms_per_day"])
    alarm_pct_col = first_existing_column(df, ["total_alarm_time_percentage", "alarm_time_percentage"])
    detection_col = first_existing_column(df, ["failure_detection_rate", "event_recall", "recall"])

    if model_col is None:
        return None

    for col in [threshold_col, duration_col, events_col, evaluated_col, lead_col, false_col, alarm_pct_col, detection_col]:
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if detection_col:
        df["_detection_rate"] = df[detection_col]
    elif events_col and evaluated_col:
        df["_detection_rate"] = df[events_col] / df[evaluated_col].replace(0, np.nan)
    else:
        df["_detection_rate"] = np.nan

    df["_false_alarm_sort"] = df[false_col].fillna(999999) if false_col else 999999
    df["_alarm_time_sort"] = df[alarm_pct_col].fillna(999999) if alarm_pct_col else 999999
    df["_lead_time_sort"] = df[lead_col].fillna(0) if lead_col else 0
    df["_events_detected"] = df[events_col].fillna(0) if events_col else df["_detection_rate"].fillna(0)

    df = df.sort_values(
        by=["_detection_rate", "_events_detected", "_false_alarm_sort", "_alarm_time_sort", "_lead_time_sort"],
        ascending=[False, False, True, True, False]
    )

    display_cols = []
    for col in [model_col, threshold_col, duration_col, events_col, evaluated_col, lead_col, false_col, alarm_pct_col]:
        if col and col not in display_cols:
            display_cols.append(col)

    return df[display_cols].head(max_rows)


def show_table(df, title=None, height=320, max_rows=300):
    if title:
        st.subheader(title)

    if df is None or df.empty:
        st.info("No data available.")
        return

    display_df = df.head(max_rows).copy()
    st.dataframe(display_df, width="stretch", height=height)

    if len(df) > max_rows:
        st.caption(f"Showing first {max_rows:,} rows from {len(df):,} rows to keep the dashboard responsive.")


def risk_badge(level):
    level = str(level)

    if level == "Critical":
        return "🔴 Critical"
    if level == "High":
        return "🟠 High"
    if level == "Medium":
        return "🟡 Medium"
    if level == "Low":
        return "🔵 Low"

    return "🟢 Normal"


def downsample_for_chart(df, max_points=2000):
    if df is None or df.empty:
        return df

    if len(df) <= max_points:
        return df

    step = max(1, int(np.ceil(len(df) / max_points)))
    return df.iloc[::step].copy()


def estimate_sample_interval_minutes(data):
    diffs = data["timestamp"].diff().dropna()

    if diffs.empty:
        return 0.0

    seconds = diffs.dt.total_seconds().median()

    if pd.isna(seconds) or seconds <= 0:
        return 0.0

    return seconds / 60


def create_persistent_predictions_fast(data, score_col, threshold, min_duration_minutes, merge_gap_minutes=10):
    """
    Recalculates alarm episodes from timestamp-level model scores.

    This version is vectorised enough for short dashboard date ranges:
    - calculate raw alarm
    - group consecutive alarms into episodes
    - keep only episodes lasting at least min_duration_minutes
    """
    working = data[["timestamp", "y_true", "warning_12h_event_id", score_col]].copy()
    working = working.sort_values("timestamp").reset_index(drop=True)

    working["raw_alarm"] = (working[score_col] >= threshold).astype(int)
    working["persistent_alarm"] = 0

    positive_positions = np.flatnonzero(working["raw_alarm"].values == 1)

    episode_columns = [
        "episode_id",
        "episode_start",
        "episode_end",
        "duration_minutes",
        "positive_points",
        "overlaps_warning_window",
        "event_ids_overlapped"
    ]

    if len(positive_positions) == 0:
        return working["persistent_alarm"].values, pd.DataFrame(columns=episode_columns)

    positives = working.iloc[positive_positions].copy()
    positives["original_position"] = positive_positions

    time_gap = positives["timestamp"].diff()
    new_episode = time_gap.isna() | (time_gap > pd.Timedelta(minutes=merge_gap_minutes))
    positives["episode_id"] = new_episode.cumsum().astype(int)

    sample_interval_minutes = estimate_sample_interval_minutes(working)
    episode_records = []
    accepted_positions = []

    for episode_id, group in positives.groupby("episode_id", sort=False):
        start = group["timestamp"].iloc[0]
        end = group["timestamp"].iloc[-1]

        duration_minutes = max((end - start).total_seconds() / 60, 0) + sample_interval_minutes

        event_ids = sorted([
            x for x in group["warning_12h_event_id"].dropna().astype(str).unique().tolist()
            if x != "None" and x != "nan"
        ])

        overlaps_warning = len(event_ids) > 0

        if duration_minutes >= min_duration_minutes:
            accepted_positions.extend(group["original_position"].tolist())

            episode_records.append({
                "episode_id": int(episode_id),
                "episode_start": start,
                "episode_end": end,
                "duration_minutes": float(duration_minutes),
                "positive_points": int(len(group)),
                "overlaps_warning_window": bool(overlaps_warning),
                "event_ids_overlapped": ", ".join(event_ids) if event_ids else "None"
            })

    if accepted_positions:
        working.loc[accepted_positions, "persistent_alarm"] = 1

    episodes = pd.DataFrame(episode_records, columns=episode_columns)

    return working["persistent_alarm"].values, episodes


def evaluate_alarm_simulation(data, y_pred, episodes):
    working = data.copy()
    working["simulated_alarm"] = np.asarray(y_pred).astype(int)

    start = working["timestamp"].min()
    end = working["timestamp"].max()

    operating_hours = max((end - start).total_seconds() / 3600, 1e-9)
    operating_days = operating_hours / 24

    events_in_range = sorted([
        x for x in working["warning_12h_event_id"].dropna().astype(str).unique().tolist()
        if x != "None" and x != "nan"
    ])

    event_rows = []

    for _, event in FAILURE_EVENTS.iterrows():
        event_id = event["event_id"]

        if event_id not in events_in_range:
            continue

        event_alert_rows = working[
            (working["warning_12h_event_id"].astype(str) == str(event_id)) &
            (working["simulated_alarm"] == 1)
        ]

        detected = len(event_alert_rows) > 0

        if detected:
            first_alert = event_alert_rows["timestamp"].min()
            lead_time_hours = (event["failure_start"] - first_alert).total_seconds() / 3600
        else:
            first_alert = pd.NaT
            lead_time_hours = np.nan

        event_rows.append({
            "event_id": event_id,
            "failure_start": event["failure_start"],
            "detected": detected,
            "first_alert_time": first_alert,
            "lead_time_hours": lead_time_hours
        })

    event_detail = pd.DataFrame(event_rows)

    if event_detail.empty:
        events_evaluated = 0
        events_detected = 0
        failure_detection_rate = np.nan
        median_lead_time_hours = np.nan
    else:
        events_evaluated = len(event_detail)
        events_detected = int(event_detail["detected"].sum())
        failure_detection_rate = events_detected / events_evaluated
        median_lead_time_hours = event_detail[event_detail["detected"] == True]["lead_time_hours"].median()

    if episodes is None or episodes.empty:
        total_alarm_episodes = 0
        true_alarm_episodes = 0
        false_alarm_episodes = 0
        total_alarm_time_hours = 0.0
        median_alarm_duration_minutes = np.nan
    else:
        total_alarm_episodes = int(len(episodes))
        true_alarm_episodes = int(episodes["overlaps_warning_window"].sum())
        false_alarm_episodes = total_alarm_episodes - true_alarm_episodes
        total_alarm_time_hours = float(episodes["duration_minutes"].sum() / 60)
        median_alarm_duration_minutes = float(episodes["duration_minutes"].median())

    false_alarm_episodes_per_day = false_alarm_episodes / operating_days
    total_alarm_time_percentage = total_alarm_time_hours / operating_hours * 100

    y_true = working["y_true"].values
    y_alarm = working["simulated_alarm"].values

    true_positive = int(((y_true == 1) & (y_alarm == 1)).sum())
    false_positive = int(((y_true == 0) & (y_alarm == 1)).sum())
    false_negative = int(((y_true == 1) & (y_alarm == 0)).sum())

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    return {
        "events_evaluated": events_evaluated,
        "events_detected": events_detected,
        "failure_detection_rate": failure_detection_rate,
        "median_lead_time_hours": median_lead_time_hours,
        "total_alarm_episodes": total_alarm_episodes,
        "true_alarm_episodes": true_alarm_episodes,
        "false_alarm_episodes": false_alarm_episodes,
        "false_alarm_episodes_per_day": false_alarm_episodes_per_day,
        "total_alarm_time_hours": total_alarm_time_hours,
        "total_alarm_time_percentage": total_alarm_time_percentage,
        "median_alarm_duration_minutes": median_alarm_duration_minutes,
        "precision": precision,
        "recall": recall
    }, event_detail


prediction_file = read_existing([
    OUTPUT_PREDICTIONS_DIR / "test_predictions_refined_alarm_rules.csv",
    OUTPUT_PREDICTIONS_DIR / "final_risk_predictions_test_set.csv",
    OUTPUT_PREDICTIONS_DIR / "test_predictions_selected_thresholds.csv"
])

validation_model_comparison = load_small_table(str(OUTPUT_TABLES_DIR / "validation_model_comparison_record_level.csv"))
selected_thresholds = load_small_table(str(OUTPUT_TABLES_DIR / "selected_thresholds_from_validation.csv"))
best_model_selection = load_small_table(str(OUTPUT_TABLES_DIR / "best_model_selection_from_validation.csv"))
test_event_results = load_small_table(str(OUTPUT_TABLES_DIR / "test_event_level_results_selected_thresholds.csv"))
alarm_burden_comparison = load_small_table(str(OUTPUT_TABLES_DIR / "alarm_burden_comparison_original_vs_refined.csv"))
test_alarm_burden_results = load_small_table(str(OUTPUT_TABLES_DIR / "test_alarm_burden_sensitivity_results.csv"), max_rows=20000)
selected_alarm_rules = load_small_table(str(OUTPUT_TABLES_DIR / "selected_alarm_burden_rules_validation.csv"), max_rows=2000)
global_shap = load_small_table(str(OUTPUT_TABLES_DIR / "global_shap_feature_importance.csv"), max_rows=100)
local_shap = load_small_table(str(OUTPUT_TABLES_DIR / "local_shap_case_explanations.csv"), max_rows=100)
risk_rules = load_small_table(str(OUTPUT_TABLES_DIR / "risk_framework_rules.csv"), max_rows=200)
risk_distribution = load_small_table(str(OUTPUT_TABLES_DIR / "risk_level_distribution_test_set.csv"), max_rows=200)
framework_summary = load_small_table(str(OUTPUT_TABLES_DIR / "shap_risk_framework_summary.csv"), max_rows=200)

validation_summary = get_validation_summary(best_model_selection)
best_alarm_summary, best_alarm_row = get_best_alarm_summary(test_alarm_burden_results)


st.sidebar.title("🚆 MetroPT Dashboard")
st.sidebar.caption("Predictive Maintenance Decision Support")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model and Alarm Evaluation",
        "Interactive Alarm Simulator",
        "Timeline Explorer",
        "Explainability and Risk Framework"
    ]
)

st.sidebar.divider()
st.sidebar.write("**Project root**")
st.sidebar.code(str(PROJECT_ROOT), language="text")

st.sidebar.write("**Prediction output**")
if prediction_file:
    st.sidebar.success(prediction_file.name)
else:
    st.sidebar.error("Prediction file not found")


st.title("MetroPT-3 Predictive Maintenance Decision Support Dashboard")
st.caption(
    "Interactive dashboard for exploring model scores, alarm thresholds, event detection, "
    "false-alarm burden, explainability outputs and maintenance risk levels."
)


if page == "Overview":
    st.header("1. Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", "MetroPT-3")
    c2.metric("Failure mode", "Air leak")
    c3.metric("Validation-selected model", validation_summary["model"])
    c4.metric("Event-level alarm highlight", best_alarm_summary["model"])

    st.divider()

    st.subheader("Project Aim")
    st.markdown(
        """
        This dashboard presents an explainable predictive maintenance framework for early failure detection
        in a metro train air compressor. The framework combines time-series preprocessing, early-warning label
        generation, model comparison, alarm-threshold analysis, SHAP-based explanation and risk-level decision support.
        """
    )

    st.subheader("Framework Pipeline")
    st.markdown(
        """
        **Raw sensor data → preprocessing → feature engineering → early-warning labels → chronological split → model training → threshold selection → event-level evaluation → alarm-burden analysis → explainability → risk-level decision support**
        """
    )

    st.subheader("Key Model Summary")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Validation model", validation_summary["model"])
    k2.metric("Validation threshold", validation_summary["threshold"])
    k3.metric("Best alarm setting", best_alarm_summary["model"])
    k4.metric("Detected events", best_alarm_summary["events"])

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Alarm threshold", best_alarm_summary["threshold"])
    k6.metric("Persistence duration", best_alarm_summary["duration"])
    k7.metric("Lead time", best_alarm_summary["lead_time"])
    k8.metric("False alarms", best_alarm_summary["false_alarms"])

    st.info(
        "The validation-selected model and the strongest event-level alarm setting are shown separately because "
        "record-level model performance and event-level early-warning performance are not always identical in rare-event predictive maintenance."
    )

    st.subheader("Failure Events Used in the Framework")
    event_display = FAILURE_EVENTS[[
        "event_id",
        "failure_type",
        "failure_start",
        "failure_end",
        "warning_12h_start",
        "warning_12h_end"
    ]].copy()
    st.dataframe(event_display, width="stretch", height=220)

    st.warning(
        "The dashboard is intended as a decision-support prototype and would require further validation before operational use."
    )

elif page == "Model and Alarm Evaluation":
    st.header("2. Model and Alarm Evaluation")

    st.write(
        "This section brings together model comparison, selected thresholds, event-level detection results "
        "and alarm-burden analysis."
    )

    m1, m2, m3, m4 = st.columns(4)
    model_count = len(validation_model_comparison) if validation_model_comparison is not None else 0
    m1.metric("Models compared", model_count)
    m2.metric("Validation-selected model", validation_summary["model"])
    m3.metric("Event-level alarm highlight", best_alarm_summary["model"])
    m4.metric("Best detected events", best_alarm_summary["events"])

    st.divider()

    show_table(validation_model_comparison, "Validation Model Comparison", height=260, max_rows=100)
    show_table(selected_thresholds, "Selected Thresholds from Validation", height=260, max_rows=100)
    show_table(test_event_results, "Test Event-Level Results", height=300, max_rows=100)

    if test_event_results is not None and not test_event_results.empty:
        model_col = first_existing_column(test_event_results, ["model", "model_name", "classifier"])
        rate_col = first_existing_column(test_event_results, ["failure_detection_rate", "event_recall", "recall"])

        if model_col and rate_col:
            st.subheader("Event Detection Rate")
            chart_df = test_event_results[[model_col, rate_col]].copy()
            chart_df[model_col] = chart_df[model_col].astype(str)
            chart_df[rate_col] = pd.to_numeric(chart_df[rate_col], errors="coerce")
            st.bar_chart(chart_df.set_index(model_col)[rate_col])

    st.divider()

    show_table(alarm_burden_comparison, "Original vs Refined Alarm-Burden Comparison", height=320, max_rows=100)

    top_alarm_settings = prepare_top_alarm_settings(test_alarm_burden_results, max_rows=10)
    show_table(top_alarm_settings, "Highest-Ranked Alarm Settings", height=320, max_rows=10)

    if selected_alarm_rules is not None and not selected_alarm_rules.empty:
        show_table(selected_alarm_rules, "Validation-Selected Alarm Rules", height=260, max_rows=100)

    fig_path = OUTPUT_FIGURES_DIR / "test_alarm_burden_original_vs_refined.png"

    if fig_path.exists():
        st.image(str(fig_path), caption="Original vs refined alarm burden", width="stretch")

    st.info(
        "The evaluation focuses on event detection, lead time and false-alarm burden because these measures are more suitable for early-warning maintenance than accuracy alone."
    )

elif page == "Interactive Alarm Simulator":
    st.header("3. Interactive Alarm Simulator")

    st.write(
        "This section recalculates alarm episodes from saved timestamp-level model scores. "
        "It supports exploration of how threshold and persistence settings affect early-warning performance."
    )

    if prediction_file is None:
        st.error("No prediction file found in the outputs/predictions directory.")
        st.stop()

    preview = load_csv(str(prediction_file), nrows=5)

    if preview is None:
        st.stop()

    all_columns = preview.columns.tolist()

    required_cols = ["timestamp", "y_true", "warning_12h_event_id"]
    missing_required = [c for c in required_cols if c not in all_columns]

    if missing_required:
        st.error(f"Missing required columns in prediction file: {missing_required}")
        st.stop()

    score_columns = [c for c in all_columns if c.endswith("_score") or c == "risk_score"]

    if not score_columns:
        st.error("No score columns found in prediction file.")
        st.stop()

    min_ts, max_ts = get_prediction_date_range(str(prediction_file))

    if min_ts is None or max_ts is None:
        st.error("Could not read timestamp range from prediction file.")
        st.stop()

    min_date = min_ts.date()
    max_date = max_ts.date()

    default_start = max(min_date, pd.Timestamp("2020-07-14").date())
    default_end = min(max_date, pd.Timestamp("2020-07-16").date())

    if default_start > default_end:
        default_start = min_date
        default_end = min(max_date, min_date + pd.Timedelta(days=2))

    model_options = {
        normalise_name(col.replace("_score", "")): col
        for col in score_columns
    }

    with st.form("alarm_recalculation_form"):
        st.subheader("Simulation Controls")

        c1, c2, c3 = st.columns(3)

        default_model_index = (
            list(model_options.values()).index("isolation_forest_score")
            if "isolation_forest_score" in model_options.values()
            else 0
        )

        selected_label = c1.selectbox(
            "Model score",
            list(model_options.keys()),
            index=default_model_index
        )

        selected_score_col = model_options[selected_label]

        threshold = c2.slider(
            "Alarm threshold",
            min_value=0.00,
            max_value=1.00,
            value=0.45 if selected_score_col == "isolation_forest_score" else 0.50,
            step=0.01
        )

        min_duration = c3.selectbox(
            "Minimum persistent alarm duration",
            options=[0, 5, 10, 15, 30, 60],
            index=1
        )

        d1, d2, d3 = st.columns(3)

        start_date = d1.date_input(
            "Start date",
            value=default_start,
            min_value=min_date,
            max_value=max_date
        )

        end_date = d2.date_input(
            "End date",
            value=default_end,
            min_value=min_date,
            max_value=max_date
        )

        max_chart_points = d3.slider(
            "Max chart points",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500
        )

        run_button = st.form_submit_button("Run simulation")

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        st.stop()

    if not run_button:
        st.info(
            "Select the model score, threshold, persistence duration and date range, then click **Run simulation**."
        )
        st.stop()

    progress_bar = st.progress(0, text="Starting simulation...")
    status_text = st.empty()

    try:
        status_text.info("Step 1/5: Loading selected data range...")
        progress_bar.progress(10, text="Loading selected data range")

        usecols = tuple(required_cols + [selected_score_col])

        selected_data = load_prediction_date_range(
            str(prediction_file),
            usecols,
            str(start_date),
            str(end_date)
        )

        if selected_data.empty:
            progress_bar.empty()
            status_text.warning("No data found for the selected date range.")
            st.stop()

        status_text.info(f"Step 2/5: Processing {len(selected_data):,} selected records...")
        progress_bar.progress(35, text="Processing selected records")

        y_sim, episodes = create_persistent_predictions_fast(
            selected_data,
            score_col=selected_score_col,
            threshold=threshold,
            min_duration_minutes=min_duration
        )

        status_text.info("Step 3/5: Calculating detection and alarm-burden metrics...")
        progress_bar.progress(60, text="Calculating alarm-burden metrics")

        summary, event_detail = evaluate_alarm_simulation(selected_data, y_sim, episodes)

        status_text.info("Step 4/5: Preparing visual outputs...")
        progress_bar.progress(80, text="Preparing visual outputs")

        plot_data = selected_data[["timestamp", selected_score_col]].copy()
        plot_data["simulated_alarm"] = y_sim
        plot_data = downsample_for_chart(plot_data, max_points=max_chart_points)
        plot_data = plot_data.set_index("timestamp")

        progress_bar.progress(100, text="Analysis complete")
        status_text.success("Step 5/5: Analysis complete.")

    except Exception as exc:
        progress_bar.empty()
        status_text.error(f"Simulation failed: {exc}")
        st.stop()

    st.subheader("Simulation Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Events detected", f"{summary['events_detected']}/{summary['events_evaluated']}")
    m2.metric("Lead time", format_number(summary["median_lead_time_hours"], 2) + " h")
    m3.metric("False alarms/day", format_number(summary["false_alarm_episodes_per_day"], 2))
    m4.metric("Alarm time", format_number(summary["total_alarm_time_hours"], 2) + " h")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Alarm time %", format_number(summary["total_alarm_time_percentage"], 2) + "%")
    m6.metric("Total episodes", str(summary["total_alarm_episodes"]))
    m7.metric("Precision", format_number(summary["precision"], 3))
    m8.metric("Recall", format_number(summary["recall"], 3))

    st.caption(
        f"Rows evaluated: {len(selected_data):,}. "
        f"Rows plotted after downsampling: {len(plot_data):,}."
    )

    st.subheader("Selected Score Timeline")
    st.line_chart(plot_data[[selected_score_col]])

    st.subheader("Simulated Alarm Timeline")
    st.line_chart(plot_data[["simulated_alarm"]])

    with st.expander("Event detection detail"):
        show_table(event_detail, height=240, max_rows=100)

    with st.expander("Accepted alarm episodes"):
        show_table(episodes, height=320, max_rows=300)

    st.success(
        "Alarm analysis completed for the selected model score, threshold, persistence duration and date range."
    )


elif page == "Timeline Explorer":
    st.header("4. Timeline Explorer")

    st.write(
        "This section provides timeline-based exploration of model scores, alarm outputs and warning labels "
        "across the selected date range."
    )

    if prediction_file is None:
        st.error("No prediction file found in the outputs/predictions directory.")
        st.stop()

    preview = load_csv(str(prediction_file), nrows=5)

    if preview is None:
        st.stop()

    all_columns = preview.columns.tolist()

    if "timestamp" not in all_columns:
        st.error("The prediction file does not contain a timestamp column.")
        st.stop()

    min_ts, max_ts = get_prediction_date_range(str(prediction_file))

    if min_ts is None or max_ts is None:
        st.error("Could not read the timestamp range from the prediction file.")
        st.stop()

    min_date = min_ts.date()
    max_date = max_ts.date()

    score_columns = [
        c for c in all_columns
        if c.endswith("_score") or c == "risk_score"
    ]

    alarm_columns = [
        c for c in all_columns
        if "prediction" in c.lower()
        or "alarm_flag" in c.lower()
        or "alarm" in c.lower()
        or c == "y_true"
        or "warning" in c.lower()
    ]

    other_columns = [
        c for c in all_columns
        if c not in ["timestamp"] + score_columns + alarm_columns
    ]

    with st.form("timeline_full_range_form"):
        st.subheader("Timeline Controls")

        d1, d2 = st.columns(2)

        start_date = d1.date_input(
            "Start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="timeline_free_start"
        )

        end_date = d2.date_input(
            "End date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="timeline_free_end"
        )

        c1, c2, c3 = st.columns(3)

        selected_score_columns = c1.multiselect(
            "Score columns to plot",
            options=score_columns,
            default=score_columns[:1] if score_columns else []
        )

        selected_alarm_columns = c2.multiselect(
            "Alarm / label columns to plot",
            options=alarm_columns,
            default=["y_true"] if "y_true" in alarm_columns else alarm_columns[:1]
        )

        selected_extra_columns = c3.multiselect(
            "Extra columns to include in preview table",
            options=other_columns,
            default=[]
        )

        max_chart_points = st.slider(
            "Maximum chart points after downsampling",
            min_value=500,
            max_value=10000,
            value=3000,
            step=500
        )

        load_timeline = st.form_submit_button("Load timeline")

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        st.stop()

    if not load_timeline:
        st.info(
            "Select the required date range and columns, then click **Load timeline**."
        )
        st.stop()

    selected_columns = (
        ["timestamp"]
        + selected_score_columns
        + selected_alarm_columns
        + selected_extra_columns
    )

    selected_columns = list(dict.fromkeys(selected_columns))

    if len(selected_columns) == 1:
        st.warning("Please select at least one score, alarm, label, or extra column.")
        st.stop()

    progress_bar = st.progress(0, text="Starting timeline load...")
    status_text = st.empty()

    try:
        status_text.info("Step 1/4: Reading selected date range and columns...")
        progress_bar.progress(20, text="Reading selected data")

        timeline = load_prediction_date_range(
            str(prediction_file),
            tuple(selected_columns),
            str(start_date),
            str(end_date)
        )

        if timeline.empty:
            progress_bar.empty()
            status_text.warning("No timeline data found for the selected range.")
            st.stop()

        status_text.info(f"Step 2/4: Loaded {len(timeline):,} rows. Preparing chart data...")
        progress_bar.progress(50, text="Preparing chart data")

        plot_columns = [
            c for c in selected_score_columns + selected_alarm_columns
            if c in timeline.columns
        ]

        if not plot_columns:
            progress_bar.empty()
            status_text.warning("No selected chart columns were found in the loaded data.")
            st.stop()

        plot_data = timeline[["timestamp"] + plot_columns].copy()
        plot_data = plot_data.dropna(subset=["timestamp"])

        status_text.info("Step 3/4: Downsampling chart for smoother display...")
        progress_bar.progress(75, text="Downsampling chart")

        plot_data = downsample_for_chart(plot_data, max_points=max_chart_points)
        plot_data = plot_data.set_index("timestamp")

        progress_bar.progress(100, text="Timeline loaded")
        status_text.success("Step 4/4: Timeline ready.")

    except Exception as exc:
        progress_bar.empty()
        status_text.error(f"Timeline loading failed: {exc}")
        st.stop()

    st.caption(
        f"Date range: {start_date} to {end_date}. "
        f"Rows loaded: {len(timeline):,}. "
        f"Rows plotted after downsampling: {len(plot_data):,}."
    )

    if selected_score_columns:
        available_score_cols = [
            c for c in selected_score_columns
            if c in plot_data.columns
        ]

        if available_score_cols:
            st.subheader("Selected Score Timeline")
            st.line_chart(plot_data[available_score_cols])

    if selected_alarm_columns:
        available_alarm_cols = [
            c for c in selected_alarm_columns
            if c in plot_data.columns
        ]

        if available_alarm_cols:
            st.subheader("Selected Alarm / Label Timeline")
            st.line_chart(plot_data[available_alarm_cols])

    with st.expander("Loaded timeline data preview"):
        st.dataframe(timeline.head(300), width="stretch", height=360)

    st.success(
        "Timeline visualisation completed for the selected date range and columns."
    )


elif page == "Explainability and Risk Framework":
    st.header("5. Explainability and Risk Framework")

    st.write(
        "This section combines SHAP-based model explanation with the maintenance-facing risk framework."
    )

    st.subheader("SHAP Explainability")

    show_table(global_shap, "Top Global SHAP Features", height=320, max_rows=50)

    c1, c2 = st.columns(2)

    shap_bar_path = read_existing([
        OUTPUT_FIGURES_DIR / "global_shap_top_features.png",
        OUTPUT_SHAP_DIR / "global_shap_top_features.png"
    ])

    shap_summary_path = read_existing([
        OUTPUT_FIGURES_DIR / "shap_summary_plot.png",
        OUTPUT_SHAP_DIR / "shap_summary_plot.png"
    ])

    with c1:
        if shap_bar_path:
            st.image(str(shap_bar_path), caption="Global SHAP Top Features", width="stretch")
        else:
            st.info("Global SHAP feature plot not found.")

    with c2:
        if shap_summary_path:
            st.image(str(shap_summary_path), caption="SHAP Summary Plot", width="stretch")
        else:
            st.info("SHAP summary plot not found.")

    show_table(local_shap, "Local SHAP Case Explanations", height=320, max_rows=50)

    st.warning(
        "SHAP explains model behaviour and should not be interpreted as direct proof of physical causality."
    )

    st.divider()

    st.subheader("Risk-Level Decision Support")

    show_table(risk_rules, "Risk Framework Rules", height=260, max_rows=100)

    if risk_distribution is not None and not risk_distribution.empty:
        st.subheader("Risk-Level Distribution")

        risk_plot = risk_distribution.copy()

        if "risk_level" in risk_plot.columns and "count" in risk_plot.columns:
            risk_plot["display_level"] = risk_plot["risk_level"].apply(risk_badge)
            st.bar_chart(risk_plot.set_index("display_level")["count"])

        show_table(risk_distribution, "Risk Distribution Table", height=220, max_rows=100)

    fig_path = OUTPUT_FIGURES_DIR / "risk_level_distribution_test_set.png"

    if fig_path.exists():
        st.image(str(fig_path), caption="Risk-Level Distribution", width="stretch")

    show_table(framework_summary, "SHAP and Risk Framework Summary", height=280, max_rows=100)

    st.info(
        "The risk framework converts model outputs into maintenance-facing categories."
    )