import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="MetroPT Predictive Maintenance Dashboard",
    page_icon="🚆",
    layout="wide"
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
APP_PATH = Path(__file__).resolve()
PROJECT_ROOT = APP_PATH.parent.parent

OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUTPUT_SHAP_DIR = PROJECT_ROOT / "outputs" / "shap"

# ---------------------------------------------------------------------
# General helper functions
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str, usecols=None, nrows=None):
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        return pd.read_csv(file_path, usecols=usecols, nrows=nrows)
    except Exception as exc:
        st.warning(f"Could not load {file_path.name}: {exc}")
        return None


@st.cache_data(show_spinner=True)
def load_prediction_subset(path: str, usecols_tuple):
    file_path = Path(path)
    usecols = list(usecols_tuple)
    df = pd.read_csv(file_path, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if "warning_12h_event_id" in df.columns:
        df["warning_12h_event_id"] = df["warning_12h_event_id"].fillna("None").astype(str)

    if "y_true" in df.columns:
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int)

    return df


@st.cache_data(show_spinner=False)
def get_prediction_date_range(path: str):
    file_path = Path(path)
    min_ts = None
    max_ts = None

    for chunk in pd.read_csv(file_path, usecols=["timestamp"], chunksize=100_000):
        ts = pd.to_datetime(chunk["timestamp"], errors="coerce").dropna()
        if ts.empty:
            continue
        cmin = ts.min()
        cmax = ts.max()
        min_ts = cmin if min_ts is None else min(min_ts, cmin)
        max_ts = cmax if max_ts is None else max(max_ts, cmax)

    return min_ts, max_ts


@st.cache_data(show_spinner=True)
def load_prediction_date_range(path: str, usecols_tuple, start_date_str: str, end_date_str: str):
    file_path = Path(path)
    usecols = list(usecols_tuple)
    start_ts = pd.Timestamp(start_date_str)
    end_ts = pd.Timestamp(end_date_str) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    pieces = []

    for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=100_000):
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


def downsample_for_chart(df, max_points=5000):
    if len(df) <= max_points:
        return df
    step = max(1, int(len(df) / max_points))
    return df.iloc[::step].copy()


def file_status(path: Path):
    return "✅ Found" if path.exists() else "❌ Missing"


def format_number(value, decimals=3):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{value:,.{decimals}f}"
        return str(value)
    except Exception:
        return str(value)


def metric_from_df(df, column, default="N/A", decimals=3):
    if df is None or df.empty or column not in df.columns:
        return default
    return format_number(df[column].iloc[0], decimals=decimals)


def show_table(df, title=None, height=360):
    if title:
        st.subheader(title)
    if df is None or df.empty:
        st.info("No data available for this table.")
    else:
        st.dataframe(df, width="stretch", height=height)


def read_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


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


# ---------------------------------------------------------------------
# Alarm simulation helper functions
# ---------------------------------------------------------------------
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


def estimate_sample_interval_minutes(data):
    diffs = data["timestamp"].diff().dropna()
    if diffs.empty:
        return 0.0
    seconds = diffs.dt.total_seconds().median()
    if pd.isna(seconds) or seconds <= 0:
        return 0.0
    return seconds / 60


def create_persistent_predictions(data, score_col, threshold, min_duration_minutes, merge_gap_minutes=10):
    working = data[["timestamp", "y_true", "warning_12h_event_id", score_col]].copy()
    working["raw_alarm"] = (working[score_col] >= threshold).astype(int)
    working["persistent_alarm"] = 0

    positives = working[working["raw_alarm"] == 1].copy()

    episode_columns = [
        "episode_id",
        "episode_start",
        "episode_end",
        "duration_minutes",
        "positive_points",
        "overlaps_warning_window",
        "event_ids_overlapped",
        "accepted_by_persistence"
    ]

    if positives.empty:
        return working["persistent_alarm"].values, pd.DataFrame(columns=episode_columns)

    positives = positives.sort_values("timestamp").reset_index()
    positives = positives.rename(columns={"index": "original_index"})

    time_gap = positives["timestamp"].diff()
    new_episode = time_gap.isna() | (time_gap > pd.Timedelta(minutes=merge_gap_minutes))
    positives["episode_id"] = new_episode.cumsum()

    sample_interval_minutes = estimate_sample_interval_minutes(data)
    episode_records = []

    for episode_id, group in positives.groupby("episode_id"):
        start = group["timestamp"].min()
        end = group["timestamp"].max()
        duration_minutes = max((end - start).total_seconds() / 60, 0) + sample_interval_minutes

        event_ids = sorted([
            x for x in group["warning_12h_event_id"].dropna().unique().tolist()
            if str(x) != "None"
        ])

        overlaps_warning = len(event_ids) > 0
        accepted = duration_minutes >= min_duration_minutes

        if accepted:
            original_indices = group["original_index"].tolist()
            working.loc[original_indices, "persistent_alarm"] = 1

        episode_records.append({
            "episode_id": int(episode_id),
            "episode_start": start,
            "episode_end": end,
            "duration_minutes": float(duration_minutes),
            "positive_points": int(len(group)),
            "overlaps_warning_window": bool(overlaps_warning),
            "event_ids_overlapped": ", ".join(event_ids) if event_ids else "None",
            "accepted_by_persistence": bool(accepted)
        })

    episodes = pd.DataFrame(episode_records, columns=episode_columns)
    episodes = episodes[episodes["accepted_by_persistence"] == True].copy()

    return working["persistent_alarm"].values, episodes


def evaluate_alarm_simulation(data, y_pred, episodes):
    working = data.copy()
    working["simulated_alarm"] = np.asarray(y_pred).astype(int)

    start = working["timestamp"].min()
    end = working["timestamp"].max()
    operating_hours = max((end - start).total_seconds() / 3600, 1e-9)
    operating_days = operating_hours / 24

    events_in_range = sorted([
        x for x in working["warning_12h_event_id"].dropna().unique().tolist()
        if str(x) != "None"
    ])

    event_rows = []

    for _, event in FAILURE_EVENTS.iterrows():
        event_id = event["event_id"]
        if event_id not in events_in_range:
            continue

        event_alert_rows = working[
            (working["warning_12h_event_id"] == event_id) &
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
        failure_detection_rate = events_detected / events_evaluated if events_evaluated else np.nan
        median_lead_time_hours = event_detail[event_detail["detected"] == True]["lead_time_hours"].median()

    if episodes.empty:
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

    summary = {
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
    }

    return summary, event_detail


# ---------------------------------------------------------------------
# Load key files
# ---------------------------------------------------------------------
validation_model_comparison = load_csv(OUTPUT_TABLES_DIR / "validation_model_comparison_record_level.csv")
selected_thresholds = load_csv(OUTPUT_TABLES_DIR / "selected_thresholds_from_validation.csv")
best_model_selection = load_csv(OUTPUT_TABLES_DIR / "best_model_selection_from_validation.csv")
test_event_results = load_csv(OUTPUT_TABLES_DIR / "test_event_level_results_selected_thresholds.csv")
alarm_burden_comparison = load_csv(OUTPUT_TABLES_DIR / "alarm_burden_comparison_original_vs_refined.csv")
test_alarm_burden_results = load_csv(OUTPUT_TABLES_DIR / "test_alarm_burden_sensitivity_results.csv")
global_shap = load_csv(OUTPUT_TABLES_DIR / "global_shap_feature_importance.csv")
local_shap = load_csv(OUTPUT_TABLES_DIR / "local_shap_case_explanations.csv")
risk_rules = load_csv(OUTPUT_TABLES_DIR / "risk_framework_rules.csv")
risk_distribution = load_csv(OUTPUT_TABLES_DIR / "risk_level_distribution_test_set.csv")
framework_summary = load_csv(OUTPUT_TABLES_DIR / "shap_risk_framework_summary.csv")
threshold_summary = load_csv(OUTPUT_TABLES_DIR / "threshold_event_evaluation_summary.csv")
alarm_summary = load_csv(OUTPUT_TABLES_DIR / "alarm_burden_sensitivity_summary.csv")

prediction_file = read_existing([
    OUTPUT_PREDICTIONS_DIR / "test_predictions_refined_alarm_rules.csv",
    OUTPUT_PREDICTIONS_DIR / "final_risk_predictions_test_set.csv",
    OUTPUT_PREDICTIONS_DIR / "test_predictions_selected_thresholds.csv"
])

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("🚆 MetroPT Dashboard")
st.sidebar.caption("Explainable Predictive Maintenance Framework")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Interactive Alarm Simulator",
        "Model Comparison",
        "Alarm Timeline",
        "Alarm Burden Analysis",
        "SHAP Explainability",
        "Risk Framework",
        "Files and Evidence"
    ]
)

st.sidebar.divider()
st.sidebar.write("**Project root**")
st.sidebar.code(str(PROJECT_ROOT), language="text")

st.sidebar.write("**Main prediction file**")
if prediction_file:
    st.sidebar.success(prediction_file.name)
else:
    st.sidebar.error("No prediction file found")

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("MetroPT-3 Predictive Maintenance Decision Support Dashboard")
st.caption(
    "A lightweight dashboard for presenting model comparison, interactive alarm settings, "
    "event-level evaluation, alarm-burden analysis, SHAP explainability and risk-based recommendations."
)

# ---------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------
if page == "Overview":
    st.header("1. Project Overview")

    best_model = "N/A"
    best_threshold = "N/A"

    if best_model_selection is not None and not best_model_selection.empty:
        best_model = str(best_model_selection.get("model", pd.Series(["N/A"])).iloc[0])
        best_threshold = metric_from_df(best_model_selection, "threshold", decimals=2)

    refined_isolation = None
    if alarm_burden_comparison is not None and not alarm_burden_comparison.empty:
        temp = alarm_burden_comparison[
            (alarm_burden_comparison["approach"].astype(str).str.contains("refined", case=False, na=False)) &
            (alarm_burden_comparison["model"].astype(str).str.contains("Isolation", case=False, na=False))
        ]
        if not temp.empty:
            refined_isolation = temp.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", "MetroPT-3")
    c2.metric("Best validation model", best_model)
    c3.metric("Selected threshold", best_threshold)
    c4.metric("Framework status", "Prototype")

    st.divider()

    st.subheader("Key Final Finding")

    if refined_isolation is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test event detected", f"{int(refined_isolation.get('events_detected', 0))}/{int(refined_isolation.get('events_evaluated', 0))}")
        m2.metric("Lead time", f"{float(refined_isolation.get('median_lead_time_hours', 0)):.2f} h")
        m3.metric("Alarm time", f"{float(refined_isolation.get('total_alarm_time_hours', 0)):.2f} h")
        m4.metric("Alarm time %", f"{float(refined_isolation.get('total_alarm_time_percentage', 0)):.2f}%")

        st.info(
            "The refined alarm-burden analysis shows that the persistence-based Isolation Forest alarm rule "
            "detected the test failure event while substantially reducing total alarm time."
        )
    else:
        st.info(
            "The dashboard could not locate the refined Isolation Forest row. "
            "Run Step 7B or check alarm_burden_comparison_original_vs_refined.csv."
        )

    st.subheader("Framework Pipeline")
    st.markdown(
        """
        **Raw MetroPT-3 data → preprocessing → feature engineering → early-warning labels → chronological split → model training → threshold selection → event-level evaluation → alarm-burden sensitivity → SHAP explanation → risk-level decision support**
        """
    )

    st.warning(
        "This dashboard presents a research prototype. It should not be interpreted as a production-ready "
        "railway maintenance system. The results require further validation with more failure events and engineering review."
    )

# ---------------------------------------------------------------------
# Interactive Alarm Simulator page
# ---------------------------------------------------------------------
elif page == "Interactive Alarm Simulator":
    st.header("2. Interactive Alarm Simulator")
    st.write(
        "Change the model, threshold, persistence duration and date range, then press **Run simulation**. "
        "This page uses cached chunks so it does not load the full prediction file into memory."
    )

    if prediction_file is None:
        st.error("No prediction file found. Run Step 6, Step 7 and Step 7B first.")
        st.stop()

    preview = load_csv(prediction_file, nrows=5)
    if preview is None:
        st.stop()

    all_columns = preview.columns.tolist()
    score_columns = [c for c in all_columns if c.endswith("_score") or c == "risk_score"]

    if not score_columns:
        st.error("No score columns found in prediction file.")
        st.stop()

    required_cols = ["timestamp", "y_true", "warning_12h_event_id"]
    missing_required = [c for c in required_cols if c not in all_columns]
    if missing_required:
        st.error(f"Missing required columns for simulation: {missing_required}")
        st.stop()

    min_ts, max_ts = get_prediction_date_range(str(prediction_file))

    if min_ts is None or max_ts is None:
        st.error("Could not read timestamp range from prediction file.")
        st.stop()

    min_date = min_ts.date()
    max_date = max_ts.date()

    # Default to the F4 failure area if it exists in the prediction period.
    default_start = max(min_date, pd.Timestamp("2020-07-14").date())
    default_end = min(max_date, pd.Timestamp("2020-07-16").date())

    if default_start > default_end:
        default_start = min_date
        default_end = min(max_date, min_date + pd.Timedelta(days=2))

    model_options = {
        col.replace("_score", "").replace("_", " ").title(): col
        for col in score_columns
    }

    with st.form("alarm_simulator_form"):
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
            min_value=1000,
            max_value=10000,
            value=4000,
            step=1000
        )

        run_button = st.form_submit_button("Run simulation")

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        st.stop()

    if not run_button:
        st.info(
            "Choose settings and press **Run simulation**. "
            "Default dates focus around the final test failure event to keep the page fast."
        )
        st.stop()

    usecols = tuple(required_cols + [selected_score_col])

    filtered = load_prediction_date_range(
        str(prediction_file),
        usecols,
        str(start_date),
        str(end_date)
    )

    if filtered.empty:
        st.warning("No rows available for the selected date range.")
        st.stop()

    with st.spinner("Calculating simulated alarms..."):
        y_sim, episodes = create_persistent_predictions(
            filtered,
            score_col=selected_score_col,
            threshold=threshold,
            min_duration_minutes=min_duration
        )

        summary, event_detail = evaluate_alarm_simulation(filtered, y_sim, episodes)

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

    st.caption(f"Rows evaluated: {len(filtered):,}. Chart is downsampled for speed.")

    st.subheader("Risk Score and Simulated Alarm Timeline")

    plot_data = filtered[["timestamp", selected_score_col]].copy()
    plot_data["simulated_alarm"] = y_sim
    plot_data = downsample_for_chart(plot_data, max_points=max_chart_points)
    plot_data = plot_data.set_index("timestamp")

    st.line_chart(plot_data[[selected_score_col]])
    st.line_chart(plot_data[["simulated_alarm"]])

    with st.expander("Show event detection detail"):
        show_table(event_detail, height=240)

    with st.expander("Show accepted alarm episodes"):
        show_table(episodes, height=320)

    st.info(
        "This page is intentionally limited to model scores, threshold, persistence and date range. "
        "It keeps the dashboard interactive without adding messy raw dataset upload or retraining."
    )

# ---------------------------------------------------------------------
# Model Comparison page
# ---------------------------------------------------------------------
elif page == "Model Comparison":
    st.header("3. Model Comparison")

    show_table(validation_model_comparison, "Validation Model Comparison", height=300)
    show_table(selected_thresholds, "Selected Thresholds from Validation", height=300)
    show_table(test_event_results, "Original Test Event-Level Results", height=350)

    if test_event_results is not None and not test_event_results.empty:
        st.subheader("Test Event Detection by Model")

        if "model" in test_event_results.columns and "failure_detection_rate" in test_event_results.columns:
            chart_df = test_event_results[["model", "failure_detection_rate"]].copy()
            st.bar_chart(chart_df.set_index("model")["failure_detection_rate"])

    st.info(
        "Use this page to explain why record-level metrics alone are not enough. "
        "For predictive maintenance, event detection and false-alarm burden are more meaningful than accuracy."
    )

# ---------------------------------------------------------------------
# Alarm Timeline page
# ---------------------------------------------------------------------
elif page == "Alarm Timeline":
    st.header("4. Alarm Timeline")

    if prediction_file is None:
        st.error("No prediction file found. Run Step 7, Step 7B or Step 8 first.")
    else:
        st.write(f"Using prediction file: `{prediction_file.name}`")

        preview = load_csv(prediction_file, nrows=5)
        if preview is None:
            st.stop()

        all_columns = preview.columns.tolist()
        score_columns = [c for c in all_columns if c.endswith("_score") or c == "risk_score"]
        alarm_columns = [
            c for c in all_columns
            if "prediction" in c.lower() or "alarm_flag" in c.lower()
        ]

        col1, col2, col3 = st.columns(3)
        score_col = col1.selectbox("Risk/score column", score_columns, index=0)
        alarm_col = col2.selectbox("Alarm/prediction column", alarm_columns, index=0 if alarm_columns else None)
        max_rows = col3.slider("Rows to plot", min_value=1000, max_value=100000, value=20000, step=1000)

        usecols = ["timestamp", "y_true"]
        if "warning_12h_event_id" in all_columns:
            usecols.append("warning_12h_event_id")
        if score_col and score_col not in usecols:
            usecols.append(score_col)
        if alarm_col and alarm_col not in usecols:
            usecols.append(alarm_col)

        timeline = load_csv(prediction_file, usecols=usecols, nrows=max_rows)

        if timeline is not None and not timeline.empty:
            timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], errors="coerce")
            timeline = timeline.dropna(subset=["timestamp"]).sort_values("timestamp")

            st.subheader("Risk Score Over Time")
            st.line_chart(timeline.set_index("timestamp")[score_col])

            if alarm_col:
                st.subheader("Alarm Flags Over Time")
                st.line_chart(timeline.set_index("timestamp")[alarm_col])

            st.subheader("Preview")
            st.dataframe(timeline.head(100), width="stretch")

# ---------------------------------------------------------------------
# Alarm Burden Analysis page
# ---------------------------------------------------------------------
elif page == "Alarm Burden Analysis":
    st.header("5. Alarm Burden Analysis")

    show_table(alarm_burden_comparison, "Original vs Refined Alarm Burden Comparison", height=420)
    show_table(test_alarm_burden_results, "Step 7B Test Alarm-Burden Sensitivity Results", height=420)

    if alarm_burden_comparison is not None and not alarm_burden_comparison.empty:
        st.subheader("Total Alarm Time Comparison")
        plot_df = alarm_burden_comparison.dropna(subset=["total_alarm_time_hours"]).copy()
        if not plot_df.empty:
            plot_df["label"] = plot_df["approach"].astype(str) + " | " + plot_df["model"].astype(str)
            st.bar_chart(plot_df.set_index("label")["total_alarm_time_hours"])

        st.subheader("False Alarm Episodes per Day")
        if "false_alarm_episodes_per_day" in plot_df.columns:
            st.bar_chart(plot_df.set_index("label")["false_alarm_episodes_per_day"])

    fig_path = OUTPUT_FIGURES_DIR / "test_alarm_burden_original_vs_refined.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Original vs refined alarm burden", width="stretch")

    st.success(
        "This page shows that the framework evaluates not only whether a failure was detected, "
        "but also whether the alarm burden is manageable."
    )

# ---------------------------------------------------------------------
# SHAP page
# ---------------------------------------------------------------------
elif page == "SHAP Explainability":
    st.header("6. SHAP Explainability")

    show_table(global_shap.head(30) if global_shap is not None else None, "Top Global SHAP Features", height=420)

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

    show_table(local_shap, "Local SHAP Case Explanations", height=420)

    st.warning(
        "SHAP values explain the model's behaviour. They should not be interpreted as proof of physical causality."
    )

# ---------------------------------------------------------------------
# Risk Framework page
# ---------------------------------------------------------------------
elif page == "Risk Framework":
    st.header("7. Risk-Level Decision Support")

    show_table(risk_rules, "Risk Framework Rules", height=300)

    if risk_distribution is not None and not risk_distribution.empty:
        st.subheader("Risk-Level Distribution on Test Set")
        risk_plot = risk_distribution.copy()
        if "risk_level" in risk_plot.columns and "count" in risk_plot.columns:
            risk_plot["display_level"] = risk_plot["risk_level"].apply(risk_badge)
            st.bar_chart(risk_plot.set_index("display_level")["count"])
        show_table(risk_distribution, "Risk Distribution Table", height=250)

    fig_path = OUTPUT_FIGURES_DIR / "risk_level_distribution_test_set.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Risk-Level Distribution", width="stretch")

    show_table(framework_summary, "SHAP and Risk Framework Summary", height=350)

    st.info(
        "The risk framework converts model outputs into maintenance-facing categories. "
        "It should be described as a prototype decision-support layer, not as a validated operational rule."
    )

# ---------------------------------------------------------------------
# Files and Evidence page
# ---------------------------------------------------------------------
elif page == "Files and Evidence":
    st.header("8. Files and Evidence")

    evidence_files = [
        OUTPUT_TABLES_DIR / "validation_model_comparison_record_level.csv",
        OUTPUT_TABLES_DIR / "selected_thresholds_from_validation.csv",
        OUTPUT_TABLES_DIR / "best_model_selection_from_validation.csv",
        OUTPUT_TABLES_DIR / "test_event_level_results_selected_thresholds.csv",
        OUTPUT_TABLES_DIR / "validation_alarm_burden_sensitivity_search.csv",
        OUTPUT_TABLES_DIR / "selected_alarm_burden_rules_validation.csv",
        OUTPUT_TABLES_DIR / "test_alarm_burden_sensitivity_results.csv",
        OUTPUT_TABLES_DIR / "alarm_burden_comparison_original_vs_refined.csv",
        OUTPUT_TABLES_DIR / "global_shap_feature_importance.csv",
        OUTPUT_TABLES_DIR / "local_shap_case_explanations.csv",
        OUTPUT_TABLES_DIR / "risk_framework_rules.csv",
        OUTPUT_TABLES_DIR / "risk_level_distribution_test_set.csv",
        OUTPUT_TABLES_DIR / "shap_risk_framework_summary.csv",
        OUTPUT_PREDICTIONS_DIR / "test_predictions_refined_alarm_rules.csv",
        OUTPUT_PREDICTIONS_DIR / "final_risk_predictions_test_set.csv",
    ]

    evidence_df = pd.DataFrame({
        "file": [str(p.relative_to(PROJECT_ROOT)) for p in evidence_files],
        "status": [file_status(p) for p in evidence_files]
    })

    st.dataframe(evidence_df, width="stretch", height=500)

    st.subheader("Recommended Dissertation Wording")
    st.markdown(
        """
        The dashboard was developed as a lightweight demonstration layer for the predictive maintenance framework.
        It does not retrain the models and does not accept arbitrary external datasets. Instead, it visualises and
        interactively explores the saved outputs from the implemented pipeline, including model comparison,
        event-level alarm evaluation, alarm-burden sensitivity analysis, SHAP explanations and risk-based
        maintenance recommendations.
        """
    )