import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    go = None
    PLOTLY_AVAILABLE = False

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="MetroPT Predictive Maintenance Dashboard",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_PATH = Path(__file__).resolve()
PROJECT_ROOT = APP_PATH.parent.parent

OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs_deploy"
if not OUTPUT_BASE_DIR.exists():
    OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs"

OUTPUT_TABLES_DIR = OUTPUT_BASE_DIR / "tables"
OUTPUT_FIGURES_DIR = OUTPUT_BASE_DIR / "figures"
OUTPUT_PREDICTIONS_DIR = OUTPUT_BASE_DIR / "predictions"
OUTPUT_SHAP_DIR = OUTPUT_BASE_DIR / "shap"

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


def inject_visual_theme():
    """Apply a vibrant professional industrial dashboard theme."""
    st.markdown(
        """
        <style>
        :root {
            --bg-0: #050b14;
            --bg-1: #081321;
            --panel: #0d1b2b;
            --panel-2: #10243a;
            --border: rgba(111, 163, 214, 0.20);
            --text: #edf6ff;
            --muted: #98abc3;
            --cyan: #21d4fd;
            --blue: #3b82f6;
            --purple: #8b5cf6;
            --pink: #ec4899;
            --green: #20d69f;
            --amber: #ffbe3f;
            --red: #ff5573;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 16% 0%, rgba(33, 212, 253, 0.12), transparent 30rem),
                radial-gradient(circle at 88% 10%, rgba(139, 92, 246, 0.12), transparent 32rem),
                radial-gradient(circle at 55% 88%, rgba(32, 214, 159, 0.06), transparent 28rem),
                linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: rgba(5, 11, 20, 0.78);
            backdrop-filter: blur(14px);
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }

        .block-container {
            max-width: 1540px;
            padding-top: 1.25rem;
            padding-bottom: 3.2rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            width: 320px !important;
            min-width: 320px !important;
            background:
                radial-gradient(circle at 30% 2%, rgba(33,212,253,0.15), transparent 15rem),
                linear-gradient(180deg, #08111e 0%, #0b1727 50%, #070e18 100%);
            border-right: 1px solid rgba(65, 158, 255, 0.20);
        }
        section[data-testid="stSidebar"] > div { width: 320px !important; }
        [data-testid="stSidebar"] * { color: #eaf4ff; }

        .metro-brand {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(33,212,253,0.22);
            background: linear-gradient(135deg, rgba(33,212,253,0.13), rgba(139,92,246,0.14));
            border-radius: 20px;
            padding: 1.05rem;
            margin: 0.30rem 0 0.85rem 0;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 14px 36px rgba(0,0,0,0.22);
        }
        .metro-brand:after {
            content: "";
            position: absolute;
            width: 100px;
            height: 100px;
            border-radius: 999px;
            right: -36px;
            top: -42px;
            background: radial-gradient(circle, rgba(33,212,253,0.35), transparent 70%);
        }
        .metro-brand-title { font-size: 1.07rem; font-weight: 820; letter-spacing: -0.01em; margin-bottom: 0.18rem; }
        .metro-brand-subtitle { color: #9bb3cd !important; font-size: 0.77rem; line-height: 1.35; }

        .metro-status {
            display: inline-flex;
            align-items: center;
            gap: 0.48rem;
            width: 100%;
            box-sizing: border-box;
            padding: 0.58rem 0.72rem;
            margin-bottom: 0.65rem;
            border-radius: 12px;
            border: 1px solid rgba(32,214,159,0.25);
            background: linear-gradient(90deg, rgba(32,214,159,0.12), rgba(33,212,253,0.06));
            color: #c5ffeb !important;
            font-size: 0.76rem;
            font-weight: 700;
        }
        .metro-status-dot {
            width: 8px;
            height: 8px;
            flex: 0 0 auto;
            border-radius: 999px;
            background: var(--green);
            box-shadow: 0 0 0 5px rgba(32,214,159,0.10), 0 0 18px rgba(32,214,159,0.50);
            animation: metroPulse 2s ease-in-out infinite;
        }
        @keyframes metroPulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: .55; transform: scale(.82); } }

        .side-section-title {
            color: #7093b9 !important;
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin: 0.9rem 0 0.35rem 0;
        }
        .side-mini-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem; margin: 0.7rem 0 0.25rem 0; }
        .side-mini-card { min-width: 0; border-radius: 12px; border: 1px solid rgba(111,163,214,0.15); background: rgba(17,36,58,0.62); padding: 0.62rem 0.65rem; }
        .side-mini-label { color: #7190ae !important; font-size: 0.62rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 800; }
        .side-mini-value { margin-top: 0.12rem; color: #f4f9ff !important; font-size: 0.78rem; line-height: 1.22; font-weight: 750; overflow-wrap: anywhere; }

        [data-testid="stSidebar"] [data-baseweb="radio"] > div { gap: 0.42rem; }
        [data-testid="stSidebar"] label[data-baseweb="radio"] {
            min-height: 48px;
            box-sizing: border-box;
            padding: 0.62rem 0.68rem;
            border-radius: 13px;
            border: 1px solid transparent;
            background: rgba(255,255,255,0.018);
            transition: background .18s ease, border-color .18s ease, transform .18s ease;
            align-items: flex-start;
        }
        [data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
            background: linear-gradient(90deg, rgba(33,212,253,.09), rgba(139,92,246,.08));
            border-color: rgba(33,212,253,.18);
            transform: translateX(2px);
        }
        [data-testid="stSidebar"] label[data-baseweb="radio"] p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere !important;
            line-height: 1.22 !important;
            font-size: 0.80rem !important;
            font-weight: 700 !important;
        }

        /* Hero */
        .hero-shell {
            position: relative;
            overflow: hidden;
            display: grid;
            grid-template-columns: minmax(0, 1.4fr) minmax(260px, .6fr);
            gap: 1.2rem;
            align-items: center;
            border: 1px solid rgba(71, 173, 255, 0.22);
            background: linear-gradient(120deg, rgba(10,28,47,.97) 0%, rgba(16,42,69,.96) 52%, rgba(38,28,76,.93) 100%);
            border-radius: 24px;
            padding: 1.55rem 1.65rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 22px 60px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,.04);
        }
        .hero-shell:before {
            content: "";
            position: absolute;
            width: 360px;
            height: 360px;
            right: -120px;
            top: -200px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(33,212,253,0.22), transparent 66%);
        }
        .hero-shell:after {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            left: 42%;
            bottom: -220px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(139,92,246,0.22), transparent 70%);
        }
        .hero-main, .hero-side { position: relative; z-index: 1; min-width: 0; }
        .hero-kicker { color: #58e5ff; font-size: 0.71rem; font-weight: 850; letter-spacing: 0.17em; text-transform: uppercase; }
        .hero-title { color: #f7fbff; font-size: clamp(1.7rem, 2.65vw, 2.75rem); font-weight: 840; line-height: 1.08; margin: 0.38rem 0 0.55rem 0; letter-spacing: -0.035em; }
        .hero-copy { color: #b6c9dc; max-width: 920px; font-size: 0.94rem; line-height: 1.58; margin: 0; }
        .hero-chips { display: flex; flex-wrap: wrap; gap: 0.48rem; margin-top: 0.9rem; }
        .hero-chip { display: inline-flex; align-items: center; gap: 0.34rem; padding: 0.36rem 0.58rem; border-radius: 999px; border: 1px solid rgba(255,255,255,0.10); background: rgba(255,255,255,0.055); color: #d9e8f7; font-size: 0.72rem; font-weight: 650; }
        .hero-side-card { border: 1px solid rgba(255,255,255,.10); border-radius: 18px; background: rgba(4,12,23,.45); padding: .95rem; backdrop-filter: blur(8px); }
        .hero-side-label { color: #7f9fbd; font-size: .63rem; font-weight: 850; letter-spacing: .12em; text-transform: uppercase; margin-bottom: .55rem; }
        .hero-flow { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; gap: .34rem; align-items: center; }
        .hero-flow-node { border-radius: 11px; text-align: center; padding: .58rem .35rem; border: 1px solid rgba(255,255,255,.08); background: rgba(255,255,255,.05); color: #eaf5ff; font-size: .68rem; font-weight: 750; line-height: 1.2; }
        .hero-flow-arrow { color: #55dffc; font-weight: 900; font-size: .9rem; }

        /* Section typography */
        .section-title-wrap { margin-top: 0.25rem; margin-bottom: 0.75rem; }
        .section-eyebrow { color: #55dffc; font-size: 0.69rem; font-weight: 850; text-transform: uppercase; letter-spacing: 0.14em; }
        .section-title { color: #f0f7ff; font-size: 1.58rem; font-weight: 800; line-height: 1.18; margin: 0.20rem 0 0.30rem 0; }
        .section-copy { color: #94a9bf; font-size: 0.91rem; line-height: 1.52; margin: 0; max-width: 1040px; }
        h1, h2, h3, h4, h5, h6 { color: #f1f7ff !important; }
        .stMarkdown p, .stMarkdown li { color: #b9cadc; }
        [data-testid="stCaptionContainer"] { color: #839ab3; }

        /* KPI cards */
        .kpi-card {
            --accent: #21d4fd;
            position: relative;
            overflow: hidden;
            min-height: 142px;
            height: 100%;
            box-sizing: border-box;
            border-radius: 18px;
            border: 1px solid rgba(111,163,214,.18);
            background: linear-gradient(145deg, rgba(18,42,67,.94), rgba(9,23,39,.96));
            padding: .95rem 1rem .92rem 1rem;
            box-shadow: 0 12px 28px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.03);
            transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
        }
        .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 18px 34px rgba(0,0,0,.28); }
        .kpi-card:after { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, var(--accent), transparent 88%); }
        .kpi-head { display: flex; justify-content: space-between; align-items: flex-start; gap: .55rem; }
        .kpi-label {
            min-width: 0;
            color: #8fa6bd;
            font-size: 0.69rem;
            font-weight: 820;
            letter-spacing: .065em;
            text-transform: uppercase;
            line-height: 1.25;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        .kpi-icon { flex: 0 0 auto; display: grid; place-items: center; width: 31px; height: 31px; border-radius: 10px; background: rgba(33,212,253,.08); border: 1px solid rgba(111,163,214,.18); color: var(--accent); font-size: .88rem; }
        .kpi-value {
            margin-top: .62rem;
            color: #f6fbff;
            font-size: clamp(1.06rem, 1.35vw, 1.62rem);
            line-height: 1.14;
            font-weight: 840;
            letter-spacing: -.025em;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: normal;
        }
        .kpi-note { margin-top: .42rem; color: #708aa5; font-size: .67rem; line-height: 1.28; white-space: normal; overflow-wrap: anywhere; }

        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(18,42,67,.94), rgba(9,23,39,.96));
            border: 1px solid rgba(111,163,214,.18);
            border-radius: 18px;
            padding: 0.95rem;
            min-height: 142px;
            overflow: visible;
        }
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricValue"] {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere !important;
        }
        [data-testid="stMetricLabel"] p { color: #8fa6bd !important; font-size: .69rem !important; line-height: 1.2 !important; }
        [data-testid="stMetricValue"] { color: #f6fbff !important; font-size: clamp(1.06rem, 1.35vw, 1.62rem) !important; }

        /* Content cards */
        .insight-card { height: 100%; box-sizing: border-box; border: 1px solid rgba(111,163,214,0.18); background: linear-gradient(145deg, rgba(16,38,62,.92), rgba(10,25,42,.95)); border-radius: 18px; padding: 1.02rem 1.08rem; box-shadow: 0 10px 28px rgba(0,0,0,.18); }
        .insight-label { color: #5fe3ff; font-size: .68rem; font-weight: 850; text-transform: uppercase; letter-spacing: .12em; margin-bottom: .38rem; }
        .insight-title { color: #eff7ff; font-size: 1.00rem; font-weight: 770; line-height: 1.28; margin-bottom: .34rem; overflow-wrap: anywhere; }
        .insight-copy { color: #98adc3; font-size: .84rem; line-height: 1.50; overflow-wrap: anywhere; }
        .insight-copy b { color: #dff4ff; }

        /* Pipeline and explanatory diagrams */
        .pipeline { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: .58rem; margin: .45rem 0 1rem 0; }
        .pipeline-step { position: relative; min-width: 0; min-height: 112px; border: 1px solid rgba(111,163,214,.17); background: linear-gradient(155deg, rgba(18,43,69,.96), rgba(8,22,38,.96)); border-radius: 16px; padding: .78rem .72rem; overflow: hidden; }
        .pipeline-step:after { content: ""; position: absolute; width: 74px; height: 74px; right: -30px; bottom: -34px; border-radius: 999px; background: radial-gradient(circle, rgba(33,212,253,.12), transparent 70%); }
        .pipeline-no { width: 28px; height: 28px; display: grid; place-items: center; border-radius: 9px; background: linear-gradient(135deg, rgba(33,212,253,.16), rgba(139,92,246,.15)); border: 1px solid rgba(33,212,253,.20); color: #62e7ff; font-size: .69rem; font-weight: 850; margin-bottom: .48rem; }
        .pipeline-step strong { color: #f1f8ff; font-size: .76rem; line-height: 1.25; display: block; overflow-wrap: anywhere; }
        .pipeline-step span { color: #7f97ae; font-size: .66rem; line-height: 1.32; display: block; margin-top: .18rem; overflow-wrap: anywhere; }

        .logic-flow { display: grid; grid-template-columns: minmax(0,1fr) auto minmax(0,1fr) auto minmax(0,1fr) auto minmax(0,1fr); gap: .42rem; align-items: stretch; margin: .55rem 0 1rem 0; }
        .logic-node { min-width: 0; border-radius: 15px; border: 1px solid rgba(111,163,214,.17); background: rgba(13,31,51,.92); padding: .78rem .74rem; text-align: center; }
        .logic-icon { font-size: 1.14rem; margin-bottom: .24rem; }
        .logic-title { color: #f1f8ff; font-size: .75rem; line-height: 1.2; font-weight: 780; overflow-wrap: anywhere; }
        .logic-copy { color: #7f97ae; font-size: .64rem; line-height: 1.3; margin-top: .22rem; overflow-wrap: anywhere; }
        .logic-arrow { display: grid; place-items: center; color: #4edfff; font-size: 1.15rem; font-weight: 900; }

        .signal-legend { display: flex; flex-wrap: wrap; gap: .48rem; margin: .30rem 0 .65rem 0; }
        .signal-pill { display: inline-flex; align-items: center; gap: .42rem; border: 1px solid rgba(111,163,214,.15); background: rgba(12,29,48,.78); border-radius: 999px; padding: .36rem .62rem; color: #b9cce0; font-size: .69rem; font-weight: 650; }
        .signal-dot { width: 8px; height: 8px; border-radius: 999px; flex:0 0 auto; }

        .eval-guide { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.55rem; margin:.5rem 0 .85rem 0; }
        .eval-card { min-width:0; border-radius:15px; border:1px solid rgba(111,163,214,.17); background:linear-gradient(150deg,rgba(16,38,62,.92),rgba(9,23,39,.96)); padding:.78rem .82rem; }
        .eval-card strong { display:block; color:#f2f8ff; font-size:.78rem; margin-bottom:.18rem; }
        .eval-card span { display:block; color:#7f97ae; font-size:.66rem; line-height:1.35; }
        .eval-up { color:#2edca8 !important; }
        .eval-down { color:#ffbe3f !important; }

        .risk-legend { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.42rem; margin:.45rem 0 .82rem 0; }
        .risk-chip { min-width:0; border-radius:12px; padding:.62rem .58rem; text-align:center; border:1px solid rgba(255,255,255,.08); font-size:.70rem; font-weight:780; color:#eef7ff; overflow-wrap:anywhere; }

        /* Forms and controls */
        [data-testid="stForm"] { background: linear-gradient(145deg, rgba(15,35,57,.94), rgba(8,21,36,.96)); border: 1px solid rgba(111,163,214,.18); border-radius: 20px; padding: 1rem 1rem .52rem 1rem; box-shadow: 0 12px 32px rgba(0,0,0,.20); }
        div[data-baseweb="select"] > div,
        [data-testid="stDateInput"] div[data-baseweb="input"] > div,
        [data-testid="stTextInput"] div[data-baseweb="input"] > div,
        [data-testid="stNumberInput"] div[data-baseweb="input"] > div {
            background: #10243a !important;
            border-color: rgba(111,163,214,.25) !important;
            color: #eef7ff !important;
            min-height: 46px;
        }
        div[data-baseweb="select"] * { color: #eaf4ff !important; }
        input { color: #eef7ff !important; }
        label p { color: #a8bbcf !important; font-size: .78rem !important; }

        .stButton > button, [data-testid="stFormSubmitButton"] button {
            border-radius: 12px;
            border: 1px solid rgba(61,203,255,.32);
            min-height: 2.8rem;
            font-weight: 780;
            background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 55%, #8b5cf6 100%);
            color: white;
            box-shadow: 0 10px 24px rgba(37,99,235,.22);
        }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { transform: translateY(-1px); border-color: rgba(104,224,255,.55); box-shadow: 0 13px 30px rgba(87,92,255,.30); }

        /* Tables, expanders, tabs */
        [data-testid="stDataFrame"] { border: 1px solid rgba(111,163,214,.18); border-radius: 15px; overflow: hidden; box-shadow: 0 9px 24px rgba(0,0,0,.16); }
        [data-testid="stExpander"] { background: rgba(11,28,47,.82); border: 1px solid rgba(111,163,214,.17); border-radius: 14px; }
        [data-testid="stExpander"] summary p { color:#dbeeff !important; font-weight:700 !important; }
        [data-testid="stTabs"] button { font-weight: 730; color: #91a8bf; }
        [data-testid="stTabs"] button[aria-selected="true"] { color: #65e4ff !important; }
        [data-testid="stAlert"] { border-radius: 14px; border: 1px solid rgba(111,163,214,.18); }
        [data-testid="stProgress"] > div > div { background: linear-gradient(90deg,#21d4fd,#8b5cf6) !important; }
        hr { border-color: rgba(111,163,214,.15) !important; }

        @media (max-width: 1180px) {
            .pipeline { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .hero-shell { grid-template-columns: 1fr; }
            .hero-side { max-width: 540px; }
            .risk-legend { grid-template-columns: repeat(3,minmax(0,1fr)); }
        }
        @media (max-width: 760px) {
            section[data-testid="stSidebar"] { width: 280px !important; min-width: 280px !important; }
            section[data-testid="stSidebar"] > div { width: 280px !important; }
            .block-container { padding-left: .72rem; padding-right: .72rem; }
            .hero-shell { padding: 1.2rem 1rem; border-radius: 18px; }
            .pipeline { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .logic-flow { grid-template-columns: 1fr; }
            .logic-arrow { transform: rotate(90deg); min-height: 20px; }
            .eval-guide { grid-template-columns: 1fr; }
            .risk-legend { grid-template-columns: repeat(2,minmax(0,1fr)); }
            .kpi-card { min-height: 128px; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_hero():
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-main">
                <div class="hero-kicker">MetroPT-3 · Predictive Maintenance</div>
                <div class="hero-title">Compressor early-warning & maintenance intelligence</div>
                <p class="hero-copy">
                    A decision-support view for model scores, failure-warning windows, alarm persistence,
                    event detection, false-alarm burden, SHAP explanations and maintenance risk.
                </p>
                <div class="hero-chips">
                    <span class="hero-chip">● Event-based evaluation</span>
                    <span class="hero-chip">⚡ Alarm tuning</span>
                    <span class="hero-chip">◆ SHAP explainability</span>
                    <span class="hero-chip">⚙ Risk decision support</span>
                </div>
            </div>
            <div class="hero-side">
                <div class="hero-side-card">
                    <div class="hero-side-label">Decision flow</div>
                    <div class="hero-flow">
                        <div class="hero-flow-node">Sensor<br>score</div>
                        <div class="hero-flow-arrow">›</div>
                        <div class="hero-flow-node">Alarm<br>rule</div>
                        <div class="hero-flow-arrow">›</div>
                        <div class="hero-flow-node">Maintenance<br>action</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_section_header(number, title, description):
    st.markdown(
        f"""
        <div class="section-title-wrap">
            <div class="section-eyebrow">Module {number}</div>
            <div class="section-title">{title}</div>
            <p class="section-copy">{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_pipeline():
    steps = [
        ("01", "Sensor data", "Raw compressor telemetry"),
        ("02", "Feature engineering", "Leakage-safe time-series features"),
        ("03", "Early-warning labels", "12-hour pre-failure windows"),
        ("04", "Model selection", "Chronological validation"),
        ("05", "Alarm evaluation", "Threshold + persistence tuning"),
        ("06", "Decision support", "SHAP + maintenance risk"),
    ]
    cards = "".join(
        f'<div class="pipeline-step"><div class="pipeline-no">{n}</div><strong>{title}</strong><span>{copy}</span></div>'
        for n, title, copy in steps
    )
    st.markdown(f'<div class="pipeline">{cards}</div>', unsafe_allow_html=True)


def render_kpi_card(label, value, icon="●", accent="#21d4fd", note=None):
    safe_label = str(label)
    safe_value = str(value)
    note_html = f'<div class="kpi-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent:{accent};">
            <div class="kpi-head">
                <div class="kpi-label">{safe_label}</div>
                <div class="kpi-icon">{icon}</div>
            </div>
            <div class="kpi-value">{safe_value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alarm_logic_diagram():
    st.markdown(
        """
        <div class="logic-flow">
            <div class="logic-node">
                <div class="logic-icon" style="color:#21d4fd;">〽</div>
                <div class="logic-title">Model score</div>
                <div class="logic-copy">Timestamp-level anomaly or risk score</div>
            </div>
            <div class="logic-arrow">→</div>
            <div class="logic-node">
                <div class="logic-icon" style="color:#ffbe3f;">━</div>
                <div class="logic-title">Threshold</div>
                <div class="logic-copy">Score must cross the selected alarm cut-off</div>
            </div>
            <div class="logic-arrow">→</div>
            <div class="logic-node">
                <div class="logic-icon" style="color:#8b5cf6;">◷</div>
                <div class="logic-title">Persistence filter</div>
                <div class="logic-copy">Alarm must continue for the chosen duration</div>
            </div>
            <div class="logic-arrow">→</div>
            <div class="logic-node">
                <div class="logic-icon" style="color:#20d69f;">✓</div>
                <div class="logic-title">Maintenance signal</div>
                <div class="logic-copy">Evaluate event detection, lead time and alarm burden</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_legend():
    st.markdown(
        """
        <div class="signal-legend">
            <span class="signal-pill"><span class="signal-dot" style="background:#21d4fd"></span>Model score</span>
            <span class="signal-pill"><span class="signal-dot" style="background:#ffbe3f"></span>Alarm threshold</span>
            <span class="signal-pill"><span class="signal-dot" style="background:#ff5573"></span>Persistent alarm</span>
            <span class="signal-pill"><span class="signal-dot" style="background:#8b5cf6"></span>12-hour warning window</span>
            <span class="signal-pill"><span class="signal-dot" style="background:#f43f5e"></span>Failure interval</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evaluation_guide():
    st.markdown(
        """
        <div class="eval-guide">
            <div class="eval-card"><strong class="eval-up">↑ Event detection</strong><span>Higher is better: how many documented failure events produced an early warning.</span></div>
            <div class="eval-card"><strong class="eval-up">↑ Lead time</strong><span>More usable warning time can give maintenance teams greater opportunity to intervene.</span></div>
            <div class="eval-card"><strong class="eval-down">↓ False alarms/day</strong><span>Lower is better: frequent false alerts reduce the operational usefulness of an alarm system.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_legend():
    st.markdown(
        """
        <div class="risk-legend">
            <div class="risk-chip" style="background:rgba(32,214,159,.18);border-color:rgba(32,214,159,.35);">Normal</div>
            <div class="risk-chip" style="background:rgba(59,130,246,.18);border-color:rgba(59,130,246,.35);">Low</div>
            <div class="risk-chip" style="background:rgba(255,190,63,.18);border-color:rgba(255,190,63,.38);">Medium</div>
            <div class="risk-chip" style="background:rgba(249,115,22,.18);border-color:rgba(249,115,22,.38);">High</div>
            <div class="risk-chip" style="background:rgba(255,85,115,.18);border-color:rgba(255,85,115,.38);">Critical</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def plotly_layout(fig, height=390, title=None, legend_orientation="h"):
    if not PLOTLY_AVAILABLE:
        return fig
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(color="#f2f8ff", size=17)) if title else None,
        margin=dict(l=20, r=20, t=58 if title else 26, b=24),
        paper_bgcolor="#0b1727",
        plot_bgcolor="#0b1727",
        font=dict(color="#a9bdd2", size=12),
        hoverlabel=dict(bgcolor="#07111f", font_color="#f5fbff", bordercolor="#2f5275"),
        legend=dict(
            orientation=legend_orientation,
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#b8cadc", size=11),
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor="rgba(111,163,214,.18)",
            tickfont=dict(color="#8fa6bd"),
            title_font=dict(color="#9fb6cc"),
            rangeslider=dict(bgcolor="#08111e", bordercolor="rgba(111,163,214,.18)", borderwidth=1),
        ),
        yaxis=dict(
            gridcolor="rgba(111,163,214,.10)",
            zeroline=False,
            linecolor="rgba(111,163,214,.18)",
            tickfont=dict(color="#8fa6bd"),
            title_font=dict(color="#9fb6cc"),
        ),
    )
    return fig

def event_timeline_figure(events):
    if not PLOTLY_AVAILABLE or events is None or events.empty:
        return None

    rows = []
    for _, row in events.iterrows():
        rows.append({
            "event_id": row["event_id"],
            "period": "12h warning window",
            "start": row["warning_12h_start"],
            "end": row["warning_12h_end"],
        })
        rows.append({
            "event_id": row["event_id"],
            "period": "Failure interval",
            "start": row["failure_start"],
            "end": row["failure_end"],
        })

    timeline_df = pd.DataFrame(rows)
    fig = px.timeline(
        timeline_df,
        x_start="start",
        x_end="end",
        y="event_id",
        color="period",
        color_discrete_map={
            "12h warning window": "#8b5cf6",
            "Failure interval": "#ff5573",
        },
        category_orders={"event_id": list(reversed(events["event_id"].tolist()))},
        custom_data=["period", "start", "end"],
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Start: %{customdata[1]}<br>End: %{customdata[2]}<extra></extra>"
    )
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None, rangeslider_visible=True)
    return plotly_layout(fig, height=350)


def event_detection_figure(df, model_col, rate_col):
    if not PLOTLY_AVAILABLE:
        return None
    chart_df = df[[model_col, rate_col]].copy()
    chart_df[model_col] = chart_df[model_col].astype(str).map(normalise_name)
    chart_df[rate_col] = pd.to_numeric(chart_df[rate_col], errors="coerce")
    chart_df = chart_df.dropna(subset=[rate_col]).sort_values(rate_col, ascending=False)
    fig = px.bar(
        chart_df,
        x=model_col,
        y=rate_col,
        text=rate_col,
        color=rate_col,
        color_continuous_scale=[[0, "#16344f"], [0.45, "#21d4fd"], [1, "#8b5cf6"]],
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(title="Detection rate", tickformat=".0%", range=[0, max(1.05, float(chart_df[rate_col].max() * 1.15))])
    fig.update_xaxes(title=None)
    return plotly_layout(fig, height=360)


def score_alarm_figure(plot_df, score_col, threshold, selected_data=None):
    if not PLOTLY_AVAILABLE:
        return None

    df = plot_df.reset_index() if "timestamp" not in plot_df.columns else plot_df.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df[score_col],
        mode="lines",
        name=normalise_name(score_col),
        line=dict(color="#21d4fd", width=2.2),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Score: %{y:.3f}<extra></extra>",
    ))

    alarm_df = df[df["simulated_alarm"] == 1]
    if not alarm_df.empty:
        fig.add_trace(go.Scatter(
            x=alarm_df["timestamp"],
            y=alarm_df[score_col],
            mode="markers",
            name="Persistent alarm",
            marker=dict(color="#ff5573", size=7, symbol="circle"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Alarm score: %{y:.3f}<extra></extra>",
        ))

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#ffbe3f",
        annotation_text=f"Threshold {threshold:.2f}",
        annotation_position="top right",
    )

    if selected_data is not None and not selected_data.empty:
        range_start = selected_data["timestamp"].min()
        range_end = selected_data["timestamp"].max()
        for _, event in FAILURE_EVENTS.iterrows():
            warning_start = max(event["warning_12h_start"], range_start)
            warning_end = min(event["warning_12h_end"], range_end)
            if warning_start <= warning_end:
                fig.add_vrect(
                    x0=warning_start,
                    x1=warning_end,
                    fillcolor="#8b5cf6",
                    opacity=0.07,
                    line_width=0,
                    annotation_text=f"{event['event_id']} warning",
                    annotation_position="top left",
                )

            fail_start = max(event["failure_start"], range_start)
            fail_end = min(event["failure_end"], range_end)
            if fail_start <= fail_end:
                fig.add_vrect(
                    x0=fail_start,
                    x1=fail_end,
                    fillcolor="#ff5573",
                    opacity=0.09,
                    line_width=0,
                )

    fig.update_yaxes(title="Model score", range=[0, max(1.0, float(df[score_col].max(skipna=True) * 1.08) if not df[score_col].dropna().empty else 1.0)])
    fig.update_xaxes(title=None, rangeslider_visible=True)
    return plotly_layout(fig, height=470)


def alarm_episode_figure(plot_df):
    if not PLOTLY_AVAILABLE:
        return None
    df = plot_df.reset_index() if "timestamp" not in plot_df.columns else plot_df.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["simulated_alarm"],
        mode="lines",
        line=dict(color="#ff5573", width=2, shape="hv"),
        fill="tozeroy",
        fillcolor="rgba(255,85,115,0.18)",
        name="Persistent alarm",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Alarm: %{y:.0f}<extra></extra>",
    ))
    fig.update_yaxes(title=None, range=[-0.05, 1.15], tickvals=[0, 1], ticktext=["Normal", "Alarm"])
    fig.update_xaxes(title=None, rangeslider_visible=False)
    return plotly_layout(fig, height=255)


def multiseries_timeline_figure(plot_df, columns, title=None):
    if not PLOTLY_AVAILABLE:
        return None
    df = plot_df.reset_index() if "timestamp" not in plot_df.columns else plot_df.copy()
    fig = go.Figure()
    palette = ["#21d4fd", "#8b5cf6", "#20d69f", "#ffbe3f", "#ec4899", "#3b82f6"]
    for idx, col in enumerate(columns):
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df[col],
            mode="lines",
            name=normalise_name(col),
            line=dict(width=1.8, color=palette[idx % len(palette)]),
            hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>{normalise_name(col)}: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_xaxes(title=None, rangeslider_visible=True)
    fig.update_yaxes(title=None)
    return plotly_layout(fig, height=430, title=title)


def alarm_timeline_figure(plot_df, columns):
    if not PLOTLY_AVAILABLE:
        return None
    df = plot_df.reset_index() if "timestamp" not in plot_df.columns else plot_df.copy()
    fig = go.Figure()
    palette = ["#ff5573", "#ffbe3f", "#21d4fd", "#20d69f", "#8b5cf6"]
    for idx, col in enumerate(columns):
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df[col],
            mode="lines",
            name=normalise_name(col),
            line=dict(width=1.8, shape="hv", color=palette[idx % len(palette)]),
            hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>{normalise_name(col)}: %{{y}}<extra></extra>",
        ))
    fig.update_xaxes(title=None, rangeslider_visible=True)
    fig.update_yaxes(title=None)
    return plotly_layout(fig, height=360)


def alarm_tradeoff_figure(df):
    """Interactive alarm trade-off diagram using existing evaluation outputs only."""
    if not PLOTLY_AVAILABLE or df is None or df.empty:
        return None

    model_col = first_existing_column(df, ["model", "model_name", "classifier"])
    threshold_col = first_existing_column(df, ["threshold", "selected_threshold"])
    duration_col = first_existing_column(df, ["min_duration_minutes", "minimum_duration_minutes", "persistence_minutes"])
    lead_col = first_existing_column(df, ["median_lead_time_hours", "lead_time_hours", "median_detection_lead_time_hours"])
    false_col = first_existing_column(df, ["false_alarm_episodes_per_day", "false_alarms_per_day"])
    detection_col = first_existing_column(df, ["failure_detection_rate", "event_recall", "recall"])
    events_col = first_existing_column(df, ["events_detected", "detected_events"])
    evaluated_col = first_existing_column(df, ["events_evaluated", "total_events"])

    if model_col is None or lead_col is None or false_col is None:
        return None

    cols = [model_col, lead_col, false_col]
    for col in [threshold_col, duration_col, detection_col, events_col, evaluated_col]:
        if col and col not in cols:
            cols.append(col)

    chart_df = df[cols].copy()
    for col in cols:
        if col != model_col:
            chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce")

    if detection_col is None:
        if events_col and evaluated_col:
            chart_df["_detection_rate"] = chart_df[events_col] / chart_df[evaluated_col].replace(0, np.nan)
        else:
            chart_df["_detection_rate"] = 0.0
        detection_plot_col = "_detection_rate"
    else:
        detection_plot_col = detection_col

    chart_df = chart_df.dropna(subset=[lead_col, false_col])
    if chart_df.empty:
        return None

    chart_df[model_col] = chart_df[model_col].astype(str).map(normalise_name)
    chart_df[detection_plot_col] = pd.to_numeric(chart_df[detection_plot_col], errors="coerce").fillna(0)

    hover_data = {}
    if threshold_col:
        hover_data[threshold_col] = ":.2f"
    if duration_col:
        hover_data[duration_col] = ":.0f"
    hover_data[detection_plot_col] = ":.1%"

    fig = px.scatter(
        chart_df,
        x=false_col,
        y=lead_col,
        color=detection_plot_col,
        symbol=model_col,
        hover_name=model_col,
        hover_data=hover_data,
        color_continuous_scale=[[0, "#ff5573"], [0.5, "#ffbe3f"], [1, "#20d69f"]],
        labels={
            false_col: "False alarm episodes / day",
            lead_col: "Median lead time (hours)",
            detection_plot_col: "Event detection",
            model_col: "Model",
        },
    )
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="rgba(255,255,255,.35)")))
    fig.update_coloraxes(colorbar=dict(title="Detection", tickformat=".0%"))
    fig.update_xaxes(title="False alarms per day  ← lower is better")
    fig.update_yaxes(title="Median lead time (hours)  ↑ higher is better")
    return plotly_layout(fig, height=440)


def global_shap_figure(df):
    if not PLOTLY_AVAILABLE or df is None or df.empty:
        return None

    feature_col = first_existing_column(df, ["feature", "feature_name", "variable", "Feature"])
    value_col = first_existing_column(df, ["mean_abs_shap", "mean_absolute_shap", "importance", "shap_importance", "mean_shap_value"])
    if feature_col is None or value_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if feature_col is None and len(df.columns) > 0:
            feature_col = df.columns[0]
        if value_col is None and numeric_cols:
            value_col = numeric_cols[0]

    if feature_col is None or value_col is None:
        return None

    chart_df = df[[feature_col, value_col]].copy()
    chart_df[value_col] = pd.to_numeric(chart_df[value_col], errors="coerce")
    chart_df = chart_df.dropna(subset=[value_col]).nlargest(15, value_col).sort_values(value_col)
    if chart_df.empty:
        return None

    fig = px.bar(
        chart_df,
        x=value_col,
        y=feature_col,
        orientation="h",
        text=value_col,
        color=value_col,
        color_continuous_scale=[[0, "#17354d"], [0.5, "#21d4fd"], [1, "#8b5cf6"]],
    )
    fig.update_coloraxes(showscale=False)
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
    fig.update_xaxes(title="Mean |SHAP| contribution")
    fig.update_yaxes(title=None)
    return plotly_layout(fig, height=470)


def risk_distribution_figure(df):
    if not PLOTLY_AVAILABLE or df is None or df.empty:
        return None
    if "risk_level" not in df.columns or "count" not in df.columns:
        return None

    chart_df = df[["risk_level", "count"]].copy()
    chart_df["count"] = pd.to_numeric(chart_df["count"], errors="coerce")
    chart_df = chart_df.dropna(subset=["count"])
    color_map = {
        "Critical": "#ff5573",
        "High": "#f97316",
        "Medium": "#ffbe3f",
        "Low": "#3b82f6",
        "Normal": "#20d69f",
    }
    fig = px.pie(
        chart_df,
        names="risk_level",
        values="count",
        hole=0.58,
        color="risk_level",
        color_discrete_map=color_map,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="%{label}<br>Count: %{value:,}<br>Share: %{percent}<extra></extra>")
    fig.update_layout(showlegend=True)
    return plotly_layout(fig, height=390)


inject_visual_theme()



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


st.sidebar.markdown(
    """
    <div class="metro-brand">
        <div class="metro-brand-title">🚆 MetroPT Command Centre</div>
        <div class="metro-brand-subtitle">Explainable predictive-maintenance decision support</div>
    </div>
    """,
    unsafe_allow_html=True
)

if prediction_file:
    st.sidebar.markdown(
        '<div class="metro-status"><span class="metro-status-dot"></span>System ready · prediction data connected</div>',
        unsafe_allow_html=True
    )
else:
    st.sidebar.error("Prediction data unavailable")

st.sidebar.markdown(
    """
    <div class="side-mini-grid">
        <div class="side-mini-card"><div class="side-mini-label">Dataset</div><div class="side-mini-value">MetroPT-3</div></div>
        <div class="side-mini-card"><div class="side-mini-label">Failure mode</div><div class="side-mini-value">Air leak</div></div>
    </div>
    <div class="side-section-title">Navigation</div>
    """,
    unsafe_allow_html=True,
)
page_map = {
    "▦  Overview": "Overview",
    "◫  Model & Alarm Evaluation": "Model and Alarm Evaluation",
    "⚡  Alarm Simulator": "Interactive Alarm Simulator",
    "⌁  Timeline Explorer": "Timeline Explorer",
    "◆  Explainability & Risk": "Explainability and Risk Framework",
}

selected_page_label = st.sidebar.radio(
    "Navigation",
    list(page_map.keys()),
    label_visibility="collapsed"
)
page = page_map[selected_page_label]

st.sidebar.divider()
with st.sidebar.expander("System & file status", expanded=False):
    st.caption("Project root")
    st.code(str(PROJECT_ROOT), language="text")
    st.caption("Prediction output")
    if prediction_file:
        st.success(prediction_file.name)
    else:
        st.error("Prediction file not found")
    st.caption(f"Plotly charts: {'enabled' if PLOTLY_AVAILABLE else 'fallback mode'}")

st.sidebar.caption("Metro compressor · Explainable ML · Event-focused evaluation")

render_hero()


if page == "Overview":
    render_section_header(
        "01",
        "Overview",
        "A high-level operational view of the predictive-maintenance framework, selected model, alarm configuration and documented failure events."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Dataset", "MetroPT-3", "▦", "#21d4fd", "Compressor time-series data")
    with c2:
        render_kpi_card("Failure mode", "Air leak", "⚠", "#ff5573", "Documented failure type")
    with c3:
        render_kpi_card("Validation model", validation_summary["model"], "◆", "#8b5cf6", "Chronological validation selection")
    with c4:
        render_kpi_card("Event alarm model", best_alarm_summary["model"], "⚡", "#20d69f", "Highlighted event-level setting")

    st.markdown("#### Framework pipeline")
    render_pipeline()

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-label">Project aim</div>
                <div class="insight-title">Move from timestamp predictions to actionable early-warning decisions</div>
                <div class="insight-copy">
                    The dashboard combines time-series preprocessing, early-warning labels, model comparison,
                    alarm-threshold analysis, persistence rules, event-level evaluation, SHAP explanation and
                    maintenance-facing risk categorisation. The emphasis is not only whether a record is classified
                    correctly, but whether a failure event is detected early enough with a manageable alarm burden.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with right:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-label">Operational highlight</div>
                <div class="insight-title">{best_alarm_summary['events']} events detected at the highlighted alarm setting</div>
                <div class="insight-copy">
                    Threshold <b>{best_alarm_summary['threshold']}</b> · persistence <b>{best_alarm_summary['duration']}</b> ·
                    median lead time <b>{best_alarm_summary['lead_time']}</b> · false alarms <b>{best_alarm_summary['false_alarms']}</b>.
                    Use the simulator to test how these trade-offs change under alternative settings.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### Key model and alarm settings")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Validation threshold", validation_summary["threshold"], "━", "#3b82f6")
    with k2:
        render_kpi_card("Alarm threshold", best_alarm_summary["threshold"], "━", "#ffbe3f")
    with k3:
        render_kpi_card("Persistence duration", best_alarm_summary["duration"], "◷", "#8b5cf6")
    with k4:
        render_kpi_card("Median lead time", best_alarm_summary["lead_time"], "⏱", "#20d69f")

    st.info(
        "Validation-selected record-level performance and the strongest event-level alarm setting are shown separately. "
        "In rare-event predictive maintenance, the best classifier at record level is not necessarily the most useful alarm configuration operationally."
    )

    st.markdown("#### Failure and warning-window timeline")
    timeline_fig = event_timeline_figure(FAILURE_EVENTS)
    if timeline_fig is not None:
        st.plotly_chart(timeline_fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})
    else:
        st.caption("Install Plotly for the interactive failure-event timeline.")

    with st.expander("View documented failure-event table", expanded=False):
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
        "Decision-support prototype only: operational deployment would require additional external validation, engineering review and controlled testing."
    )

elif page == "Model and Alarm Evaluation":
    render_section_header(
        "02",
        "Model and Alarm Evaluation",
        "Compare record-level model performance with event-level detection, lead time and false-alarm burden — the measures that matter most for early-warning maintenance."
    )

    model_count = len(validation_model_comparison) if validation_model_comparison is not None else 0
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_kpi_card("Models compared", model_count, "▦", "#21d4fd")
    with m2:
        render_kpi_card("Validation-selected", validation_summary["model"], "◆", "#8b5cf6")
    with m3:
        render_kpi_card("Event alarm highlight", best_alarm_summary["model"], "⚡", "#ffbe3f")
    with m4:
        render_kpi_card("Detected events", best_alarm_summary["events"], "✓", "#20d69f")

    render_evaluation_guide()

    tab_perf, tab_alarm, tab_tables = st.tabs([
        "Performance view",
        "Alarm-burden view",
        "Detailed tables",
    ])

    with tab_perf:
        if test_event_results is not None and not test_event_results.empty:
            model_col = first_existing_column(test_event_results, ["model", "model_name", "classifier"])
            rate_col = first_existing_column(test_event_results, ["failure_detection_rate", "event_recall", "recall"])

            if model_col and rate_col:
                st.markdown("#### Event detection rate by model")
                fig = event_detection_figure(test_event_results, model_col, rate_col)
                if fig is not None:
                    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
                else:
                    chart_df = test_event_results[[model_col, rate_col]].copy()
                    chart_df[rate_col] = pd.to_numeric(chart_df[rate_col], errors="coerce")
                    st.bar_chart(chart_df.set_index(model_col)[rate_col])

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(
                """
                <div class="insight-card">
                    <div class="insight-label">Why event metrics matter</div>
                    <div class="insight-title">Accuracy alone can hide rare-event failure behaviour</div>
                    <div class="insight-copy">A maintenance alarm must detect failure episodes early enough to support intervention. Event detection and lead time therefore complement record-level precision and recall.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-label">Current highlighted setting</div>
                    <div class="insight-title">{best_alarm_summary['model']} · threshold {best_alarm_summary['threshold']}</div>
                    <div class="insight-copy">Persistence {best_alarm_summary['duration']}; median lead time {best_alarm_summary['lead_time']}; false alarms {best_alarm_summary['false_alarms']}.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        show_table(test_event_results, "Test Event-Level Results", height=300, max_rows=100)

    with tab_alarm:
        st.markdown("#### Alarm trade-off map")
        st.caption("Best operational area: move upward for more lead time and leftward for fewer false alarms. Point colour indicates event-detection performance.")
        tradeoff_fig = alarm_tradeoff_figure(test_alarm_burden_results)
        if tradeoff_fig is not None:
            st.plotly_chart(tradeoff_fig, width="stretch", config={"displaylogo": False})

        top_alarm_settings = prepare_top_alarm_settings(test_alarm_burden_results, max_rows=10)
        show_table(top_alarm_settings, "Highest-Ranked Alarm Settings", height=320, max_rows=10)
        show_table(alarm_burden_comparison, "Original vs Refined Alarm-Burden Comparison", height=320, max_rows=100)

        fig_path = OUTPUT_FIGURES_DIR / "test_alarm_burden_original_vs_refined.png"
        if fig_path.exists():
            with st.expander("Original static alarm-burden figure", expanded=False):
                st.image(str(fig_path), caption="Original vs refined alarm burden", width="stretch")

        if selected_alarm_rules is not None and not selected_alarm_rules.empty:
            with st.expander("Validation-selected alarm rules", expanded=False):
                show_table(selected_alarm_rules, height=260, max_rows=100)

    with tab_tables:
        show_table(validation_model_comparison, "Validation Model Comparison", height=300, max_rows=100)
        show_table(selected_thresholds, "Selected Thresholds from Validation", height=260, max_rows=100)
        show_table(test_event_results, "Test Event-Level Results", height=300, max_rows=100)

    

elif page == "Interactive Alarm Simulator":
    render_section_header(
        "03",
        "Interactive Alarm Simulator",
        "Tune the score threshold, persistence rule and date range, then immediately inspect the impact on event detection, lead time and false-alarm burden."
    )

    if prediction_file is None:
        st.error(f"No prediction file found in the {OUTPUT_PREDICTIONS_DIR} directory.")
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

    st.markdown("#### How the alarm simulator works")
    render_alarm_logic_diagram()

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

    st.markdown("#### Simulation results")

    detection_rate = summary["failure_detection_rate"]
    detection_delta = None if pd.isna(detection_rate) else f"{detection_rate:.0%} event recall"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_kpi_card("Events detected", f"{summary['events_detected']}/{summary['events_evaluated']}", "✓", "#20d69f", detection_delta)
    with m2:
        render_kpi_card("Median lead time", format_number(summary["median_lead_time_hours"], 2) + " h", "⏱", "#21d4fd")
    with m3:
        render_kpi_card("False alarms/day", format_number(summary["false_alarm_episodes_per_day"], 2), "!", "#ffbe3f")
    with m4:
        render_kpi_card("Alarm time", format_number(summary["total_alarm_time_hours"], 2) + " h", "◷", "#8b5cf6")

    m5, m6, m7, m8 = st.columns(4)
    with m5:
        render_kpi_card("Alarm time %", format_number(summary["total_alarm_time_percentage"], 2) + "%", "%", "#ec4899")
    with m6:
        render_kpi_card("Accepted episodes", str(summary["total_alarm_episodes"]), "▤", "#3b82f6")
    with m7:
        render_kpi_card("Precision", format_number(summary["precision"], 3), "P", "#21d4fd")
    with m8:
        render_kpi_card("Recall", format_number(summary["recall"], 3), "R", "#20d69f")

    st.caption(
        f"Rows evaluated: {len(selected_data):,} · "
        f"rows plotted after downsampling: {len(plot_data):,} · "
        f"threshold: {threshold:.2f} · persistence: {min_duration} min"
    )

    st.markdown("#### Score, threshold and alarm activity")
    render_signal_legend()
    score_fig = score_alarm_figure(plot_data, selected_score_col, threshold, selected_data=selected_data)
    if score_fig is not None:
        st.plotly_chart(score_fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})
    else:
        st.line_chart(plot_data[[selected_score_col]])

    alarm_fig = alarm_episode_figure(plot_data)
    if alarm_fig is not None:
        st.plotly_chart(alarm_fig, width="stretch", config={"displaylogo": False})
    else:
        st.line_chart(plot_data[["simulated_alarm"]])

    detail_tab, episode_tab = st.tabs(["Event detection detail", "Accepted alarm episodes"])
    with detail_tab:
        show_table(event_detail, height=260, max_rows=100)
    with episode_tab:
        show_table(episodes, height=340, max_rows=300)

    st.success(
        "Simulation complete. Adjust the controls above and rerun to compare the operational trade-off between earlier detection and alarm burden."
    )


elif page == "Timeline Explorer":
    render_section_header(
        "04",
        "Timeline Explorer",
        "Inspect score trajectories, alarm outputs and labels over any selected time range with hover detail, zoom controls and a range slider."
    )

    if prediction_file is None:
        st.error(f"No prediction file found in the {OUTPUT_PREDICTIONS_DIR} directory.")
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
            st.markdown("#### Score timeline")
            score_timeline_fig = multiseries_timeline_figure(plot_data, available_score_cols)
            if score_timeline_fig is not None:
                st.plotly_chart(score_timeline_fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})
            else:
                st.line_chart(plot_data[available_score_cols])

    if selected_alarm_columns:
        available_alarm_cols = [
            c for c in selected_alarm_columns
            if c in plot_data.columns
        ]

        if available_alarm_cols:
            st.markdown("#### Alarm and label timeline")
            alarm_timeline_fig = alarm_timeline_figure(plot_data, available_alarm_cols)
            if alarm_timeline_fig is not None:
                st.plotly_chart(alarm_timeline_fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})
            else:
                st.line_chart(plot_data[available_alarm_cols])

    with st.expander("Loaded timeline data preview", expanded=False):
        st.dataframe(timeline.head(300), width="stretch", height=360)

    st.success(
        "Timeline visualisation completed. Use drag-to-zoom, hover values and the range slider to inspect specific operating periods."
    )


elif page == "Explainability and Risk Framework":
    render_section_header(
        "05",
        "Explainability and Risk Framework",
        "Translate model behaviour into interpretable feature contributions and maintenance-facing risk categories without treating explanation as physical causality."
    )

    global_tab, local_tab, risk_tab = st.tabs([
        "Global explainability",
        "Local case explanations",
        "Maintenance risk",
    ])

    with global_tab:
        st.markdown("#### Global SHAP feature importance")
        shap_fig = global_shap_figure(global_shap)
        if shap_fig is not None:
            st.plotly_chart(shap_fig, width="stretch", config={"displaylogo": False})

        show_table(global_shap, "Top Global SHAP Features", height=300, max_rows=50)

        shap_bar_path = read_existing([
            OUTPUT_FIGURES_DIR / "global_shap_top_features.png",
            OUTPUT_SHAP_DIR / "global_shap_top_features.png"
        ])
        shap_summary_path = read_existing([
            OUTPUT_FIGURES_DIR / "shap_summary_plot.png",
            OUTPUT_SHAP_DIR / "shap_summary_plot.png"
        ])

        with st.expander("Original saved SHAP figures", expanded=False):
            c1, c2 = st.columns(2, gap="large")
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

        st.warning(
            "SHAP describes how the trained model uses features to form predictions. It should not be interpreted as direct proof of physical or engineering causality."
        )

    with local_tab:
        st.markdown("#### Local prediction explanations")
        st.write(
            "Use the local explanation table to inspect which features pushed individual cases toward higher or lower predicted risk."
        )
        show_table(local_shap, "Local SHAP Case Explanations", height=430, max_rows=100)

    with risk_tab:
        st.markdown("#### Risk-level decision support")
        render_risk_legend()

        if risk_distribution is not None and not risk_distribution.empty:
            c1, c2 = st.columns([1.05, 0.95], gap="large")
            with c1:
                risk_fig = risk_distribution_figure(risk_distribution)
                if risk_fig is not None:
                    st.plotly_chart(risk_fig, width="stretch", config={"displaylogo": False})
                else:
                    risk_plot = risk_distribution.copy()
                    if "risk_level" in risk_plot.columns and "count" in risk_plot.columns:
                        risk_plot["display_level"] = risk_plot["risk_level"].apply(risk_badge)
                        st.bar_chart(risk_plot.set_index("display_level")["count"])

            with c2:
                st.markdown(
                    """
                    <div class="insight-card">
                        <div class="insight-label">Decision-support interpretation</div>
                        <div class="insight-title">Risk categories convert model output into maintenance language</div>
                        <div class="insight-copy">
                            The risk framework is a communication layer for triage and prioritisation. It should be used together with alarm persistence, event context, explainability and engineering judgement rather than as a standalone automated maintenance instruction.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        show_table(risk_rules, "Risk Framework Rules", height=280, max_rows=100)
        show_table(risk_distribution, "Risk Distribution Table", height=240, max_rows=100)

        fig_path = OUTPUT_FIGURES_DIR / "risk_level_distribution_test_set.png"
        if fig_path.exists():
            with st.expander("Static risk-distribution figure", expanded=False):
                st.image(str(fig_path), caption="Risk-Level Distribution", width="stretch")

        show_table(framework_summary, "SHAP and Risk Framework Summary", height=300, max_rows=100)

        st.info(
            "The maintenance risk framework is a decision-support layer: it translates model outputs into categories that can be reviewed alongside the underlying evidence."
        )

