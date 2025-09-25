# app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time as dtime
import re

st.set_page_config(page_title="KLGA METAR vs LAMP", layout="wide")

# --- compact spacing between tables ---
st.markdown("""
<style>
div[data-testid="stDataFrame"] { margin: 0 !important; }
div.stCaption, p.stCaption { margin-top: .25rem !important; margin-bottom: .25rem !important; }
</style>
""", unsafe_allow_html=True)

LAMP_URL = "http://34.9.231.83/glmp_temperature_klga.csv"
METAR_URL = "http://34.9.231.83/klga_metar.csv"

@st.cache_data(ttl=10)
def load_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def _safe_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def pick_default_date_et(metar_df: pd.DataFrame):
    if metar_df.empty or "observation_time_et" not in metar_df.columns:
        return datetime.now().date()
    m = _safe_to_datetime(metar_df, "observation_time_et")
    dates = sorted(m["observation_time_et"].dropna().dt.date.unique())
    return dates[-1] if dates else datetime.now().date()

def prepare_metar_day_table(metar_df: pd.DataFrame, date_et) -> pd.DataFrame:
    """2-row METAR table: EST TIME (METAR) + METAR °F for chosen ET date (temps formatted to 2 decimals)."""
    m = _safe_to_datetime(metar_df, "observation_time_et")
    day = m[m["observation_time_et"].dt.date == date_et].copy()

    if day.empty:
        cols = [f"{h:02d}:00" for h in range(24)]
        return pd.DataFrame([cols, [""] * 24],
                            index=["EST TIME (METAR)", "METAR °F"],
                            columns=cols)

    minute_mode = int(day["observation_time_et"].dt.minute.mode().iloc[0])
    cols = [f"{h:02d}:{minute_mode:02d}" for h in range(24)]

    day["hhmm"] = day["observation_time_et"].dt.strftime("%H:%M")
    reading_map = dict(zip(day["hhmm"], day["temp_F"]))

    temps = []
    for c in cols:
        val = reading_map.get(c, None)
        if pd.isna(val) or val is None:
            temps.append("")
        else:
            try:
                temps.append(f"{float(val):.2f}")
            except Exception:
                temps.append("")
    return pd.DataFrame([cols, temps],
                        index=["EST TIME (METAR)", "METAR °F"],
                        columns=cols)

# --------- LAMP helpers ----------
def _label_from_run_est(run_est: pd.Timestamp) -> str:
    """Timestamp (EST) -> 'GLMP HH:MM EST MM/DD'."""
    if pd.isna(run_est):
        return "GLMP"
    return f"GLMP {run_est.strftime('%H:%M')} EST {run_est.strftime('%m/%d')}"

def _standardize_valid_est(series: pd.Series) -> pd.Series:
    """Force valid_time_est to naive EST."""
    s = pd.to_datetime(series, errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(s):
        s = s.dt.tz_convert("UTC").dt.tz_localize(None) - pd.Timedelta(hours=4)
    return s

def _ensure_dt_and_make_run_est(lamp_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime types and add 'run_est' (naive EST)."""
    df = lamp_df.copy()
    obs_utc = pd.to_datetime(df["observation_time_utc"], errors="coerce", utc=True)
    df["run_est"] = obs_utc.dt.tz_convert(None) - pd.Timedelta(hours=4)
    df["valid_time_est"] = _standardize_valid_est(df["valid_time_est"])
    return df

def _pick_best_lamp_minute(valid_time_est_series: pd.Series) -> int:
    mins = valid_time_est_series.dropna().dt.minute.astype(int)
    if mins.empty:
        return 0
    vc = mins.value_counts()
    max_count = vc.max()
    cands = vc[vc == max_count].index.tolist()
    return 51 if 51 in cands else max(cands)

def _expected_full_day_times(date_et, minute: int):
    times = [pd.Timestamp.combine(date_et, dtime(h, minute)) for h in range(1, 24)]
    times.append(pd.Timestamp.combine(date_et + timedelta(days=1), dtime(0, minute)))
    return times

def prepare_lamp_day_table_aligned(lamp_df: pd.DataFrame, date_et, metar_cols):
    expected = {"valid_time_est", "observation_time_utc", "temp_F"}
    missing = expected - set(lamp_df.columns)
    if missing:
        return pd.DataFrame({"ERROR": [f"Missing columns in LAMP CSV: {', '.join(sorted(missing))}"]}), None, None

    l = _ensure_dt_and_make_run_est(lamp_df)
    daymask = l["valid_time_est"].dt.date.isin([date_et, date_et + timedelta(days=1)])
    day = l[daymask].copy()
    if day.empty:
        return pd.DataFrame([[""] * len(metar_cols)], index=["EST TIME (LAMP)"], columns=metar_cols), None, None

    lamp_minute = _pick_best_lamp_minute(day["valid_time_est"])
    pivot = day.pivot_table(index="run_est", columns="valid_time_est", values="temp_F", aggfunc="first")
    pivot = pivot.sort_index(ascending=False)

    source_dts = []
    for col in metar_cols:
        hh = int(col.split(":")[0])
        lamp_hour = (hh + 1) % 24
        base_date = date_et + (timedelta(days=1) if hh == 23 else timedelta(0))
        source_dts.append(pd.Timestamp.combine(base_date, dtime(lamp_hour, lamp_minute)))

    pivot = pivot.reindex(columns=sorted(set(pivot.columns) | set(source_dts)))

    aligned_num = pd.DataFrame(index=pivot.index, columns=metar_cols, dtype=float)
    for tgt_label, src_dt in zip(metar_cols, source_dts):
        aligned_num[tgt_label] = pivot[src_dt]

    # drop empty runs
    nonempty_mask = aligned_num.notna().any(axis=1)
    aligned_num = aligned_num.loc[nonempty_mask]

    def fmt2(x): return "" if pd.isna(x) else f"{float(x):.2f}"
    aligned_disp = aligned_num.applymap(fmt2)
    aligned_disp.index = [_label_from_run_est(ts) for ts in aligned_num.index]

    header_labels = [dt.strftime("%H:%M") for dt in source_dts]
    top = pd.DataFrame([header_labels], index=["EST TIME (LAMP)"], columns=metar_cols)
    display_df = pd.concat([top, aligned_disp], axis=0)
    return display_df, aligned_num, source_dts

# ---- Red highlight logic ----
def red_columns_from_full_day_run(lamp_df: pd.DataFrame, date_et, metar_cols):
    l = _ensure_dt_and_make_run_est(lamp_df)
    daymask = l["valid_time_est"].dt.date.isin([date_et, date_et + timedelta(days=1)])
    day = l[daymask].copy()
    if day.empty:
        return [], False

    minute = _pick_best_lamp_minute(day["valid_time_est"])
    expected_times = _expected_full_day_times(date_et, minute)

    pivot = day.pivot_table(index="run_est", columns="valid_time_est", values="temp_F", aggfunc="first")
    pivot = pivot.reindex(columns=sorted(set(pivot.columns) | set(expected_times)))
    if pivot.empty:
        return [], False

    full_day_mask = pivot[expected_times].notna().all(axis=1)
    full_day_runs = pivot.index[full_day_mask].tolist()
    if not full_day_runs:
        return [], False

    chosen_run = min(full_day_runs)
    s = pivot.loc[chosen_run, expected_times]
    top3 = s.nlargest(3)

    source_dts = []
    for col in metar_cols:
        hh = int(col.split(":")[0])
        lamp_hour = (hh + 1) % 24
        base_date = date_et + (timedelta(days=1) if hh == 23 else timedelta(0))
        source_dts.append(pd.Timestamp.combine(base_date, dtime(lamp_hour, minute)))
    dt_to_col = dict(zip(source_dts, metar_cols))

    red_cols = [dt_to_col[dt] for dt in top3.index if dt in dt_to_col]
    return red_cols, True

def style_columns(df: pd.DataFrame, red_cols):
    def _col_style(col: pd.Series):
        if col.name in red_cols:
            return ['color: #d11; font-weight:700;'] * len(col)
        return [''] * len(col)
    return df.style.apply(_col_style, axis=0)

def compute_global_biases(lamp_df: pd.DataFrame, metar_df: pd.DataFrame) -> dict:
    """
    Compute mean (forecast - observed) by 30-min lead time buckets using ALL rows.
    Matching rule: for each LAMP valid_time_est (naive ET), pair with the nearest
    METAR observation_time_et within ±20 minutes (tolerance adjustable).
    """
    need_lamp = {"valid_time_est", "observation_time_utc", "temp_F"}
    need_metar = {"observation_time_et", "temp_F"}
    if lamp_df.empty or not need_lamp.issubset(lamp_df.columns):
        return {}
    if metar_df.empty or not need_metar.issubset(metar_df.columns):
        return {}

    l = _ensure_dt_and_make_run_est(lamp_df).copy()
    l = l.dropna(subset=["valid_time_est", "run_est", "temp_F"])
    l = l.rename(columns={"temp_F": "fcst_temp_F"})
    l = l[["valid_time_est", "run_est", "fcst_temp_F"]].copy()

    m = _safe_to_datetime(metar_df.copy(), "observation_time_et")
    m = m.dropna(subset=["observation_time_et", "temp_F"])
    m = m.rename(columns={"temp_F": "obs_temp_F"})
    m = m[["observation_time_et", "obs_temp_F"]].copy()

    if l.empty or m.empty:
        return {}

    l = l.sort_values("valid_time_est")
    m = m.sort_values("observation_time_et")

    tol = pd.Timedelta(minutes=20)
    matched = pd.merge_asof(
        l, m,
        left_on="valid_time_est",
        right_on="observation_time_et",
        direction="nearest",
        tolerance=tol,
    )

    matched = matched.dropna(subset=["obs_temp_F"])
    if matched.empty:
        return {}

    lead_hours = (matched["valid_time_est"] - matched["run_est"]).dt.total_seconds() / 3600.0
    matched["lead_half_hours"] = (lead_hours * 2).round() / 2.0  # nearest 0.5h
    matched["err"] = matched["fcst_temp_F"].astype(float) - matched["obs_temp_F"].astype(float)

    grp = matched.groupby("lead_half_hours")["err"].mean()
    if grp.empty:
        return {}

    out = {}
    for lead, mean_err in grp.sort_index().items():
        total_minutes = int(round(abs(lead) * 60))
        hours, minutes = divmod(total_minutes, 60)
        label = f"-{hours:02d}:{minutes:02d}"
        out[label] = float(mean_err)
    return out

def display_bias_cards(bias_dict: dict, title="Forecast Bias by Lead Time"):
    if not bias_dict:
        st.info("No bias data available.")
        return
    st.subheader(title)
    items = list(bias_dict.items())
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]
    for row_items in chunks(items, 6):
        cols = st.columns(len(row_items))
        for (lead, bias), col in zip(row_items, cols):
            color = "#d11" if bias > 0 else "#1565c0"
            col.markdown(f"""
            <div style="
                border:1px solid #e5e7eb;
                border-radius:14px;
                padding:1rem;
                text-align:center;
                background:#fafafa;
                box-shadow:2px 2px 6px rgba(0,0,0,0.06);
                min-height:90px;
                display:flex;
                flex-direction:column;
                justify-content:center;
            ">
              <div style="font-size:.9rem;font-weight:600;opacity:.8;">{lead}</div>
              <div style="font-size:1.6rem;font-weight:700;color:{color};">
                {bias:.2f}°F
              </div>
            </div>
            """, unsafe_allow_html=True)

# ===== NEW: Bias-adjusted latest run helper =====
def _lead_label_from_minutes(minutes: float) -> str:
    """Convert +/- minutes to your '-HH:MM' bucket label using nearest 30-min increments."""
    half_hours = round((minutes / 60.0) * 2) / 2.0
    total_minutes = int(round(abs(half_hours) * 60))
    hh, mm = divmod(total_minutes, 60)
    return f"-{hh:02d}:{mm:02d}"

def prepare_bias_adjusted_latest_run_table(
    lamp_df: pd.DataFrame,
    date_et,
    metar_cols,
    bias_dict: dict
) -> pd.DataFrame:
    """
    Build a 3-row table for the *latest* GLMP run:
      1) GLMP (bias-adjusted) °F
      2) Bias applied (°F)
      3) Datetime (LAMP valid)
    Aligned to METAR/LAMP columns. Adjustment uses bucketed lead-time bias.
    """
    expected = {"valid_time_est", "observation_time_utc", "temp_F"}
    if lamp_df.empty or not expected.issubset(lamp_df.columns):
        return pd.DataFrame()

    l = _ensure_dt_and_make_run_est(lamp_df).copy()
    l = l.dropna(subset=["valid_time_est", "run_est"])

    if l.empty:
        return pd.DataFrame()

    latest_run = l["run_est"].max()
    run_rows = l[l["run_est"] == latest_run].copy()
    if run_rows.empty:
        return pd.DataFrame()

    # Prefer the minute mode within this run
    try:
        lamp_minute = int(run_rows["valid_time_est"].dt.minute.mode().iloc[0])
    except Exception:
        lamp_minute = _pick_best_lamp_minute(l["valid_time_est"])

    # Build the valid timestamps corresponding to each METAR column (like LAMP alignment)
    source_dts = []
    for col in metar_cols:
        hh = int(col.split(":")[0])
        lamp_hour = (hh + 1) % 24
        base_date = date_et + (timedelta(days=1) if hh == 23 else timedelta(0))
        source_dts.append(pd.Timestamp.combine(base_date, dtime(lamp_hour, lamp_minute)))

    # Map valid_time_est -> temp for latest run
    run_map = dict(zip(run_rows["valid_time_est"], run_rows["temp_F"]))

    # Compute adjusted forecasts + biases
    adj_vals = []
    bias_vals = []
    time_labels = []

    for vdt in source_dts:
        fcst = run_map.get(vdt, float("nan"))
        # Lead time in minutes for latest run
        lead_minutes = (vdt - latest_run).total_seconds() / 60.0
        label = _lead_label_from_minutes(lead_minutes)
        bias = float(bias_dict.get(label, 0.0))  # forecast - observed; subtract to correct
        if pd.isna(fcst):
            adj_vals.append("")
            bias_vals.append("")
        else:
            adj_vals.append(f"{(float(fcst) - bias):.2f}")
            bias_vals.append(f"{bias:+.2f}")
        time_labels.append(vdt.strftime("%H:%M"))

    df = pd.DataFrame(
        [adj_vals, bias_vals, time_labels],
        index=[
            f"{_label_from_run_est(latest_run)} (bias-adjusted °F)",
            "Bias applied (°F)",
            "Datetime (LAMP valid)",
        ],
        columns=metar_cols,
    )
    return df

# ---------- Load data ----------
try:
    lamp_df = load_csv(LAMP_URL)
except Exception as e:
    lamp_df = pd.DataFrame()
    st.sidebar.error(f"Failed to load LAMP CSV: {e}")

try:
    metar_df = load_csv(METAR_URL)
except Exception as e:
    metar_df = pd.DataFrame()
    st.sidebar.error(f"Failed to load METAR CSV: {e}")

# ---------- Sidebar ----------
st.sidebar.header("Controls")
location = st.sidebar.text_input("Location (ICAO)", value="KLGA")
st.sidebar.caption("Currently reads KLGA files from your URLs. The location label here is for display/context.")
selected_date = st.sidebar.date_input("Date (ET / UTC-4)", value=pick_default_date_et(metar_df))

st.sidebar.markdown("---")
st.sidebar.subheader("Data sources")
st.sidebar.write(f"**LAMP CSV:** {LAMP_URL}")
st.sidebar.write(f"**METAR CSV:** {METAR_URL}")

# ---------- Main ----------
st.title(f"METAR vs LAMP: Temperature ({location})")
st.caption("Times shown are **ET (UTC-4)**. Blanks indicate missing values at that time label.")

if metar_df.empty:
    st.warning("No METAR data loaded.")
    metar_table = pd.DataFrame()
else:
    metar_table = prepare_metar_day_table(metar_df, selected_date)

st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

# ---- Compute all-time bias once (used below) ----
global_bias = compute_global_biases(lamp_df=lamp_df, metar_df=metar_df)

if lamp_df.empty:
    st.info("LAMP CSV not available.")
else:
    metar_cols = list(metar_table.columns) if not metar_table.empty else [f"{h:02d}:00" for h in range(24)]

    # --- METAR table ---
    if not metar_table.empty:
        st.write(style_columns(metar_table, []), unsafe_allow_html=True)
    else:
        st.dataframe(metar_table, use_container_width=True)

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    # ===== NEW: Bias-adjusted latest GLMP run (inserted BETWEEN METAR and LAMP) =====
    bias_adj_df = prepare_bias_adjusted_latest_run_table(
        lamp_df=lamp_df,
        date_et=selected_date,
        metar_cols=metar_cols,
        bias_dict=global_bias
    )
    if not bias_adj_df.empty:
        st.write(style_columns(bias_adj_df, []), unsafe_allow_html=True)
        st.caption("Latest GLMP run adjusted by lead-time bias (bias = mean[forecast − observed] per bucket; we subtract it).")
    else:
        st.info("Bias-adjusted GLMP table unavailable for this date/run.")

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    # --- LAMP aligned table + highlights ---
    lamp_display, lamp_numeric, source_dts = prepare_lamp_day_table_aligned(lamp_df, selected_date, metar_cols)
    red_cols, has_full = red_columns_from_full_day_run(lamp_df, selected_date, metar_cols)

    if isinstance(lamp_display, pd.DataFrame) and not lamp_display.empty:
        st.write(style_columns(lamp_display, red_cols), unsafe_allow_html=True)
        st.caption("Top row shows the LAMP valid times used for each aligned METAR column.")
        if not has_full:
            st.caption("No GLMP run with full-day coverage (01→23 + next-day 00) — highlights disabled.")
    else:
        st.dataframe(lamp_display, use_container_width=True)

    # ---- ALL-TIME bias section ----
    display_bias_cards(global_bias, title="Forecast Bias by Lead Time (All-Time)")

# Optional raw previews
with st.expander("Show raw LAMP & METAR data"):
    if not lamp_df.empty:
        st.markdown("**LAMP temperature file (preview):**")
        st.dataframe(lamp_df.head(200), use_container_width=True)
    else:
        st.info("LAMP CSV not available.")
    if not metar_df.empty:
        st.markdown("**METAR file (preview):**")
        st.dataframe(metar_df.head(200), use_container_width=True)
    else:
        st.info("METAR CSV not available.")
