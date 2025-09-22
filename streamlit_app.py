# app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time as dtime
import re
import math

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

@st.cache_data(ttl=300)
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
    """
    Return valid_time_est as **naive EST**:
      - If tz-aware, convert to UTC -> drop tz -> subtract 4h.
      - If tz-naive, assume it's already EST and leave as-is.
    """
    s = pd.to_datetime(series, errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(s):
        # aware: bring to naive UTC, then shift to naive EST
        s = s.dt.tz_convert("UTC").dt.tz_localize(None) - pd.Timedelta(hours=4)
    # if naive: already EST by contract of this CSV
    return s

def _ensure_dt_and_make_run_est(lamp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure datetime types and add 'run_est' **naive EST (UTC-4)**:
      - observation_time_utc (UTC) -> aware UTC -> drop tz -> subtract 4h
      - valid_time_est -> force to naive EST (handle aware inputs)
    """
    df = lamp_df.copy()
    # observation_time_utc as aware UTC
    obs_utc = pd.to_datetime(df["observation_time_utc"], errors="coerce", utc=True)
    # run_est = naive EST
    df["run_est"] = obs_utc.dt.tz_convert(None) - pd.Timedelta(hours=4)

    # valid_time_est as naive EST (handles aware/naive inputs)
    df["valid_time_est"] = _standardize_valid_est(df["valid_time_est"])
    return df

def _pick_best_lamp_minute(valid_time_est_series: pd.Series) -> int:
    """
    Choose the minute used for alignment:
    - most frequent minute; tie-breaker: prefer :51 else the largest minute.
    """
    mins = valid_time_est_series.dropna().dt.minute.astype(int)
    if mins.empty:
        return 0
    vc = mins.value_counts()
    max_count = vc.max()
    cands = vc[vc == max_count].index.tolist()
    return 51 if 51 in cands else max(cands)

def _expected_full_day_times(date_et, minute: int):
    """Return the 24 expected valid_time_est datetimes: 01..23 on date, + 00 on next day, all at `minute`."""
    times = [pd.Timestamp.combine(date_et, dtime(h, minute)) for h in range(1, 24)]
    times.append(pd.Timestamp.combine(date_et + timedelta(days=1), dtime(0, minute)))
    return times  # length 24

def prepare_lamp_day_table_aligned(lamp_df: pd.DataFrame, date_et, metar_cols):
    """
    Build the display & numeric LAMP tables aligned to METAR columns.
    Row order: newest run first by run_est (all naive EST).
    Drops runs that have no data for the mapped hours.
    """
    expected = {"valid_time_est", "observation_time_utc", "temp_F"}
    missing = expected - set(lamp_df.columns)
    if missing:
        return pd.DataFrame({"ERROR": [f"Missing columns in LAMP CSV: {', '.join(sorted(missing))}"]}), None, None

    l = _ensure_dt_and_make_run_est(lamp_df)

    # include next day for 23->00 rollover
    daymask = l["valid_time_est"].dt.date.isin([date_et, date_et + timedelta(days=1)])
    day = l[daymask].copy()
    if day.empty:
        return pd.DataFrame([[""] * len(metar_cols)], index=["EST TIME (LAMP)"], columns=metar_cols), None, None

    lamp_minute = _pick_best_lamp_minute(day["valid_time_est"])

    # Pivot: index=run_est (naive EST), columns=valid_time_est (naive EST)
    pivot = day.pivot_table(index="run_est", columns="valid_time_est", values="temp_F", aggfunc="first")
    pivot = pivot.sort_index(ascending=False)  # newest run first

    # Build the LAMP datetimes used for each METAR column (hour+1 mapping)
    source_dts = []
    for col in metar_cols:
        hh = int(col.split(":")[0])
        lamp_hour = (hh + 1) % 24
        base_date = date_et + (timedelta(days=1) if hh == 23 else timedelta(0))
        source_dts.append(pd.Timestamp.combine(base_date, dtime(lamp_hour, lamp_minute)))

    # make sure all these LAMP dts exist in pivot columns
    pivot = pivot.reindex(columns=sorted(set(pivot.columns) | set(source_dts)))

    # aligned numeric values (float)
    aligned_num = pd.DataFrame(index=pivot.index, columns=metar_cols, dtype=float)
    for tgt_label, src_dt in zip(metar_cols, source_dts):
        aligned_num[tgt_label] = pivot[src_dt]

    # ---- drop runs that have no data for any mapped hour (empty rows) ----
    nonempty_mask = aligned_num.notna().any(axis=1)
    aligned_num = aligned_num.loc[nonempty_mask]

    # pretty display (2 decimals) + row labels
    def fmt2(x): return "" if pd.isna(x) else f"{float(x):.2f}"
    aligned_disp = aligned_num.applymap(fmt2)
    aligned_disp.index = [_label_from_run_est(ts) for ts in aligned_num.index]

    # top header row shows the LAMP valid times used per column
    header_labels = [dt.strftime("%H:%M") for dt in source_dts]
    top = pd.DataFrame([header_labels], index=["EST TIME (LAMP)"], columns=metar_cols)

    display_df = pd.concat([top, aligned_disp], axis=0)
    return display_df, aligned_num, source_dts

# ---- Red highlight logic: choose ONE run with FULL-DAY coverage and mark its top-3 hours ----
def red_columns_from_full_day_run(lamp_df: pd.DataFrame, date_et, metar_cols):
    """
    Your rule:
      - find a run with full-day coverage (01..23 on date + 00 next day at consistent minute)
      - from that run, pick top-3 temps
      - map those times to aligned columns
      - color those columns; else none
    """
    l = _ensure_dt_and_make_run_est(lamp_df)

    # keep only valid times for date and next day (for 00)
    daymask = l["valid_time_est"].dt.date.isin([date_et, date_et + timedelta(days=1)])
    day = l[daymask].copy()
    if day.empty:
        return [], False

    minute = _pick_best_lamp_minute(day["valid_time_est"])
    expected_times = _expected_full_day_times(date_et, minute)

    # pivot by run_est so each row is a run; ensure expected columns exist
    pivot = day.pivot_table(index="run_est", columns="valid_time_est", values="temp_F", aggfunc="first")
    pivot = pivot.reindex(columns=sorted(set(pivot.columns) | set(expected_times)))

    if pivot.empty:
        return [], False

    # runs that have ALL 24 expected times present (no NaN)
    full_day_mask = pivot[expected_times].notna().all(axis=1)
    full_day_runs = pivot.index[full_day_mask].tolist()
    if not full_day_runs:
        return [], False

    # choose the earliest run among full-day runs
    chosen_run = min(full_day_runs)

    # top-3 temps within that run among the expected 24 hours
    s = pivot.loc[chosen_run, expected_times]
    top3 = s.nlargest(3)  # Series indexed by the LAMP datetimes

    # build the mapping used by the display: METAR col -> LAMP datetime (inverse mapping for lookup)
    source_dts = []
    for col in metar_cols:
        hh = int(col.split(":")[0])
        lamp_hour = (hh + 1) % 24
        base_date = date_et + (timedelta(days=1) if hh == 23 else timedelta(0))
        source_dts.append(pd.Timestamp.combine(base_date, dtime(lamp_hour, minute)))
    dt_to_col = dict(zip(source_dts, metar_cols))

    red_cols = [dt_to_col[dt] for dt in top3.index if dt in dt_to_col]
    return red_cols, True

# ---- Styling helpers (color full columns) ----
def style_columns(df: pd.DataFrame, red_cols):
    def _col_style(col: pd.Series):
        if col.name in red_cols:
            return ['color: #d11; font-weight:700;'] * len(col)
        return [''] * len(col)
    return df.style.apply(_col_style, axis=0)

# ---- Bias computation (robust, tz-clean, aligned with your table) ----
def compute_biases_by_lead_from_aligned(lamp_numeric: pd.DataFrame,
                                        source_dts: list,
                                        metar_df: pd.DataFrame,
                                        date_et,
                                        metar_cols: list) -> dict:
    """
    Compute bias per lead time using the already-aligned LAMP table:
      - lamp_numeric: rows=run_est (naive EST), cols=metar_cols (HH:MM)
      - source_dts:   LAMP valid timestamps (naive EST) for each metar column
      - metar_df:     used to fetch actual METAR obs per metar column
      - bias = mean(forecast - observed) grouped by lead time (rounded to nearest 0.5h),
               where lead = source_dt - run_est (both in naive EST)
    Returns: dict { "-0.5h": bias, "-1.5h": bias, ... } sorted by increasing lead.
    """
    if lamp_numeric is None or lamp_numeric.empty:
        return {}

    # Build METAR map for the selected date keyed by "HH:MM" (use actual minute mode for that date)
    m = _safe_to_datetime(metar_df.copy(), "observation_time_et")
    metar_day = m[m["observation_time_et"].dt.date == date_et].copy()
    if metar_day.empty:
        return {}

    metar_map = dict(zip(metar_day["observation_time_et"].dt.strftime("%H:%M"), metar_day["temp_F"]))

    # Pair each METAR column with its LAMP valid datetime
    col_to_dt = dict(zip(metar_cols, source_dts))

    # Aggregate errors by half-hour lead buckets
    bucket_errors = {}  # { numeric_lead_hours: [errors] }

    for run_est, row in lamp_numeric.iterrows():
        for col in metar_cols:
            fcst = row[col]
            if pd.isna(fcst):
                continue

            # obs at METAR column time (same label used in the table)
            obs = metar_map.get(col, None)
            if obs is None or pd.isna(obs):
                continue

            # lead = LAMP valid dt - run_est (both naive EST), bucket to nearest 0.5h
            src_dt = col_to_dt[col]
            lead_hours = (src_dt - run_est).total_seconds() / 3600.0
            # Round to nearest 0.5 h robustly
            lead_half_hours = round(lead_hours * 2) / 2.0
            bucket_errors.setdefault(lead_half_hours, []).append(fcst - obs)

    if not bucket_errors:
        return {}

    # Mean bias per bucket; display as negative “time before outcome” (–Xh)
    # Sorting by increasing lead (0.5, 1.0, 1.5, ...)
    out = {}
    for lead in sorted(bucket_errors.keys()):
        vals = bucket_errors[lead]
        if not vals:
            continue
        mean_err = float(sum(vals) / len(vals))
        label = f"–{lead:.1f}h"
        out[label] = mean_err
    return out

def display_bias_cards(bias_dict: dict):
    if not bias_dict:
        st.info("No bias data available for the selected day.")
        return

    st.subheader("Forecast Bias by Lead Time")
    items = list(bias_dict.items())

    # chunk into rows of up to 6 cards for nice layout
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
st.caption("Times shown are **ET (UTC-4)** (naive). Blanks indicate missing values at that time label.")

# METAR table (build first; styling later)
if metar_df.empty:
    st.warning("No METAR data loaded.")
    metar_table = pd.DataFrame()
else:
    metar_table = prepare_metar_day_table(metar_df, selected_date)

st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

# LAMP table (aligned + display)
if lamp_df.empty:
    st.info("LAMP CSV not available.")
else:
    metar_cols = list(metar_table.columns) if not metar_table.empty else [f"{h:02d}:00" for h in range(24)]
    lamp_display, lamp_numeric, source_dts = prepare_lamp_day_table_aligned(lamp_df, selected_date, metar_cols)

    # compute highlight columns strictly from a FULL-DAY run
    red_cols, has_full = red_columns_from_full_day_run(lamp_df, selected_date, metar_cols)

    # render METAR with column highlights
    if not metar_table.empty:
        st.write(style_columns(metar_table, red_cols), unsafe_allow_html=True)
        if not has_full:
            st.caption("No GLMP run with full-day coverage (01→23 + next-day 00) — highlights disabled.")
    else:
        st.dataframe(metar_table, use_container_width=True)

    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    # render LAMP with column highlights
    if isinstance(lamp_display, pd.DataFrame) and not lamp_display.empty:
        st.write(style_columns(lamp_display, red_cols), unsafe_allow_html=True)
        st.caption("Top row shows the LAMP valid times used for each aligned METAR column.")
    else:
        st.dataframe(lamp_display, use_container_width=True)

    # ---- Bias section (uses the aligned numeric table to avoid tz mismatches) ----
    if lamp_numeric is not None and not lamp_numeric.empty and not metar_df.empty:
        bias_dict = compute_biases_by_lead_from_aligned(
            lamp_numeric=lamp_numeric,
            source_dts=source_dts,
            metar_df=metar_df,
            date_et=selected_date,
            metar_cols=metar_cols,
        )
        display_bias_cards(bias_dict)

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
