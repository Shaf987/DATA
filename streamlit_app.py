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

# ----- helpers for LAMP labels & sorting -----
def _label_from_file_path_est(file_path: str) -> str:
    """file_path 'glmp...tHHMMz...' -> 'GLMP HH:MM EST' (UTC->EST)."""
    m = re.search(r"t(\d{2})(\d{2})z", str(file_path))
    if not m:
        return "GLMP"
    hh, mm = int(m.group(1)), int(m.group(2))
    est_hh = (hh - 4) % 24
    return f"GLMP {est_hh:02d}:{mm:02d} EST"

def _est_minutes_from_file_path(file_path: str) -> int:
    """Return EST minutes since midnight from file_path's tHHMMz; -1 if not parseable."""
    m = re.search(r"t(\d{2})(\d{2})z", str(file_path))
    if not m:
        return -1
    hh_utc, mm = int(m.group(1)), int(m.group(2))
    hh_est = (hh_utc - 4) % 24
    return hh_est * 60 + mm

def prepare_lamp_day_table_aligned(lamp_df: pd.DataFrame, date_et, metar_cols):
    """
    LAMP table aligned to METAR columns:
      - Columns: exactly METAR columns (visual alignment).
      - First row ("EST TIME (LAMP)"): the *LAMP source time* used for each column (HH:MM).
      - Rows: one per file_path, values shown to 2 decimals (strings; blanks for NaN).
      - Row order: DESC by GLMP EST time (e.g., 15:30 → 06:30).
      - Mapping: for each METAR 'HH:mm', use LAMP at hour '(HH+1)' with day rollover;
                 minute = dominant LAMP minute for that date.
    Returns: (display_df, numeric_df, source_datetimes)
    """
    expected = {"valid_time_est", "file_path", "temp_F"}
    missing = expected - set(lamp_df.columns)
    if missing:
        return pd.DataFrame({"ERROR": [f"Missing columns in LAMP CSV: {', '.join(sorted(missing))}"]}), None, None

    l = _safe_to_datetime(lamp_df, "valid_time_est")

    # include next day to handle 23:51 -> next-day 00:MM mapping
    daymask = l["valid_time_est"].dt.date.isin([date_et, date_et + timedelta(days=1)])
    day = l[daymask].copy()
    if day.empty:
        return pd.DataFrame([[""] * len(metar_cols)], index=["EST TIME (LAMP)"], columns=metar_cols), None, None

    lamp_minute = int(day["valid_time_est"].dt.minute.mode().iloc[0])
    pivot = day.pivot_table(index="file_path", columns="valid_time_est", values="temp_F", aggfunc="first")

    # sort rows latest → earliest by GLMP EST time
    order = pd.Series({idx: _est_minutes_from_file_path(idx) for idx in pivot.index}) \
              .sort_values(ascending=False).index
    pivot = pivot.reindex(index=order)

    # build LAMP datetime used for each METAR column (hour+1, rollover if needed)
    source_dts = []
    for col in metar_cols:
        hh = int(col.split(":")[0])
        lamp_hour = (hh + 1) % 24
        base_date = date_et + (timedelta(days=1) if hh == 23 else timedelta(0))
        source_dts.append(pd.Timestamp.combine(base_date, dtime(lamp_hour, lamp_minute)))

    # ensure those datetime columns exist
    pivot = pivot.reindex(columns=sorted(set(pivot.columns) | set(source_dts)))

    # aligned numeric values (float)
    aligned_num = pd.DataFrame(index=pivot.index, columns=metar_cols, dtype=float)
    for tgt_label, src_dt in zip(metar_cols, source_dts):
        aligned_num[tgt_label] = pivot[src_dt]

    # pretty display: force two decimals everywhere numbers appear
    def fmt2(x):
        return "" if pd.isna(x) else f"{float(x):.2f}"
    aligned_disp = aligned_num.applymap(fmt2)
    aligned_disp.index = [_label_from_file_path_est(fp) for fp in aligned_disp.index]

    header_labels = [dt.strftime("%H:%M") for dt in source_dts]
    top = pd.DataFrame([header_labels], index=["EST TIME (LAMP)"], columns=metar_cols)

    display_df = pd.concat([top, aligned_disp], axis=0)
    return display_df, aligned_num, source_dts

# ---- Top-3 hottest hours from the earliest GLMP run (all red) ----
def top3_hot_hours_from_earliest(lamp_df: pd.DataFrame, date_et, metar_cols):
    """
    Returns (red_cols, orange_cols, full_day) where:
      - red_cols: the 3 hottest hours (METAR column labels) from the *earliest* GLMP run
      - orange_cols: empty (not used)
      - full_day: True only if the earliest GLMP run has coverage for ALL mapped hours (no NaN)
    If the earliest run does not cover the entire day, returns no highlights and full_day=False.
    """
    l = _safe_to_datetime(lamp_df.copy(), "valid_time_est")
    daymask = l["valid_time_est"].dt.date.isin([date_et, date_et + timedelta(days=1)])
    day = l[daymask].copy()
    if day.empty:
        return [], [], False

    lamp_minute = int(day["valid_time_est"].dt.minute.mode().iloc[0])
    pivot = day.pivot_table(index="file_path", columns="valid_time_est", values="temp_F", aggfunc="first")

    # Map METAR columns to LAMP datetimes (same mapping used for the table)
    source_dts = []
    for col in metar_cols:
        hh = int(col.split(":")[0])
        lamp_hour = (hh + 1) % 24
        base_date = date_et + (timedelta(days=1) if hh == 23 else timedelta(0))
        source_dts.append(pd.Timestamp.combine(base_date, dtime(lamp_hour, lamp_minute)))

    pivot = pivot.reindex(columns=sorted(set(pivot.columns) | set(source_dts)))

    # Earliest GLMP run by EST time
    order_asc = pd.Series({idx: _est_minutes_from_file_path(idx) for idx in pivot.index}) \
                  .sort_values(ascending=True).index
    if len(order_asc) == 0:
        return [], [], False
    earliest_idx = order_asc[0]

    # Require *full-day* coverage (no NaN in any mapped hour)
    s_all = pivot.loc[earliest_idx, source_dts]
    full_day = s_all.notna().all()
    if not full_day:
        return [], [], False

    # With full coverage, pick top-3 hottest hours (no NaN present)
    dt_to_col = dict(zip(source_dts, metar_cols))
    top3 = s_all.nlargest(min(3, len(s_all)))
    cols_in_order = [dt_to_col[dt] for dt in top3.index]

    red_cols = cols_in_order[:3]   # all three in red
    orange_cols = []               # none in orange
    return red_cols, orange_cols, True

# ---- Styling helpers (color full columns) ----
def style_columns(df: pd.DataFrame, red_cols, orange_cols):
    def _col_style(col: pd.Series):
        if col.name in red_cols:
            return ['color: #d11; font-weight:700;'] * len(col)
        if col.name in orange_cols:
            return ['color: #d97900; font-weight:700;'] * len(col)
        return [''] * len(col)
    return df.style.apply(_col_style, axis=0)

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

    # compute top-3 hottest hours from earliest GLMP run
    red_cols, orange_cols, full_day = top3_hot_hours_from_earliest(lamp_df, selected_date, metar_cols)

    # render METAR with column highlights (values already formatted to 2 decimals)
    if not metar_table.empty:
        st.write(style_columns(metar_table, red_cols, orange_cols), unsafe_allow_html=True)
        if not full_day:
            st.caption("Earliest GLMP run does not cover the full day — red highlights disabled for today.")
    else:
        st.dataframe(metar_table, use_container_width=True)

    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    # render LAMP with column highlights (values formatted to 2 decimals)
    if isinstance(lamp_display, pd.DataFrame) and not lamp_display.empty:
        st.write(style_columns(lamp_display, red_cols, orange_cols), unsafe_allow_html=True)
        st.caption("Top row shows the original LAMP valid times used for each aligned METAR column (with day rollover handled).")
    else:
        st.dataframe(lamp_display, use_container_width=True)

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
