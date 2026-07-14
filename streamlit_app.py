import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import requests
import streamlit as st

from fetch_weather_hourly_cache import API_URL, extract_business_hours

st.set_page_config(
    page_title="Kohi Scheduler",
    page_icon="📅",
    layout="wide",
)

# Streamlit's own columns give each column min-width: calc(100% - 24px), so
# below ~640px viewport width they always stack vertically no matter how the
# app calls st.columns(). Same scoped override as the pastry planner: keep
# rows as rows on phones and let overflow become a horizontal swipe.
st.markdown("""
<style>
@media (max-width: 640px) {
  [data-testid="stHorizontalBlock"] {
      flex-wrap: nowrap !important;
      overflow-x: auto !important;
      -webkit-overflow-scrolling: touch;
      gap: 0.6rem !important;
      padding-bottom: 4px;
  }
  [data-testid="stColumn"] {
      min-width: 118px !important;
      flex: 0 0 auto !important;
      width: auto !important;
  }
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────

# All day-boundary logic runs on Kohi local time, never the server clock —
# Streamlit Cloud containers run in UTC and would otherwise flip "today"
# at 6/7 PM Central.
CENTRAL = ZoneInfo("America/Chicago")

MODEL_PATH = Path(__file__).parent / "models" / "Hourly_Items_LightGBM_1.pkl"
CACHE_PATH = Path(__file__).parent / "data" / "weather_hourly_cache.json"
CACHE_MAX_AGE_HOURS = 24

HOURS = list(range(6, 15))          # business hour bins: 6 AM open → 2 PM hour
HOUR_LABELS = ["6 AM", "7 AM", "8 AM", "9 AM", "10 AM",
               "11 AM", "12 PM", "1 PM", "2 PM"]
DAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

# 37-column dummy-encoded feature order the rank-1 LightGBM model was trained
# on (goal2_scheduler_modeling/goal2_main.py). Reference (dropped) levels:
# weather=clear, day=FRIDAY, weekday/weekend=Weekday, event=No, month=April,
# hour=6.
DUMMY_COLS = [
    "temp_f", "precip_in", "humidity_pct", "wind_mph",
    "Total Occ %", "Group Occ %", "Transient Occ %",
    "weather_condition_cloudy", "weather_condition_rainy", "weather_condition_snowy",
    "Day of Week_MONDAY", "Day of Week_SATURDAY", "Day of Week_SUNDAY",
    "Day of Week_THURSDAY", "Day of Week_TUESDAY", "Day of Week_WEDNESDAY",
    "Weekday or Weekend_Weekend",
    "Event_Yes",
    "Month_August", "Month_December", "Month_February", "Month_January",
    "Month_July", "Month_June", "Month_March", "Month_May",
    "Month_November", "Month_October", "Month_September",
    "Hour_7", "Hour_8", "Hour_9", "Hour_10", "Hour_11",
    "Hour_12", "Hour_13", "Hour_14",
]

# Last-resort weather if both the cache and the live API fail: per-hour
# medians from the training data (sum_hour_kohi.csv). Weather is a background
# feature in this app, so a typical-day stand-in beats blocking the forecast.
TYPICAL_WEATHER = {
    6:  {"temp_f": 54.2, "precip_in": 0.0, "humidity_pct": 82.5, "wind_mph": 7.2, "weather_condition": "clear"},
    7:  {"temp_f": 54.0, "precip_in": 0.0, "humidity_pct": 84.0, "wind_mph": 7.3, "weather_condition": "clear"},
    8:  {"temp_f": 54.1, "precip_in": 0.0, "humidity_pct": 83.0, "wind_mph": 7.2, "weather_condition": "clear"},
    9:  {"temp_f": 56.9, "precip_in": 0.0, "humidity_pct": 78.0, "wind_mph": 7.4, "weather_condition": "clear"},
    10: {"temp_f": 59.6, "precip_in": 0.0, "humidity_pct": 71.0, "wind_mph": 7.5, "weather_condition": "clear"},
    11: {"temp_f": 62.9, "precip_in": 0.0, "humidity_pct": 65.0, "wind_mph": 7.8, "weather_condition": "cloudy"},
    12: {"temp_f": 65.0, "precip_in": 0.0, "humidity_pct": 59.0, "wind_mph": 8.3, "weather_condition": "cloudy"},
    13: {"temp_f": 67.6, "precip_in": 0.0, "humidity_pct": 54.0, "wind_mph": 8.2, "weather_condition": "cloudy"},
    14: {"temp_f": 68.1, "precip_in": 0.0, "humidity_pct": 53.0, "wind_mph": 8.2, "weather_condition": "cloudy"},
}

# Sequential ramp in the brand red: one hue, light→dark — magnitude only.
KOHI_CMAP = LinearSegmentedColormap.from_list(
    "kohi_red", ["#FCF5F3", "#EFC4B8", "#D97C63", "#B0402E", "#8B1A1A"])

# ── Weather: cached file first, live API fallback (background only) ────────────
# A GitHub Actions job (.github/workflows/weather_cache.yml) refreshes
# data/weather_hourly_cache.json at 5 AM Central daily. The scheduler never
# shows weather to the user — it's fetched here and fed straight to the model.

def load_cached_hourly() -> dict | None:
    """{date_iso: {hour_str: weather}} if the cache file exists and is
    fresher than CACHE_MAX_AGE_HOURS, else None."""
    try:
        cache = json.loads(CACHE_PATH.read_text())
        generated = datetime.fromisoformat(cache["generated_at"])
        age_hours = (datetime.now(CENTRAL) - generated).total_seconds() / 3600
        if age_hours > CACHE_MAX_AGE_HOURS or not cache.get("days"):
            return None
        return cache["days"]
    except Exception:
        return None


@st.cache_data(ttl=1800)
def fetch_live_hourly() -> dict:
    """Live Open-Meteo fallback — same extraction as the cache builder."""
    r = requests.get(API_URL, timeout=15)
    r.raise_for_status()
    return extract_business_hours(r.json()["hourly"])


def get_hourly_weather(week_dates: list) -> tuple[dict, str]:
    """Hourly weather for the requested dates, {date_iso: {hour_str: w}}.
    Falls back cache → live API → per-hour training medians."""
    need = {d.isoformat() for d in week_dates}
    cached = load_cached_hourly()
    if cached and need.issubset(cached.keys()):
        return cached, "cached"
    try:
        live = fetch_live_hourly()
        if need.issubset(live.keys()):
            return live, "live"
    except Exception:
        pass
    typical = {d.isoformat(): {str(h): TYPICAL_WEATHER[h] for h in HOURS}
               for d in week_dates}
    return typical, "typical"

# ── Model ──────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ── Feature engineering ────────────────────────────────────────────────────────

def build_feature_row(w: dict, occ_total: float, occ_group: float,
                      occ_transient: float, event: str, data_date, hour: int) -> dict:
    dow      = data_date.strftime("%A").upper()
    is_wkend = dow in ("SATURDAY", "SUNDAY")
    month    = data_date.strftime("%B")
    cond     = w["weather_condition"]

    row = dict.fromkeys(DUMMY_COLS, 0)
    row.update({
        "temp_f":          w["temp_f"],
        "precip_in":       w["precip_in"],
        "humidity_pct":    w["humidity_pct"],
        "wind_mph":        w["wind_mph"],
        "Total Occ %":     occ_total,
        "Group Occ %":     occ_group,
        "Transient Occ %": occ_transient,
    })
    if cond != "clear" and f"weather_condition_{cond}" in row:
        row[f"weather_condition_{cond}"] = 1
    if f"Day of Week_{dow}" in row:
        row[f"Day of Week_{dow}"] = 1
    if is_wkend:
        row["Weekday or Weekend_Weekend"] = 1
    if event == "Yes":
        row["Event_Yes"] = 1
    if month != "April" and f"Month_{month}" in row:
        row[f"Month_{month}"] = 1
    if hour != 6:
        row[f"Hour_{hour}"] = 1
    return row

# ── Heatmap ────────────────────────────────────────────────────────────────────

def plot_week_heatmap(matrix: np.ndarray, week_start, week_end):
    """Sun-Sat × business-hour heatmap; hour labels mirrored on both sides."""
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    vmax = max(matrix.max(), 1)
    im = ax.imshow(matrix, cmap=KOHI_CMAP, aspect="auto", vmin=0, vmax=vmax)

    ax.set_xticks(range(len(DAY_LABELS)))
    ax.set_xticklabels(DAY_LABELS, fontsize=10)
    ax.set_yticks(range(len(HOUR_LABELS)))
    ax.set_yticklabels(HOUR_LABELS, fontsize=9)
    # Hour labels mirrored left/right, day labels mirrored top/bottom
    ax.tick_params(left=True, right=True, labelleft=True, labelright=True,
                   top=True, bottom=True, labeltop=True, labelbottom=True)

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            v = matrix[r, c]
            ink = "white" if v > 0.55 * vmax else "#1A1A1A"
            ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                    fontsize=9, color=ink)

    ax.set_title(
        f"Forecasted Items per Hour — {week_start.strftime('%b %d')} – "
        f"{week_end.strftime('%b %d, %Y')}", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.09)
    cbar.set_label("Items per hour", fontsize=9)
    plt.tight_layout()
    return fig

# ── Staffing insights ──────────────────────────────────────────────────────────

def hour_label(h: int) -> str:
    return HOUR_LABELS[HOURS.index(h)]


def staffing_insights_md(matrix: np.ndarray) -> str:
    col_totals = matrix.sum(axis=0)   # per day
    row_means  = matrix.mean(axis=1)  # per hour

    peak_day  = DAY_LABELS[int(col_totals.argmax())]
    quiet_day = DAY_LABELS[int(col_totals.argmin())]
    peak_hour  = hour_label(HOURS[int(row_means.argmax())])
    quiet_hour = hour_label(HOURS[int(row_means.argmin())])

    rush = [h for h, v in zip(HOURS, row_means) if v >= 0.8 * row_means.max()]
    rush_window = f"{hour_label(rush[0])}–{hour_label(rush[-1])}" if rush else "n/a"

    peak_by_day = "  \n".join(
        f"&nbsp;&nbsp;• **{DAY_LABELS[c]}**: {hour_label(HOURS[int(matrix[:, c].argmax())])}"
        f" (~{matrix[:, c].max():.0f} items)"
        for c in range(7))

    return f"""
- **Busiest day: {peak_day}** (~{col_totals.max():.0f} items for the day) — schedule the fullest lineup here; lightest day is **{quiet_day}** (~{col_totals.min():.0f} items).
- **Rush window: {rush_window}** — these hours run at ≥80% of peak volume. Keep full coverage through the whole window, not just the single busiest hour.
- **Peak hour overall: {peak_hour}** (~{row_means.max():.0f} items on an average day); **quietest hour: {quiet_hour}** (~{row_means.min():.0f} items) — best slot for breaks, prep, and restocking.
- **Peak hour by day:**  \n{peak_by_day}
"""

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    today = datetime.now(CENTRAL).date()

    # Next full Sunday-Saturday week, always strictly in the future
    # (Mon Jul 13 → Sun Jul 19; a Sunday rolls to the following Sunday).
    days_to_sunday = (6 - today.weekday()) % 7 or 7
    week_start = today + timedelta(days=days_to_sunday)
    week_end   = week_start + timedelta(days=6)
    week_dates = [week_start + timedelta(days=i) for i in range(7)]

    st.title("📅 Kohi Scheduler")
    st.markdown(
        f"**Peak-hour forecast for next week:** "
        f"{week_start.strftime('%A, %B %d')} – {week_end.strftime('%A, %B %d, %Y')}")
    st.divider()

    st.subheader("🏨 Hotel Occupancy & Events")
    st.caption(
        "From your weekly hotel email: enter each day's projected occupancy "
        "(use the previous day's figure — those guests are the next day's customers), "
        "and flip Event or Holiday to Yes where it applies. "
        "Hourly weather is loaded automatically in the background."
    )

    edit_df = pd.DataFrame([{
        "Day":              d.strftime("%a %b %d"),
        "Total Occ %":      70.0,
        "Group Occ %":      20.0,
        "Transient Occ %":  50.0,
        "Event or Holiday": "No",
    } for d in week_dates])

    edited = st.data_editor(
        edit_df,
        disabled=["Day"],
        column_config={
            "Total Occ %":      st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
            "Group Occ %":      st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
            "Transient Occ %":  st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
            "Event or Holiday": st.column_config.SelectboxColumn(options=["No", "Yes"]),
        },
        hide_index=True,
        use_container_width=True,
        key="week_editor",
    )
    st.divider()

    if st.button("📅  Forecast Next Week", type="primary",
                 use_container_width=True, key="predict"):
        with st.spinner("Forecasting all 63 hours…"):
            weather, wx_source = get_hourly_weather(week_dates)
            model = load_model()

            rows = []
            for c, d in enumerate(week_dates):
                day_wx  = weather.get(d.isoformat(), {})
                day_in  = edited.iloc[c]
                for h in HOURS:
                    w = day_wx.get(str(h), TYPICAL_WEATHER[h])
                    rows.append(build_feature_row(
                        w,
                        float(day_in["Total Occ %"]),
                        float(day_in["Group Occ %"]),
                        float(day_in["Transient Occ %"]),
                        str(day_in["Event or Holiday"]),
                        d, h,
                    ))

            X = pd.DataFrame(rows)[DUMMY_COLS]
            preds = np.clip(model.predict(X), 0, None)
            matrix = preds.reshape(7, len(HOURS)).T  # rows=hours, cols=days

        st.pyplot(plot_week_heatmap(matrix, week_start, week_end),
                  use_container_width=True)

        st.subheader("👥 Staffing Insights")
        st.markdown(staffing_insights_md(matrix))

        note = {"cached": "", "live": "",
                "typical": "  ·  ⚠️ live weather unavailable — typical seasonal weather used"}[wx_source]
        st.caption(
            f"Generated {datetime.now(CENTRAL).strftime('%I:%M %p')} Central  ·  "
            f"model: LightGBM (Goal 2 rank 1){note}"
        )


if __name__ == "__main__":
    main()
