# Kohi Scheduler

Streamlit app for Kohi Coffee's manager: forecasts item volume for every
business hour (6 AM–3 PM) of the **next full Sunday–Saturday week** so
staffing can be planned ahead. Goal 2 of the MABA Practicum II project
(peak-hour forecasting), Kohi location only.

## How it works

- **Inputs**: projected hotel occupancy (Total / Group / Transient %) and
  an Event-or-Holiday flag per day — 7 rows in one editable table.
- **Weather**: hourly Bentonville, AR weather is loaded automatically in
  the background (never shown) from `data/weather_hourly_cache.json`,
  refreshed daily at 5 AM Central by the GitHub Actions workflow. Falls
  back to a live Open-Meteo call, then to seasonal per-hour medians.
- **Model**: LightGBM (`models/Hourly_Items_LightGBM_1.pkl`), rank 1 of 5
  models trained on `sum_hour_kohi.csv` — total items sold per hour
  (Drink + Food + Pastry) over Aug 2024–Apr 2026. Test RMSE 7.65,
  R² 0.704. Training code lives in the main project's
  `goal2_scheduler_modeling/`.
- **Output**: a Sun–Sat × business-hour heatmap of forecasted item volume
  plus a staffing-insights summary (busiest day, rush window, peak hour
  per day, quietest slot).

## Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | The app |
| `models/Hourly_Items_LightGBM_1.pkl` | Rank-1 trained model |
| `fetch_weather_hourly_cache.py` | Hourly weather cache builder (stdlib only) |
| `data/weather_hourly_cache.json` | 14-day hourly weather cache |
| `.github/workflows/weather_cache.yml` | Daily 5 AM Central refresh (DST-guarded) |

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
