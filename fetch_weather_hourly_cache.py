#!/usr/bin/env python3
"""
fetch_weather_hourly_cache.py
Kohi Scheduler — hourly weather cache builder.

Goal 2 companion to the pastry app's fetch_weather_cache.py, but at
HOURLY grain: the scheduler predicts demand per business hour, so each
hour keeps its own weather reading instead of a business-hours average.

Pulls hourly Open-Meteo data for Bentonville, AR over a 14-day horizon —
enough to cover the app's "next full Sunday-Saturday week", which ends at
most 13 days out (when today is a Sunday). Keeps the nine business-hour
readings per day (06:00-14:00 Central, Kohi open 6 AM-3 PM) and writes
data/weather_hourly_cache.json.

Run daily at 5 AM Central by .github/workflows/weather_cache.yml.
Stdlib only — no third-party dependencies, so CI needs no pip install.
"""

import json
import os
import ssl
import sys
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

try:  # some local Python installs lack the system CA bundle; CI doesn't need this
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

CENTRAL = ZoneInfo("America/Chicago")
LAT, LON = 36.3729, -94.2088
BUSINESS_HOURS = range(6, 15)  # hourly readings 06:00 through 14:00 inclusive
FORECAST_DAYS = 14             # covers next full Sun-Sat even from a Sunday

API_URL = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LAT}&longitude={LON}"
    "&hourly=temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m,weather_code"
    "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
    f"&timezone=America%2FChicago&forecast_days={FORECAST_DAYS}"
)

WMO_MAP = {
    0: "clear",
    1: "cloudy", 2: "cloudy", 3: "cloudy", 45: "cloudy", 48: "cloudy",
    51: "rainy", 53: "rainy", 55: "rainy", 56: "rainy", 57: "rainy",
    61: "rainy", 63: "rainy", 65: "rainy", 66: "rainy", 67: "rainy",
    71: "snowy", 73: "snowy", 75: "snowy", 77: "snowy",
    80: "rainy", 81: "rainy", 82: "rainy", 85: "snowy", 86: "snowy",
    95: "rainy", 96: "rainy", 99: "rainy",
}


def extract_business_hours(hourly: dict) -> dict:
    """{date_iso: {hour_str: weather}} for the 6 AM-2 PM readings."""
    days = {}
    for i, ts in enumerate(hourly["time"]):  # e.g. "2026-07-13T06:00"
        date_str, hour = ts[:10], int(ts[11:13])
        if hour not in BUSINESS_HOURS or hourly["temperature_2m"][i] is None:
            continue
        days.setdefault(date_str, {})[str(hour)] = {
            "temp_f":            round(hourly["temperature_2m"][i], 1),
            "precip_in":         round(hourly["precipitation"][i], 3),
            "humidity_pct":      round(hourly["relative_humidity_2m"][i], 1),
            "wind_mph":          round(hourly["wind_speed_10m"][i], 1),
            "weather_condition": WMO_MAP.get(hourly["weather_code"][i], "clear"),
        }
    return days


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base_dir, "data", "weather_hourly_cache.json")

    with urllib.request.urlopen(API_URL, timeout=30, context=SSL_CONTEXT) as resp:
        payload = json.load(resp)

    days = extract_business_hours(payload["hourly"])
    now_central = datetime.now(CENTRAL)
    today_str = now_central.date().isoformat()
    if today_str not in days:
        print(f"ERROR: API response has no business-hours data for today ({today_str})",
              file=sys.stderr)
        return 1

    cache = {
        "generated_at": now_central.isoformat(timespec="seconds"),
        "timezone": "America/Chicago",
        "location": {"lat": LAT, "lon": LON},
        "business_hours": "hourly readings 06:00-14:00 (Kohi open 6 AM-3 PM Central)",
        "grain": "hourly — one entry per business hour per day",
        "days": days,
    }
    with open(out_path, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"  generated_at: {cache['generated_at']}  ({len(days)} days cached)")
    for d in sorted(days):
        hours = days[d]
        temps = [hours[h]["temp_f"] for h in sorted(hours, key=int)]
        print(f"  {d}: {len(hours)} hours, {min(temps)}-{max(temps)}°F")
    return 0


if __name__ == "__main__":
    sys.exit(main())
