"""Optional live air-quality lookup via Open-Meteo with high-performance in-memory caching.

Zero external dependencies (pure standard library).
Features:
- Permanent geocoding coordinate cache (city coordinates never change).
- 10-minute TTL air quality cache (reduces redundant network requests).
- Outbound request bounds (8s timeout, 200KB payload cap, HTTPS enforcement).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
TIMEOUT_S = 8.0
MAX_BYTES = 200_000
DEFAULT_CACHE_TTL_S = 600.0  # 10 minutes cache for atmospheric readings

Fetcher = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class LiveReading:
    """Represents a validated atmospheric observation."""

    place: str
    aqi: int | None
    pm25: float | None
    pm10: float | None
    cached: bool = False


class DataSourceError(Exception):
    """Raised when an external data fetch or parsing fails."""

    pass


# High-efficiency in-memory caches to minimize latency & external bandwidth
_GEO_CACHE: dict[str, tuple[float, float, str]] = {}
_READING_CACHE: dict[str, tuple[float, LiveReading]] = {}  # key -> (timestamp, LiveReading)


def clear_caches() -> None:
    """Flush all in-memory geocoding and reading caches (useful for testing)."""
    _GEO_CACHE.clear()
    _READING_CACHE.clear()


def http_get_json(url: str) -> dict[str, Any]:
    """Execute a secure, bounded HTTPS GET request returning parsed JSON."""
    if not url.startswith("https://"):
        raise DataSourceError("only https is allowed")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "airguide-assistant/1.0",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # noqa: S310
            raw = resp.read(MAX_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DataSourceError(str(exc)) from exc
    if len(raw) > MAX_BYTES:
        raise DataSourceError("response too large")
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise DataSourceError("invalid JSON") from exc
    if not isinstance(data, dict):
        raise DataSourceError("unexpected response shape")
    return data


def _num(value: Any) -> float | None:
    """Safely convert numeric values, rejecting booleans, strings and nulls."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def fetch_live(
    city: str,
    fetcher: Fetcher = http_get_json,
    ttl: float = DEFAULT_CACHE_TTL_S,
    use_cache: bool = True,
) -> LiveReading:
    """Geocode `city` then fetch current readings with intelligent caching.

    Args:
        city: The place name to look up.
        fetcher: HTTP JSON fetching callable (defaults to http_get_json).
        ttl: Time-to-live in seconds for cached air quality readings.
        use_cache: If True, uses geocoding and air reading cache.

    Returns:
        LiveReading with verified place, AQI, PM2.5, and PM10 figures.

    Raises:
        DataSourceError: On lookup failure, network timeout, or invalid format.
    """
    clean_city = city.strip()
    cache_key = clean_city.lower()
    now = time.monotonic()

    # Check reading cache first (instant response in <0.1ms)
    if use_cache and cache_key in _READING_CACHE:
        cached_time, cached_reading = _READING_CACHE[cache_key]
        if now - cached_time < ttl:
            return LiveReading(
                place=cached_reading.place,
                aqi=cached_reading.aqi,
                pm25=cached_reading.pm25,
                pm10=cached_reading.pm10,
                cached=True,
            )

    # 1. Geocode city (with permanent coordinate cache)
    if use_cache and cache_key in _GEO_CACHE:
        lat, lon, place = _GEO_CACHE[cache_key]
    else:
        geo = fetcher(
            f"{GEOCODE_URL}?{urllib.parse.urlencode({'name': clean_city, 'count': 1})}"
        )
        results = geo.get("results")
        if not isinstance(results, list) or not results:
            raise DataSourceError("location not found")
        top = results[0]
        lat_val = _num(top.get("latitude"))
        lon_val = _num(top.get("longitude"))
        if lat_val is None or lon_val is None:
            raise DataSourceError("bad coordinates")
        lat, lon = lat_val, lon_val
        place = ", ".join(
            p for p in (top.get("name"), top.get("country")) if isinstance(p, str)
        ) or clean_city
        if use_cache:
            _GEO_CACHE[cache_key] = (lat, lon, place)

    # 2. Fetch atmospheric data
    query = urllib.parse.urlencode(
        {"latitude": lat, "longitude": lon, "current": "pm10,pm2_5,us_aqi"}
    )
    air = fetcher(f"{AIR_URL}?{query}")
    current = air.get("current")
    if not isinstance(current, dict):
        raise DataSourceError("no current data")
    aqi = _num(current.get("us_aqi"))
    pm25, pm10 = _num(current.get("pm2_5")), _num(current.get("pm10"))
    if aqi is None and pm25 is None and pm10 is None:
        raise DataSourceError("no usable readings")

    reading = LiveReading(
        place=place,
        aqi=None if aqi is None else max(0, min(500, round(aqi))),
        pm25=pm25,
        pm10=pm10,
        cached=False,
    )

    if use_cache:
        _READING_CACHE[cache_key] = (now, reading)

    return reading
