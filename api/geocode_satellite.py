"""
Geocoding and satellite tile URL via Google Maps APIs.
Uses GOOGLE_MAPS_API_KEY for both Geocoding API and Maps Static API.
"""

import os
from typing import Any

import httpx

API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GEOCODING_API_KEY")


def geocode(address: str) -> dict[str, Any] | None:
    """
    Return { "lat": float, "lng": float, "formatted_address": str } or None.
    """
    if not API_KEY:
        return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": API_KEY}
    try:
        r = httpx.get(url, params=params, timeout=10.0)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK" or not data.get("results"):
            return None
        loc = data["results"][0]["geometry"]["location"]
        return {
            "lat": loc["lat"],
            "lng": loc["lng"],
            "formatted_address": data["results"][0].get("formatted_address", address),
        }
    except Exception:
        return None


def satellite_tile_url(lat: float, lng: float, zoom: int = 20, width: int = 640, height: int = 640) -> str | None:
    """
    Google Maps Static API satellite view URL (no labels).
    Frontend can use as img src. Requires API key.
    """
    if not API_KEY:
        return None
    # maptype=satellite, no markers
    base = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lng}",
        "zoom": zoom,
        "size": f"{width}x{height}",
        "maptype": "satellite",
        "key": API_KEY,
    }
    q = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{base}?{q}"


def fetch_satellite_image_bytes(lat: float, lng: float, zoom: int = 20, width: int = 640, height: int = 640) -> bytes | None:
    """Download satellite tile as bytes (for inference)."""
    url = satellite_tile_url(lat, lng, zoom=zoom, width=width, height=height)
    if not url:
        return None
    try:
        r = httpx.get(url, timeout=15.0)
        r.raise_for_status()
        return r.content
    except Exception:
        return None
