from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any, Dict, Tuple

import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

app = FastAPI()

# -----------------------------
# Helpers
# -----------------------------
def _round_key(lat: float, lon: float, ndigits: int = 3) -> Tuple[float, float]:
    # 3位小数约110m网格；hover体验好且大幅减少请求
    return (round(lat, ndigits), round(lon, ndigits))

def _get_json(url: str, params: Dict[str, Any], timeout: float = 20) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "smallbiz-platform/1.0"})
    r.raise_for_status()
    return r.json()

# -----------------------------
# A) Census Geocoder: coords -> tract/state/county
# -----------------------------
@lru_cache(maxsize=50_000)
def census_geographies(lat_key: float, lon_key: float) -> Dict[str, str]:
    # US Census Geocoder (coordinates geoLookup) :contentReference[oaicite:7]{index=7}
    url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    params = {
        "x": lon_key,
        "y": lat_key,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json",
    }
    data = _get_json(url, params)

    try:
        geos = data["result"]["geographies"]
        tract_info = geos["Census Tracts"][0]
        county_info = geos["Counties"][0]
        state_info = geos["States"][0]
        return {
            "state_fips": state_info["STATE"],
            "county_fips": county_info["COUNTY"],
            "tract": tract_info["TRACT"],
        }
    except Exception:
        return {}

# -----------------------------
# B) ACS: tract -> income/pop
# -----------------------------
@lru_cache(maxsize=50_000)
def acs_features(state_fips: str, county_fips: str, tract: str) -> Dict[str, Any]:
    # ACS 5-year via Census Data API :contentReference[oaicite:8]{index=8}
    # 这里用 2022/acs/acs5 作为示例；你也可改成最新年份
    url = "https://api.census.gov/data/2022/acs/acs5"
    params = {
        "get": "B19013_001E,B01003_001E",  # median income, population
        "for": f"tract:{tract}",
        "in": f"state:{state_fips} county:{county_fips}",
    }
    data = _get_json(url, params)

    cols = data[0]
    vals = data[1]
    row = dict(zip(cols, vals))

    def to_int(x: str):
        try:
            return int(x)
        except Exception:
            return None

    return {
        "population": to_int(row.get("B01003_001E", "")),
        "median_household_income": to_int(row.get("B19013_001E", "")),
    }

# -----------------------------
# C) Overpass: POI counts within radius
# -----------------------------
@lru_cache(maxsize=50_000)
def poi_counts(lat_key: float, lon_key: float, radius_m: int) -> Dict[str, int]:
    # Overpass API endpoint/query examples :contentReference[oaicite:9]{index=9}
    overpass_url = "https://overpass-api.de/api/interpreter"

    def count_for(filter_expr: str) -> int:
        q = f"""
        [out:json][timeout:25];
        (
          node(around:{radius_m},{lat_key},{lon_key})[{filter_expr}];
          way(around:{radius_m},{lat_key},{lon_key})[{filter_expr}];
          relation(around:{radius_m},{lat_key},{lon_key})[{filter_expr}];
        );
        out ids;
        """
        r = requests.post(overpass_url, data=q.encode("utf-8"),
                          timeout=35, headers={"User-Agent": "smallbiz-platform/1.0"})
        r.raise_for_status()
        j = r.json()
        return len(j.get("elements", []))

    return {
        "restaurants": count_for('amenity"="restaurant'),
        "bars": count_for('amenity"="bar'),
        "cafes": count_for('amenity"="cafe'),
        "shops": count_for("shop"),
    }

# -----------------------------
# D) Baltimore Police ArcGIS: crime counts within radius
# -----------------------------
@lru_cache(maxsize=50_000)
def crime_counts(lat_key: float, lon_key: float, radius_m: int) -> Dict[str, Any]:
    # Public_Crime_Map_Last3Months FeatureServer :contentReference[oaicite:10]{index=10}
    base = "https://arcgisportal.baltimorepolice.org/gis/rest/services/Crime/Public_Crime_Map_Last3Months/FeatureServer/0/query"

    # 先拿总数（returnCountOnly）
    params_total = {
        "where": "1=1",
        "geometry": f"{lon_key},{lat_key}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "distance": str(radius_m),
        "units": "esriSRUnit_Meter",
        "returnCountOnly": "true",
        "f": "json",
    }
    total = _get_json(base, params_total).get("count", 0)

    # 再按 CRIME_TYPE 做 group-by 统计（outStatistics + groupByFieldsForStatistics）
    out_stats = json.dumps([{
        "statisticType": "count",
        "onStatisticField": "OBJECTID",
        "outStatisticFieldName": "ct"
    }])
    params_by = {
        "where": "1=1",
        "geometry": f"{lon_key},{lat_key}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "distance": str(radius_m),
        "units": "esriSRUnit_Meter",
        "groupByFieldsForStatistics": "CRIME_TYPE",
        "outStatistics": out_stats,
        "outFields": "CRIME_TYPE",
        "returnGeometry": "false",
        "f": "json",
    }
    feats = _get_json(base, params_by).get("features", [])
    by_type = {}
    for f in feats:
        a = f.get("attributes", {})
        k = a.get("CRIME_TYPE") or "UNKNOWN"
        by_type[k] = int(a.get("ct", 0) or 0)

    return {"total_last_3mo": int(total), "by_type": by_type}

# -----------------------------
# E) Transit (MVP版)
# -----------------------------
def transit_stub(lat: float, lon: float, radius_m: int) -> Dict[str, Any]:
    # 这里先给MVP：先把字段打通，后续接入GTFS静态数据计算 stops/nearest
    # 你会在下一步把 MTA GTFS 下载并预处理（官方提供GTFS下载）:contentReference[oaicite:11]{index=11}
    return {"nearest_stop_m": None, "stops_within_radius": None, "note": "GTFS pending"}

# -----------------------------
# API: features
# -----------------------------
@app.get("/api/features")
def api_features(
    lat: float = Query(...),
    lon: float = Query(...),
    radius: int = Query(500, ge=100, le=2000),
):
    lat_k, lon_k = _round_key(lat, lon, ndigits=3)

    geo = census_geographies(lat_k, lon_k)
    census = dict(geo)

    if geo:
        census.update(acs_features(geo["state_fips"], geo["county_fips"], geo["tract"]))

    resp = {
        "lat": lat,
        "lon": lon,
        "radius_m": radius,
        "census": census,
        "poi": poi_counts(lat_k, lon_k, radius),
        "crime": crime_counts(lat_k, lon_k, radius),
        "transit": transit_stub(lat, lon, radius),
    }
    return resp

# -----------------------------
# Simple frontend page
# -----------------------------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>SmallBiz Map Hover MVP</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    html, body { height: 100%; margin: 0; }
    #wrap { display: flex; height: 100%; }
    #map { flex: 1; }
    #panel {
      width: 360px; padding: 12px; font-family: Arial, sans-serif;
      border-left: 1px solid #ddd; overflow: auto;
    }
    pre { white-space: pre-wrap; word-break: break-word; }
  </style>
</head>
<body>
<div id="wrap">
  <div id="map"></div>
  <div id="panel">
    <h3>Hover Data</h3>
    <div id="meta"></div>
    <pre id="out">Move mouse on map...</pre>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  const map = L.map('map').setView([39.2904, -76.6122], 13); // Baltimore downtown
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);

  let timer = null;
  let lastKey = null;

  function debounceFetch(lat, lon) {
    if (timer) clearTimeout(timer);
    timer = setTimeout(async () => {
      const radius = 500;
      // 前端也做一次“网格化”避免重复请求
      const key = `${lat.toFixed(3)},${lon.toFixed(3)},${radius}`;
      if (key === lastKey) return;
      lastKey = key;

      document.getElementById('meta').innerText = `lat=${lat.toFixed(5)}, lon=${lon.toFixed(5)}, r=${radius}m`;
      try {
        const res = await fetch(`/api/features?lat=${lat}&lon=${lon}&radius=${radius}`);
        const j = await res.json();
        document.getElementById('out').innerText = JSON.stringify(j, null, 2);
      } catch (e) {
        document.getElementById('out').innerText = String(e);
      }
    }, 300);
  }

  map.on('mousemove', (e) => {
    debounceFetch(e.latlng.lat, e.latlng.lng);
  });
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML
