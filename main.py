from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse


app = FastAPI(title="SmallBiz Map Hover MVP")


# -----------------------------
# Basic helpers
# -----------------------------
class ApiError(RuntimeError):
    pass


@dataclass
class RateLimiter:
    calls: int
    period_sec: float
    _timestamps: Optional[list[float]] = None

    def __post_init__(self) -> None:
        self._timestamps = []

    def wait(self) -> None:
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < self.period_sec]
        if len(self._timestamps) >= self.calls:
            sleep_for = self.period_sec - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._timestamps.append(time.time())


def _round_key(lat: float, lon: float, ndigits: int = 3) -> Tuple[float, float]:
    # 3 位小数约 110m 网格，hover 模式必须网格化避免请求爆炸
    return round(lat, ndigits), round(lon, ndigits)


def request_json(
    method: str,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[bytes] = None,
    timeout: float = 25,
    limiter: Optional[RateLimiter] = None,
) -> Dict[str, Any]:
    if limiter:
        limiter.wait()

    headers = {"User-Agent": "smallbiz-platform/1.0 (hover-mvp)"}
    r = requests.request(method, url, params=params, data=data, timeout=timeout, headers=headers)
    if r.status_code >= 400:
        raise ApiError(f"HTTP {r.status_code}: {r.text[:200]}")
    try:
        return r.json()
    except Exception as e:
        raise ApiError(f"Non-JSON response: {str(e)[:120]} | body={r.text[:200]}")


def safe_call(fn, default):
    """
    任何外部源失败都不让 /api/features 500；
    返回 (value, note). note 为 None 表示成功。
    """
    try:
        return fn(), None
    except Exception as e:
        return default, f"{type(e).__name__}: {str(e)[:180]}"


# -----------------------------
# A) Census Geocoder: coords -> tract/state/county
# -----------------------------
_census_limiter = RateLimiter(calls=8, period_sec=1.0)


@lru_cache(maxsize=20000)
def census_geographies(lat_key: float, lon_key: float) -> Dict[str, str]:
    """
    给定坐标（网格化后），返回 state/county/tract FIPS.
    """
    url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    params = {
        "x": lon_key,
        "y": lat_key,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json",
    }
    data = request_json("GET", url, params=params, timeout=20, limiter=_census_limiter)

    geos = (data.get("result") or {}).get("geographies") or {}
    try:
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
_acs_limiter = RateLimiter(calls=8, period_sec=1.0)


@lru_cache(maxsize=20000)
def acs_features(state_fips: str, county_fips: str, tract: str) -> Dict[str, Any]:
    """
    ACS 5-year. 这里用 2022 做示例；你后面需要更新年份只改这里一行即可。
    """
    url = "https://api.census.gov/data/2022/acs/acs5"
    params = {
        "get": "B19013_001E,B01003_001E",  # median household income, total population
        "for": f"tract:{tract}",
        "in": f"state:{state_fips} county:{county_fips}",
    }
    data = request_json("GET", url, params=params, timeout=20, limiter=_acs_limiter)

    cols = data[0]
    vals = data[1]
    row = dict(zip(cols, vals))

    def to_int(x: str) -> Optional[int]:
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
_overpass_limiter = RateLimiter(calls=2, period_sec=1.0)  # 保守，避免 Overpass 压力过大


@lru_cache(maxsize=20000)
def poi_counts(lat_key: float, lon_key: float, radius_m: int) -> Dict[str, Any]:
    """
    周边 POI 密度（按类计数）。
    关键：filter_expr 要写成 Overpass 支持的表达式，例如:
      '"amenity"="restaurant"'  ->  node(...)[ "amenity"="restaurant" ]
      'shop'                    ->  node(...)[ shop ]
    """
    overpass_url = "https://overpass-api.de/api/interpreter"

    def count_for(filter_expr: str) -> int:
        query = f"""
        [out:json][timeout:25];
        (
          node(around:{radius_m},{lat_key},{lon_key})[{filter_expr}];
          way(around:{radius_m},{lat_key},{lon_key})[{filter_expr}];
          relation(around:{radius_m},{lat_key},{lon_key})[{filter_expr}];
        );
        out ids;
        """
        data = request_json(
            "POST",
            overpass_url,
            data=query.encode("utf-8"),
            timeout=35,
            limiter=_overpass_limiter,
        )
        return len(data.get("elements", []))

    return {
        "restaurants": count_for('"amenity"="restaurant"'),
        "bars": count_for('"amenity"="bar"'),
        "cafes": count_for('"amenity"="cafe"'),
        "shops": count_for("shop"),
    }


# -----------------------------
# D) Baltimore Police ArcGIS: crime counts within radius
# -----------------------------
_crime_limiter = RateLimiter(calls=6, period_sec=1.0)


@lru_cache(maxsize=20000)
def crime_counts(lat_key: float, lon_key: float, radius_m: int) -> Dict[str, Any]:
    """
    近三个月公共犯罪地图（ArcGIS FeatureServer）。
    返回：总数 + 按 CRIME_TYPE 分组统计。
    """
    base = "https://arcgisportal.baltimorepolice.org/gis/rest/services/Crime/Public_Crime_Map_Last3Months/FeatureServer/0/query"

    # 1) 总数
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
    total_data = request_json("GET", base, params=params_total, timeout=25, limiter=_crime_limiter)
    total = int(total_data.get("count") or 0)

    # 2) 分类型统计
    out_stats = json.dumps(
        [{
            "statisticType": "count",
            "onStatisticField": "OBJECTID",
            "outStatisticFieldName": "ct",
        }]
    )
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
    by_data = request_json("GET", base, params=params_by, timeout=25, limiter=_crime_limiter)
    feats = by_data.get("features", []) or []

    by_type: Dict[str, int] = {}
    for f in feats:
        a = f.get("attributes", {}) or {}
        k = a.get("CRIME_TYPE") or "UNKNOWN"
        by_type[k] = int(a.get("ct") or 0)

    return {"total_last_3mo": total, "by_type": by_type}


# -----------------------------
# E) Transit (MVP stub)
# -----------------------------
def transit_stub(lat: float, lon: float, radius_m: int) -> Dict[str, Any]:
    # 先打通字段与展示；后续接入 MTA GTFS 可计算 nearest_stop_m / stops_within_radius
    return {
        "nearest_stop_m": None,
        "stops_within_radius": None,
        "note": "GTFS not implemented yet",
    }


# -----------------------------
# API endpoint: features
# -----------------------------
@app.get("/api/features")
def api_features(
    lat: float = Query(...),
    lon: float = Query(...),
    radius: int = Query(500, ge=100, le=2000),
):
    lat_k, lon_k = _round_key(lat, lon, ndigits=3)
    notes: Dict[str, str] = {}

    geo, note = safe_call(lambda: census_geographies(lat_k, lon_k), {})
    if note:
        notes["census_geo"] = note

    census = dict(geo)
    if geo:
        acs, note = safe_call(lambda: acs_features(geo["state_fips"], geo["county_fips"], geo["tract"]), {})
        if note:
            notes["census_acs"] = note
        census.update(acs)

    poi_default = {"restaurants": None, "bars": None, "cafes": None, "shops": None}
    poi, note = safe_call(lambda: poi_counts(lat_k, lon_k, radius), poi_default)
    if note:
        notes["poi"] = note

    crime_default = {"total_last_3mo": None, "by_type": {}}
    crime, note = safe_call(lambda: crime_counts(lat_k, lon_k, radius), crime_default)
    if note:
        notes["crime"] = note

    transit, note = safe_call(lambda: transit_stub(lat, lon, radius), {"nearest_stop_m": None, "stops_within_radius": None})
    if note:
        notes["transit"] = note

    return JSONResponse(
        {
            "lat": lat,
            "lon": lon,
            "radius_m": radius,
            "census": census,
            "poi": poi,
            "crime": crime,
            "transit": transit,
            "notes": notes,  # ✅ 如果某个源失败，这里会显示原因
        }
    )


# -----------------------------
# Frontend page (Leaflet)
# -----------------------------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>SmallBiz Map Hover MVP</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />

  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    html, body { height: 100%; margin: 0; }
    #wrap { display: flex; height: 100%; }
    #map { flex: 1; }
    #panel {
      width: 380px; padding: 14px; font-family: Arial, sans-serif;
      border-left: 1px solid #ddd; overflow: auto;
    }
    pre { white-space: pre-wrap; word-break: break-word; }
    .muted { color: #666; font-size: 12px; }
  </style>
</head>
<body>
<div id="wrap">
  <div id="map"></div>
  <div id="panel">
    <h2>Hover Data</h2>
    <div id="meta" class="muted">Move mouse on map...</div>
    <pre id="out"></pre>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  const map = L.map('map').setView([39.2904, -76.6122], 13); // Baltimore
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {  // ✅ HTTPS
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);

  let timer = null;
  let lastKey = null;

  async function fetchFeatures(lat, lon) {
    const radius = 500;

    // 前端也做网格化，减少重复请求
    const key = `${lat.toFixed(3)},${lon.toFixed(3)},${radius}`;
    if (key === lastKey) return;
    lastKey = key;

    document.getElementById('meta').innerText = `lat=${lat.toFixed(5)}, lon=${lon.toFixed(5)}, r=${radius}m`;

    try {
      const res = await fetch(`/api/features?lat=${lat}&lon=${lon}&radius=${radius}`);
      const text = await res.text();

      if (!res.ok) {
        document.getElementById('out').innerText = `HTTP ${res.status}\\n` + text;
        return;
      }

      try {
        const j = JSON.parse(text);
        document.getElementById('out').innerText = JSON.stringify(j, null, 2);
      } catch (e) {
        document.getElementById('out').innerText = `JSON parse error: ${e}\\n` + text;
      }
    } catch (e) {
      document.getElementById('out').innerText = String(e);
    }
  }

  function debounce(lat, lon) {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => fetchFeatures(lat, lon), 300);
  }

  map.on('mousemove', (e) => {
    debounce(e.latlng.lat, e.latlng.lng);
  });
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"status": "ok"}
