#!/usr/bin/env python3
"""
Module A1 — Jataka Fetch
Fetches birth chart data from Prokerala API
and saves raw responses to patient folder.

Usage:
  python jataka_fetch.py \
    --name "Anurag" \
    --dob "1968-10-14" \
    --time "20:30" \
    --place "Shimoga, Karnataka"
"""

import os, json, argparse, asyncio, time
import httpx
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz

from config import PROKERALA_CLIENT_ID, PROKERALA_CLIENT_SECRET

_cached_token     = None
_token_expires_at = None


class ChartServiceError(Exception):
    """Prokerala refused or failed (no credits, bad keys, outage). Raised
    before any patient folder is created, so a failed generate leaves
    nothing behind and does not use up the account's chart allowance."""


def _api_failure(label: str, r) -> str:
    try:
        detail = r.json()["errors"][0]["detail"]
    except Exception:
        detail = r.text[:200]
    return f"{label} {r.status_code} - {detail}"

# ── Cache Logic ─────────────────────────────
CACHE_DIR = "charts_cache"
PATIENTS_DIR = "patients"
PATIENT_INDEX_FILE = os.path.join(PATIENTS_DIR, "index.json")

def get_request_key(name: str, dob: str, birth_time: str, lat: float, lng: float) -> str:
    """Creates a unique key for the birth details."""
    import hashlib
    # Normalize to avoid duplicate entries for same person
    raw_key = f"{name.lower().strip()}|{dob}|{birth_time}|{round(lat, 4)}|{round(lng, 4)}"
    return hashlib.md5(raw_key.encode()).hexdigest()

def get_cached_data(req_key: str) -> dict:
    """Checks if valid raw data exists in the master cache folder."""
    cache_path = os.path.join(CACHE_DIR, req_key)
    chart_file = os.path.join(cache_path, "raw_chart.json")
    dasha_file = os.path.join(cache_path, "raw_dasha.json")
    
    if os.path.exists(chart_file) and os.path.exists(dasha_file):
        with open(chart_file, "r") as f:
            raw_chart = json.load(f)
        with open(dasha_file, "r") as f:
            raw_dasha = json.load(f)
        
        # Check if the cached chart is restricted
        svg = raw_chart.get("chart_svg", "")
        if "SVG image restricted" in svg or not svg:
            print(f"⚠️ Cache found but data is restricted or empty. Bypassing...")
            return None

        return {"raw_chart": raw_chart, "raw_dasha": raw_dasha}
    return None

def save_to_master_cache(req_key: str, raw_chart: dict, raw_dasha: dict):
    """Saves raw API responses to the master cache folder."""
    cache_path = os.path.join(CACHE_DIR, req_key)
    os.makedirs(cache_path, exist_ok=True)
    
    with open(os.path.join(cache_path, "raw_chart.json"), "w") as f:
        json.dump(raw_chart, f, indent=2)
    with open(os.path.join(cache_path, "raw_dasha.json"), "w") as f:
        json.dump(raw_dasha, f, indent=2)

# ── Counter logic ───────────────────────────
def _read_json(path: str, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _write_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_patient_index() -> dict:
    index = _read_json(PATIENT_INDEX_FILE, {})
    return index if isinstance(index, dict) else {}


def _find_existing_patient(req_key: str) -> str | None:
    if not os.path.isdir(PATIENTS_DIR):
        return None

    matches = []
    for entry in os.scandir(PATIENTS_DIR):
        if not entry.is_dir():
            continue
        raw_chart = _read_json(os.path.join(entry.path, "raw_chart.json"), {})
        if not raw_chart:
            continue
        try:
            existing_key = get_request_key(
                raw_chart.get("name", ""),
                raw_chart.get("dob", ""),
                raw_chart.get("birth_time", ""),
                float(raw_chart.get("lat", 0)),
                float(raw_chart.get("lng", 0)),
            )
        except (TypeError, ValueError):
            continue
        if existing_key == req_key:
            matches.append(entry.name)

    if not matches:
        return None
    return max(matches, key=lambda value: int(value) if value.isdigit() else -1)


def get_or_create_patient(req_key: str) -> tuple[str, str, bool]:
    os.makedirs(PATIENTS_DIR, exist_ok=True)
    index = _load_patient_index()
    patient_id = index.get(req_key)

    if patient_id and os.path.isdir(os.path.join(PATIENTS_DIR, str(patient_id))):
        return str(patient_id), get_patient_folder(str(patient_id)), False

    patient_id = _find_existing_patient(req_key)
    created = patient_id is None
    if created:
        patient_id = get_next_patient_id()

    index[req_key] = patient_id
    _write_json(PATIENT_INDEX_FILE, index)
    return patient_id, get_patient_folder(patient_id), created


def get_next_patient_id() -> str:
    counter_file = "counter.txt"
    if not os.path.exists(counter_file):
        with open(counter_file, "w") as f:
            f.write("0")
    with open(counter_file, "r") as f:
        count = int(f.read().strip() or 0)
    # counter.txt is gitignored, so a fresh copy of the project starts it at 0
    # and would hand out IDs that already have folders (merging a new patient
    # into an old one). Never go below the highest existing patient folder.
    if os.path.isdir(PATIENTS_DIR):
        existing = [int(e.name) for e in os.scandir(PATIENTS_DIR) if e.is_dir() and e.name.isdigit()]
        count = max([count] + existing)
    count += 1
    with open(counter_file, "w") as f:
        f.write(str(count))
    return str(count).zfill(3)

def get_patient_folder(patient_id: str) -> str:
    folder = os.path.join(PATIENTS_DIR, patient_id)
    os.makedirs(folder, exist_ok=True)
    return folder


def update_patient_records(
    folder: str,
    patient_id: str,
    req_key: str,
    profile: dict,
    source: str,
    created: bool,
):
    profile_file = os.path.join(folder, "profile.json")
    existing_profile = _read_json(profile_file, {})
    now = datetime.now().isoformat()
    merged_profile = {
        **existing_profile,
        **{key: value for key, value in profile.items() if value not in (None, "")},
        "patient_id": patient_id,
        "chart_key": req_key,
        "updated_at": now,
    }
    merged_profile.setdefault("created_at", now)
    _write_json(profile_file, merged_profile)

    history_file = os.path.join(folder, "run_history.json")
    history = _read_json(history_file, [])
    if not isinstance(history, list):
        history = []
    history.append({
        "run_at": now,
        "source": source,
        "created_patient": created,
    })
    _write_json(history_file, history)

# ── Token ───────────────────────────────────
async def get_token() -> str:
    global _cached_token, _token_expires_at

    if _cached_token and time.time() < _token_expires_at:
        return _cached_token

    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.prokerala.com/token",
            data={
                "grant_type":    "client_credentials",
                "client_id":     PROKERALA_CLIENT_ID,
                "client_secret": PROKERALA_CLIENT_SECRET,
            }
        )

    if r.status_code != 200:
        if r.status_code == 401:
            raise ChartServiceError(f"Prokerala login failed: 401 (Unauthorized). "
                                    f"Check PROKERALA_CLIENT_ID and PROKERALA_CLIENT_SECRET in .env. "
                                    f"Original error: {r.text[:200]}")
        raise ChartServiceError(_api_failure("Prokerala login failed:", r))

    d = r.json()
    _cached_token     = d["access_token"]
    _token_expires_at = time.time() + d["expires_in"] - 300  # refresh 5 min early
    print("[OK] Token fetched")
    return _cached_token


# ── Timezone ────────────────────────────────
def get_tz_offset(lat: float, lng: float) -> str:
    """Returns UTC offset string e.g. '+05:30' for a given lat/lng."""
    try:
        tf      = TimezoneFinder()
        tz_name = tf.timezone_at(lat=lat, lng=lng)
        if not tz_name:
            return "+05:30"
        tz     = pytz.timezone(tz_name)
        offset = datetime.now(tz).strftime("%z")   # "+0530"
        return f"{offset[:3]}:{offset[3:]}"        # "+05:30"
    except Exception:
        return "+05:30"

# ── Main fetch ──────────────────────────────
async def fetch_and_save(
    name: str,
    dob: str,
    birth_time: str,
    lat: float,
    lng: float,
    place: str,
    gender: str = "Unknown",
    father_name: str = "",
    mother_name: str = ""
) -> dict:
    """
    Fetches chart from Prokerala API.
    Saves raw files to patients/00X/ folder.
    Returns dict with:
      {
        "patient_id": "001",
        "folder": "patients/001",
        "planets": [...],
        "houses": [...],
        "ascendant": {...},
        "svg_raw": "...",
        "dasha_raw": {...}
      }
    """
    tz_offset  = get_tz_offset(lat, lng)
    dt_iso     = f"{dob}T{birth_time}:00{tz_offset}"
    coords     = f"{lat},{lng}"

    # Step 0: Check Master Cache Folder
    req_key = get_request_key(name, dob, birth_time, lat, lng)
    cached = get_cached_data(req_key)

    profile = {
        "name": name,
        "dob": dob,
        "birth_time": birth_time,
        "birth_place": place,
        "lat": lat,
        "lng": lng,
        "gender": gender,
        "father_name": father_name,
        "mother_name": mother_name,
    }

    if cached:
        patient_id, folder, created = get_or_create_patient(req_key)
        print(f"\n[Local Cache] Master data found for {name} (Key: {req_key[:8]})")
        print(f"              Reusing API responses from {CACHE_DIR}/")
        raw_chart = cached["raw_chart"]
        raw_dasha = cached["raw_dasha"]
        raw_chart["patient_id"] = patient_id
        raw_dasha["patient_id"] = patient_id
        raw_chart.update({
            "name": name,
            "gender": gender,
            "dob": dob,
            "birth_time": birth_time,
            "place": place,
            "lat": lat,
            "lng": lng,
        })
        
        # Update gender if it was unknown in cache
        if raw_chart.get("gender") == "Unknown" and gender != "Unknown":
            raw_chart["gender"] = gender

        # Still save a copy to the patient folder for record-keeping
        with open(f"{folder}/raw_chart.json", "w") as f: json.dump(raw_chart, f, indent=2)
        with open(f"{folder}/raw_dasha.json", "w") as f: json.dump(raw_dasha, f, indent=2)
        update_patient_records(
            folder, patient_id, req_key, profile, "Local Cache", created
        )
        
        from jataka_logic import build_chart_dict
        chart_data = build_chart_dict(raw_chart, raw_dasha)
        
        return {
            "patient_id": patient_id,
            "folder": folder,
            "planets": chart_data["planets"],
            "houses": chart_data["houses"],
            "ascendant": chart_data["ascendant"],
            "svg_raw": chart_data["svg_raw"],
            "dasha_raw": chart_data["dasha_raw"],
            "source": "Local Cache"
        }

    print(f"\n[Prokerala API] Fetching fresh data for {name}...")
    token      = await get_token()

    async with httpx.AsyncClient(
            timeout=30.0) as client:

        # Call 1: Planet positions
        r1 = await client.get(
            "https://api.prokerala.com"
            "/v2/astrology/planet-position",
            headers={"Authorization":
                     f"Bearer {token}"},
            params={"ayanamsa": 1,
                    "coordinates": coords,
                    "datetime": dt_iso,
                    "la": "en"}
        )

        # Call 2: SVG Chart
        r2 = await client.get(
            "https://api.prokerala.com"
            "/v2/astrology/chart",
            headers={"Authorization":
                     f"Bearer {token}"},
            params={"ayanamsa": 1,
                    "coordinates": coords,
                    "datetime": dt_iso,
                    "chart_type": "rasi",
                    "chart_style": "south-indian",
                    "la": "en"}
        )

        # Call 3: Birth details
        r3 = await client.get(
            "https://api.prokerala.com"
            "/v2/astrology/birth-details",
            headers={"Authorization":
                     f"Bearer {token}"},
            params={"ayanamsa": 1,
                    "coordinates": coords,
                    "datetime": dt_iso,
                    "la": "en"}
        )

        # Call 4: Dasha periods
        r4 = await client.get(
            "https://api.prokerala.com"
            "/v2/astrology/dasha-periods",
            headers={"Authorization":
                     f"Bearer {token}"},
            params={"ayanamsa": 1,
                    "coordinates": coords,
                    "datetime": dt_iso,
                    "la": "en"}
        )

    # Planets, chart and dasha are what every algorithm runs on. If any of
    # them failed, stop here: no patient folder, no empty chart.
    failed = [_api_failure(label, r) for label, r in
              (("Planet API", r1), ("Chart API", r2), ("Dasha API", r4)) if r.status_code != 200]
    if failed:
        raise ChartServiceError("; ".join(failed))   # server.py prints each reason
    if r3.status_code != 200:
        print(f"[Warning] Birth details error: {r3.status_code} - {r3.text[:200]} (chart continues without them)")

    patient_id, folder, created = get_or_create_patient(req_key)

    # Save raw chart data
    raw_chart = {
        "patient_id":  patient_id,
        "name":        name,
        "gender":      gender,
        "dob":         dob,
        "birth_time":  birth_time,
        "place":       place,
        "lat":         lat,
        "lng":         lng,
        "fetched_at":  datetime.now().isoformat(),
        "planet_position": r1.json(),
        "chart_svg":   r2.text,
        "birth_details": r3.json() if r3.status_code == 200 else {},
    }

    # Save raw dasha data
    raw_dasha = {
        "patient_id": patient_id,
        "dasha_data": r4.json()
    }

    # CRITICAL FIX: Only save to Master Cache if the data is valid and complete
    # Check if we got actual chart data and SVG (not restricted/error message)
    is_valid = (
        r1.status_code == 200 and 
        r2.status_code == 200 and 
        "SVG image restricted" not in r2.text and
        r4.status_code == 200
    )

    if is_valid:
        save_to_master_cache(req_key, raw_chart, raw_dasha)
        print(f"✅ Master Cache updated for {name}")
    else:
        print(f"⚠️ Data incomplete or restricted. Skipping Master Cache for {name}")

    with open(f"{folder}/raw_chart.json", "w") as f:
        json.dump(raw_chart, f, indent=2)

    with open(f"{folder}/raw_dasha.json", "w") as f:
        json.dump(raw_dasha, f, indent=2)

    update_patient_records(
        folder, patient_id, req_key, profile, "Prokerala API", created
    )

    print(f"✅ Patient {patient_id} saved → {folder}/")
    print(f"   Planet API: {r1.status_code}")
    print(f"   Chart API:  {r2.status_code}")
    print(f"   Details:    {r3.status_code}")
    print(f"   Dasha:      {r4.status_code}")
    
    from jataka_logic import build_chart_dict
    chart_data = build_chart_dict(raw_chart, raw_dasha)
    
    return {
        "patient_id": patient_id,
        "folder": folder,
        "planets": chart_data["planets"],
        "houses": chart_data["houses"],
        "ascendant": chart_data["ascendant"],
        "svg_raw": chart_data["svg_raw"],
        "dasha_raw": chart_data["dasha_raw"],
        "source": "Prokerala API"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",  required=True)
    parser.add_argument("--dob",   required=True,
        help="YYYY-MM-DD")
    parser.add_argument("--time",  required=True,
        help="HH:MM")
    parser.add_argument("--place", required=True)
    parser.add_argument("--lat", type=float, default=28.6139)
    parser.add_argument("--lng", type=float, default=77.2090)
    args = parser.parse_args()

    res = asyncio.run(fetch_and_save(
        args.name, args.dob,
        args.time, args.lat, args.lng, args.place
    ))
    print(f"\nPatient ID: {res['patient_id']}")
