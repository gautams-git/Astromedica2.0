import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Settings and secrets (AUTH_ENABLED, Clerk keys, Firestore...) from a local
# .env file next to this one — gitignored, never committed. Must run before
# any import that reads the environment. Values already set in the terminal win.
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from flask import (Flask, request, g,
                   jsonify, send_from_directory)
from flask_cors import CORS
from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
import httpx
import asyncio
import time
import hmac
import json as _json

from rules import (
    apply_rule_1,
    generate_chat_questions,
    analyse_dasha,
    complete_risk_analysis
)

from jataka_fetch import fetch_and_save, ChartServiceError, _api_failure
from jataka_logic import run_logic
from disease_diagnose import run_diagnose
from combine_output import run_combine, write_rl_report, ensure_rl_report
from imp_rashi_distance import (
    build_imp_rashi_distance_analysis,
    build_birth_current_distance_analysis,
)
from second_ascendant import calculate_second_ascendant
from knowledge_base import (
    NAKSHATRA_DISEASES,
    PLANET_DISEASES,
    RASHI_ORGANS,
    NAKSHATRA_LORDS,
    filter_nakshatra_diseases_by_gender,
    filter_planet_diseases_by_gender,
    filter_rashi_organs_by_gender,
    filter_terms_by_gender,
    get_nakshatra_diseases,
    normalize_gender,
)
from zone_engine import run_zone_analysis
from chart_specific_overrides import apply_bhadravathi_overrides, apply_person2_overrides
from algorithm_comparison_v2 import build_combined_algo_summary_v2
from priority_cascade import build_full_cascade_view, build_cascade_tiers, build_drishti_catalysts, build_ascendant_reference, build_house_occupant_reference
from disease_priority_cascade_logic import run_disease_priority_cascade_logic
from disease_priority_cascade_logic_v2 import run_disease_priority_cascade_logic_v2, build_organ_priority_view
from shadbala_weighted_disease_priority import run_shadbala_weighted_disease_priority
from dasha_weighted_disease_priority import run_dasha_weighted_disease_priority
import os


def _json_fallback_message(message: str, question_type: str = "context") -> str:
    return _json.dumps({
        "reasoning": "System fallback response due to temporary model/API issue.",
        "cluster_rank": 0,
        "question_type": question_type,
        "chat_message": message
    })

app = Flask(__name__)

# Abhishek's RL enhancer (rl_enhancer/), merged in: his page at /rl plus his
# /api/... and /healthz endpoints. Replaces running his separate FastAPI app.
from rl_integration import rl_bp, init_rl
app.register_blueprint(rl_bp)

# Admin page (/admin): accounts, charts, delete / reset. Admin-only data.
from admin_routes import admin_bp
app.register_blueprint(admin_bp)
from tester_routes import tester_bp
app.register_blueprint(tester_bp)
import tester_access

init_rl()
CORS(app)

from config import (
    PROKERALA_CLIENT_ID,
    PROKERALA_CLIENT_SECRET,
    GEMINI_MODEL,
    LLM_PROVIDER,
    SECRET_KEY,
)
import gemini_client

app.secret_key = SECRET_KEY

# Login (Clerk) + main DB (Firestore) come from the colleague's app_core/,
# wired in through app_auth.py. Login is switched with the AUTH_ENABLED env
# var (off by default). The old email/password login (auth.py, login.html,
# seed_users.py) is no longer used.
import db_layer
from app_auth import (
    AUTH_ENABLED,
    AUTH_ERRORS,
    check_patient_access,
    db_error_hint,
    describe_main_db,
    is_admin,
    main_db_label,
    login_required,
    print_startup_status,
    register_auth_errors,
    user_owns_patient,
)
from app_core.runtime import get_services
from chart_access import (
    ALGO_VERSION,
    ChartLimitError,
    chart_status,
    check_can_generate,
    load_chart_source,
    mark_source_version,
    record_new_chart,
    save_chart_source,
)
register_auth_errors(app)

# Single Orchestrator instance — the only thing that should call the astro
# engine, deidentification, chat_module, or the RAG pipeline directly.
# Replaces the old standalone conversation_store dict; chat_module still
# gets a plain dict to mutate, it's just orchestrator's internal one now.
from orchestrator import Orchestrator
orchestrator = Orchestrator()

# Credentials now imported from config.py

# ── Token cache (auto-refreshes before expiry) ─────────────────────────────
_cached_token     = None
_token_expires_at = None


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
    print("✅ Token fetched")
    return _cached_token


def get_timezone_offset(lat: float, lng: float) -> str:
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


def get_timezone_name(lat: float, lng: float) -> str:
    try:
        tf = TimezoneFinder()
        return tf.timezone_at(lat=lat, lng=lng) or "Asia/Kolkata"
    except Exception:
        return "Asia/Kolkata"


def build_planet_position_snapshot(data: dict) -> dict:
    """Parse Prokerala planet-position payload into UI-friendly planet rows."""
    NAKSHATRAS = [
        "Ashwini","Bharani","Krithika","Rohini",
        "Mrigashira","Aridra","Punarvasu","Pushya",
        "Ashlesha","Magha","Purvaphalguni",
        "Uttara Phalguni","Hasta","Chitha","Swathi",
        "Vishaka","Anuradha","Jyesta","Moola",
        "Poorvashada","Uttaraashada","Shravana",
        "Dhanishta","Shathabhisha","Poorvabhadra",
        "Uttara Bhadrapada","Revathi"
    ]
    ZODIAC_IDX = {
        "Mesha": 0, "Vrishabha": 1, "Mithuna": 2, "Karka": 3,
        "Simha": 4, "Kanya": 5, "Tula": 6, "Vrischika": 7,
        "Dhanu": 8, "Makara": 9, "Kumbha": 10, "Meena": 11
    }

    def calc_nakshatra(sign_name, deg, mins):
        sign_idx = ZODIAC_IDX.get(sign_name, 0)
        full_lon = sign_idx * 30 + deg + mins / 60.0
        nk_idx = int(full_lon / 13.3333) % 27
        pada = int((full_lon % 13.3333) / 3.3333) + 1
        return {"name": NAKSHATRAS[nk_idx], "pada": min(pada, 4)}

    nakshatra_map = {}
    for p in data.get("planet_position", []):
        pname = p.get("name", "")
        sign_name = p.get("rasi", {}).get("name", "Mesha")
        deg_raw = p.get("degree", 0)
        deg = int(deg_raw)
        mins = int((deg_raw % 1) * 60)
        nakshatra_map[pname] = calc_nakshatra(sign_name, deg, mins)

    ascendant = {}
    for p in data.get("planet_position", []):
        if p.get("name") == "Ascendant":
            asc_id = p.get("rasi", {}).get("id", 0)
            ascendant = {
                "name": p.get("rasi", {}).get("name", "—"),
                "id": asc_id,
                "house": 1,
                "nakshatra": nakshatra_map.get("Ascendant", {"name": "—", "pada": ""}),
            }
            break

    asc_position = ascendant["id"] + 1 if "id" in ascendant else 1
    planets = []
    for p in data.get("planet_position", []):
        pname = p.get("name", "")
        if pname == "Ascendant":
            planets.append({
                "name": "Ascendant",
                "house": 1,
                "absolute_house": p.get("position", 1),
                "sign": {"name": p.get("rasi", {}).get("name", "—")},
                "degree": int(p.get("degree", 0)),
                "minutes": int((p.get("degree", 0) % 1) * 60),
                "nakshatra": nakshatra_map.get("Ascendant", {"name": "—", "pada": ""}),
                "isRetrograde": False,
            })
            continue

        planet_position = p.get("position", 1)
        house_number = ((planet_position - asc_position) % 12) + 1
        planets.append({
            "name": pname,
            "house": house_number,
            "absolute_house": planet_position,
            "sign": {"name": p.get("rasi", {}).get("name", "—")},
            "degree": int(p.get("degree", 0)),
            "minutes": int((p.get("degree", 0) % 1) * 60),
            "nakshatra": nakshatra_map.get(pname, {"name": "—", "pada": ""}),
            "isRetrograde": p.get("is_retrograde", False),
        })
    return {"ascendant": ascendant, "planets": planets}


def _legacy_build_imp_rashi_distance_analysis_unused(chart_data: dict, rule1_result: dict, current_positions: dict, dasha_result: dict) -> dict:
    planets = chart_data.get("planets", []) or []
    houses = chart_data.get("houses", []) or []
    current_planets = (current_positions or {}).get("planets", []) or []
    current_by_name = {p.get("name"): p for p in current_planets if p.get("name")}

    moon = next((p for p in planets if p.get("name") == "Moon"), None)
    if not moon:
        return {
            "available": False,
            "reason": "Moon not found in birth chart planets.",
            "rows": [],
            "trail": {"error": "Moon not found"},
        }

    imp_house = int(moon.get("house", 0) or 0)
    imp_rashi = ""
    for house in houses:
        if int(house.get("house", 0) or 0) == imp_house:
            imp_rashi = ((house.get("sign") or {}).get("name", ""))
            break

    active_dasha = {
        "mahadasha": ((dasha_result or {}).get("mahadasha") or {}).get("planet", "—"),
        "antardasha": ((dasha_result or {}).get("antardasha") or {}).get("planet", "—"),
        "pratyantardasha": ((dasha_result or {}).get("pratyantardasha") or {}).get("planet", "—"),
    }

    increase_set = {2, 6, 7, 8, 12}
    lower_set = {1, 3, 4, 5, 9, 10, 11}

    role_rows = []

    def add_role_rows(role_code: str, role_label: str, items):
        if not items:
            return
        iterable = items if isinstance(items, list) else [items]
        for index, item in enumerate(iterable, start=1):
            if not item:
                continue
            planet_name = item.get("name") if isinstance(item, dict) else ""
            if not planet_name:
                continue
            current_planet = current_by_name.get(planet_name) or {}
            current_abs_house = int(current_planet.get("absolute_house", 0) or 0)
            current_sign = ((current_planet.get("sign") or {}).get("name", "—"))

            raw_gap = ((current_abs_house - imp_house + 12) % 12) if imp_house and current_abs_house else None
            distance = (raw_gap + 1) if raw_gap is not None else None

            active_levels = []
            if planet_name == active_dasha["mahadasha"]:
                active_levels.append("Mahadasha")
            if planet_name == active_dasha["antardasha"]:
                active_levels.append("Antardasha")
            if planet_name == active_dasha["pratyantardasha"]:
                active_levels.append("Pratyantardasha")

            bucket = "unmapped"
            effect = "No rule"
            if distance in increase_set:
                bucket = "increase"
                effect = "Increase Severity" if active_levels else "Increase Pattern Present"
            elif distance in lower_set:
                bucket = "lower"
                effect = "Lower / Nullify" if active_levels else "Lower / Nullify Pattern"

            role_rows.append({
                "role": role_code if len(iterable) == 1 else f"{role_code}.{index}",
                "role_base": role_code,
                "role_label": role_label,
                "moon_house": imp_house,
                "planet": planet_name,
                "birth_reference_sign": imp_rashi or "—",
                "current_absolute_house": current_abs_house or None,
                "current_sign": current_sign or "—",
                "raw_gap": raw_gap,
                "distance": distance,
                "distance_bucket": bucket,
                "dasha_active_levels": active_levels,
                "effect": effect,
                "note": (
                    f"Distance {distance} is in the {'increase' if bucket == 'increase' else 'lower/nullify' if bucket == 'lower' else 'unmapped'} group."
                    + (f" Active dasha: {', '.join(active_levels)}." if active_levels else " Planet is not active in current dasha.")
                ) if distance is not None else "Current planetary position not found for this planet.",
                "backend_trail": {
                    "imp_house": imp_house,
                    "target_absolute_house": current_abs_house or None,
                    "raw_gap_formula": f"(({current_abs_house} - {imp_house} + 12) % 12) = {raw_gap}" if current_abs_house else "Target house unavailable",
                    "distance_formula": f"{raw_gap} + 1 = {distance}" if raw_gap is not None else "Distance unavailable",
                    "increase_set": sorted(increase_set),
                    "lower_set": sorted(lower_set),
                    "active_dasha_levels": active_levels,
                }
            })

    add_role_rows("P1", "Occupant", rule1_result.get("rogkaraka_1", []))
    add_role_rows("P2", "Dispositor", rule1_result.get("rogkaraka_3"))
    add_role_rows("P3", "Drishti", [{"name": row.get("planet", "")} for row in (rule1_result.get("drishti_on_6th", []) or []) if row.get("planet")])
    add_role_rows("P4", "6th House Lord", rule1_result.get("rogkaraka_2"))

    return {
        "available": True,
        "title": "IMP Rashi Distance",
        "imp_rashi": {
            "moon_house": imp_house,
            "sign": imp_rashi or "—",
            "moon_sign": ((moon.get("sign") or {}).get("name", "—")) if isinstance(moon.get("sign"), dict) else moon.get("sign", "—"),
        },
        "rules": {
            "increase_distances": sorted(increase_set),
            "lower_distances": sorted(lower_set),
            "distance_mode": "Astrological house counting (same house = 1)",
            "raw_gap_mode": "Circular gap (same house = 0)",
        },
        "current_dasha": active_dasha,
        "rows": role_rows,
        "trail": {
            "birth_moon": {
                "house": imp_house,
                "sign": ((moon.get("sign") or {}).get("name", "—")) if isinstance(moon.get("sign"), dict) else moon.get("sign", "—"),
                "degree": moon.get("degree", 0),
                "minutes": moon.get("minutes", 0),
            },
            "imp_rashi_resolution": {
                "house_number_used": imp_house,
                "house_sign_found": imp_rashi or "—",
            },
            "current_planetary_timestamp": (current_positions or {}).get("as_of_local", "—"),
            "row_count": len(role_rows),
            "rows": role_rows,
        }
    }


async def fetch_current_planetary_positions(lat: float, lng: float) -> dict:
    token = await get_token()
    tz_name = get_timezone_name(lat, lng)
    tz = pytz.timezone(tz_name)
    current_local = datetime.now(tz)
    dt_iso = current_local.isoformat(timespec="seconds")
    coords = f"{lat},{lng}"

    print(f"\n[TRANSIT] /planet-position datetime={dt_iso} coords={coords}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(
            "https://api.prokerala.com/v2/astrology/planet-position",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "ayanamsa": 1,
                "coordinates": coords,
                "datetime": dt_iso,
                "la": "en",
            }
        )

    if r.status_code != 200:
        raise ChartServiceError(_api_failure("Current planet-position API", r))

    snapshot = build_planet_position_snapshot(r.json().get("data", {}) or {})
    return {
        "as_of_local": current_local.isoformat(),
        "timezone": tz_name,
        "ascendant": snapshot.get("ascendant", {}),
        "planets": snapshot.get("planets", []),
    }


async def fetch_chart(lat: float, lng: float, dob: str, birth_time: str) -> dict:
    """
    Calls the Prokerala /kundli endpoint.
    Returns: { ascendant, planets, houses, svg_url }
    """
    token     = await get_token()
    tz_offset = get_timezone_offset(lat, lng)
    dt_iso    = f"{dob}T{birth_time}:00{tz_offset}"
    coords    = f"{lat},{lng}"

    print(f"\n[CHART] /kundli  datetime={dt_iso}  coords={coords}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Call 1: Get planetary data
        r = await client.get(
            "https://api.prokerala.com/v2/astrology/planet-position",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "ayanamsa":    1,
                "coordinates": coords,
                "datetime":    dt_iso,
                "la":          "en"
            }
        )

        # Call 2: Get SVG Chart Image
        r_svg = await client.get(
            "https://api.prokerala.com/v2/astrology/chart",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "ayanamsa":    1,
                "coordinates": coords,
                "datetime":    dt_iso,
                "chart_type":  "rasi",
                "chart_style": "south-indian",
                "la":          "en"
            }
        )

        # Call 3: Get birth details (nakshatra etc.)
        r_details = await client.get(
            "https://api.prokerala.com/v2/astrology/birth-details",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "ayanamsa":    1,
                "coordinates": coords,
                "datetime":    dt_iso,
                "la":          "en"
            }
        )

        # Call 4: Get dasha periods
        r_dasha = await client.get(
            "https://api.prokerala.com/v2/astrology/dasha-periods",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "ayanamsa":    1,
                "coordinates": coords,
                "datetime":    dt_iso,
                "la":          "en"
            }
        )

    print(f"   Data Status:    {r.status_code}")
    print(f"   SVG Status:     {r_svg.status_code}")
    print(f"   Details Status: {r_details.status_code}")
    print(f"   Dasha Status:   {r_dasha.status_code}")

    dasha_raw = {}
    if r_dasha.status_code == 200:
        dasha_raw = r_dasha.json().get("data",{})
        print(f"   Dasha keys: {list(dasha_raw.keys())}")
    else:
        print(f"   Dasha failed: {r_dasha.text[:200]}")

    if r.status_code != 200:
        print(f"   Error body: {r.text}")
        raise Exception(f"kundli API error {r.status_code}: {r.text}")

    data = r.json().get("data", {})

    # ── Nakshatra map ─────────────────────────────────────────────────────
    NAKSHATRAS = [
        "Ashwini","Bharani","Krithika","Rohini",
        "Mrigashira","Aridra","Punarvasu","Pushya",
        "Ashlesha","Magha","Purvaphalguni",
        "Uttara Phalguni","Hasta","Chitha","Swathi",
        "Vishaka","Anuradha","Jyesta","Moola",
        "Poorvashada","Uttaraashada","Shravana",
        "Dhanishta","Shathabhisha","Poorvabhadra",
        "Uttara Bhadrapada","Revathi"
    ]

    ZODIAC_IDX = {
        "Mesha": 0, "Vrishabha": 1, "Mithuna": 2, "Karka": 3,
        "Simha": 4, "Kanya": 5, "Tula": 6, "Vrischika": 7,
        "Dhanu": 8, "Makara": 9, "Kumbha": 10, "Meena": 11
    }

    def calc_nakshatra(sign_name, deg, mins):
        sign_idx = ZODIAC_IDX.get(sign_name, 0)
        full_lon = sign_idx * 30 + deg + mins / 60.0
        nk_idx  = int(full_lon / 13.3333) % 27
        pada    = int((full_lon % 13.3333) / 3.3333) + 1
        return {"name": NAKSHATRAS[nk_idx], "pada": min(pada, 4)}

    # birth-details only returns the Lagna nakshatra, NOT per-planet nakshatra.
    # So we always calculate nakshatra from planet longitude for all planets.
    print("   Calculating nakshatra from planet longitudes")
    nakshatra_map = {}
    for p in data.get("planet_position", []):
        pname     = p.get("name", "")
        sign_name = p.get("rasi", {}).get("name", "Mesha")
        deg_raw   = p.get("degree", 0)
        deg       = int(deg_raw)
        mins      = int((deg_raw % 1) * 60)
        nakshatra_map[pname] = calc_nakshatra(sign_name, deg, mins)
    print(f"   Nakshatras calculated for: {list(nakshatra_map.keys())}")

    # ── Planets and Ascendant ──────────────────────────────────────────────────
    planets = []
    ascendant = {}
    
    # First pass: Identify Ascendant
    for p in data.get("planet_position", []):
        if p.get("name") == "Ascendant":
            asc_id = p.get("rasi", {}).get("id", 0)
            nk_data = nakshatra_map.get("Ascendant", {"name": "—", "pada": ""})
            ascendant = {
                "name": p.get("rasi", {}).get("name", "—"),
                "id": asc_id,
                "house": 1,
                "nakshatra": nk_data,
                "nakshatra_pada": nk_data.get("pada", 1)
            }
            break

    asc_position = ascendant["id"] + 1 if "id" in ascendant else 1

    # Second pass: Process all planets
    for p in data.get("planet_position", []):
        pname = p.get("name", "")
        if pname == "Ascendant":
            planets.append({
                "name":         "Ascendant",
                "house":        1,
                "sign":         {"name": p.get("rasi", {}).get("name", "—")},
                "degree":       int(p.get("degree", 0)),
                "minutes":      int((p.get("degree", 0) % 1) * 60),
                "nakshatra":    nakshatra_map.get("Ascendant", {"name": "—", "pada": ""}),
                "isRetrograde": False,
            })
            continue

        planet_position = p.get("position", 1)
        house_number = ((planet_position - asc_position) % 12) + 1

        planets.append({
            "name":         pname,
            "house":        house_number,
            "sign":         {"name": p.get("rasi", {}).get("name", "—")},
            "degree":       int(p.get("degree", 0)),
            "minutes":      int((p.get("degree", 0) % 1) * 60),
            "nakshatra":    nakshatra_map.get(pname, {"name": "—", "pada": ""}),
            "isRetrograde": p.get("is_retrograde", False),
        })

    # ── Houses (calculated from Ascendant) ───────────────────────────────────
    ZODIAC = ["", "Mesha", "Vrishabha", "Mithuna",
              "Karka", "Simha", "Kanya", "Tula",
              "Vrischika", "Dhanu", "Makara",
              "Kumbha", "Meena"]

    houses = []
    if "id" in ascendant:
        for i in range(12):
            sign_pos = ((asc_position - 1 + i) % 12) + 1
            houses.append({
                "house": i + 1,
                "sign":  {"name": ZODIAC[sign_pos]}
            })

    # ── Chart SVG image from second API call ──────────────────────────────────
    svg_raw = ""
    if r_svg.status_code == 200:
        svg_raw = r_svg.text

    print(f"[OK] Chart ready — {len(planets)} planets | {len(houses)} houses | asc={ascendant.get('name','?')} | svg={'yes' if svg_raw else 'no'}")

    return {
        "planets":   planets,
        "houses":    houses,
        "ascendant": ascendant,
        "svg_raw":   svg_raw,
        "dasha_raw": dasha_raw
    }



# ══════════════════════════════════════════════════════════════
# LOGIN (Clerk) — the browser side is static/session.js
# ══════════════════════════════════════════════════════════════
@app.get("/public-config")
def public_config():
    # Only the publishable key — it is meant to be public. Sent only when
    # login is on, so the pages don't show a Sign in button that does nothing.
    settings = get_services().settings
    return jsonify({
        "clerkPublishableKey": settings.clerk_publishable_key if AUTH_ENABLED else "",
        "frontendUrl": settings.frontend_url,
    })


@app.get("/me/records")
@login_required(rate_limit="read")
def my_records():
    # What the main DB holds for the signed-in user (or 'local-dev' when
    # login is off) — handy for checking that chart saves arrived.
    records = get_services().repository.list_records(g.user["internal_user_id"])
    return jsonify({"success": True, "records": records})


# ══════════════════════════════════════════════════════════════
# SERVE FRONTEND ROUTES
# ══════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return send_from_directory(
        ".", "astrohealth-landing.html")

# The logo / "New Analysis" links on results.html and priority6.html use this file name.
@app.route("/astrohealth-landing.html")
def landing_html():
    return send_from_directory(
        ".", "astrohealth-landing.html")

@app.route("/results")
def results_page():
    return send_from_directory(
        ".", "results.html")

@app.route("/results.html")
def results_html():
    return send_from_directory(
        ".", "results.html")

@app.route("/print_report.js")
def print_report_js():
    return send_from_directory(
        ".", "print_report.js")

@app.route("/print-report")
@app.route("/print_report.html")
def print_report_html():
    return send_from_directory(
        ".", "print_report.html")

@app.route("/priority6")
def priority6_page():
    return send_from_directory(
        ".", "priority6.html")

@app.route("/favicon.ico")
def favicon():
    return "", 204


def _read_json_if_exists(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return _json.load(f)


def _resolve_patient_gender(chart_data=None, diagnosis=None, payload=None, processed=None, raw_chart=None):
    for source in (chart_data, diagnosis, payload, processed, raw_chart):
        if not isinstance(source, dict):
            continue
        gender = source.get("gender")
        if normalize_gender(gender):
            return gender
    return ""


def _build_latest_patient_payload(patient_id: str) -> dict:
    folder = os.path.join("patients", str(patient_id))
    chart_data = _read_json_if_exists(os.path.join(folder, "chart_data.json"))
    processed = _read_json_if_exists(os.path.join(folder, "processed_logic.json"))
    diagnosis = _read_json_if_exists(os.path.join(folder, "diagnosis.json"))
    raw_dasha = _read_json_if_exists(os.path.join(folder, "raw_dasha.json"))
    raw_chart = _read_json_if_exists(os.path.join(folder, "raw_chart.json"))

    payload = {}
    if raw_chart:
        payload.update({
            "name": raw_chart.get("name", ""),
            "dob": raw_chart.get("dob", ""),
            "birth_time": raw_chart.get("birth_time", ""),
            "birth_place": raw_chart.get("birth_place", raw_chart.get("place", "")),
        })

    if processed:
        payload.update({
            "ascendant": processed.get("ascendant", {}),
            "planets": processed.get("planets", []),
            "houses": processed.get("houses", []),
            "rule1": processed.get("rule1", {}),
            "dasha": processed.get("dasha", {}),
            "gender": processed.get("gender", ""),
            "complete_analysis": processed.get("complete_analysis", {}),
            "hitlist": processed.get("hitlist", {}),
            "most_probable": processed.get("most_probable", []),
            "top_diseases": processed.get("complete_analysis", {}).get("top_diseases", []),
            "rashi_correlation": processed.get("rashi_correlation", {}),
            "organ_truth_correlation": processed.get("organ_truth_correlation", {}),
            "common_findings": processed.get("common_findings", {}),
            "related_house_severity": processed.get("related_house_severity", {}),
            "dasha_chat_priority": processed.get("dasha_chat_priority", {}),
            "health_forecast": processed.get("health_forecast", []),
        })

    if chart_data:
        payload.update({
            "gender": chart_data.get("gender", payload.get("gender", "")),
            "disease_filter": chart_data.get("disease_filter", payload.get("disease_filter", {})),
            "rashi_correlation": chart_data.get("rashi_correlation", payload.get("rashi_correlation", {})),
            "current_planetary_positions": chart_data.get("current_planetary_positions", {}),
            "second_ascendant": chart_data.get("second_ascendant", {}),
            "imp_rashi_distance": chart_data.get("imp_rashi_distance", {}),
            "birth_current_distance": chart_data.get("birth_current_distance", {}),
            "svg_raw": chart_data.get("svg_raw", ""),
        })

        if not payload.get("zone_analysis") and chart_data.get("second_ascendant"):
            try:
                payload["zone_analysis"] = run_zone_analysis(chart_data, chart_data.get("second_ascendant"))
            except Exception:
                payload["zone_analysis"] = {}

    current_dasha = payload.get("dasha") or {}
    if payload.get("rule1") and raw_dasha and not current_dasha.get("antardasha_groups"):
        try:
            dasha_raw_data = raw_dasha.get("dasha_data", {}).get("data", {})
            enriched_dasha = analyse_dasha(dasha_raw_data, payload.get("rule1", {}))
            current_dasha.update({
                "mahadasha_periods": enriched_dasha.get("mahadasha_periods", []),
                "antardasha_groups": enriched_dasha.get("antardasha_groups", []),
            })
            payload["dasha"] = current_dasha
        except Exception:
            pass

    df = payload.get("disease_filter") or {}
    eq = (df.get("relation_equation") or {}) if isinstance(df, dict) else {}
    if payload.get("rule1") and payload.get("dasha") and not eq:
        try:
            from rules import apply_disease_filter
            gender_value = _resolve_patient_gender(chart_data, diagnosis, payload, processed, raw_chart)
            payload["disease_filter"] = apply_disease_filter(payload.get("rule1", {}), payload.get("dasha", {}), gender=gender_value)
        except Exception:
            pass

    if diagnosis:
        payload["diagnosis"] = diagnosis

    if not payload.get("common_findings") and (payload.get("hitlist") or payload.get("rashi_correlation")):
        try:
            from common_findings_engine import build_common_findings
            payload["common_findings"] = build_common_findings(
                payload.get("hitlist", {}),
                payload.get("rashi_correlation", {}),
            )
        except Exception:
            payload["common_findings"] = {}

    if not payload.get("related_house_severity") and payload.get("rule1") and payload.get("dasha"):
        try:
            from related_house_severity_engine import build_related_house_severity
            payload["related_house_severity"] = build_related_house_severity(
                payload.get("rule1", {}),
                payload.get("dasha", {}),
                payload.get("planets", []),
            )
        except Exception:
            payload["related_house_severity"] = {}

    if payload.get("planets") and payload.get("rule1"):
        try:
            from organ_truth_correlation_engine import run_organ_truth_correlation_analysis
            gender_value = _resolve_patient_gender(chart_data, diagnosis, payload, processed, raw_chart)
            gender_code = normalize_gender(gender_value)
            live_chart = {
                "planets": payload.get("planets", []),
                "ascendant": payload.get("ascendant", {}),
                "houses": payload.get("houses", []),
            }
            payload["organ_truth_correlation"] = run_organ_truth_correlation_analysis(
                live_chart,
                payload.get("rule1", {}),
                gender=gender_code,
            )
        except Exception:
            pass
        try:
            from disease_first_logic import run_disease_first_logic, build_current_dasha_priority_sources
            gender_value = _resolve_patient_gender(chart_data, diagnosis, payload, processed, raw_chart)
            gender_code = normalize_gender(gender_value)
            live_chart = {
                "planets": payload.get("planets", []),
                "ascendant": payload.get("ascendant", {}),
                "houses": payload.get("houses", []),
            }
            payload["disease_first_logic"] = run_disease_first_logic(
                live_chart,
                payload.get("rule1", {}),
                gender=gender_code,
            )
            payload["current_dasha_priority_sources"] = build_current_dasha_priority_sources(
                live_chart,
                payload.get("rule1", {}),
                payload.get("dasha", {}),
                gender=gender_code,
            )
        except Exception:
            pass
        try:
            from disease_compare_logic import run_disease_compare_logic
            gender_value = _resolve_patient_gender(chart_data, diagnosis, payload, processed, raw_chart)
            gender_code = normalize_gender(gender_value)
            live_chart = {
                "planets": payload.get("planets", []),
                "ascendant": payload.get("ascendant", {}),
                "houses": payload.get("houses", []),
            }
            payload["disease_compare_logic"] = run_disease_compare_logic(
                live_chart,
                payload.get("rule1", {}),
                gender=gender_code,
            )
        except Exception:
            pass

    gender_value = _resolve_patient_gender(chart_data, diagnosis, payload, processed, raw_chart)
    payload["gender"] = gender_value or payload.get("gender", "")
    payload["kb_planet_diseases"] = filter_planet_diseases_by_gender(PLANET_DISEASES, gender_value)
    payload["kb_nakshatra_diseases"] = filter_nakshatra_diseases_by_gender(NAKSHATRA_DISEASES, gender_value)
    payload["kb_rashi_organs"] = filter_rashi_organs_by_gender(RASHI_ORGANS, gender_value)
    payload["kb_nakshatra_lords"] = dict(NAKSHATRA_LORDS)

    try:
        def _nk_name(obj):
            nk = (obj or {}).get("nakshatra", {})
            if isinstance(nk, dict):
                return nk.get("name", "")
            return nk or ""

        def _nk_lord(name):
            return NAKSHATRA_LORDS.get(name, "—")

        planets = payload.get("planets", []) or []
        planets_by_name = {p.get("name"): p for p in planets if p.get("name")}
        rule1 = payload.get("rule1", {}) or {}

        sixth_occupants = []
        for occ in rule1.get("rogkaraka_1", []) or []:
            planet = planets_by_name.get(occ.get("name", ""))
            nk_name = _nk_name(planet)
            sixth_occupants.append({
                "planet": occ.get("name", "—"),
                "nakshatra": nk_name or "—",
                "lord": _nk_lord(nk_name) if nk_name else "—",
            })

        eleventh_occupants = []
        for planet in planets:
            if int(planet.get("house", 0) or 0) != 11:
                continue
            nk_name = _nk_name(planet)
            eleventh_occupants.append({
                "planet": planet.get("name", "—"),
                "nakshatra": nk_name or "—",
                "lord": _nk_lord(nk_name) if nk_name else "—",
            })

        sixth_lord_planet = planets_by_name.get(rule1.get("sixth_house_lord", ""))
        sixth_lord_nk = _nk_name(sixth_lord_planet)
        dispositor_planet = planets_by_name.get((rule1.get("rogkaraka_3") or {}).get("name", ""))
        dispositor_nk = _nk_name(dispositor_planet)
        asc_lord_planet = planets_by_name.get(rule1.get("ascendant_lord", ""))
        asc_lord_nk = _nk_name(asc_lord_planet)

        eleventh_lord_name = ""
        houses = payload.get("houses", []) or []
        eleventh_house = next((h for h in houses if int(h.get("house", 0) or 0) == 11), {}) or {}
        eleventh_sign = ((eleventh_house.get("sign") or {}).get("name", ""))
        sign_lords = {
            "Aries": "Mars", "Mesha": "Mars",
            "Taurus": "Venus", "Vrishabha": "Venus", "Rishabha": "Venus",
            "Gemini": "Mercury", "Mithuna": "Mercury", "Mithun": "Mercury",
            "Cancer": "Moon", "Karka": "Moon", "Kataka": "Moon",
            "Leo": "Sun", "Simha": "Sun",
            "Virgo": "Mercury", "Kanya": "Mercury",
            "Libra": "Venus", "Tula": "Venus",
            "Scorpio": "Mars", "Vrischika": "Mars", "Vrishchika": "Mars",
            "Sagittarius": "Jupiter", "Dhanu": "Jupiter", "Dhanur": "Jupiter",
            "Capricorn": "Saturn", "Makara": "Saturn",
            "Aquarius": "Saturn", "Kumbha": "Saturn",
            "Pisces": "Jupiter", "Meena": "Jupiter",
        }
        eleventh_lord_name = sign_lords.get(eleventh_sign, "")
        eleventh_lord_planet = planets_by_name.get(eleventh_lord_name, {})
        eleventh_lord_nk = _nk_name(eleventh_lord_planet)

        payload["nakshatra_lord_reference"] = {
            "sixth_occupants": sixth_occupants,
            "sixth_lord": {
                "planet": rule1.get("sixth_house_lord", "—"),
                "nakshatra": sixth_lord_nk or "—",
                "lord": _nk_lord(sixth_lord_nk) if sixth_lord_nk else "—",
            },
            "dispositor": {
                "planet": (rule1.get("rogkaraka_3") or {}).get("name", "—"),
                "nakshatra": dispositor_nk or "—",
                "lord": _nk_lord(dispositor_nk) if dispositor_nk else "—",
            },
            "ascendant_lord": {
                "planet": rule1.get("ascendant_lord", "—"),
                "nakshatra": asc_lord_nk or "—",
                "lord": _nk_lord(asc_lord_nk) if asc_lord_nk else "—",
            },
            "eleventh_occupants": eleventh_occupants,
            "eleventh_lord": {
                "planet": eleventh_lord_name or "—",
                "nakshatra": eleventh_lord_nk or "—",
                "lord": _nk_lord(eleventh_lord_nk) if eleventh_lord_nk else "—",
            },
        }
    except Exception:
        payload["nakshatra_lord_reference"] = {}

    apply_bhadravathi_overrides(payload, raw_chart or payload)
    apply_person2_overrides(payload, raw_chart or payload)
    payload.setdefault("data_source", "Local Cache")
    return payload


# Algorithm outputs saved to the main DB, one document each. The kb_* keys
# (static knowledge base, same for everyone) and the SVG are left out.
_MAIN_DB_ALGORITHM_KEYS = (
    "rule1", "dasha", "complete_analysis", "disease_filter", "hitlist",
    "rashi_correlation", "organ_truth_correlation", "dasha_chat_priority",
    "nakshatra_diseases", "planet_diseases", "most_probable", "top_diseases",
    "diagnosis", "zone_analysis", "second_ascendant", "imp_rashi_distance",
    "birth_current_distance", "health_forecast",
)


def _save_chart_to_main_db(patient_id: str, source: str, response: dict, fetch_result: dict) -> None:
    """Copy of this chart run in the main DB. The patients/<id>/ files stay
    the working copy, so a DB failure is logged and the chart still returns."""
    from combined_algo_clusters import compute_combined_algo_summary
    try:
        chart_input = {key: response.get(key) for key in (
            "name", "dob", "birth_time", "birth_place", "father_name",
            "mother_name", "gender", "lat", "lng")}
        results = {key: response[key] for key in _MAIN_DB_ALGORITHM_KEYS if key in response}
        results["combined_algo"] = compute_combined_algo_summary(os.path.join("patients", patient_id))
        results["prokerala_result"] = {k: v for k, v in fetch_result.items() if k != "svg_raw"}
        saved = db_layer.save_chart_generation(patient_id, chart_input, source, results)
        user_id = g.user["internal_user_id"]
        print(f"[DB] Saved chart for patient {patient_id} to {describe_main_db()}")
        print(f"[DB]   under users/{user_id[:8]}.../  (input_id {saved['input_id']})")
        print(f"[DB]   inputs, chart_inputs  <- birth details")
        print(f"[DB]   prokerala_results     <- raw Prokerala chart data (no SVG)")
        names = ", ".join(doc_id[len(saved["input_id"]) + 1:] for doc_id in saved["algorithm_ids"])
        print(f"[DB]   algorithm_results     <- {saved['total_saved']} docs: {names}")
        print(f"[DB]   combined_results      <- links the {saved['total_saved']} algorithm docs to this run")
        print(f"[DB]   patients/{patient_id}          <- marks this patient as this user's")
    except Exception as e:
        print(f"[DB] [FAIL] could not save chart for patient {patient_id} to the main DB: {db_error_hint(e)}")


# ══════════════════════════════════════════════════════════════
# Chart pipeline — everything after Prokerala. /generate-chart runs it on
# freshly fetched data; _prepare_patient_chart re-runs it on a saved chart
# when the algorithms have changed (ALGO_VERSION), with no Prokerala call.
# ══════════════════════════════════════════════════════════════
def _run_chart_pipeline(patient_id: str, fetch_result: dict, birth: dict,
                        current_planetary_positions: dict, source: str) -> dict:
    name        = birth.get("name", "")
    dob         = birth.get("dob", "")
    birth_time  = birth.get("birth_time", "08:00")
    lat         = float(birth.get("lat") or 0)
    lng         = float(birth.get("lng") or 0)
    birth_place = birth.get("birth_place", "")
    father_name = birth.get("father_name", "")
    mother_name = birth.get("mother_name", "")
    gender      = birth.get("gender", "Unknown")

    # Step 2: Run logic processing
    processed = run_logic(patient_id)

    # Orchestrator: caches the Combined Algo consensus + de-identifies,
    # under patient_id so a later chat session for this same patient can
    # find it (there's no session_id yet at this point in the flow).
    # Must run after run_logic(): the Combined Algo it caches is computed
    # from the processed_logic.json that run_logic() just wrote.
    orchestrator.handle_chart_generation(
        patient_id=patient_id,
        name=name,
        dob=dob,
        birth_place=birth_place,
        chart_data=fetch_result,
        father_name=father_name,
        mother_name=mother_name,
    )

    # RL handoff file: the Combined Algo prediction is known now; the
    # post-chat section stays empty until /combine-output runs after the chat.
    write_rl_report(os.path.join("patients", patient_id))

    # Step 3: Run disease diagnosis
    diagnosis = run_diagnose(patient_id)

    second_ascendant = calculate_second_ascendant(
        dob=dob,
        birth_time=birth_time,
        lat=lat,
        lng=lng,
        planets=fetch_result["planets"],
        houses=fetch_result["houses"],
    )
    # Step 4: Run zone analysis
    zone_analysis = run_zone_analysis(fetch_result, second_ascendant)
    imp_rashi_distance = build_imp_rashi_distance_analysis(
        fetch_result,
        processed["rule1"],
        current_planetary_positions,
        processed["dasha"],
    )
    birth_current_distance = build_birth_current_distance_analysis(
        fetch_result,
        processed["rule1"],
        current_planetary_positions,
        processed["dasha"],
    )

    from rules import apply_disease_filter
    disease_filter = apply_disease_filter(processed["rule1"], processed["dasha"], gender=gender)

    chart_data_path = f"patients/{patient_id}/chart_data.json"
    with open(chart_data_path, "w") as f:
        _json.dump({
            **fetch_result,
            "rule1":           processed["rule1"],
            "dasha":           processed["dasha"],
            "complete":        processed["complete_analysis"],
            "disease_filter":  disease_filter,
            "rashi_correlation": processed.get("rashi_correlation", {}),
            "organ_truth_correlation": processed.get("organ_truth_correlation", {}),
            "dasha_chat_priority": processed.get("dasha_chat_priority", {}),
            "zone_analysis":   zone_analysis,
            "current_planetary_positions": current_planetary_positions,
            "second_ascendant": second_ascendant,
            "imp_rashi_distance": imp_rashi_distance,
            "birth_current_distance": birth_current_distance,
        }, f, indent=2)

    processed_file = f"patients/{patient_id}/processed_logic.json"
    with open(processed_file, "r") as f:
        proc_dict = _json.load(f)

    # --- Diagnostic Logic moved to rules.py ---
    most_probable = processed["complete_analysis"].get("most_probable", [])
    
    proc_dict["disease_filter"] = disease_filter
    proc_dict["most_probable"] = most_probable
    proc_dict["gender"] = gender
    proc_dict["second_ascendant"] = second_ascendant
    proc_dict["zone_analysis"] = zone_analysis
    proc_dict["imp_rashi_distance"] = imp_rashi_distance
    proc_dict["birth_current_distance"] = birth_current_distance
    # Which version of the algorithms produced these results (re-run on open when older).
    proc_dict["algo_version"] = ALGO_VERSION

    with open(processed_file, "w") as f:
        _json.dump(proc_dict, f, indent=2)

    # Build mappings for legacy UI components if needed
    rule1_result = processed["rule1"]
    nakshatra_diseases = {}
    planet_diseases = {}
    rk_planets = (
        rule1_result.get("rogkaraka_1", []) +
        ([rule1_result["rogkaraka_2"]] if rule1_result.get("rogkaraka_2") else []) +
        ([rule1_result["rogkaraka_3"]] if rule1_result.get("rogkaraka_3") else [])
    )
    conditional_moon = rule1_result.get("conditional_moon") or {}
    if conditional_moon.get("include") and (conditional_moon.get("planet") or {}).get("name") == "Moon":
        rk_planets.append(conditional_moon["planet"])
    for p in rk_planets:
        if p.get("name"):
            nakshatra_diseases[p["name"]] = filter_terms_by_gender(
                get_nakshatra_diseases(p.get("nakshatra"), p.get("nakshatra_pada")),
                gender,
            )
            diseases = PLANET_DISEASES.get(p["name"], [])
            planet_diseases[p["name"]] = filter_terms_by_gender(diseases, gender)

    # Step 4: Build response exactly as before
    response = {
        "success":           True,
        "name":              name,
        "dob":               dob,
        "birth_time":        birth_time,
        "birth_place":       birth_place,
        "father_name":       father_name,
        "mother_name":       mother_name,
        "gender":            gender,
        "lat":               lat,
        "lng":               lng,
        "patient_id":        patient_id,
        "ascendant":         fetch_result["ascendant"],
        "planets":           fetch_result["planets"],
        "houses":            fetch_result["houses"],
        "svg_raw":           fetch_result["svg_raw"],
        "rule1":             processed["rule1"],
        "dasha":             processed["dasha"],
        "complete_analysis": processed["complete_analysis"],
        "disease_filter":    disease_filter,
        "hitlist":           processed.get("hitlist", {}),
        "rashi_correlation": processed.get("rashi_correlation", {}),
        "organ_truth_correlation": processed.get("organ_truth_correlation", {}),
        "dasha_chat_priority": processed.get("dasha_chat_priority", {}),
        "nakshatra_diseases": nakshatra_diseases,
        "planet_diseases":   planet_diseases,
        "kb_planet_diseases": filter_planet_diseases_by_gender(PLANET_DISEASES, gender),
        "kb_nakshatra_diseases": filter_nakshatra_diseases_by_gender(NAKSHATRA_DISEASES, gender),
        "kb_rashi_organs": filter_rashi_organs_by_gender(RASHI_ORGANS, gender),
        "most_probable":     most_probable,
        "top_diseases":      processed["complete_analysis"].get("top_diseases", []),
        "diagnosis":         diagnosis,
        "zone_analysis":     zone_analysis,
        "current_planetary_positions": current_planetary_positions,
        "second_ascendant":  second_ascendant,
        "imp_rashi_distance": imp_rashi_distance,
        "birth_current_distance": birth_current_distance,
        "health_forecast":   processed.get("health_forecast", []),
        "data_source":       source,
    }
    return response


def _save_chart_source(patient_id: str, birth: dict, current_planetary_positions: dict) -> None:
    """The raw Prokerala chart + dasha in Firestore, so the patient folder can be
    rebuilt and the algorithms re-run later without calling Prokerala."""
    folder = os.path.join("patients", patient_id)
    try:
        size = save_chart_source(
            patient_id,
            _read_json_if_exists(os.path.join(folder, "raw_chart.json")) or {},
            _read_json_if_exists(os.path.join(folder, "raw_dasha.json")) or {},
            birth,
            current_planetary_positions,
        )
        print(f"[DB]   chart_sources/{patient_id}     <- raw chart + dasha ({size // 1024} KB), "
              f"algo v{ALGO_VERSION} - used to reopen and re-run without Prokerala")
    except Exception as e:
        print(f"[DB] [FAIL] could not save the chart source for patient {patient_id}: {db_error_hint(e)}")


CHART_LIMIT_MESSAGE = ("Your birth chart is ready — continue your health chat. "
                       "Each account can create one chart and correct it once.")
CHART_SERVICE_MESSAGE = ("The birth chart service is unavailable right now. "
                         "Please try again later - this attempt did not count towards your chart.")


# ══════════════════════════════════════════════════════════════
# POST /generate-chart
# ══════════════════════════════════════════════════════════════
@app.route("/generate-chart", methods=["POST"])
@login_required(rate_limit="chart")
def generate_chart():
    try:
        body = request.get_json(force=True)

        name        = body.get("name", "").strip()
        dob         = body.get("dob", "")          # "YYYY-MM-DD"
        birth_time  = body.get("birth_time", "08:00")  # "HH:MM"
        lat         = float(body.get("lat", 0))
        lng         = float(body.get("lng", 0))
        birth_place = body.get("birth_place", "")
        father_name = body.get("father_name", "")
        mother_name = body.get("mother_name", "")
        gender      = body.get("gender", "Unknown")

        if not name or not dob or not lat or not lng:
            return jsonify({"success": False, "error": "name, dob, lat, lng are required"}), 400

        # One chart + one correction per account — checked before any Prokerala call.
        check_can_generate()
        # Testers: a daily cap on NEW Prokerala charts (cached charts are not counted).
        tester_access.check_daily_cap(name, dob, birth_time, lat, lng)

        print(f"\n{'='*50}")
        print(f"  Patient     : {name}")
        print(f"  DOB         : {dob}  Time: {birth_time}")
        print(f"  Birth Place : {birth_place}")
        print(f"  Lat / Lng   : {lat}, {lng}")
        print(f"{'='*50}")

        # Today's planets first: it needs no patient, so if Prokerala is down
        # we stop before a patient folder exists.
        current_planetary_positions = asyncio.run(fetch_current_planetary_positions(lat, lng))

        # Step 1: Fetch and save raw data
        fetch_result = asyncio.run(fetch_and_save(
            name       = name,
            dob        = dob,
            birth_time = birth_time,
            lat        = lat,
            lng        = lng,
            place      = birth_place,
            gender     = gender,
            father_name = father_name,
            mother_name = mother_name
        ))

        patient_id = fetch_result["patient_id"]
        source     = fetch_result.get("source", "Unknown")
        print(f"Data Source: {source}")
        print(f"Patient ID: {patient_id}")

        birth = {
            "name": name, "dob": dob, "birth_time": birth_time, "birth_place": birth_place,
            "lat": lat, "lng": lng, "gender": gender,
            "father_name": father_name, "mother_name": mother_name,
        }
        response = _run_chart_pipeline(patient_id, fetch_result, birth,
                                       current_planetary_positions, source)

        _save_chart_to_main_db(patient_id, source, response, fetch_result)
        _save_chart_source(patient_id, birth, current_planetary_positions)
        try:
            record_new_chart(patient_id)
        except Exception as e:
            print(f"[Allowance] [FAIL] could not record the chart for this account: {db_error_hint(e)}")
        if tester_access.is_tester():
            try:
                tester_access.record_run(patient_id, birth, source)
            except Exception as e:
                print(f"[Tester] [FAIL] could not save this run: {db_error_hint(e)}")
        return jsonify(response)

    except tester_access.TesterCapError:
        cap = tester_access.daily_cap()
        return jsonify({"success": False, "code": "tester_cap",
                        "error": f"Daily limit reached: {cap} new charts per day. "
                                 f"Charts you already made can be re-run any time, and the limit resets at midnight (IST)."}), 429
    except ChartLimitError:
        return jsonify({"success": False, "code": "chart_limit", "error": CHART_LIMIT_MESSAGE}), 403
    except ChartServiceError as e:
        # Nothing was created and the allowance is untouched; the reason is
        # in the console, the user gets a plain message.
        for reason in str(e).split("; "):
            print(f"[Prokerala] [FAIL] {reason}")
        print("[Prokerala] [FAIL] Chart not created - no patient folder made, allowance not used -> 503")
        return jsonify({"success": False, "code": "chart_service",
                        "error": CHART_SERVICE_MESSAGE}), 503
    except AUTH_ERRORS:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Error] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# Reopening a saved chart — Firestore first, re-run if the algorithms changed
# ══════════════════════════════════════════════════════════════
_LOADED_FROM_DB: dict = {}   # (user, patient) -> when the chart was last pulled from Firestore
_RELOAD_AFTER_SECONDS = 60   # the results page asks for the same patient twice on load


def _prepare_patient_chart(patient_id: str) -> None:
    """Refresh patients/<id>/ from the user's saved chart in Firestore, and re-run
    the algorithms on it when the saved results are from an older ALGO_VERSION.
    Never calls Prokerala."""
    folder = os.path.join("patients", patient_id)
    key = (g.user["internal_user_id"], patient_id)
    source = None

    since = time.time() - _LOADED_FROM_DB.get(key, 0)
    if since < _RELOAD_AFTER_SECONDS:
        print(f"[DB] Patient {patient_id}: chart already loaded from {main_db_label()} {int(since)}s ago - reusing it")
    else:
        try:
            source = load_chart_source(patient_id)
        except AUTH_ERRORS as e:
            print(f"[DB] [FAIL] could not read patient {patient_id} from {main_db_label()}: {db_error_hint(e)} "
                  f"- falling back to local files patients/{patient_id}/")
        if source:
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "raw_chart.json"), "w", encoding="utf-8") as f:
                _json.dump(source["raw_chart"], f, indent=2)
            with open(os.path.join(folder, "raw_dasha.json"), "w", encoding="utf-8") as f:
                _json.dump(source["raw_dasha"], f, indent=2)
            profile = _read_json_if_exists(os.path.join(folder, "profile.json")) or {}
            with open(os.path.join(folder, "profile.json"), "w", encoding="utf-8") as f:
                _json.dump({**profile, **source["profile"], "patient_id": patient_id}, f, indent=2)
            _LOADED_FROM_DB[key] = time.time()
            print(f"[DB] Patient {patient_id} chart loaded from {main_db_label()}: {source['path']} "
                  f"(raw chart + dasha, {source['bytes'] // 1024} KB) -> working copy patients/{patient_id}/ refreshed")
        else:
            print(f"[DB] Patient {patient_id}: no saved chart in {main_db_label()} for this account "
                  f"- using local files patients/{patient_id}/")

    processed = _read_json_if_exists(os.path.join(folder, "processed_logic.json"))
    have = processed.get("algo_version", 1) if processed else None
    if have == ALGO_VERSION:
        print(f"[Algo] Patient {patient_id}: results are v{have} (current algorithms) - no re-run needed")
        return

    raw_chart = _read_json_if_exists(os.path.join(folder, "raw_chart.json"))
    raw_dasha = _read_json_if_exists(os.path.join(folder, "raw_dasha.json"))
    if not raw_chart or not raw_dasha:
        print(f"[Algo] Patient {patient_id}: no saved chart to re-run the algorithms on")
        return

    reason = "no results on this machine yet" if processed is None else f"results v{have} -> v{ALGO_VERSION}"
    print(f"[Algo] Patient {patient_id}: {reason} - re-running the 5 algorithms + Combined Algo "
          f"on the saved chart (no Prokerala call)")
    if processed is not None:
        # Older results stay for admins to compare.
        with open(os.path.join(folder, f"processed_logic.v{have}.json"), "w", encoding="utf-8") as f:
            _json.dump(processed, f)

    from jataka_logic import build_chart_dict
    chart = build_chart_dict(raw_chart, raw_dasha)
    fetch_result = {
        "patient_id": patient_id, "folder": folder,
        "planets": chart["planets"], "houses": chart["houses"], "ascendant": chart["ascendant"],
        "svg_raw": chart["svg_raw"], "dasha_raw": chart["dasha_raw"], "source": "Saved chart",
    }
    profile = _read_json_if_exists(os.path.join(folder, "profile.json")) or {}
    birth = {
        "name": raw_chart.get("name") or profile.get("name", ""),
        "dob": raw_chart.get("dob") or profile.get("dob", ""),
        "birth_time": raw_chart.get("birth_time") or profile.get("birth_time", ""),
        "birth_place": raw_chart.get("place") or profile.get("birth_place", ""),
        "lat": raw_chart.get("lat") or profile.get("lat"),
        "lng": raw_chart.get("lng") or profile.get("lng"),
        "gender": raw_chart.get("gender") or profile.get("gender", "Unknown"),
        "father_name": profile.get("father_name", ""),
        "mother_name": profile.get("mother_name", ""),
    }
    # The planet positions of the day the chart was made (refreshing them would need Prokerala).
    current = (source or {}).get("current_positions") or (
        (_read_json_if_exists(os.path.join(folder, "chart_data.json")) or {}).get("current_planetary_positions")) or {}

    response = _run_chart_pipeline(patient_id, fetch_result, birth, current,
                                   f"Saved chart (algorithms v{ALGO_VERSION})")
    print(f"[Algo] Patient {patient_id}: re-run done - results are now v{ALGO_VERSION}")
    if source:
        # Only for the account's own saved chart: an admin viewing someone else's
        # patient must not have it saved under the admin's account.
        _save_chart_to_main_db(patient_id, f"Algorithm re-run v{ALGO_VERSION}", response, fetch_result)
        try:
            mark_source_version(patient_id)
        except Exception as e:
            print(f"[DB] [FAIL] could not update the chart source version: {db_error_hint(e)}")


@app.get("/my-chart")
@login_required(rate_limit="read")
def my_chart():
    """What the home and results pages show: the form, or 'continue your chat'."""
    return jsonify({"success": True, **chart_status()})



# GET /health
@app.get("/health")
def health():
    return jsonify({"status": "ok", "server": "AstroMedica"})


DEMO_LIST_HIDDEN_PATIENT_IDS = {"007"}


@app.get("/patients-list")
@login_required(rate_limit="read")
def patients_list():
    root = "patients"
    out = []
    if os.path.isdir(root):
        for entry in sorted(os.listdir(root)):
            if entry in DEMO_LIST_HIDDEN_PATIENT_IDS:
                continue
            # With login on, only the signed-in user's own patients.
            if not user_owns_patient(entry):
                continue
            folder = os.path.join(root, entry)
            if not os.path.isdir(folder):
                continue
            profile = _read_json_if_exists(os.path.join(folder, "profile.json"))
            if not profile:
                continue
            out.append({
                "patient_id": entry,
                "name": profile.get("name", entry),
                "dob": profile.get("dob", ""),
                "birth_place": profile.get("birth_place", ""),
                "gender": profile.get("gender", ""),
            })
    return jsonify({"success": True, "patients": out})


@app.get("/patient-data/<patient_id>")
@login_required(rate_limit="read")
def patient_data(patient_id):
    check_patient_access(patient_id)
    try:
        # The account's saved chart from Firestore, re-run if the algorithms changed.
        _prepare_patient_chart(patient_id)
        payload = _build_latest_patient_payload(patient_id)
        if not payload:
            return jsonify({"success": False, "error": "patient data not found"}), 404

        # Patients opened from the list (not re-generated) never went through
        # /generate-chart, so make sure their RL file exists here. An existing
        # current file is left alone so its chat_log isn't wiped.
        try:
            ensure_rl_report(os.path.join("patients", str(patient_id)))
        except Exception as e:
            print(f"[RL] could not create For RL Report.json for {patient_id}: {e}")

        return jsonify({"success": True, "patient_id": patient_id, "data": payload})
    except AUTH_ERRORS:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.get("/priority6-data/<patient_id>")
@login_required(rate_limit="read")
def priority6_data(patient_id):
    """
    New priority-cascade page. Pulls ONLY the cached patient/chart facts
    (ascendant, planets, houses, rule1, dasha, gender, name/dob/birth_time/
    birth_place) from the existing cache -- never the old disease/organ
    logic outputs. The 5 new v2 algorithms + v2 combined summary are always
    computed fresh, with no chart-specific overrides applied.
    """
    check_patient_access(patient_id)
    try:
        payload = _build_latest_patient_payload(patient_id)
        if not payload:
            return jsonify({"success": False, "error": "patient data not found"}), 404

        folder = os.path.join("patients", str(patient_id))
        cached_chart_data = _read_json_if_exists(os.path.join(folder, "chart_data.json")) or {}
        shadbala = _read_json_if_exists(os.path.join(folder, "shadbala.json")) or {}

        chart_facts = {
            "name": payload.get("name", ""),
            "dob": payload.get("dob", ""),
            "birth_time": payload.get("birth_time", ""),
            "birth_place": payload.get("birth_place", ""),
            "ascendant": payload.get("ascendant", {}),
            "planets": payload.get("planets", []),
            "houses": payload.get("houses", []),
            "dasha": payload.get("dasha", {}),
            "gender": payload.get("gender", ""),
            "svg_raw": cached_chart_data.get("svg_raw", ""),
        }
        rule1_result = payload.get("rule1", {}) or {}
        gender = payload.get("gender", "")

        combined_v2 = build_combined_algo_summary_v2(chart_facts, rule1_result, gender)
        cascade_view = build_full_cascade_view(rule1_result)
        cascade_detail = build_cascade_tiers(chart_facts, rule1_result, gender)
        drishti_detail = build_drishti_catalysts(chart_facts, rule1_result, gender)
        ascendant_reference = build_ascendant_reference(chart_facts, rule1_result, gender)
        second_house_reference = build_house_occupant_reference(chart_facts, rule1_result, 2, gender)
        disease_priority_cascade = run_disease_priority_cascade_logic(chart_facts, rule1_result, gender)
        disease_priority_cascade_v2 = run_disease_priority_cascade_logic_v2(chart_facts, rule1_result, gender)
        organ_priority_v2 = build_organ_priority_view(disease_priority_cascade_v2)
        shadbala_weighted_priority = run_shadbala_weighted_disease_priority(chart_facts, rule1_result, shadbala, gender)
        dasha_weighted_priority = run_dasha_weighted_disease_priority(chart_facts, rule1_result, chart_facts.get("dasha", {}), gender)

        return jsonify({
            "success": True,
            "patient_id": patient_id,
            "cascade_view": cascade_view,
            "cascade_detail": cascade_detail,
            "drishti_detail": drishti_detail,
            "ascendant_reference": ascendant_reference,
            "second_house_reference": second_house_reference,
            "chart": chart_facts,
            "rule1": rule1_result,
            "combined_algo_summary_v2": combined_v2,
            "shadbala": shadbala,
            "disease_priority_cascade": disease_priority_cascade,
            "disease_priority_cascade_v2": disease_priority_cascade_v2,
            "organ_priority_v2": organ_priority_v2,
            "shadbala_weighted_priority": shadbala_weighted_priority,
            "dasha_weighted_priority": dasha_weighted_priority,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


async def call_gemini(
    system_prompt: str,
    conversation:  list
) -> str:
    """
    Calls Gemini APIs with full
    conversation history.
    Returns AI response text.
    """
    # Vertex AI with the ADC login (gemini_client.py) - API keys are blocked.
    url = gemini_client.generate_url(GEMINI_MODEL)
    headers = await gemini_client.auth_headers()

    # Build contents from conversation history
    contents = []

    for msg in conversation:
        role = msg["role"]
        text = msg["content"]

        # Gemini uses "user" and "model" roles
        gemini_role = (
            "model" if role == "assistant"
            else "user"
        )

        contents.append({
            "role":  gemini_role,
            "parts": [{"text": text}]
        })

    if "gemma" in GEMINI_MODEL.lower():
        if contents and contents[0]["role"] == "user":
            orig = contents[0]["parts"][0]["text"]
            contents[0]["parts"][0]["text"] = f"{system_prompt}\n\n[USER]: {orig}"
        else:
            contents.insert(0, {
                "role": "user",
                "parts": [{"text": f"{system_prompt}\n\n[USER]: Hello. Please begin."}]
            })
            
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature":     0.7,
                "maxOutputTokens": 1200,
                "topP":            0.95
            }
        }
    else:
        generation_config = {
            "temperature":     0.7,
            # gemini-2.5 counts its thinking tokens against maxOutputTokens;
            # at 1200 long thinking cut the JSON reply off mid-way.
            "maxOutputTokens": 2048,
            "topP":            0.95,
            # The chat always expects a JSON object back.
            "responseMimeType": "application/json",
        }
        if "2.5" in GEMINI_MODEL:
            generation_config["thinkingConfig"] = {"thinkingBudget": 512}
        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": contents,
            "generationConfig": generation_config
        }

    # 5xx from Gemini ("high demand", overloaded) and 429 (Vertex per-minute
    # quota) are usually gone within seconds.
    retry_waits = [2, 5]
    async with httpx.AsyncClient() as client:
        for attempt in range(len(retry_waits) + 1):
            r = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == len(retry_waits):
                break
            print(f"[Warning] Gemini {r.status_code}, retrying in {retry_waits[attempt]}s "
                  f"(attempt {attempt + 2} of {len(retry_waits) + 1})")
            await asyncio.sleep(retry_waits[attempt])

    if r.status_code != 200:
        if r.status_code == 429:
            # Raised, not returned as a chat message: chat_module.ask_gemini then
            # asks the plan's next question (or gives the final summary) itself,
            # instead of telling the patient to resend an answer already counted.
            print("[Warning] Gemini rate limit (429) still hit after retries - the chat uses its own next question")
            raise Exception("Gemini rate limit (429) after retries")

        if r.status_code in [401, 403]:
            message = gemini_client.auth_failure_message(r.status_code, r.text)
            print(f"[Gemini] [FAIL] {message}")
            raise Exception(f"[Error] Gemini login failed: {r.status_code}. {message}")

        print(f"Gemini error: {r.status_code}")
        print(f"Response: {r.text[:300]}")
        raise Exception(f"Gemini API Error: {r.status_code} — {r.text}")

    data = r.json()
    
    # Safely extract text, handling potential safety blocks
    try:
        candidate = data.get("candidates", [])[0]
        
        # Check if blocked by safety
        if candidate.get("finishReason") == "SAFETY":
            print("[Warning] Gemini blocked response due to SAFETY.")
            return _json_fallback_message(
                "I apologize, but I am unable to discuss this specific symptom further. Could we focus on your general wellness routine or daily habits instead?"
            )
            
        response = candidate["content"]["parts"][0]["text"]
    except (IndexError, KeyError) as e:
        print(f"Gemini API unexpected format: {data}")
        raise Exception(f"Failed to parse Gemini response: {str(e)}")

    finish = candidate.get("finishReason")
    if finish and finish != "STOP":
        print(f"[Warning] Gemini reply ended early ({finish}) after {len(response)} chars")
    print(f"[OK] Gemini response "
          f"({len(response)} chars)")

    # Safely strip markdown code blocks if the AI accidentally adds them
    if response.strip().startswith("```"):
        import re
        # This matches ```json {data} ``` or just ``` {data} ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            response = match.group(1)

    return response.strip()

@app.route("/start-consultation", methods=["POST"])
@login_required(rate_limit="chat")
def start_consultation():
    """
    Starts a new AI health consultation.
    Reads patient data from files via chat_module.
    """
    try:
        body = request.get_json(force=True)

        session_id = body.get("session_id", "")
        patient_id = body.get("patient_id", "")

        if not session_id:
            return jsonify({"success": False, "error": "session_id required"}), 400
        if not patient_id:
            return jsonify({"success": False, "error": "patient_id required"}), 400
        check_patient_access(patient_id)

        print(f"\n[AI] Starting consultation: {session_id} | patient: {patient_id}")

        opening_result = orchestrator.start_chat_session(
            session_id,
            patient_id,
            call_gemini
        )
        if tester_access.is_tester():
            # Tester chats are for reviewing the chatbot: saved with the tester's
            # run (send_message) and kept out of RL's chat_logs (chat_module).
            orchestrator.get_session(session_id)["skip_rl"] = True

        return jsonify({
            "success":  True,
            "response": opening_result["response"],
            "turn":     1,
            "is_final": False,
            "chat_debug": opening_result.get("debug_context", {})
        })

    except AUTH_ERRORS:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Start consultation error:")
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/send-message", methods=["POST"])
@login_required(rate_limit="chat")
def send_message():
    """
    Handles each patient message via chat_module.
    """
    try:
        body = request.get_json(force=True)

        session_id   = body.get("session_id", "")
        user_message = body.get("message", "").strip()

        if not session_id:
            return jsonify({"success": False, "error": "session_id required"}), 400

        if not orchestrator.session_exists(session_id):
            # Sessions live in server memory: a restart or a second gunicorn
            # worker loses them. The page offers to restart the chat.
            print(f"[Chat] Session {session_id} not found (server restarted, or more than one worker?)")
            return jsonify({"success": False, "code": "session_ended", "error": "Session not found."}), 400
        # The request only carries session_id, so check the session's patient.
        check_patient_access(orchestrator.get_session(session_id)["patient_id"])

        print(f"\n💬 Turn {orchestrator.get_session(session_id)['turn_count']+1} — User: {user_message[:50]}...")

        result = orchestrator.handle_chat_message(
            session_id,
            user_message,
            call_gemini
        )

        if tester_access.is_tester():
            try:
                tester_access.sync_chat(orchestrator.get_session(session_id)["patient_id"])
            except Exception as e:
                print(f"[Tester] [FAIL] could not save the chat: {db_error_hint(e)}")

        # Deferred cleanup after final
        if result["is_final"]:
            import threading
            def cleanup():
                import time
                time.sleep(120)
                orchestrator.remove_session(session_id)
            threading.Thread(target=cleanup, daemon=True).start()

        return jsonify({
            "success":  True,
            "response": result["response"],
            "turn":     result["turn"],
            "is_final": result["is_final"],
            "chat_debug": result.get("debug_context", {})
        })

    except AUTH_ERRORS:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Send message error:")
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500



@app.route("/combine-output", methods=["POST"])
@login_required(rate_limit="chart")
def combine_output():
    """
    Module C aggregation endpoint.
    Synthesizes diagnosis and chat history into final_output.json.
    """
    try:
        body = request.get_json(force=True)
        patient_id = body.get("patient_id")

        if not patient_id:
            return jsonify({"success": False, "error": "patient_id required"}), 400
        check_patient_access(patient_id)

        # Verify chat log exists
        log_file = f"patients/{patient_id}/chat_log.json"
        if not os.path.exists(log_file):
            return jsonify({
                "success": False, 
                "error": "Chat record not found for this patient."
            }), 400

        print(f"🔄 Aggregating data for patient {patient_id}...")
        result = run_combine(patient_id)

        if not result:
            return jsonify({
                "success": False, 
                "error": "Aggregation engine failed to produce output."
            }), 500

        if tester_access.is_tester():
            try:
                tester_access.sync_chat(patient_id)
            except Exception as e:
                print(f"[Tester] [FAIL] could not save the final report: {db_error_hint(e)}")

        return jsonify({
            "success": True,
            "final_output": result
        })

    except AUTH_ERRORS:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Aggregation error:")
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# RL Report under the LLM report on results.html, for comparing the two while
# testing. RL_REPORT_VISIBLE_TO in .env: all (default) | admins | none.
RL_REPORT_VISIBLE_TO = os.environ.get("RL_REPORT_VISIBLE_TO", "all").strip().lower()


def _rl_report_visible() -> bool:
    if RL_REPORT_VISIBLE_TO == "admins":
        return is_admin()
    return RL_REPORT_VISIBLE_TO != "none"


@app.route("/rl-report/<patient_id>", methods=["GET"])
@login_required()
def rl_report(patient_id):
    check_patient_access(patient_id)
    if not _rl_report_visible():
        # Same answer whether or not a report exists - nothing to see here.
        return jsonify({"visible": False})
    from rl_integration import patient_rl_report
    return jsonify({"visible": True, **patient_rl_report(patient_id)})


# Password for the Dev side of the VIEW MODE toggle on results.html. Kept in
# .env so it never sits in the page source. Only hides the Dev cards from
# casual viewers - their data is still sent to the browser.
DEV_MODE_PASSWORD = os.environ.get("DEV_MODE_PASSWORD", "").strip()


@app.route("/dev-mode/unlock", methods=["POST"])
@login_required(rate_limit="unlock")
def dev_mode_unlock():
    if not DEV_MODE_PASSWORD:
        print("[Dev view] Unlock refused - DEV_MODE_PASSWORD is not set in .env")
        return jsonify({"success": False, "error": "Dev view is not set up on this server."}), 503
    typed = str((request.get_json(silent=True) or {}).get("password", "")).strip()
    if hmac.compare_digest(typed.encode(), DEV_MODE_PASSWORD.encode()):
        print("[Dev view] Unlocked")
        return jsonify({"success": True})
    print("[Dev view] Wrong password")
    return jsonify({"success": False, "error": "Wrong password."}), 403


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[OK] AstroMedica server starting on port {port}")
    print_startup_status()
    gemini_client.print_startup_status(GEMINI_MODEL)

    # host="0.0.0.0" opens the port so Render can see it
    app.run(host="0.0.0.0", port=port, debug=False)
