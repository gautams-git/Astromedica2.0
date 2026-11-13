from __future__ import annotations

from datetime import datetime, timedelta
from math import acos, asin, atan, cos, degrees, floor, radians, sin, tan

import pytz
from timezonefinder import TimezoneFinder


ZODIAC_SIGNS = [
    "Mesha", "Vrishabha", "Mithuna", "Karka", "Simha", "Kanya",
    "Tula", "Vrischika", "Dhanu", "Makara", "Kumbha", "Meena",
]

NAKSHATRAS = [
    "Ashwini", "Bharani", "Krithika", "Rohini",
    "Mrigashira", "Aridra", "Punarvasu", "Pushya",
    "Ashlesha", "Magha", "Purvaphalguni", "Uttara Phalguni",
    "Hasta", "Chitha", "Swathi", "Vishaka",
    "Anuradha", "Jyesta", "Moola", "Poorvashada",
    "Uttaraashada", "Shravana", "Dhanishta", "Shathabhisha",
    "Poorvabhadra", "Uttara Bhadrapada", "Revathi",
]

SIGN_INDEX = {name: idx for idx, name in enumerate(ZODIAC_SIGNS)}
NAKSHATRA_SPAN = 360.0 / 27.0
PADA_SPAN = NAKSHATRA_SPAN / 4.0
SIGN_LORDS = {
    "Mesha": "Mars",
    "Vrishabha": "Venus",
    "Mithuna": "Mercury",
    "Karka": "Moon",
    "Simha": "Sun",
    "Kanya": "Mercury",
    "Tula": "Venus",
    "Vrischika": "Mars",
    "Dhanu": "Jupiter",
    "Makara": "Saturn",
    "Kumbha": "Saturn",
    "Meena": "Jupiter",
}


def _normalize_degrees(value: float) -> float:
    return value % 360.0


def _timezone_name(lat: float, lng: float) -> str:
    tf = TimezoneFinder()
    return tf.timezone_at(lat=lat, lng=lng) or "Asia/Kolkata"


def _planet_absolute_degree(planet: dict) -> float:
    sign_name = ((planet or {}).get("sign") or {}).get("name", "")
    sign_idx = SIGN_INDEX.get(sign_name, 0)
    degree = float((planet or {}).get("degree", 0) or 0)
    minutes = float((planet or {}).get("minutes", 0) or 0)
    return sign_idx * 30.0 + degree + (minutes / 60.0)


def _absolute_degree_to_rashi(absolute_degree: float) -> dict:
    normalized = _normalize_degrees(absolute_degree)
    sign_idx = int(normalized // 30.0) % 12
    sign_degree = normalized - sign_idx * 30.0
    deg = int(sign_degree)
    mins = int(round((sign_degree - deg) * 60.0))
    if mins == 60:
        deg += 1
        mins = 0
    if deg == 30:
        sign_idx = (sign_idx + 1) % 12
        deg = 0
    return {
        "name": ZODIAC_SIGNS[sign_idx],
        "index": sign_idx,
        "degree": deg,
        "minutes": mins,
    }


def _absolute_degree_to_nakshatra(absolute_degree: float) -> dict:
    normalized = _normalize_degrees(absolute_degree)
    nak_index = int(normalized // NAKSHATRA_SPAN) % 27
    offset_inside_nak = normalized - (nak_index * NAKSHATRA_SPAN)
    pada = min(4, int(offset_inside_nak // PADA_SPAN) + 1)
    return {
        "name": NAKSHATRAS[nak_index],
        "index": nak_index,
        "pada": pada,
    }


def _absolute_degree_to_house(absolute_degree: float, houses: list[dict]) -> dict:
    rashi = _absolute_degree_to_rashi(absolute_degree)
    sign_name = rashi["name"]
    for house in houses or []:
        house_sign = ((house or {}).get("sign") or {}).get("name", "")
        if house_sign == sign_name:
            return {
                "house": house.get("house"),
                "sign": sign_name,
            }
    return {
        "house": None,
        "sign": sign_name,
    }


def _compute_local_sunrise(dob: str, lat: float, lng: float, tz_name: str) -> dict:
    date_obj = datetime.strptime(dob, "%Y-%m-%d").date()
    day_of_year = date_obj.timetuple().tm_yday
    lng_hour = lng / 15.0
    zenith = radians(90.833)

    t = day_of_year + ((6.0 - lng_hour) / 24.0)
    mean_anomaly = (0.9856 * t) - 3.289
    true_longitude = mean_anomaly + (1.916 * sin(radians(mean_anomaly))) + (0.020 * sin(radians(2 * mean_anomaly))) + 282.634
    true_longitude = _normalize_degrees(true_longitude)

    right_ascension = degrees(atan(0.91764 * tan(radians(true_longitude))))
    right_ascension = _normalize_degrees(right_ascension)
    l_quadrant = floor(true_longitude / 90.0) * 90.0
    ra_quadrant = floor(right_ascension / 90.0) * 90.0
    right_ascension = (right_ascension + (l_quadrant - ra_quadrant)) / 15.0

    sin_declination = 0.39782 * sin(radians(true_longitude))
    cos_declination = cos(asin(sin_declination))

    cos_hour_angle = (
        cos(zenith) - (sin_declination * sin(radians(lat)))
    ) / (cos_declination * cos(radians(lat)))

    if cos_hour_angle < -1 or cos_hour_angle > 1:
        raise ValueError("Sunrise could not be computed for this location/date.")

    hour_angle = 360.0 - degrees(acos(cos_hour_angle))
    hour_angle_hours = hour_angle / 15.0

    local_mean_time = hour_angle_hours + right_ascension - (0.06571 * t) - 6.622
    utc_hour = (local_mean_time - lng_hour) % 24.0

    utc_datetime = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=utc_hour)
    utc_aware = pytz.utc.localize(utc_datetime)
    local_zone = pytz.timezone(tz_name)
    local_dt = utc_aware.astimezone(local_zone)

    return {
        "timezone": tz_name,
        "sunrise_local": local_dt,
        "sunrise_utc": utc_aware,
        "trail": {
            "day_of_year": day_of_year,
            "longitude_hour": round(lng_hour, 6),
            "approx_time": round(t, 6),
            "mean_anomaly": round(mean_anomaly, 6),
            "true_longitude": round(true_longitude, 6),
            "right_ascension_hours": round(right_ascension, 6),
            "sin_declination": round(sin_declination, 6),
            "cos_declination": round(cos_declination, 6),
            "cos_hour_angle": round(cos_hour_angle, 6),
            "hour_angle_degrees": round(hour_angle, 6),
            "local_mean_time": round(local_mean_time, 6),
            "utc_hour": round(utc_hour, 6),
        }
    }


def calculate_second_ascendant(
    *,
    dob: str,
    birth_time: str,
    lat: float,
    lng: float,
    planets: list[dict],
    houses: list[dict],
) -> dict:
    tz_name = _timezone_name(lat, lng)
    local_zone = pytz.timezone(tz_name)
    birth_dt = local_zone.localize(datetime.strptime(f"{dob} {birth_time}", "%Y-%m-%d %H:%M"))

    sun = next((planet for planet in planets or [] if planet.get("name") == "Sun"), None)
    if not sun:
        raise ValueError("Sun position not found in chart data.")

    sunrise_data = _compute_local_sunrise(dob, lat, lng, tz_name)
    sunrise_local = sunrise_data["sunrise_local"]
    gap = birth_dt - sunrise_local
    gap_minutes_total = (gap.total_seconds() / 60.0)
    if gap_minutes_total < 0:
        gap_minutes_total += 24.0 * 60.0

    gap_hours_whole = int(gap_minutes_total // 60)
    gap_remaining_minutes = gap_minutes_total - (gap_hours_whole * 60)
    gap_display_minutes = int(round(gap_remaining_minutes))
    gap_display_hours = gap_hours_whole
    if gap_display_minutes == 60:
        gap_display_minutes = 0
        gap_display_hours += 1
    hour_degree = gap_hours_whole * 30.0
    minute_degree = (gap_remaining_minutes / 60.0) * 30.0
    gap_degree_total = hour_degree + minute_degree

    sun_absolute_degree = _planet_absolute_degree(sun)
    raw_total_degree = sun_absolute_degree + gap_degree_total
    normalized_total_degree = _normalize_degrees(raw_total_degree)

    rashi = _absolute_degree_to_rashi(normalized_total_degree)
    nakshatra = _absolute_degree_to_nakshatra(normalized_total_degree)
    house = _absolute_degree_to_house(normalized_total_degree, houses)

    return {
        "title": "2nd Ascendant",
        "sun_absolute_degree": round(sun_absolute_degree, 4),
        "sun_position": {
            "sign": ((sun.get("sign") or {}).get("name", "—")),
            "degree": int(sun.get("degree", 0) or 0),
            "minutes": int(sun.get("minutes", 0) or 0),
        },
        "birth": {
            "date": dob,
            "time": birth_time,
            "timezone": tz_name,
            "latitude": lat,
            "longitude": lng,
        },
        "sunrise": {
            "local_time": sunrise_local.strftime("%H:%M:%S"),
            "local_iso": sunrise_local.isoformat(),
            "utc_iso": sunrise_data["sunrise_utc"].isoformat(),
            "calculation_method": "Local astronomical sunrise calculation (NOAA-style solar position formula)",
        },
        "gap": {
            "total_minutes": round(gap_minutes_total, 4),
            "hours_component": gap_hours_whole,
            "minutes_component": round(gap_remaining_minutes, 4),
            "formatted": f"{gap_display_hours:02d}:{gap_display_minutes:02d}",
        },
        "degree_math": {
            "hour_formula": f"{gap_hours_whole} x 30 = {round(hour_degree, 4)}",
            "minute_formula": f"({round(gap_remaining_minutes, 4)} / 60) x 30 = {round(minute_degree, 4)}",
            "hour_degree": round(hour_degree, 4),
            "minute_degree": round(minute_degree, 4),
            "gap_degree_total": round(gap_degree_total, 4),
        },
        "final_degree": {
            "raw_total": round(raw_total_degree, 4),
            "normalized_total": round(normalized_total_degree, 4),
        },
        "result": {
            "house": house["house"],
            "rashi": rashi["name"],
            "planet": SIGN_LORDS.get(rashi["name"], "—"),
            "rashi_degree": rashi["degree"],
            "rashi_minutes": rashi["minutes"],
            "nakshatra": nakshatra["name"],
            "pada": nakshatra["pada"],
        },
        "trail": {
            "sun_absolute_formula": (
                f"({SIGN_INDEX.get(((sun.get('sign') or {}).get('name', '')), 0)} x 30) + "
                f"{int(sun.get('degree', 0) or 0)} + ({int(sun.get('minutes', 0) or 0)} / 60)"
            ),
            "sunrise_math": sunrise_data["trail"],
            "normalization_formula": (
                f"{round(raw_total_degree, 4)} % 360 = {round(normalized_total_degree, 4)}"
            ),
            "house_resolution": (
                f"Resolved sign {house['sign']} against house-sign sequence to get House {house['house']}"
                if house["house"] is not None else
                f"Resolved sign {house['sign']} but no matching house sign was found."
            ),
            "nakshatra_formula": (
                f"nakshatra_index = floor({round(normalized_total_degree, 4)} / {round(NAKSHATRA_SPAN, 6)}), "
                f"pada = floor(remainder / {round(PADA_SPAN, 6)}) + 1"
            ),
        },
    }
