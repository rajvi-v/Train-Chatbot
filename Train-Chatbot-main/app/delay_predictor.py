from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import joblib
import numpy as np

MODEL_PATH_WAT2WEY = "model_rf_WAT2WEY.pkl"
MODEL_PATH_WEY2WAT = "model_rf_WEY2WAT.pkl"

STATION_NAME_TO_CODE = {
    "london waterloo": "WAT", "waterloo": "WAT", "wat": "WAT",
    "clapham junction": "CLJ", "clapham": "CLJ", "clj": "CLJ",
    "woking": "WOK", "wok": "WOK",
    "basingstoke": "BSK", "bsk": "BSK",
    "winchester": "WIN", "win": "WIN",
    "shawford": "SHW", "shw": "SHW",
    "eastleigh": "ESL", "esl": "ESL",
    "southampton airport parkway": "SOA", "southampton airport": "SOA", "soa": "SOA",
    "swaythling": "SWG", "swg": "SWG",
    "st denys": "SDN", "sdn": "SDN",
    "southampton central": "SOU", "southampton": "SOU", "sou": "SOU",
    "totton": "TTN", "ttn": "TTN",
    "ashurst new forest": "ANF", "ashurst": "ANF", "anf": "ANF",
    "beaulieu road": "BEU", "beaulieu": "BEU", "beu": "BEU",
    "brockenhurst": "BCU", "bcu": "BCU",
    "sway": "SWY", "swy": "SWY",
    "new milton": "NWM", "nwm": "NWM",
    "hinton admiral": "HNA", "hna": "HNA",
    "christchurch": "CHR", "chr": "CHR",
    "pokesdown": "POK", "pok": "POK",
    "bournemouth": "BMH", "bmh": "BMH",
    "boscombe": "BSM", "bsm": "BSM",
    "poole": "POO", "poo": "POO",
    "hamworthy": "HAM", "ham": "HAM",
    "holton heath": "HOL", "hol": "HOL",
    "wareham": "WRM", "wrm": "WRM",
    "wool": "WOO", "woo": "WOO",
    "moreton": "MTN", "mtn": "MTN",
    "dorchester south": "DCH", "dorchester": "DCH", "dch": "DCH",
    "upwey": "UPW", "upw": "UPW",
    "weymouth": "WEY", "wey": "WEY",
    "guildford": "GLD", "gld": "GLD",
    "havant": "HAV", "hav": "HAV",
    "fareham": "FRM", "frm": "FRM",
}

STATION_ORDER_WAT2WEY = [
    "WAT", "CLJ", "WOK", "BSK", "WIN", "SHW", "ESL", "SOA",
    "SWG", "SDN", "SOU", "TTN", "ANF", "BEU", "BCU", "SWY",
    "NWM", "HNA", "CHR", "POK", "BMH", "BSM", "PKS", "POO",
    "HAM", "HOL", "WRM", "WOO", "MTN", "DCH", "UPW", "WEY",
]
STATION_ORDER_WEY2WAT = list(reversed(STATION_ORDER_WAT2WEY))

FEATURES = [
    "planned_dep_min",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak",
    "hour",
    "station_num",
    "journey_progress",
    "stations_remaining",
    "current_delay",
    "destination_num",
    "stops_to_dest",
]

JOURNEY_TIMES_FROM_WAT = {
    "WAT": 0,   "CLJ": 10,  "WOK": 28,  "BSK": 47,  "WIN": 63,
    "SHW": 68,  "ESL": 74,  "SOA": 78,  "SWG": 82,  "SDN": 85,
    "SOU": 90,  "TTN": 97,  "ANF": 103, "BEU": 109, "BCU": 115,
    "SWY": 120, "NWM": 126, "HNA": 131, "CHR": 136, "POK": 139,
    "BMH": 143, "BSM": 147, "PKS": 150, "POO": 155, "HAM": 160,
    "HOL": 165, "WRM": 170, "WOO": 177, "MTN": 183, "DCH": 189,
    "UPW": 194, "WEY": 200,
}

def _load_model(direction: str) -> dict:
    path = MODEL_PATH_WAT2WEY if direction == "WAT2WEY" else MODEL_PATH_WEY2WAT
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise RuntimeError(
            f"Model file '{path}' not found. Please run train_model.py first."
        )


def _resolve_station(name: str) -> str | None:
    return STATION_NAME_TO_CODE.get(name.strip().lower())


def _parse_time(value: str) -> datetime | None:
    text = value.strip().lower()
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if not match:
        return None
    hour   = int(match.group(1))
    minute = int(match.group(2) or 0)
    period = (match.group(3) or "").lower()
    if period == "pm" and hour != 12:
        hour += 12
    elif period == "am" and hour == 12:
        hour = 0
    now = datetime.now()
    try:
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    except ValueError:
        return None


def _parse_delay(value: str) -> int:
    if not value:
        return 0
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else 0


def _infer_direction(origin_code: str, dest_code: str) -> str:
    try:
        origin_idx = STATION_ORDER_WAT2WEY.index(origin_code)
    except ValueError:
        origin_idx = -1
    try:
        dest_idx = STATION_ORDER_WAT2WEY.index(dest_code)
    except ValueError:
        dest_idx = len(STATION_ORDER_WAT2WEY)
    return "WAT2WEY" if origin_idx <= dest_idx else "WEY2WAT"

def predict_delay_ml(slots: dict[str, Any]) -> dict[str, Any]:
    
    origin               = slots.get("origin", "Unknown")
    destination          = slots.get("destination", "Unknown")
    train_service        = slots.get("train_service", "South Western Railway")
    planned_arrival_str  = slots.get("planned_arrival_time", "")
    current_station      = slots.get("current_station", "Unknown")
    current_delay_str    = slots.get("current_delay", "0")

    origin_code  = _resolve_station(origin)          or origin.upper()[:3]
    dest_code    = _resolve_station(destination)      or destination.upper()[:3]
    current_code = _resolve_station(current_station)  or current_station.upper()[:3]

    direction = slots.get("direction", "").upper()
    if direction not in ("WAT2WEY", "WEY2WAT"):
        direction = _infer_direction(origin_code, dest_code)

    station_order = STATION_ORDER_WAT2WEY if direction == "WAT2WEY" else STATION_ORDER_WEY2WAT
    n_stations    = len(station_order)
    station_map   = {s: i for i, s in enumerate(station_order)}

    payload = _load_model(direction)
    model   = payload["model"]

    planned_arrival_dt = _parse_time(planned_arrival_str) if planned_arrival_str else datetime.now()
    if planned_arrival_dt is None:
        planned_arrival_dt = datetime.now()

    if direction == "WAT2WEY":
        origin_mins  = JOURNEY_TIMES_FROM_WAT.get(origin_code, 0)
        dest_mins    = JOURNEY_TIMES_FROM_WAT.get(dest_code, 200)
        journey_mins = max(dest_mins - origin_mins, 0)
    else:
        origin_mins  = 200 - JOURNEY_TIMES_FROM_WAT.get(origin_code, 200)
        dest_mins    = 200 - JOURNEY_TIMES_FROM_WAT.get(dest_code, 0)
        journey_mins = max(dest_mins - origin_mins, 0)

    real_departure_dt = planned_arrival_dt - timedelta(minutes=journey_mins)

    current_delay_mins = _parse_delay(str(current_delay_str))

    planned_dep_min    = real_departure_dt.hour * 60 + real_departure_dt.minute
    day_of_week        = real_departure_dt.weekday()
    month              = real_departure_dt.month
    is_weekend         = int(day_of_week >= 5)
    is_peak            = int(420 <= planned_dep_min <= 570 or 990 <= planned_dep_min <= 1140)
    hour               = real_departure_dt.hour

    station_num        = float(station_map.get(current_code, n_stations // 2))
    destination_num    = float(station_map.get(dest_code, n_stations - 1))
    journey_progress   = station_num / (n_stations - 1)
    stations_remaining = (n_stations - 1) - station_num
    stops_to_dest      = max(destination_num - station_num, 0)

    features = np.array([[
        planned_dep_min,
        day_of_week,
        month,
        is_weekend,
        is_peak,
        hour,
        station_num,
        journey_progress,
        stations_remaining,
        current_delay_mins,
        destination_num,
        stops_to_dest,
    ]])

    ml_additional_delay = float(model.predict(features)[0])

    """ ml_additional_delay = max(0.0, round(ml_additional_delay, 1))
    total_delay = round(current_delay_mins + ml_additional_delay, 1) """
    ml_additional_delay = max(0.0, round(ml_additional_delay))  # round to whole number
    total_delay = current_delay_mins + ml_additional_delay  # then calculate

    expected_arrival_dt = planned_arrival_dt + timedelta(minutes=total_delay)
    expected_arrival_str = expected_arrival_dt.strftime("%H:%M")

    if total_delay <= 5:
        risk_level = "low"
    elif total_delay <= 20:
        risk_level = "medium"
    else:
        risk_level = "high"

    reasons = _build_reasons(
        is_peak, is_weekend, day_of_week,
        current_delay_mins, ml_additional_delay,
        current_code, dest_code,
    )

    return {
        "origin":                  origin,
        "destination":             destination,
        "train_service":           train_service,
        "direction":               direction,
        "current_station":         current_station,
        "current_delay_minutes":   current_delay_mins,
        "ml_additional_delay":     ml_additional_delay,
        "ml_additional_delay_display": format_additional_delay(ml_additional_delay),
        "planned_arrival_time":    planned_arrival_str,
        "expected_arrival_time":   expected_arrival_str,
        "predicted_delay_minutes": total_delay,
        "risk_level":              risk_level,
        "confidence":              round(1.0 - (payload.get("mae", 6) / 30), 2),
        "operator":                "South Western Railway",
        "reasons":                 reasons,
        "recommendation":          _recommendation(risk_level),
        "model_used":              payload.get("model_name", "XGBoost ML Model"),
    }

def _build_reasons(
    is_peak, is_weekend, day_of_week,
    current_delay, ml_delay,
    current_code, dest_code,
) -> list[str]:
    total_del = current_delay + ml_delay
    reasons = []
    
    if total_del > 20:
        reasons.append(f"Your train is already {current_delay} minutes late and is expected to arrive around {round(total_del)} minutes behind schedule. Historical data shows significant further delay risk for this route")
    elif total_del > 5:
        reasons.append(f"Your train is {current_delay} minutes late but is not expected to delay much further. No significant delay patterns found for this route and time.")
    else:
        reasons.append(f"Minor delay only — train should arrive close to schedule. No significant delay patterns found for this route and time.")
    return reasons


def _recommendation(risk_level: str) -> str:
    if risk_level == "high":
        return (
            "Allow significant extra time and consider contacting your destination "
        )
    if risk_level == "medium":
        return (
            "Leave some buffer time and check the live service status again "
            "closer to arrival."
        )
    return (
        "The service looks likely to arrive close to schedule, "
        "but always check live updates before travelling."
    )


def format_additional_delay(ml_delay: float) -> str:
    if ml_delay < 1:
        return "Less than 1 minute"
    elif ml_delay < 60:
        return f"{round(ml_delay)} minutes"
    else:
        hours, mins = divmod(round(ml_delay), 60)
        return f"{hours}h {mins}min"