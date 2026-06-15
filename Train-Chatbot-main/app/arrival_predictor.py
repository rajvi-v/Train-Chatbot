from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "arrival_delay_model.joblib"
METRICS_PATH = ROOT / "models" / "arrival_delay_metrics.json"


DEFAULT_ALIASES = {
    "weymouth": "WEY",
    "waterloo": "WAT",
    "london": "WAT",
    "london waterloo": "WAT",
    "waterloo london": "WAT",
    "southampton": "SOU",
    "southampton central": "SOU",
    "southampton airport": "SOA",
    "southampton airport parkway": "SOA",
    "bournemouth": "BMH",
    "brockenhurst": "BCU",
    "poole": "POO",
    "hamworthy": "HAM",
    "wareham": "WRM",
    "dorchester": "DCH",
    "dorchester south": "DCH",
    "winchester": "WIN",
    "woking": "WOK",
    "clapham junction": "CLJ",
}


def predict_arrival(slots: dict[str, Any]) -> dict[str, Any]:
    artifact = load_artifact()
    metrics = load_metrics()
    if not artifact:
        return unavailable_prediction(slots, "Arrival model has not been trained yet.")

    current_station = normalise_station(slots.get("current_station"), artifact)
    destination = normalise_station(slots.get("destination"), artifact)
    direction = infer_direction(slots, current_station, destination)
    if not destination:
        destination = "WAT" if direction == "WEY2WAT" else "WEY"

    delay_minutes = parse_delay_minutes(slots.get("delay_minutes"))
    if delay_minutes is None:
        return unavailable_prediction(slots, "Delay minutes were not provided.")

    service_context = lookup_service_context(slots, artifact, current_station, destination)
    if not current_station:
        return unavailable_prediction(slots, "Current station was not recognised.")

    features = build_features(
        slots=slots,
        artifact=artifact,
        current_station=current_station,
        destination=destination,
        direction=direction,
        delay_minutes=delay_minutes,
        service_context=service_context,
    )

    model = artifact["model"]
    predicted_final_delay = float(model.predict(pd.DataFrame([features]))[0])
    predicted_final_delay = max(-30.0, min(180.0, predicted_final_delay))

    planned_arrival_min = int(features["planned_destination_arrival_min"])
    predicted_arrival_min = int(round(planned_arrival_min + predicted_final_delay))
    arrival_label = minutes_to_time(predicted_arrival_min)

    selected_model = str(artifact.get("model_name", "trained_model"))
    model_metrics = metrics.get("models", {}).get(selected_model, {})
    mae = model_metrics.get("mae_minutes")
    confidence_note = (
        f"2025 holdout MAE: about {mae:.1f} minutes."
        if isinstance(mae, (int, float))
        else "Model accuracy metrics are unavailable."
    )

    return {
        "provider": "South Western Railway operations model",
        "model": selected_model,
        "current_station": display_station(current_station),
        "current_station_code": current_station,
        "destination": display_station(destination),
        "destination_code": destination,
        "direction": direction,
        "delay_minutes_reported": int(round(delay_minutes)),
        "predicted_final_delay_minutes": int(round(predicted_final_delay)),
        "planned_destination_arrival": minutes_to_time(planned_arrival_min),
        "predicted_arrival_time": arrival_label,
        "confidence_note": confidence_note,
        "explanation": (
            "Prediction uses the current delay, current station, direction, scheduled remaining "
            "time, and time-of-service patterns from the SWR operational spreadsheets."
        ),
    }


def build_features(
    slots: dict[str, Any],
    artifact: dict[str, Any],
    current_station: str,
    destination: str,
    direction: str,
    delay_minutes: float,
    service_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if service_context:
        planned_event_min = int(service_context["planned_event_min"])
        destination_arrival_min = int(service_context["destination_planned_arrival_min"])
        remaining_sched_min = int(service_context["remaining_sched_min"])
        planned_hour = int(service_context["planned_hour"])
        day_of_week = int(service_context["day_of_week"])
        month = int(service_context["month"])
    else:
        remaining_sched_min = estimate_remaining_minutes(current_station, destination, direction)
        planned_hour = parse_hour(slots.get("travel_datetime")) or 12
        travel_date = parse_date(slots.get("date_of_service") or slots.get("travel_datetime"))
        day_of_week = travel_date.weekday()
        month = travel_date.month
        planned_event_min = planned_hour * 60
        destination_arrival_min = planned_event_min + remaining_sched_min

    return {
        "current_delay_min": delay_minutes,
        "current_station": current_station,
        "destination": destination,
        "direction": direction,
        "remaining_sched_min": remaining_sched_min,
        "planned_hour": planned_hour,
        "day_of_week": day_of_week,
        "month": month,
        "planned_destination_arrival_min": destination_arrival_min,
    }


def lookup_service_context(
    slots: dict[str, Any],
    artifact: dict[str, Any],
    current_station: str | None,
    destination: str | None,
) -> dict[str, Any] | None:
    if not current_station:
        return None

    service_id = str(slots.get("train_id") or slots.get("service_id") or "").strip()
    lookup = artifact.get("service_lookup", {})
    if service_id and service_id in lookup:
        service = lookup[service_id]
        if destination and service.get("destination") != destination:
            return None
        return service.get("stops", {}).get(current_station)
    return None


def infer_direction(slots: dict[str, Any], current_station: str | None, destination: str | None) -> str:
    text = " ".join(str(value) for value in slots.values()).casefold()
    if destination == "WEY" or "to weymouth" in text or "waterloo to weymouth" in text:
        return "WAT2WEY"
    return "WEY2WAT"


def normalise_station(value: Any, artifact: dict[str, Any] | None = None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if len(text) == 3 and text.isalpha():
        return text.upper()
    aliases = dict(DEFAULT_ALIASES)
    if artifact:
        aliases.update({str(k): str(v) for k, v in artifact.get("station_aliases", {}).items()})
    return aliases.get(text.casefold())


def parse_delay_minutes(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def parse_hour(value: Any) -> int | None:
    if not value:
        return None
    match = re.search(r"\b(\d{1,2})(?::\d{2})?\s*(am|pm)?\b", str(value), re.I)
    if not match:
        return None
    hour = int(match.group(1))
    period = (match.group(2) or "").lower()
    if period == "pm" and hour != 12:
        hour += 12
    elif period == "am" and hour == 12:
        hour = 0
    return hour if 0 <= hour <= 23 else None


def parse_date(value: Any) -> date:
    if not value:
        return date.today()
    text = str(value).strip().casefold()
    if "tomorrow" in text:
        return date.today() + timedelta(days=1)
    if "today" in text:
        return date.today()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return date.today()


def estimate_remaining_minutes(current_station: str, destination: str, direction: str) -> int:
    wey_to_wat = {
        "WEY": 165,
        "DCH": 155,
        "WRM": 140,
        "HAM": 132,
        "POO": 126,
        "BMH": 110,
        "BCU": 88,
        "SOU": 78,
        "SOA": 70,
        "WIN": 55,
        "BSK": 35,
        "WOK": 25,
        "CLJ": 8,
        "WAT": 0,
    }
    wat_to_wey = {
        "WAT": 165,
        "CLJ": 157,
        "WOK": 140,
        "BSK": 120,
        "WIN": 95,
        "SOA": 82,
        "SOU": 75,
        "BCU": 55,
        "BMH": 35,
        "POO": 28,
        "HAM": 22,
        "WRM": 15,
        "DCH": 8,
        "WEY": 0,
    }
    table = wat_to_wey if direction == "WAT2WEY" or destination == "WEY" else wey_to_wat
    return table.get(current_station, 75)


def minutes_to_time(value: int) -> str:
    value = value % 1440
    return f"{value // 60:02d}:{value % 60:02d}"


def display_station(code: str) -> str:
    names = {
        "WEY": "Weymouth",
        "WAT": "London Waterloo",
        "SOU": "Southampton Central",
        "SOA": "Southampton Airport Parkway",
        "BMH": "Bournemouth",
        "BCU": "Brockenhurst",
        "POO": "Poole",
        "HAM": "Hamworthy",
        "WRM": "Wareham",
        "DCH": "Dorchester South",
        "WIN": "Winchester",
        "WOK": "Woking",
        "CLJ": "Clapham Junction",
    }
    return names.get(code, code)


def unavailable_prediction(slots: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "provider": "South Western Railway operations model",
        "model": "unavailable",
        "current_station": str(slots.get("current_station", "Unknown")),
        "destination": str(slots.get("destination", "London Waterloo")),
        "predicted_arrival_time": "Unavailable",
        "predicted_final_delay_minutes": None,
        "confidence_note": reason,
        "explanation": "Please provide current station, delay minutes, and destination.",
    }


@lru_cache(maxsize=1)
def load_artifact() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
