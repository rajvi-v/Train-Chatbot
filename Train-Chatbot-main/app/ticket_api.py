from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import (
    TICKET_API_BASE_URL,
    TICKET_API_PASSWORD,
    TICKET_API_USERNAME,
    TICKET_API_WSDL_URL,
)

print("Loaded WSDL URL:", TICKET_API_WSDL_URL)

RAILCARD_TYPES = {
    "16-25 Railcard": "YNG",
    "Senior Railcard": "SRN",
    "Disabled Persons Railcard": "DIS",
    "Family & Friends Railcard": "FAM",
    "Two Together Railcard": "TGT",
    "Network Railcard": "NWR",
    "Veterans Railcard": "VET",
}

STATION_CRS_CODES = {
    "norwich": "NRW",
    "london": "LST",
    "london liverpool street": "LST",
    "liverpool street": "LST",
    "london waterloo": "WAT",
    "waterloo": "WAT",
    "waterloo london": "WAT",
    "cambridge": "CBG",
    "manchester": "MAN",
    "birmingham": "BHM",
    "oxford": "OXF",
}

REALTIME_ENQUIRY = "STANDARD"
FARE_CLASS = "STANDARD"
DEFAULT_SERVICE_ADDRESS = "https://ojp.nationalrail.co.uk/webservices"
SERVICE_BINDING = "{http://ojp.nationalrail.co.uk}jpservicesBinding"
NATIONAL_RAIL_DEVELOPER_LINK = "https://www.nationalrail.co.uk/developers/online-journey-planner-data-feeds/"


NON_PASSENGER_STATION_TERMS = (
    " sidings",
    " siding",
    " depot",
    " yard",
    " junction",
    " bogie store",
)

_FALLBACK_STATION_CODES = {
    "norwich": "NRW",
    "london": "LST",
    "cambridge": "CBG",
    "manchester": "MAN",
    "birmingham": "BHM",
}


def is_public_station_name(name: str, crs: str) -> bool:
    if crs.startswith("X"):
        return False
    text = f" {name.casefold()} "
    return not any(term in text for term in NON_PASSENGER_STATION_TERMS)


def load_station_codes() -> dict[str, str]:
    csv_path = Path(__file__).parent / "StationNameAndCode.csv"
    if not csv_path.exists():
        csv_path = Path(
            "C:/Users/user/Desktop/Train-Chatbot-main/Train-Chatbot-main/app/StationNameAndCode.csv"
        )

    if csv_path.exists():
        codes: dict[str, str] = {}
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_name = row["NAME"].strip()
                name = raw_name.casefold()
                crs = row["CRS"].strip().upper()
                if name and crs and is_public_station_name(raw_name, crs):
                    codes[name] = crs
        print(f"Loaded {len(codes)} public stations from CSV.")
        return codes

    print("StationNameAndCode.csv not found — using fallback station list.")
    return dict(_FALLBACK_STATION_CODES)


STATION_CRS_CODES = load_station_codes()

REALTIME_ENQUIRY = "STANDARD"
FARE_CLASS = "STANDARD"
DEFAULT_SERVICE_ADDRESS = "https://ojp.nationalrail.co.uk/webservices"
SERVICE_BINDING = "{http://ojp.nationalrail.co.uk}jpservicesBinding"

def create_ticket_client() -> Any:
    if not TICKET_API_WSDL_URL:
        raise TicketApiConfigError("TICKET_API_WSDL_URL is not configured.")

    from requests import Session
    from requests.auth import HTTPBasicAuth
    from zeep import Client, Settings
    from zeep.transports import Transport

    session = Session()
    if TICKET_API_USERNAME and TICKET_API_PASSWORD:
        session.auth = HTTPBasicAuth(TICKET_API_USERNAME, TICKET_API_PASSWORD)

    return Client(
        wsdl=TICKET_API_WSDL_URL,
        transport=Transport(session=session),
        settings=Settings(strict=False),
    )


def create_ticket_service(client: Any) -> Any:
    address = TICKET_API_BASE_URL or DEFAULT_SERVICE_ADDRESS
    return client.create_service(SERVICE_BINDING, address)

def inspect_ticket_api() -> str:
    client = create_ticket_client()
    lines = ["Ticket API WSDL loaded successfully.", ""]

    for service in client.wsdl.services.values():
        lines.append(f"Service: {service.name}")
        for port in service.ports.values():
            lines.append(f"  Port: {port.name}")
            binding = port.binding
            for operation_name, operation in binding._operations.items():
                lines.append(f"    Operation: {operation_name}")
                lines.append(f"      Input: {operation.input.signature()}")
                lines.append(f"      Output: {operation.output.signature()}")
        lines.append("")

    return "\n".join(lines).rstrip()

def search_tickets_from_api(slots: dict[str, Any]) -> dict[str, Any] | None:
    request = build_realtime_journey_plan_request(slots)

    if isinstance(request, dict) and request.get("error"):
        return request

    client = create_ticket_client()
    service = create_ticket_service(client)
    response = service.RealtimeJourneyPlan(**request)
    return normalise_ticket_response(response, slots)


def build_realtime_journey_plan_request(slots: dict[str, Any]) -> dict[str, Any]:
    origin_code = station_to_crs(slots.get("origin"))
    destination_code = station_to_crs(slots.get("destination"))
    outbound_time = parse_travel_datetime(str(slots.get("outbound_datetime", "")))
    return_time = (
        parse_travel_datetime(str(slots.get("return_datetime", "")))
        if slots.get("return_preference") == "return"
        else None
    )

    if not origin_code or not destination_code:
        print("Invalid station:", slots.get("origin"), slots.get("destination"))
        return {"error": "invalid_station"}

    if outbound_time is None:
        return {"error": "invalid_time"}

    railcard_slot = slots.get("railcard")
    railcard_codes = (
        [RAILCARD_TYPES[railcard_slot]]
        if railcard_slot and railcard_slot != "none" and railcard_slot in RAILCARD_TYPES
        else []
    )

    return {
        "origin": {"stationCRS": origin_code},
        "destination": {"stationCRS": destination_code},
        "realtimeEnquiry": REALTIME_ENQUIRY,
        "outwardTime": {"departBy": outbound_time},
        "inwardTime": {"departBy": return_time} if return_time else None,
        "directTrains": False,
        "fareRequestDetails": {
            "passengers": {"adult": 1, "child": 0},
            "fareClass": FARE_CLASS,
            "railcard": (
                [RAILCARD_TYPES.get(slots.get("railcard"))]
                if slots.get("railcard") and slots.get("railcard") != "none"
                else []
            ),
        },
        "reducedTransferTime": False,
        "onlySearchForSleeper": False,
        "overtakenTrains": False,
        "increasedInterchange": 0,
        "operator": None,
        "rawRealtimeInfo": False,
        "planAnytimeToday": False,
    }

def station_to_crs(value: Any) -> str | None:
    if not value:
        return None

    text = str(value).strip().casefold()
    text = text.replace(" station", "").strip()

    # Direct 3-letter CRS code passed in
    if len(text) == 3 and text.isalpha():
        return text.upper()

    # Exact match in loaded codes
    if text in STATION_CRS_CODES:
        return STATION_CRS_CODES[text]

    # Fuzzy match (cutoff 0.8 to avoid false positives)
    from difflib import get_close_matches
    close = get_close_matches(text, STATION_CRS_CODES.keys(), n=1, cutoff=0.8)
    if close:
        return STATION_CRS_CODES[close[0]]

    return None

@lru_cache(maxsize=1)
def station_crs_lookup() -> dict[str, str]:
    lookup = dict(STATION_CRS_CODES)
    station_file = Path("StationNameAndCode.csv")
    if not station_file.exists():
        return lookup

    with station_file.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("NAME", "")).strip()
            code = str(row.get("CRS", "")).strip().upper()
            if not name or not code:
                continue
            lookup.setdefault(name.casefold(), code)

    lookup.update(STATION_CRS_CODES)
    return lookup

def parse_travel_datetime(value: str, now: datetime | None = None) -> datetime | None:
    now = now or datetime.now()
    text = value.strip().lower()
    if not text:
        return None

    date_part = now.date()
    if "tomorrow" in text:
        date_part = (now + timedelta(days=1)).date()
    elif "today" in text:
        date_part = now.date()
    else:
        parsed_date = parse_explicit_date(text, now)
        if parsed_date is not None:
            date_part = parsed_date

    time_part = parse_time(text)
    if time_part is None:
        return None

    return datetime.combine(date_part, time_part)


def parse_explicit_date(text: str, now: datetime) -> Any:
    for pattern, fmt in (
        (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", "%d/%m/%Y"),
        (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2}\b", "%d/%m/%y"),
    ):
        match = re.search(pattern, text)
        if match:
            raw = match.group(0).replace("-", "/")
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                return None

    month_match = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|"
        r"aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b",
        text,
    )
    if not month_match:
        return None

    month_lookup = {
        "jan": 1, "january": 1, "feb": 2, "february": 2,
        "mar": 3, "march": 3, "apr": 4, "april": 4,
        "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    day = int(month_match.group(1))
    month = month_lookup[month_match.group(2)]
    try:
        candidate = datetime(now.year, month, day).date()
    except ValueError:
        return None
    if candidate < now.date():
        candidate = datetime(now.year + 1, month, day).date()
    return candidate


def parse_time(text: str) -> Any:
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    period = match.group(3)

    if period == "pm" and hour != 12:
        hour += 12
    elif period == "am" and hour == 12:
        hour = 0

    if hour > 23 or minute > 59:
        return None

    return datetime.min.replace(hour=hour, minute=minute).time()

def normalise_ticket_response(response: Any, slots: dict[str, Any]) -> dict[str, Any] | None:
    if response is None:
        return None

    data = _object_to_dict(response)
    print(json.dumps(data, indent=2, default=str))

    journeys = _as_list(data.get("outwardJourney"))
    journey = journeys[0] if journeys else {}
    fares = _as_list(journey.get("fare")) if isinstance(journey, dict) else []
    fare = cheapest_fare(fares)
    first_leg = first_journey_leg(journey)
    print(list(first_leg.keys()))

    departure_time = nested_get(first_leg, "timetable", "scheduled", "departure")
    arrival_time = nested_get(first_leg, "timetable", "scheduled", "arrival")
    departure_platform = nested_get(first_leg, "originPlatform") or "Unavailable"
    arrival_platform = nested_get(first_leg, "destinationPlatform") or "Unavailable"

    result: dict[str, Any] = {
        "provider": str(nested_get(first_leg, "operator", "name") or "National Rail OJP"),
        "origin": slots.get("origin", "Unknown"),
        "destination": slots.get("destination", "Unknown"),
        "departure_time": format_time(departure_time),
        "arrival_time": format_time(arrival_time),
        "departure_platform": (
            departure_platform
            if departure_platform != "Unavailable"
            else "Platform will be announced closer to departure"
        ),
        "arrival_platform": (
            arrival_platform
            if arrival_platform != "Unavailable"
            else "Platform will be announced closer to arrival"
        ),
        "duration": compute_duration(departure_time, arrival_time),
        "outbound_datetime": slots.get("outbound_datetime"),
        "price": format_price(fare),
        "ticket_type": (
            str(fare.get("description") or fare.get("typeCode") or "Ticket")
            if fare else "Ticket"
        ),
        "booking_url": str(TICKET_API_WSDL_URL or ""),
    }

    # Return journey 
    if slots.get("return_preference") == "return":
        return_journeys = _as_list(data.get("returnJourney") or data.get("inwardJourney"))
        return_journey = return_journeys[0] if return_journeys else {}
        return_leg = first_journey_leg(return_journey)

        ret_fares = _as_list(return_journey.get("fare")) if isinstance(return_journey, dict) else []
        ret_fare = cheapest_fare(ret_fares)

        ret_dep = nested_get(return_leg, "timetable", "scheduled", "departure")
        ret_arr = nested_get(return_leg, "timetable", "scheduled", "arrival")
        ret_dep_platform = nested_get(return_leg, "originPlatform") or "Unavailable"
        ret_arr_platform = nested_get(return_leg, "destinationPlatform") or "Unavailable"

        out_price_pence = _price_pence(fare)
        ret_price_pence = _price_pence(ret_fare)

        result.update({
            
            "outward": {
                "origin": result["origin"],
                "destination": result["destination"],
                "departure_time": result["departure_time"],
                "arrival_time": result["arrival_time"],
                "departure_platform": result["departure_platform"],
                "arrival_platform": result["arrival_platform"],
                "duration": result["duration"],
                "price": result["price"],
                "booking_url": result["booking_url"],
            },
            # Return leg
            "return": {
                "origin": slots.get("destination", "Unknown"),
                "destination": slots.get("origin", "Unknown"),
                "departure_time": format_time(ret_dep),
                "arrival_time": format_time(ret_arr),
                "departure_platform": (
                    ret_dep_platform
                    if ret_dep_platform != "Unavailable"
                    else "Platform will be announced closer to departure"
                ),
                "arrival_platform": (
                    ret_arr_platform
                    if ret_arr_platform != "Unavailable"
                    else "Platform will be announced closer to arrival"
                ),
                "duration": compute_duration(ret_dep, ret_arr),
                "price": format_price(ret_fare),
                "booking_url": str(TICKET_API_WSDL_URL or ""),
            },
            "total_price": format_price_pence(out_price_pence + ret_price_pence),
        })

    return result

def cheapest_fare(fares: list[Any]) -> dict[str, Any] | None:
    fare_dicts = [f for f in fares if isinstance(f, dict)]
    if not fare_dicts:
        return None
    return min(fare_dicts, key=lambda f: int(f.get("totalPrice") or 999_999_999))


def _price_pence(fare: dict[str, Any] | None) -> int:
    if not fare:
        return 0
    try:
        return int(fare.get("totalPrice") or 0)
    except (TypeError, ValueError):
        return 0


def format_price(fare: dict[str, Any] | None) -> str:
    if not fare:
        return "Price unavailable"
    total_price = fare.get("totalPrice")
    if total_price is None:
        return "Price unavailable"
    try:
        return f"GBP {int(total_price) / 100:.2f}"
    except (TypeError, ValueError):
        return str(total_price)


def format_price_pence(pence: int) -> str:
    if pence == 0:
        return "Price unavailable"
    return f"GBP {pence / 100:.2f}"

def first_journey_leg(journey: Any) -> dict[str, Any]:
    if not isinstance(journey, dict):
        return {}
    legs = _as_list(journey.get("leg"))
    return legs[0] if legs and isinstance(legs[0], dict) else {}


def nested_get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list):
            try:
                current = current[int(key)]
            except (IndexError, ValueError, TypeError):
                return None
        else:
            return None
        if current is None:
            return None
    return current


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def format_time(value: Any) -> str:
    if not value:
        return "Unavailable"
    try:
        if isinstance(value, datetime):
            return value.strftime("%H:%M")
        cleaned = str(value).replace(" ", "T").replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned).strftime("%H:%M")
    except Exception:
        return str(value)


def compute_duration(dep: Any, arr: Any) -> str:
    try:
        if not isinstance(dep, datetime):
            dep = datetime.fromisoformat(str(dep).replace(" ", "T"))
        if not isinstance(arr, datetime):
            arr = datetime.fromisoformat(str(arr).replace(" ", "T"))
        diff = arr - dep
        total_minutes = int(diff.total_seconds() // 60)
        return f"{total_minutes // 60}h {total_minutes % 60}m"
    except Exception:
        return "Duration unavailable"


def format_duration(value: str) -> str:

    if not value:
        return "Duration unavailable"
    if not isinstance(value, str) or not value.startswith("PT"):
        return str(value)
    hours = re.search(r"(\d+)H", value)
    minutes = re.search(r"(\d+)M", value)
    h = hours.group(1) if hours else "0"
    m = minutes.group(1) if minutes else "0"
    return f"{h}h {m}m"


def _object_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        from zeep.helpers import serialize_object
        serialised = serialize_object(value)
        if isinstance(serialised, dict):
            return serialised
    except Exception:
        pass
    return {}

def format_journey_time(departure: Any, arrival: Any, slots: dict[str, Any]) -> str:
    if departure and arrival:
        return f"{departure} to {arrival}"
    if departure:
        return str(departure)
    return str(slots.get("outbound_datetime", "requested time"))

class TicketApiConfigError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or test the configured ticket SOAP API."
    )
    parser.add_argument(
        "command", choices=["inspect", "journey"], help="Ticket API command to run"
    )
    parser.add_argument("--origin", default="Norwich", help="Origin station name or CRS code")
    parser.add_argument(
        "--destination", default="London", help="Destination station name or CRS code"
    )
    parser.add_argument("--time", default="tomorrow 8am", help="Outbound date/time")
    parser.add_argument(
        "--return-time", default=None, help="Return date/time (omit for single journey)"
    )
    args = parser.parse_args()

    if args.command == "inspect":
        print(inspect_ticket_api())
    elif args.command == "journey":
        slots: dict[str, Any] = {
            "origin": args.origin,
            "destination": args.destination,
            "outbound_datetime": args.time,
            "return_preference": "return" if args.return_time else "single",
        }
        if args.return_time:
            slots["return_datetime"] = args.return_time
        result = search_tickets_from_api(slots)
        print(result or "No live journey result returned.")


if __name__ == "__main__":
    main()