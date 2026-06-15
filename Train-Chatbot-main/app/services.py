from __future__ import annotations

import re
from datetime import datetime, timedelta, time
from typing import Any

from app.ticket_api import search_tickets_from_api

from app.delay_predictor import predict_delay_ml
from app.arrival_predictor import predict_arrival
from app.expert_system import get_contingency_advice
from app.ticket_api import search_tickets_from_api

NATIONAL_RAIL_DEVELOPER_LINK = "https://www.nationalrail.co.uk/developers/online-journey-planner-data-feeds/"

def find_cheapest_ticket(slots: dict[str, Any]) -> dict[str, Any]:
   
    # Trying to get real journey data from API
    try:
        api_ticket = search_tickets_from_api(slots) or {}
    except Exception as e:
        print("Ticket API Error:", e)
        api_ticket = {}

    # Handling API errors
    if isinstance(api_ticket, dict) and api_ticket.get("error") in ("invalid_station", "invalid_time"):
        return api_ticket
    
    # Generating a fallback fare. If the API returns a real price, it replaces this.
    mock_ticket = mock_cheapest_ticket(slots)
    
    # Merging API data into mock ticket (single journey)
    if slots.get("return_preference") != "return":
        merge_available_ticket_fields(
            mock_ticket,
            api_ticket,
            {
                "departure_time": "departure_time",
                "arrival_time": "arrival_time",
                "departure_platform": "departure_platform",
                "arrival_platform": "arrival_platform",
                "duration": "duration",
                "operator": "operator",
            },
        )
        merge_api_price(mock_ticket, api_ticket)
        return mock_ticket

    # Merging API data into mock ticket (return journey)
    outward = mock_ticket["outward"]
    return_leg = mock_ticket["return"]

    merge_available_ticket_fields(
        outward,
        api_ticket,
        {
            "departure_time": "out_departure_time",
            "arrival_time": "out_arrival_time",
            "departure_platform": "out_departure_platform",
            "arrival_platform": "out_arrival_platform",
            "duration": "out_duration",
        },
    )
    merge_available_ticket_fields(
        return_leg,
        api_ticket,
        {
            "departure_time": "ret_departure_time",
            "arrival_time": "ret_arrival_time",
            "departure_platform": "ret_departure_platform",
            "arrival_platform": "ret_arrival_platform",
            "duration": "ret_duration",
        },
    )
    api_price_used = merge_api_price(outward, api_ticket)
    if api_price_used:
        return_leg["price"] = "Included in return total"
        mock_ticket["total_price"] = outward["price"]

    return mock_ticket



def live_ticket_unavailable(slots: dict[str, Any]) -> dict[str, Any]:
    origin = slots.get("origin", "Unknown")
    destination = slots.get("destination", "Unknown")
    outbound = slots.get("outbound_datetime", "requested time")
    return {
        "provider": "National Rail OJP",
        "origin": origin,
        "destination": destination,
        "outbound_datetime": outbound,
        "price": "Live price unavailable",
        "ticket_type": "No live fare returned",
        "booking_url": NATIONAL_RAIL_DEVELOPER_LINK,
    }

def ticket_api_error(slots: dict[str, Any], error: Exception) -> dict[str, Any]:
    origin = slots.get("origin", "Unknown")
    destination = slots.get("destination", "Unknown")
    outbound = slots.get("outbound_datetime", "requested time")
    return {
        "provider": "Ticket API error",
        "origin": origin,
        "destination": destination,
        "outbound_datetime": outbound,
        "price": "Live price unavailable",
        "ticket_type": f"Live API failed: {type(error).__name__}",
        "booking_url": NATIONAL_RAIL_DEVELOPER_LINK,
    }



def predict_arrival_time(slots: dict[str, Any]) -> dict[str, Any]:
    return predict_arrival(slots)


def advise_contingency(slots: dict[str, Any]) -> dict[str, Any]:
    return get_contingency_advice(slots, str(slots.get("query", "")))

def get_route_status(storage: Any, origin: str, destination: str) -> list[dict[str, Any]]:
    return storage.get_route_status(origin, destination)


def rule_points(rules: dict[str, dict[str, Any]], code: str, default: int) -> int:
    rule = rules.get(code)
    return int(rule["points"]) if rule else default


def rule_description(rules: dict[str, dict[str, Any]], code: str, default: str) -> str:
    rule = rules.get(code)
    return str(rule["description"]) if rule else default



def merge_available_ticket_fields(
    target: dict[str, Any],
    source: dict[str, Any],
    field_map: dict[str, str],
) -> None:
    for target_key, source_key in field_map.items():
        value = source.get(source_key)
        if is_available_ticket_value(value):
            target[target_key] = value



def is_available_ticket_value(value: Any) -> bool:
    if value is None:
        return False

    text = str(value).strip()
    if not text:
        return False

    lowered = text.lower()
    unavailable_phrases = (
        "unavailable",
        "duration unavailable",
        "platform will be announced",
        "will be announced closer",
    )
    return not any(phrase in lowered for phrase in unavailable_phrases)



def merge_api_price(target: dict[str, Any], source: dict[str, Any]) -> bool:
    used_api_price = False

    price = source.get("price")
    if is_real_price_value(price):
        target["price"] = price
        used_api_price = True

    ticket_type = source.get("ticket_type")
    if is_available_ticket_value(ticket_type):
        target["ticket_type"] = ticket_type

    return used_api_price



def is_real_price_value(value: Any) -> bool:
    if value is None:
        return False

    text = str(value).strip()
    if not text:
        return False

    lowered = text.lower()
    return "unavailable" not in lowered and lowered != "price unavailable"




def mock_cheapest_ticket(slots: dict[str, Any]) -> dict[str, Any]:

    origin = slots.get("origin", "Norwich")
    destination = slots.get("destination", "London")
    outbound = slots.get("outbound_datetime", "requested time")
    return_pref = slots.get("return_preference", "").lower()
    railcard = slots.get("railcard", "").lower()
    duration_minutes, is_estimated_duration = estimate_journey_minutes(origin, destination)
    departure_time, arrival_time, duration_text = mock_journey_times(
        outbound,
        duration_minutes,
        is_estimated_duration,
    )
    return_departure_time, return_arrival_time, return_duration_text = mock_journey_times(
        slots.get("return_datetime", "17:45"),
        duration_minutes,
        is_estimated_duration,
    )

    # Railcard discount factors
    railcard_discounts = {
        "16-25 railcard": 0.66,
        "senior railcard": 0.66,
        "disabled persons railcard": 0.66,
        "family & friends railcard": 0.66,
        "two together railcard": 0.66,
        "network railcard": 0.66,
        "veterans railcard": 0.66,
    }

    # Base realistic price for Norwich → London
    base_price = estimate_base_price(origin, destination, duration_minutes)

    # Generate fare options (SINGLE)
    single_fares = [
        {"ticket_type": "Advance Single", "price": base_price * 0.50},
        {"ticket_type": "Super Off-Peak Single", "price": base_price * 0.60},
        {"ticket_type": "Off-Peak Single", "price": base_price * 0.80},
        {"ticket_type": "Anytime Single", "price": base_price * 1.00},
    ]

    # Generate fare options (RETURN)
    return_fares = [
        {"ticket_type": "Super Off-Peak Return", "price": base_price * 1.20},
        {"ticket_type": "Off-Peak Return", "price": base_price * 1.40},
        {"ticket_type": "Anytime Return", "price": base_price * 1.80},
    ]

    # Apply railcard discount
    def apply_railcard_discount(price: float, railcard: str | None) -> float:
        if not railcard:
            return round(price, 2)

        railcard_lower = railcard.lower().strip()

        for rc, factor in railcard_discounts.items():
            if rc in railcard_lower:
                return round(price * factor, 2)

        return round(price, 2)

    def format_currency(price: float) -> str:
        return f"GBP {price:.2f}"


    # SINGLE JOURNEY LOGIC
    if return_pref != "return":
        cheapest = min(single_fares, key=lambda x: x["price"]).copy()
        cheapest["price"] = apply_railcard_discount(cheapest["price"],railcard)

        return {
            "origin": origin,
            "destination": destination,
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "departure_platform": "To be announced",
            "arrival_platform": "To be announced",
            "duration": duration_text,
            "ticket_type": cheapest["ticket_type"],
            "price": format_currency(cheapest["price"]),
            "booking_url": "https://www.nationalrail.co.uk/journey-planner/",
        }

    # RETURN JOURNEY LOGIC
    cheapest_return = min(return_fares, key=lambda x: x["price"]).copy()
    cheapest_return["price"] = apply_railcard_discount(cheapest_return["price"],railcard)

    outward = {
        "origin": origin,
        "destination": destination,
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "departure_platform": "To be announced",
        "arrival_platform": "To be announced",
        "duration": duration_text,
        "ticket_type": cheapest_return["ticket_type"] + " (Outward)",
        "price": format_currency(cheapest_return["price"] / 2),
        "booking_url": "https://www.nationalrail.co.uk/journey-planner/",
    }

    return_leg = {
        "origin": destination,
        "destination": origin,
        "departure_time": return_departure_time,
        "arrival_time": return_arrival_time,
        "departure_platform": "To be announced",
        "arrival_platform": "To be announced",
        "duration": return_duration_text,
        "ticket_type": cheapest_return["ticket_type"] + " (Return)",
        "price": format_currency(cheapest_return["price"] / 2),
        "booking_url": "https://www.nationalrail.co.uk/journey-planner/",
    }

    total_price = cheapest_return["price"]

    return {
        "outward": outward,
        "return": return_leg,
        "total_price": format_currency(total_price),
    }



def estimate_journey_minutes(origin: str, destination: str) -> tuple[int, bool]:
    route = {origin.casefold(), destination.casefold()}
    known_routes = {
        frozenset(("norwich", "london")): 105,
        frozenset(("norwich", "oxford")): 190,
        frozenset(("oxford", "cambridge")): 150,
        frozenset(("cambridge", "london")): 65,
        frozenset(("manchester", "london")): 130,
        frozenset(("birmingham", "london")): 85,
    }
    matched_route = frozenset(route)
    if matched_route in known_routes:
        return known_routes[matched_route], False
    return 105, True



def estimate_base_price(origin: str, destination: str, duration_minutes: int) -> float:
    route = {origin.casefold(), destination.casefold()}
    known_prices = {
        frozenset(("norwich", "london")): 35.00,
        frozenset(("norwich", "oxford")): 58.00,
        frozenset(("oxford", "cambridge")): 42.00,
        frozenset(("cambridge", "london")): 24.00,
        frozenset(("manchester", "london")): 72.00,
        frozenset(("birmingham", "london")): 48.00,
    }

    matched_route = frozenset(route)
    if matched_route in known_prices:
        return known_prices[matched_route]

    route_key = f"{origin.casefold()}|{destination.casefold()}"
    route_variation = sum(ord(char) for char in route_key) % 18
    estimated_price = 12.00 + (duration_minutes * 0.22) + route_variation
    return round(max(8.00, min(estimated_price, 120.00)), 2)



def mock_journey_times(
    outbound_datetime: Any,
    duration_minutes: int,
    is_estimated_duration: bool,
) -> tuple[str, str, str]:
    departure = parse_time(str(outbound_datetime))
    if departure is None:
        departure = time(8, 30)

    departure_dt = datetime.combine(datetime.today(), departure)
    arrival_dt = departure_dt + timedelta(minutes=duration_minutes)

    return (
        departure_dt.strftime("%H:%M"),
        arrival_dt.strftime("%H:%M"),
        format_duration_minutes(duration_minutes, is_estimated_duration),
    )

def format_arrival_time(departure_time: str, duration_minutes: int) -> str:
    departure = parse_time(departure_time)
    if departure is None:
        return "19:30"

    departure_dt = datetime.combine(datetime.today(), departure)
    arrival_dt = departure_dt + timedelta(minutes=duration_minutes)
    return arrival_dt.strftime("%H:%M")

def format_duration_minutes(duration_minutes: int, is_estimated: bool = False) -> str:
    hours = duration_minutes // 60
    minutes = duration_minutes % 60
    if hours and minutes:
        duration = f"{hours}h {minutes}m"
    elif hours:
        duration = f"{hours}h"
    else:
        duration = f"{minutes}m"

    if is_estimated:
        return f"Estimated {duration}"
    return duration



def predict_delay(slots: dict[str, Any], storage: Any) -> dict[str, Any]:
    return predict_delay_ml(slots)

def get_route_status(storage: Any, origin: str, destination: str) -> list[dict[str, Any]]:
    return storage.get_route_status(origin, destination)


def rule_points(rules: dict[str, dict[str, Any]], code: str, default: int) -> int:
    rule = rules.get(code)
    return int(rule["points"]) if rule else default


def rule_description(rules: dict[str, dict[str, Any]], code: str, default: str) -> str:
    rule = rules.get(code)
    return str(rule["description"]) if rule else default


def is_peak_time(value: str) -> bool:
    parsed = parse_time(value)
    if parsed is None:
        return False
    return time(7, 0) <= parsed <= time(9, 30) or time(16, 30) <= parsed <= time(19, 0)


def parse_time(value: str) -> time | None:
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", value, re.I)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    period = (match.group(3) or "").lower()
    if period == "pm" and hour != 12:
        hour += 12
    elif period == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return time(hour, minute)


def score_to_prediction(score: int) -> tuple[str, int]:
    if score <= 2:
        return "low", min(5, max(0, score * 2))
    if score <= 5:
        return "medium", 5 + ((score - 3) * 5)
    return "high", min(45, 15 + ((score - 6) * 5))


def recommendation_for(risk_level: str) -> str:
    if risk_level == "high":
        return "Allow extra time, check service updates before leaving, and consider an earlier train."
    if risk_level == "medium":
        return "Leave some buffer time and check the service status again closer to departure."
    return "The service looks reasonably safe, but check live updates before travelling." 

