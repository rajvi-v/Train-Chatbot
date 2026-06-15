from __future__ import annotations

import re
from typing import Any

from app.models import Intent


STATION_PATTERN = re.compile(
    r"\bfrom\s+([a-zA-Z ]+?)\s+to\s+([a-zA-Z ]+?)(?:\s+(?:currently|now|on|at|before|after|by|be|is|will|delayed|late|tomorrow|today|next|return|single|$)|[,.!?]|$)",
    re.I,
)
TIME_PATTERN = re.compile(r"\b(?:at|before|after)?\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", re.I)
DATE_PATTERN = re.compile(
    r"\b(today|tomorrow|\d{1,2}(?:st|nd|rd|th)?\s+[a-zA-Z]+|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b",
    re.I,
)
OPERATOR_PATTERN = re.compile(r"\b(?:with|on|operator)\s+([a-zA-Z][a-zA-Z &-]+?)(?:\s+(?:from|to|at|on|tomorrow|today)|[,.!?]|$)", re.I)
SERVICE_PATTERN = re.compile(r"\b(?:service|train)\s+(?:id|number|no\.?)?\s*([A-Z]?\d{2,5}[A-Z]?)\b", re.I)
RID_PATTERN = re.compile(r"\b(?:rid|train|service)\s+(?:id|number|no\.?)?\s*(\d{10,18})\b", re.I)

CURRENT_STATION_PATTERN = re.compile(
    r"\b(?:currently at|now at|at the moment at|stopped at)\s+([a-zA-Z ]+?)(?:\s+(?:station|at|on|be|is|will|delayed|late|today|tomorrow)|[,.!?]|$)",
    re.I,
)
DELAY_MINUTES_PATTERN = re.compile(r"\b(?:delayed|delay|late)\s+(?:by|for)?\s*(\d{1,3})\s*(?:minutes?|mins?|m)?\b", re.I)
DESTINATION_PATTERN = re.compile(
    r"\b(?:destination|going to|travelling to|traveling to)\s+([a-zA-Z ]+?)(?:\s+(?:station|train|with|delayed|late|from|at|today|tomorrow)|[,.!?]|$)",
    re.I,
)


def detect_intent(message: str) -> Intent:
    text = message.lower()
    if any(
        word in text
        for word in (
            "contingency",
            "disruption plan",
            "station disruption",
            "alternative transport",
            "blocked",
            "blockage",
            "station staff",
            "signaller",
            "signallers",
            "alternative passenger",
            "passenger alternative",
            "passenger alternatives",
            "alternative route",
            "alternative routes",
            "service alteration",
            "replacement bus",
            "emergency",
            "incident",
            "evacuation",
            "crowd control",
            "taxi",
            "ticket acceptance",
            "csl2",
            "control",
            "customer",
            "what should staff",
            "what should station staff",
        )
    ):
        return "contingency_advice"
    if (
        any(word in text for word in ("arrival", "arrive", "reach", "destination", "waterloo", "weymouth", "swr"))
        and any(word in text for word in ("delay", "delayed", "late"))
    ):
        return "arrival_time_prediction"
    if any(word in text for word in ("delay", "delayed", "late", "disruption", "cancelled", "status", "on time", "arrive", "arrival")):
        return "delay_prediction"
    if any(word in text for word in ("ticket", "fare", "cheapest", "book", "journey")):
        return "ticket_search"
    return "unknown"


def extract_entities(message: str, intent: Intent) -> dict[str, Any]:
    text = message.strip()
    entities: dict[str, Any] = {}

    route = STATION_PATTERN.search(text)
    if route:
        entities["origin"] = clean_value(route.group(1))
        entities["destination"] = clean_value(route.group(2))

    date = DATE_PATTERN.search(text)
    time = TIME_PATTERN.search(text)
    if intent in {"ticket_search", "delay_prediction"}:
        parts = []
        if date:
            parts.append(date.group(1))
        if time:
            parts.append(time.group(1))
        if parts:
            datetime_slot = "travel_datetime" if intent == "delay_prediction" else "outbound_datetime"
            entities[datetime_slot] = " ".join(parts)

    if intent == "ticket_search":
        if any(word in text.lower() for word in ("return", "coming back", "round trip")):
            entities["return_preference"] = "return"
        elif any(word in text.lower() for word in ("single", "one way", "one-way")):
            entities["return_preference"] = "single"

    if intent == "delay_prediction":
        operator = OPERATOR_PATTERN.search(text)
        service = SERVICE_PATTERN.search(text)
        current_station = CURRENT_STATION_PATTERN.search(text)
        if operator:
            entities["operator"] = clean_value(operator.group(1))
        if service:
            entities["service_id"] = service.group(1).upper()
        if current_station:
            entities["current_station"] = clean_value(current_station.group(1))

    if intent == "arrival_time_prediction":
        rid = RID_PATTERN.search(text)
        service = SERVICE_PATTERN.search(text)
        current_station = CURRENT_STATION_PATTERN.search(text)
        delay_minutes = DELAY_MINUTES_PATTERN.search(text)
        destination = DESTINATION_PATTERN.search(text)
        if rid:
            entities["train_id"] = rid.group(1)
        elif service:
            entities["train_id"] = service.group(1).upper()
        if current_station:
            entities["current_station"] = clean_arrival_value(current_station.group(1))
        if delay_minutes:
            entities["delay_minutes"] = int(delay_minutes.group(1))
        if destination:
            entities["destination"] = clean_arrival_value(destination.group(1))
        elif route:
            entities["destination"] = clean_arrival_value(route.group(2))
        elif "waterloo to weymouth" in text.lower() or "to weymouth" in text.lower():
            entities["destination"] = "Weymouth"
        elif "waterloo" in text.lower() or "london" in text.lower():
            entities["destination"] = "London Waterloo"
        elif "weymouth" in text.lower():
            entities["destination"] = "Weymouth"
        if "waterloo to weymouth" in text.lower() or "from waterloo" in text.lower():
            entities["direction"] = "WAT2WEY"
        elif "weymouth to" in text.lower() or "to waterloo" in text.lower() or "london waterloo" in text.lower():
            entities["direction"] = "WEY2WAT"

    if intent == "contingency_advice":
        entities["query"] = text
        lower_text = text.lower()
        if any(
            phrase in lower_text
            for phrase in (
                "emergency",
                "incident",
                "evacuation",
                "crowd control",
                "replacement bus",
                "taxi",
                "ticket acceptance",
                "csl2",
            )
        ):
            entities["event_type"] = "general_contingency"
        if "line" in lower_text or "blocked" in lower_text or "blockage" in lower_text:
            entities["event_type"] = "line_blockage"
        if "station disruption" in lower_text or "disruption at" in lower_text or "at " in lower_text and "station" in lower_text:
            entities.setdefault("event_type", "station_disruption")
        route = re.search(r"\bbetween\s+([A-Za-z ]+?)\s+and\s+([A-Za-z ]+?)(?:[,.?]|$)", text, re.I)
        if route:
            entities["first_station"] = clean_value(route.group(1))
            entities["last_station"] = clean_value(route.group(2))
            entities["event_type"] = "line_blockage"
        route = route or re.search(
            r"\b(?:for|on|at)\s+([A-Za-z ]+?)\s+(?:-|to)\s+([A-Za-z ]+?)(?:\s+disruption|[,.?]|$)",
            text,
            re.I,
        )
        if route:
            entities["first_station"] = clean_value(route.group(1))
            entities["last_station"] = clean_value(route.group(2))
            entities["event_type"] = "line_blockage"
        station = re.search(r"\b(?:at|for|during disruption at)\s+([A-Za-z ]+?)(?:\s+station|[,.?]|$)", text, re.I)
        if station and "first_station" not in entities:
            station_value = clean_station_value(station.group(1))
            if station_value.casefold() not in {"station staff", "staff", "passenger", "customers", "control"}:
                entities["station"] = station_value
                if "blocked" not in lower_text and "blockage" not in lower_text:
                    entities["event_type"] = "station_disruption"
        elif ("alternative" in lower_text or "disruption" in lower_text) and " at " in lower_text:
            station_after_at = re.search(r"\bat\s+([A-Za-z ]+?)(?:[,.?]|$)", text, re.I)
            if station_after_at:
                entities["station"] = clean_station_value(station_after_at.group(1))
                entities.setdefault("event_type", "station_disruption")
        if "both line" in text.lower() or "both lines" in text.lower():
            entities["blockage_type"] = "both lines blocked"
            entities["severity"] = "full"
        elif "one line" in text.lower() or "single line" in text.lower():
            entities["blockage_type"] = "one line blocked"
            entities["severity"] = "partial"
        elif "partial" in lower_text:
            entities["blockage_type"] = "one line blocked"
            entities["severity"] = "partial"
        elif "full" in lower_text:
            entities["blockage_type"] = "both lines blocked"
            entities["severity"] = "full"
        if "signaller" in text.lower():
            entities["role"] = "signaller"
            entities["requested_detail"] = "signaller_info"
        elif "control" in lower_text or "controller" in lower_text:
            entities["role"] = "service_controller"
            entities["requested_detail"] = "service_alteration"
        elif "station staff" in text.lower() or "staff" in text.lower():
            entities["role"] = "station_staff"
            entities["requested_detail"] = "station_staff_info"
        elif "passenger" in text.lower() or "customer" in lower_text:
            entities["role"] = "passenger"
            entities["requested_detail"] = "passenger_info"
        if "alternative" in lower_text or "route" in lower_text or "bus" in lower_text or "taxi" in lower_text:
            entities["requested_detail"] = "alternative_passenger_journey"
        elif "service alteration" in lower_text or "divert" in lower_text or "cancel" in lower_text:
            entities["requested_detail"] = "service_alteration"
        elif "ticket acceptance" in lower_text or "csl2" in lower_text or "crowd control" in lower_text:
            entities.setdefault("requested_detail", "station_staff_info")
        elif re.search(r"\b(all|everything)\b", lower_text):
            entities["requested_detail"] = "all"
        time_match = re.search(r"\b(?:at|from|since)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", text, re.I)
        if time_match:
            entities["incident_time"] = time_match.group(1)

    return entities


def clean_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" ,.!?").title()


def clean_station_value(value: str) -> str:
    text = clean_value(value)
    text = re.sub(r"^Disruption\s+At\s+", "", text, flags=re.I).strip()
    return text


def clean_arrival_value(value: str) -> str:
    text = clean_value(value)
    text = re.sub(r"\b(?:And|Train|Train Is|Service|Service Is)$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_time_text(value: str) -> str | None:
    match = TIME_PATTERN.search(value)
    if not match:
        return None
    return next(group for group in match.groups() if group)


def has_time_expression(value: str) -> bool:
    return extract_time_text(value) is not None


def has_date_expression(value: str) -> bool:
    return DATE_PATTERN.search(value) is not None
