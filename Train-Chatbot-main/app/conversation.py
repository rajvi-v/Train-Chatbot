from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from app.models import ChatResponse, Intent
from app.nlu import detect_intent, extract_entities, has_date_expression, has_time_expression
from app.rag import answer_from_knowledge_base, format_kb_context, search_knowledge_base
from app.services import advise_contingency, find_cheapest_ticket, predict_arrival_time, predict_delay
from app.storage import ChatStorage
from app.ticket_api import STATION_CRS_CODES

# Supported railcards
RAILCARD_TYPES = {
    "16-25 Railcard": "YNG",
    "Senior Railcard": "SRN",
    "Disabled Persons Railcard": "DIS",
    "Family & Friends Railcard": "FAM",
    "Two Together Railcard": "TGT",
    "Network Railcard": "NWR",
    "Veterans Railcard": "VET",
}

REQUIRED_SLOTS: dict[Intent, list[str]] = {
    "ticket_search": [
        "origin", "destination", "outbound_datetime", "return_preference", "railcard",
    ],
    "delay_prediction": [
        "direction",
        "origin", "destination",
        "planned_arrival_time", "current_station", "current_delay",
    ],
    "arrival_time_prediction": ["current_station", "delay_minutes", "destination"],
    "contingency_advice": ["event_type", "requested_detail"],
    "unknown": [],
}

MAIN_MENU = (
    "Hello! How can I help you today? I can help you with:\n"
    "1. Find the cheapest train ticket\n"
    "2. Predict a delayed train's arrival time\n"
    "3. Get contingency plan advice (staff use)\n\n"
    "Type 1, 2, or 3 — or just describe what you need."
)

SLOT_PROMPTS = {
    # Task 1 — ticket search
    "origin": "What station are you travelling from?",
    "destination": "What is your destination station?",
    "outbound_datetime": "What date and time would you like to travel?",
    "return_datetime": "What date and time would you like to return?",
    "travel_datetime": "What date and time is the train due to travel?",
    "return_preference": "Is this a single journey or do you need a return ticket?",
    "railcard": "Do you have a railcard you'd like to use?",
    # Task 2 — delay prediction
    "direction": (
        "Which direction are you travelling? "
        "Reply 'WAT2WEY' for London Waterloo -> Weymouth, "
        "or 'WEY2WAT' for Weymouth -> London Waterloo."
    ),
    "planned_arrival_time": (
        "What is your planned arrival time at your destination? "
        "Please use HH:MM format (e.g. 10:30)."
    ),
    "current_station": "Which station is the train currently at? (e.g. Bournemouth, Wareham)",
    "current_delay": (
        "How many minutes has the train been delayed so far? "
        "Please enter a number (e.g. 10)."
    ),
    "event_type": (
        "This is the Staff Contingency Plan Assistant.\n\n"
        "What type of issue are you dealing with?\n"
        "1. Line blockage\n"
        "2. Station disruption"
    ),
    "first_station": "Which is the first station or location in the affected line section?",
    "last_station": "Which is the last station or location in the affected line section?",
    "blockage_type": "Is the blockage partial (one line blocked) or full (both lines blocked)?",
    "station": "Which station is affected?",
    "requested_detail": (
        "Do you need station staff advice , passenger advice or signallar advice ?"
    ),
}

SKIP_EXTRACTION_STATES = {
    # Task 1
    "collecting_railcard_type", "collecting_return_datetime",
    "collecting_return_preference",
    # Task 2
    "collecting_direction", "collecting_current_delay",
    "collecting_current_station", "collecting_planned_arrival_time",
    # Task 3
    "collecting_event_type", "collecting_blockage_type",
    "collecting_requested_detail", "collecting_confirm_event_type",
    "collecting_confirm_location", "collecting_confirm_blockage_type",
}

class ConversationManager:
    def __init__(self, storage: ChatStorage) -> None:
        self.storage = storage

    def handle(self, session_id: str | None, message: str) -> ChatResponse:
        session_id = session_id.strip() if session_id else str(uuid4())
        user_message = message.strip()
        self.storage.log_message(session_id, "user", user_message)

        kb_matches = search_knowledge_base(user_message)
        existing = self.storage.load_session(session_id)

        if is_start_over(user_message):
            response = ChatResponse(
                session_id=session_id,
                reply=MAIN_MENU,
                intent="unknown",
                state="awaiting_intent",
                slots={},
                complete=False,
            )
            self._persist(response)
            return response

        if is_greeting(user_message) and not existing:
            response = ChatResponse(
                session_id=session_id,
                reply=MAIN_MENU,
                intent="unknown",
                state="awaiting_intent",
                slots={},
                complete=False,
            )
            self._persist(response)
            return response
        
        menu_response = self._handle_menu_choice(session_id, user_message, existing)
        if menu_response:
            return menu_response

        detected_intent: Intent = detect_intent(user_message)
        if (
            existing
            and existing["intent"] != "unknown"
            and not should_switch_intent(existing, detected_intent, user_message)
        ):
            intent: Intent = existing["intent"]
            slots: dict[str, Any] = existing["slots"]
        else:
            intent = detected_intent
            if existing and existing["intent"] == detected_intent:
                slots = existing["slots"]
            else:
                slots = {}
        current_state: str = existing["state"] if existing else ""

        if intent == "unknown":
            reply = answer_from_knowledge_base(kb_matches) if kb_matches else (
                "Sorry, I can help with cheapest UK train tickets, delay prediction, "
                "arrival-time prediction, and railway disruption plan advice.\n\n"
                "For example:\n"
                "- 'Find a cheap ticket from Norwich to London tomorrow at 9am'\n"
                "- 'I want to predict how late my train is going to be?'\n"
                "- 'Passenger advice for both lines blocked between Havant and Portcreek Junction'\n\n"
                "Or type 1, 2, or 3 to choose from the menu."
            )
            response = ChatResponse(
                session_id=session_id,
                reply=reply,
                intent="unknown",
                state="awaiting_intent",
                slots=slots,
                complete=False,
            )
            self._persist(response)
            return response

        if current_state not in SKIP_EXTRACTION_STATES:
            extracted = extract_entities(user_message, intent)
            # Task 3: merge datetime answers intelligently
            if (
                intent == "ticket_search"
                and "outbound_datetime" in extracted
                and "outbound_datetime" in slots
            ):
                extracted["outbound_datetime"] = merge_datetime_answer(
                    str(slots.get("outbound_datetime", "")),
                    str(extracted["outbound_datetime"]),
                )
            slots.update(extracted)

        try:
            fill_slot_from_direct_answer(intent, slots, user_message, current_state)
        except MissingTimeError as e:
            datetime_slot = str(e) if str(e) else "outbound_datetime"
            time_reply = (
                "Please specify the time you'd like to return."
                if datetime_slot == "return_datetime"
                else "Please specify the time you'd like to travel."
            )
            response = ChatResponse(
                session_id=session_id,
                reply=time_reply,
                intent=intent,
                state=f"collecting_{datetime_slot}",
                slots=slots,
                complete=False,
            )
            self._persist(response)
            return response
        except ValueError as e:
            if str(e) == "invalid_station":
                response = ChatResponse(
                    session_id=session_id,
                    reply=(
                        "Sorry, I can only help with UK train stations. "
                        "Please enter a valid UK station like Norwich, London, "
                        "Cambridge, or Birmingham."
                    ),
                    intent=intent,
                    state="awaiting_intent",
                    slots={},
                    complete=False,
                )
                self._persist(response)
                return response

        if "_validation_error" in slots:
            error_msg = slots.pop("_validation_error")
            response = ChatResponse(
                session_id=session_id,
                reply=error_msg,
                intent=intent,
                state=current_state,
                slots=slots,
                complete=False,
            )
            self._persist(response)
            return response

        if intent == "ticket_search" and slots.get("railcard") == "unspecified":
            response = ChatResponse(
                session_id=session_id,
                reply=(
                    "Which railcard would you like to use? For example, enter 16-25, Senior, "
                    "Disabled Persons, Family & Friends, Two Together, Network, or Veterans."
                    if current_state == "collecting_railcard_type"
                    else
                    "Which railcard would you like to use? "
                    "You can choose from: 16-25, Senior, Disabled Persons, "
                    "Family & Friends, Two Together, Network, or Veterans."
                ),
                intent=intent,
                state="collecting_railcard_type",
                slots=slots,
                complete=False,
            )
            self._persist(response)
            return response

        missing = first_missing_slot(intent, slots)
        if missing:
            response = ChatResponse(
                session_id=session_id,
                reply=slot_prompt(intent, missing, slots),
                intent=intent,
                state=f"collecting_{missing}",
                slots=slots,
                complete=False,
            )
            self._persist(response)
            return response
        
        reply, prediction = self._complete_flow(session_id, intent, slots)
        reply = with_knowledge_base_note(reply, kb_matches)
        response = ChatResponse(
            session_id=session_id,
            reply=reply,
            intent=intent,
            state="completed",
            slots=slots,
            complete=True,
            prediction=prediction,
        )
        self._persist(response)
        return response

    def _persist(self, response: ChatResponse) -> None:
        self.storage.save_session(
            response.session_id, response.intent, response.state, response.slots
        )
        self.storage.log_message(response.session_id, "assistant", response.reply)

    def _complete_flow(
        self,
        session_id: str,
        intent: Intent,
        slots: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        if intent == "delay_prediction":
            prediction = predict_delay(slots, self.storage)
            self.storage.log_prediction(session_id, slots, prediction)
            return format_delay_reply(prediction), prediction
        if intent == "arrival_time_prediction":
            prediction = predict_arrival_time(slots)
            self.storage.log_prediction(session_id, slots, prediction)
            return format_arrival_reply(prediction), prediction
        if intent == "contingency_advice":
            advice = advise_contingency(slots)
            return format_contingency_reply(advice), advice
        return complete_flow(intent, slots), None

    def _handle_menu_choice(
        self,
        session_id: str,
        message: str,
        existing: dict[str, Any] | None,
    ) -> ChatResponse | None:
        choice = normalised_menu_choice(message)
        if not choice:
            return None
        is_menu_state = existing is None or (
            existing.get("intent") == "unknown"
            and str(existing.get("state", "")).startswith("awaiting")
        )
        if not is_menu_state:
            return None
        if choice == "1":
            response = ChatResponse(
                session_id=session_id,
                reply=SLOT_PROMPTS["origin"],
                intent="ticket_search",
                state="collecting_origin",
                slots={},
                complete=False,
            )
        elif choice == "2":
            response = ChatResponse(
            session_id=session_id,
            reply=SLOT_PROMPTS["direction"],
            intent="delay_prediction",
            state="collecting_direction",
            slots={},
            complete=False,
        )

        else:
            response = ChatResponse(
                session_id=session_id,
                reply=SLOT_PROMPTS["event_type"],
                intent="contingency_advice",
                state="collecting_event_type",
                slots={},
                complete=False,
            )
        self._persist(response)
        return response


def is_greeting(message: str) -> bool:
    return message.lower().strip() in {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    }


def is_start_over(message: str) -> bool:
    return message.casefold().strip() in {
        "start over", "restart", "reset", "menu", "main menu",
    }


def normalised_menu_choice(message: str) -> str:
    text = message.casefold().strip()
    if text in {"1", "one", "cheapest ticket", "ticket", "tickets"}:
        return "1"
    if text in {
        "2", "two", "delay", "arrival", "delay prediction",
        "arrival prediction", "delay/arrival prediction",
    }:
        return "2"
    if text in {
        "3", "three", "contingency", "contingency plan",
        "staff contingency", "disruption plan",
    }:
        return "3"
    return ""


def should_switch_intent(
    existing: dict[str, Any] | None,
    detected_intent: Intent,
    message: str,
) -> bool:
    if not existing or detected_intent == "unknown":
        return False
    current_intent = existing["intent"]
    if detected_intent == current_intent:
        return False
    text = message.casefold()
    state = str(existing.get("state", ""))
    if state == "completed":
        return True
    if detected_intent == "ticket_search" and any(
        w in text for w in ("book", "ticket", "fare", "cheapest", "journey")
    ):
        return True
    if detected_intent == "arrival_time_prediction" and any(
        w in text for w in ("arrival", "arrive", "destination", "waterloo", "weymouth", "swr")
    ) and any(w in text for w in ("delay", "delayed", "late")):
        return True
    if detected_intent == "delay_prediction" and any(
        w in text for w in ("delay", "delayed", "late", "disruption", "cancelled", "status")
    ):
        return True
    if detected_intent == "contingency_advice" and any(
        w in text for w in (
            "contingency", "disruption plan", "blocked", "blockage",
            "station staff", "signaller",
        )
    ):
        return True
    return False

def first_missing_slot(intent: Intent, slots: dict[str, Any]) -> str | None:
    if intent == "contingency_advice":
        return first_missing_contingency_slot(slots)
    for slot in REQUIRED_SLOTS[intent]:
        if (
            intent == "ticket_search"
            and slot == "railcard"
            and slots.get("return_preference") == "return"
            and slots.get("return_datetime") in (None, "")
        ):
            return "return_datetime"
        if (
            intent == "ticket_search"
            and slot == "outbound_datetime"
            and slots.get(slot)
            and not has_complete_ticket_datetime(str(slots[slot]))
        ):
            return slot
        if slots.get(slot) in (None, ""):
            return slot
    return None


def first_missing_contingency_slot(slots: dict[str, Any]) -> str | None:
    event_type = slots.get("event_type")
    if not slots.get("confirmed_event_type"):
        return "event_type"
    if event_type == "line_blockage":
        if not slots.get("first_station") or not slots.get("last_station"):
            return "line_location"
        if not slots.get("confirmed_location"):
            return "confirm_location"
        if not slots.get("blockage_type"):
            return "blockage_type"
        if not slots.get("confirmed_blockage_type"):
            return "confirm_blockage_type"
        if slots.get("requested_detail") in (None, ""):
            return "requested_detail"
    elif event_type == "station_disruption":
        if not slots.get("station"):
            return "station"
        if not slots.get("confirmed_location"):
            return "confirm_location"
        if slots.get("requested_detail") in (None, ""):
            return "requested_detail"
    elif slots.get("requested_detail") in (None, ""):
        return "requested_detail"
    return None


def slot_prompt(intent: Intent, slot: str, slots: dict[str, Any]) -> str:
    if intent == "ticket_search" and slot == "outbound_datetime":
        outbound = str(slots.get("outbound_datetime", "")).strip()
        if outbound and has_date_expression(outbound) and not has_time_expression(outbound):
            return f"What time would you like to travel on {outbound}?"
        if outbound and has_time_expression(outbound) and not has_date_expression(outbound):
            return f"What date would you like to travel at {outbound}?"
    if intent == "contingency_advice":
        if slot == "line_location":
            return "Which two stations or locations are affected?"
        if slot == "confirm_location":
            return f"I understand the affected location is {human_location(slots)}. Is that correct?"
        if slot == "confirm_blockage_type":
            return f"I understand the blockage is {slots.get('blockage_type')}. Is that correct?"
    return SLOT_PROMPTS.get(slot, f"Please provide your {slot}.")


def with_knowledge_base_note(reply: str, kb_matches: list[dict[str, Any]]) -> str:
    note = format_kb_context(kb_matches)
    return f"{reply}\n\n{note}" if note else reply


class MissingTimeError(Exception):
    pass

def is_valid_station(name: str) -> bool:
    return name.casefold() in STATION_CRS_CODES


def find_station_matches(text: str) -> list[str]:
    exact, startswith, word = [], [], []
    for name in STATION_CRS_CODES.keys():
        station_lower = name.lower()
        display = display_station_name(name)
        if station_lower == text:
            exact.append(display)
        elif station_lower.startswith(text):
            startswith.append(display)
        elif all(w in station_lower for w in text.split()):
            word.append(display)
    return (
        sorted(exact, key=len)
        + sorted(startswith, key=len)
        + sorted(word, key=len)
    )


def display_station_name(name: str) -> str:
    if name.casefold() == "london br":
        return "London"
    return name.title()


def platform_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"unavailable", "none"}:
        return "Platform: To be announced"
    lowered = text.lower()
    if "to be announced" in lowered or "will be announced" in lowered:
        return "Platform: To be announced"
    if lowered.startswith("platform"):
        return text
    return f"Platform {text}"

def normalise_railcard(message: str) -> str | None:
    text = message.strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", text)
    aliases = {
        "16-25 Railcard": ("16-25", "1625", "young", "yng"),
        "Senior Railcard": ("senior", "srn"),
        "Disabled Persons Railcard": ("disabled", "disabled persons", "dis"),
        "Family & Friends Railcard": (
            "family", "friends", "family and friends", "family & friends", "fam",
        ),
        "Two Together Railcard": ("two together", "2 together", "twotogether", "tgt"),
        "Network Railcard": ("network", "nwr"),
        "Veterans Railcard": ("veteran", "veterans", "vet"),
    }
    for railcard, keywords in aliases.items():
        for keyword in keywords:
            keyword_compact = re.sub(r"[^a-z0-9]", "", keyword)
            if keyword in text or keyword_compact == compact:
                return railcard
    for railcard in RAILCARD_TYPES:
        railcard_text = railcard.lower()
        if railcard_text in text or railcard_text.replace(" railcard", "") in text:
            return railcard
    return None

def _is_valid_time(text: str) -> bool:
    stripped = text.strip().lower()
    if not re.match(r"^\d{1,2}:\d{2}\s*(am|pm)?$", stripped):
        return False
    match = re.match(r"^(\d{1,2}):(\d{2})\s*(am|pm)?$", stripped)
    if not match:
        return False
    hour, minute = int(match.group(1)), int(match.group(2))
    period = match.group(3) or ""
    if period in ("am", "pm"):
        if not (1 <= hour <= 12):
            return False
    else:
        if not (0 <= hour <= 23):
            return False
    return 0 <= minute <= 59


def merge_datetime_answer(existing_value: str, answer: str) -> str:
    existing_value = existing_value.strip()
    answer = answer.strip()
    if not existing_value:
        return answer
    if has_date_expression(existing_value) and not has_time_expression(existing_value):
        if has_time_expression(answer):
            return f"{existing_value} {answer}"
        return existing_value
    if has_time_expression(existing_value) and not has_date_expression(existing_value):
        if has_date_expression(answer):
            return f"{answer} {existing_value}"
        return existing_value
    if existing_value and not has_time_expression(existing_value) and has_time_expression(answer):
        return f"{existing_value} {answer}"
    return answer


def has_complete_ticket_datetime(value: str) -> bool:
    return has_date_expression(value) and has_time_expression(value)

def requested_detail_from_text(text: str) -> str:
    if "signaller" in text:
        return "signaller_info"
    if "station staff" in text or "staff" in text:
        return "station_staff_info"
    if "passenger" in text:
        if "alternative" in text or "route" in text or "journey" in text:
            return "alternative_passenger_journey"
        return "passenger_info"
    if "alternative" in text or "route" in text or "bus" in text:
        return "alternative_passenger_journey"
    if "service alteration" in text or "divert" in text or "cancel" in text:
        return "service_alteration"
    return "all"


def confirmation_from_text(text: str) -> bool | None:
    yes_words = {"yes", "y", "correct", "right", "true", "that is correct", "that's correct"}
    no_words = {"no", "n", "incorrect", "wrong", "not correct", "that's wrong", "that is wrong"}
    cleaned = re.sub(r"[^a-z' ]+", " ", text.casefold())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned in yes_words:
        return True
    if cleaned in no_words:
        return False
    if cleaned.startswith(("yes ", "correct ", "right ")):
        return True
    if cleaned.startswith(("no ", "incorrect ", "wrong ")):
        return False
    return None


def human_location(slots: dict[str, Any]) -> str:
    if slots.get("event_type") == "line_blockage":
        first = str(slots.get("first_station") or "the first location").strip()
        last = str(slots.get("last_station") or "the last location").strip()
        return f"{first} to {last}"
    return str(slots.get("station") or "the affected station").strip()


def clear_event_type(slots: dict[str, Any]) -> None:
    for key in (
        "event_type", "confirmed_event_type", "confirmed_location",
        "confirmed_blockage_type", "blockage_type", "severity",
    ):
        slots.pop(key, None)
    clear_location(slots)


def clear_location(slots: dict[str, Any]) -> None:
    for key in ("first_station", "last_station", "station", "confirmed_location"):
        slots.pop(key, None)


def parse_line_location(value: str) -> tuple[str, str]:
    patterns = (
        r"\bbetween\s+([A-Za-z ]+?)\s+and\s+([A-Za-z ]+?)(?:[,.?]|$)",
        r"\bfrom\s+([A-Za-z ]+?)\s+to\s+([A-Za-z ]+?)(?:[,.?]|$)",
        r"^\s*([A-Za-z ]+?)\s+(?:-|to|and)\s+([A-Za-z ]+?)\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, value, re.I)
        if match:
            return clean_slot_value(match.group(1)), clean_slot_value(match.group(2))
    return "", ""


def clean_slot_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" ,.!?").title()

def fill_slot_from_direct_answer(
    intent: Intent,
    slots: dict[str, Any],
    message: str,
    state: str = "",
) -> None:
    from app.delay_predictor import _resolve_station

    text = message.strip()
    lower = text.lower()
    collecting = (
        state.removeprefix("collecting_")
        if state.startswith("collecting_")
        else ""
    )

    if collecting and text:

        if collecting == "railcard_type":
            if lower in ("no", "nope", "nah", "none") or "no railcard" in lower:
                slots["railcard"] = "none"
            else:
                railcard = normalise_railcard(message)
                slots["railcard"] = railcard or "unspecified"
            return

        if collecting == "return_datetime":
            time_pattern = r"\b(?:\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm))\b"
            date_pattern = (
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
                r"\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|"
                r"\btomorrow\b|\btoday\b"
            )
            if re.search(date_pattern, lower) and not re.search(time_pattern, lower):
                slots["return_datetime"] = None
                raise MissingTimeError("return_datetime")
            if re.search(time_pattern, lower):
                slots["return_datetime"] = message
            return

        if collecting == "outbound_datetime":
            time_pattern = r"\b(?:\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm))\b"
            date_pattern = (
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
                r"\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|"
                r"\btomorrow\b|\btoday\b"
            )
            current_value = str(slots.get("outbound_datetime", "")).strip()
            if current_value and has_time_expression(current_value):
                return
            merged = merge_datetime_answer(current_value, text)
            if re.search(date_pattern, merged.lower()) and not re.search(time_pattern, merged.lower()):
                slots["outbound_datetime"] = merged
                raise MissingTimeError("outbound_datetime")
            slots["outbound_datetime"] = merged
            return

        if collecting == "return_preference":
            if "return" in lower or "round trip" in lower:
                slots["return_preference"] = "return"
            elif "single" in lower or "one way" in lower or "one-way" in lower:
                slots["return_preference"] = "single"
            else:
                slots["_validation_error"] = (
                    f"Sorry, I didn't understand '{text}'. "
                    "Please reply 'single' for a one-way journey or 'return' for a return ticket."
                )
            return

        if collecting == "direction":
            if lower in ("wat2wey", "waterloo to weymouth", "london to weymouth", "towards weymouth"):
                slots["direction"] = "WAT2WEY"
            elif lower in ("wey2wat", "weymouth to waterloo", "weymouth to london", "towards waterloo", "towards london"):
                slots["direction"] = "WEY2WAT"
            else:
                slots["_validation_error"] = (
                    f"Sorry, I didn't recognise '{text}' as a valid direction. "
                    "Please reply 'WAT2WEY' for London Waterloo -> Weymouth, "
                    "or 'WEY2WAT' for Weymouth -> London Waterloo."
                )
            return

        if collecting == "origin" and intent == "delay_prediction":
            if len(text.split()) <= 5 and _resolve_station(text):
                slots["origin"] = text.title()
            else:
                slots["_validation_error"] = (
                    f"Sorry, I don't recognise '{text}' as a station on this route. "
                    "Please enter a station name such as London Waterloo, Bournemouth, or Weymouth."
                )
            return

        if collecting == "destination" and intent == "delay_prediction":
            if len(text.split()) <= 5 and _resolve_station(text):
                if text.title() == slots.get("origin"):
                    slots["_validation_error"] = (
                        "Your destination can't be the same as your origin. "
                        "Please enter a different destination station."
                    )
                else:
                    slots["destination"] = text.title()
            else:
                slots["_validation_error"] = (
                    f"Sorry, I don't recognise '{text}' as a station on this route. "
                    "Please enter a station name such as London Waterloo, Bournemouth, or Weymouth."
                )
            return

        if collecting == "current_station" and intent == "delay_prediction":
            if len(text.split()) <= 5 and _resolve_station(text):
                slots["current_station"] = text.title()
            else:
                slots["_validation_error"] = (
                    f"Sorry, I don't recognise '{text}' as a station on this route. "
                    "Please enter a station name such as Bournemouth, Wareham, or Southampton."
                )
            return

        if collecting == "planned_arrival_time":
            if _is_valid_time(text):
                slots["planned_arrival_time"] = text
            else:
                slots["_validation_error"] = (
                    f"Sorry, '{text}' doesn't look like a valid time. "
                    "Please use HH:MM format, for example 10:30 or 08:15."
                )
            return

        if collecting == "current_delay":
            if re.match(r"^\d+$", text.strip()):
                delay = int(text.strip())
                if 0 <= delay <= 180:
                    slots["current_delay"] = text
                else:
                    slots["_validation_error"] = (
                        f"Sorry, '{delay}' doesn't seem like a realistic delay. "
                        "Please enter a delay in minutes between 0 and 180."
                    )
            else:
                slots["_validation_error"] = (
                    f"Sorry, '{text}' doesn't look like a valid number. "
                    "Please enter the delay in minutes as a whole number, for example 10."
                )
            return

        if collecting == "delay_minutes":
            match = re.search(r"\d+", text)
            if match:
                slots["delay_minutes"] = int(match.group(0))
            return

        if collecting == "train_id":
            slots["train_id"] = text.upper()
            return

        if collecting == "event_type":
            if lower.strip() in {"1", "one"} or "line" in lower or "block" in lower:
                slots["event_type"] = "line_blockage"
                slots["confirmed_event_type"] = True
            elif lower.strip() in {"2", "two"} or "station" in lower or "platform" in lower:
                slots["event_type"] = "station_disruption"
                slots["confirmed_event_type"] = True
            return

        if collecting == "confirm_event_type":
            confirmation = confirmation_from_text(lower)
            if confirmation is True:
                slots["confirmed_event_type"] = True
            elif confirmation is False:
                clear_event_type(slots)
            elif "line" in lower or "block" in lower:
                slots["event_type"] = "line_blockage"
                slots["confirmed_event_type"] = True
            elif "station" in lower or "platform" in lower:
                slots["event_type"] = "station_disruption"
                slots["confirmed_event_type"] = True
            return

        if collecting == "confirm_location":
            confirmation = confirmation_from_text(lower)
            if confirmation is True:
                slots["confirmed_location"] = True
            elif confirmation is False:
                clear_location(slots)
            elif slots.get("event_type") == "line_blockage":
                first, last = parse_line_location(text)
                if first and last:
                    slots["first_station"] = first
                    slots["last_station"] = last
                    slots["confirmed_location"] = True
            elif text:
                slots["station"] = clean_slot_value(text)
                slots["confirmed_location"] = True
            return

        if collecting == "confirm_blockage_type":
            confirmation = confirmation_from_text(lower)
            if confirmation is True:
                slots["confirmed_blockage_type"] = True
            elif confirmation is False:
                slots.pop("blockage_type", None)
                slots.pop("severity", None)
                slots.pop("confirmed_blockage_type", None)
            elif "both" in lower or "full" in lower:
                slots["blockage_type"] = "both lines blocked"
                slots["severity"] = "full"
                slots["confirmed_blockage_type"] = True
            elif "one" in lower or "single" in lower or "partial" in lower:
                slots["blockage_type"] = "one line blocked"
                slots["severity"] = "partial"
                slots["confirmed_blockage_type"] = True
            return

        if collecting == "line_location":
            first, last = parse_line_location(text)
            if first and last:
                slots["first_station"] = first
                slots["last_station"] = last
                slots["confirmed_location"] = True
            return

        if collecting in {"first_station", "last_station"}:
            slots[collecting] = text.title()
            return

        if collecting == "station":
            slots["station"] = clean_slot_value(text)
            slots["confirmed_location"] = True
            return

        if collecting == "blockage_type":
            if "both" in lower or "full" in lower:
                slots["blockage_type"] = "both lines blocked"
                slots["severity"] = "full"
                slots["confirmed_blockage_type"] = True
            elif "one" in lower or "single" in lower or "partial" in lower:
                slots["blockage_type"] = "one line blocked"
                slots["severity"] = "partial"
                slots["confirmed_blockage_type"] = True
            else:
                slots["blockage_type"] = text
                slots["confirmed_blockage_type"] = True
            return

        if collecting == "requested_detail":
            slots["requested_detail"] = requested_detail_from_text(lower)
            return

        if collecting in {"origin", "destination", "current_station"} and len(text.split()) <= 5:
            slots[collecting] = text.title()
            return

    if intent == "delay_prediction" and "direction" not in slots:
        if any(k in lower for k in ("wat2wey", "waterloo to weymouth", "london to weymouth")):
            slots["direction"] = "WAT2WEY"
        elif any(k in lower for k in ("wey2wat", "weymouth to waterloo", "weymouth to london")):
            slots["direction"] = "WEY2WAT"

    if intent == "ticket_search":
        date_pattern = (
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
            r"\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|"
            r"\btomorrow\b|\btoday\b"
        )
        time_pattern = r"\b(?:\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm))\b"
        if re.search(date_pattern, lower) and not re.search(time_pattern, lower):
            slot_name = "return_datetime" if collecting == "return_datetime" else "outbound_datetime"
            slots[slot_name] = None
            raise MissingTimeError(slot_name)

    if intent == "ticket_search" and "return_preference" not in slots:
        if "return" in lower or "round trip" in lower:
            slots["return_preference"] = "return"
        elif "single" in lower or "one way" in lower or "one-way" in lower:
            slots["return_preference"] = "single"

    if intent == "ticket_search":
        if slots.get("railcard") in (None, "", "unspecified"):
            if lower in ("no", "nope", "nah", "none") or "no railcard" in lower:
                slots["railcard"] = "none"
            elif lower in ("yes", "yeah", "yep", "sure"):
                slots["railcard"] = "unspecified"
            else:
                railcard = normalise_railcard(message)
                if railcard:
                    slots["railcard"] = railcard
                elif "railcard" in lower:
                    slots["railcard"] = "unspecified"
        if not slots.get("railcard"):
            railcard = normalise_railcard(message)
            if railcard:
                slots["railcard"] = railcard

    if intent == "ticket_search":
        time_pattern = r"\b(?:\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm))\b"
        found_stations = find_station_matches(lower)
        if "origin" not in slots and found_stations:
            slots["origin"] = found_stations[0]
        elif "destination" not in slots and found_stations:
            if found_stations[0] != slots.get("origin"):
                slots["destination"] = found_stations[0]
        if "outbound_datetime" not in slots and re.search(time_pattern, lower):
            slots["outbound_datetime"] = message

    if intent == "arrival_time_prediction":
        if "delay_minutes" not in slots:
            match = re.search(r"\b(\d{1,3})\s*(?:minutes?|mins?|m)\b", lower)
            if match:
                slots["delay_minutes"] = int(match.group(1))
        if "destination" not in slots:
            if "waterloo" in lower or "london" in lower:
                slots["destination"] = "London Waterloo"
            elif "weymouth" in lower:
                slots["destination"] = "Weymouth"
        if "current_station" not in slots and len(text.split()) <= 5 and "delay" not in lower:
            slots["current_station"] = text.title()

    if intent == "contingency_advice":
        if "requested_detail" not in slots:
            detail = requested_detail_from_text(lower)
            if detail != "all":
                slots["requested_detail"] = detail

def format_delay_reply(prediction: dict[str, Any]) -> str:
    reasons = "\n".join(f"- {reason}" for reason in prediction["reasons"])
    return (
        f"Delay update for your journey from {prediction['origin']} to {prediction['destination']}:\n\n"
        f"Train service: {prediction['train_service']}\n"
        f"Current station: {prediction['current_station']}\n"
        f"Current delay: {prediction['current_delay_minutes']} minutes\n\n"
        f"Scheduled arrival at {prediction['destination']}: {prediction['planned_arrival_time']}\n"
        f"Estimated arrival at {prediction['destination']}: {prediction['expected_arrival_time']}\n"
        f"Predicted additional delay: {prediction['ml_additional_delay_display']}\n\n"
        f"Risk: {prediction['risk_level'].title()}\n\n"
        f"Possible reason for the delay:\n{reasons}\n\n"
        f"Recommendation: {prediction['recommendation']}"
    )


def format_arrival_reply(prediction: dict[str, Any]) -> str:
    if prediction.get("predicted_arrival_time") == "Unavailable":
        return (
            "I could not predict the arrival time yet. "
            f"{prediction.get('confidence_note', '')} {prediction.get('explanation', '')}"
        )
    return (
        f"Your train is expected to arrive at {prediction['destination']} "
        f"at {prediction['predicted_arrival_time']}.\n\n"
        f"It is currently at {prediction['current_station']} with a reported delay of "
        f"{prediction['delay_minutes_reported']} minutes. Based on previous SWR services, "
        f"the final delay at {prediction['destination']} is predicted to be about "
        f"{prediction['predicted_final_delay_minutes']} minutes."
    )


def format_contingency_reply(advice: dict[str, Any]) -> str:
    if advice.get("kind") == "unavailable":
        return (
            "I could not find a matching contingency plan yet. "
            "Please include the affected station or line section and the blockage type."
        )
    requested_detail = advice.get("requested_detail", "all")
    parts = [contingency_context(advice)]
    if should_include_detail(requested_detail, "service_alteration") and advice.get("service_alteration"):
        parts.append(f"Principles of Service Alteration:\n{advice['service_alteration']}")
    if should_include_detail(requested_detail, "alternative_passenger_journey") and advice.get("alternative_passenger_journey"):
        parts.append(f"Alternative Passenger Journey:\n{advice['alternative_passenger_journey']}")
    if should_include_detail(requested_detail, "station_staff_info") and advice.get("station_staff_info"):
        parts.append(f"Additional Information for Station Staff:\n{advice['station_staff_info']}")
    if should_include_detail(requested_detail, "signaller_info") and advice.get("signaller_info"):
        parts.append(f"Information for Signallers:\n{advice['signaller_info']}")
    if should_include_detail(requested_detail, "passenger_info") and advice.get("passenger_info"):
        parts.append(f"Passenger Information:\n{advice['passenger_info']}")
    """ if advice.get("kind") != "semantic_guidance":
        parts.append("Would you like to see additional contingency plans for this incident?") """
    return "\n\n".join(parts)


def contingency_context(advice: dict[str, Any]) -> str:
    title = str(advice.get("title", "")).strip()
    if advice.get("kind") == "semantic_guidance":
        return "Here is the most relevant operational guidance for this disruption."
    if advice.get("kind") == "station_disruption_plan":
        station = title.removeprefix("Station disruption plan:").strip() or "the affected station"
        return f"Station Disruption Plan\nStation: {station}"
    blockage = str(advice.get("blockage_type", "")).strip().lower()
    route = route_from_plan_title(title)
    lines = [contingency_plan_heading(title)]
    if route:
        lines.append(f"Location: {route.replace(' and ', ' - ')}")
    if blockage:
        lines.append(f"Condition: {blockage.title()}")
    return "\n".join(lines)


def contingency_plan_heading(title: str) -> str:
    match = re.search(r"\bPlan\s+(\d+\.\d+)", title)
    if match:
        return f"Contingency Plan {match.group(1)}"
    return "Contingency Plan"


def route_from_plan_title(title: str) -> str:
    _, _, section = title.partition(":")
    if " - " not in section:
        return ""
    first, last = section.split(" - ", 1)
    first = first.strip()
    last = last.strip()
    for suffix in (
        " Portsmouth Direct",
        " Bournemouth Main Line",
        " Basingstoke to Laverstock",
        " Alton Line",
    ):
        if last.endswith(suffix):
            last = last[: -len(suffix)].strip()
            break
    if not first or not last:
        return ""
    return f"{first} and {last}"


def should_include_detail(requested_detail: str, field: str) -> bool:
    if requested_detail in {"", "all", None}:
        return True
    if requested_detail == field:
        return True
    if requested_detail == "signaller_info" and field == "service_alteration":
        return True
    if requested_detail in {"alternative_passenger_journey", "passenger_info"} and field in {
        "alternative_passenger_journey", "passenger_info"
    }:
        return True
    return False

def complete_flow(intent: Intent, slots: dict[str, Any]) -> str:
    if intent != "ticket_search":
        return (
            "Sorry, I can only help with UK train tickets, delay predictions, "
            "SWR arrival predictions, and contingency advice right now."
        )
    if not slots.get("origin") or not slots.get("destination"):
        return "Please tell me both the origin and destination stations so I can find your ticket."

    ticket = find_cheapest_ticket(slots)

    if isinstance(ticket, dict) and ticket.get("error") == "invalid_station":
        return (
            "Sorry, I can help with UK train stations. "
            "Please enter valid UK station names like Norwich, London, Cambridge, or Waterloo."
        )
    if isinstance(ticket, dict) and ticket.get("error") == "invalid_time":
        return "Please provide a valid date and time for your journey."
    if isinstance(ticket, dict) and ticket.get("error") == "api_failure":
        return (
            "Sorry, the live ticket service is unavailable right now, "
            "so I couldn't check fares for that journey."
        )

    if slots.get("return_preference") == "return" and isinstance(ticket, dict) and "return" in ticket:
        out = ticket["outward"]
        ret = ticket["return"]
        total = ticket["total_price"]
        if not out or "origin" not in out or "destination" not in out:
            return (
                "Sorry, I couldn't find a valid journey for those stations. "
                "Please check the station names and try again."
            )
        return (
            "Here's the cheapest return ticket I found!\n\n"
            f"OUTWARD: {out.get('origin')} -> {out.get('destination')}\n"
            f"Departs {out.get('departure_time')} ({platform_label(out.get('departure_platform'))})\n"
            f"Arrives {out.get('arrival_time')} ({platform_label(out.get('arrival_platform'))})\n"
            f"Duration {out.get('duration')}\n"
            f"Price: {out.get('price')}\n\n"
            f"RETURN: {ret.get('origin')} -> {ret.get('destination')}\n"
            f"Departs {ret.get('departure_time')} ({platform_label(ret.get('departure_platform'))})\n"
            f"Arrives {ret.get('arrival_time')} ({platform_label(ret.get('arrival_platform'))})\n"
            f"Duration {ret.get('duration')}\n"
            f"Price: {ret.get('price')}\n\n"
            f"TOTAL: {total}\n"
            f"Booking link: {out.get('booking_url')}"
        )

    out = ticket
    if not out or "origin" not in out or "destination" not in out:
        return (
            "Sorry, I couldn't find a valid journey for those stations. "
            "Please check the station names and try again."
        )
    return (
        "Here's the cheapest ticket I found!\n\n"
        f"{out.get('origin')} -> {out.get('destination')}\n"
        f"Departs {out.get('departure_time')} ({platform_label(out.get('departure_platform'))})\n"
        f"Arrives {out.get('arrival_time')} ({platform_label(out.get('arrival_platform'))})\n"
        f"Duration {out.get('duration')}\n"
        f"Ticket: {out.get('ticket_type')}\n"
        f"Price: {out.get('price')}\n\n"
        f"Booking link: {out.get('booking_url')}"
    )