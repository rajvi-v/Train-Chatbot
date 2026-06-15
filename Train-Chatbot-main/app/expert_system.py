from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from app.config import CHROMA_DB_PATH, TASK3_CHROMA_COLLECTION_NAME, TASK3_EXPERT_DB_PATH
from app.task3_rag import search_operational_rules


ROOT = Path(__file__).resolve().parents[1]
EXPERT_DB = ROOT / TASK3_EXPERT_DB_PATH


def get_contingency_advice(slots: dict[str, Any], message: str = "") -> dict[str, Any]:
    query = " ".join(str(value) for value in [message, *slots.values()] if value).strip()
    if not EXPERT_DB.exists():
        return unavailable("Task 3 expert knowledge base has not been ingested yet.")

    exact_result: dict[str, Any] | None = None
    with sqlite3.connect(EXPERT_DB) as conn:
        conn.row_factory = sqlite3.Row
        event_type = slots.get("event_type")
        has_line_context = event_type == "line_blockage" or line_pair(query, slots) != ("", "") or blockage_phrase(query, slots)
        if has_line_context:
            exact_result = contingency_plan_advice(conn, query, slots)

        if not exact_result:
            exact_result = station_disruption_advice(conn, query, slots)

        if not exact_result and has_line_context:
            exact_result = contingency_plan_advice(conn, query, slots)

    semantic_matches = search_operational_rules(query, slots)
    if exact_result:
        return merge_semantic_matches(exact_result, semantic_matches, slots)
    if semantic_matches:
        return semantic_guidance(semantic_matches, slots)
    return unavailable("I could not find a matching contingency or station disruption plan.")


def contingency_plan_advice(conn: sqlite3.Connection, query: str, slots: dict[str, Any]) -> dict[str, Any] | None:
    first, last = line_pair(query, slots)
    blockage = blockage_phrase(query, slots)
    if not any((first, last, blockage, slots.get("event_type") == "line_blockage")):
        return None
    rows = conn.execute(
        """
        SELECT cp.id, cp.subplan, cp.name_of_line, cp.first_station, cp.last_station,
               bt.description AS blockage_type, cp.source_file
        FROM ContingencyPlan cp
        LEFT JOIN blockage_types bt ON bt.id = cp.blockage_type
        WHERE (? = '' OR lower(cp.first_station) LIKE lower(?) OR lower(cp.name_of_line) LIKE lower(?))
          AND (? = '' OR lower(cp.last_station) LIKE lower(?) OR lower(cp.name_of_line) LIKE lower(?))
          AND (? = '' OR lower(bt.description) LIKE lower(?))
        ORDER BY cp.id, cp.subplan
        LIMIT 1
        """,
        (
            first,
            f"%{first}%",
            f"%{first}%",
            last,
            f"%{last}%",
            f"%{last}%",
            blockage,
            f"%{blockage}%",
        ),
    ).fetchall()

    if not rows and first:
        rows = conn.execute(
            """
            SELECT cp.id, cp.subplan, cp.name_of_line, cp.first_station, cp.last_station,
                   bt.description AS blockage_type, cp.source_file
            FROM ContingencyPlan cp
            LEFT JOIN blockage_types bt ON bt.id = cp.blockage_type
            WHERE lower(cp.name_of_line) LIKE lower(?)
            ORDER BY cp.id, cp.subplan
            LIMIT 1
            """,
            (f"%{first}%",),
        ).fetchall()

    if not rows:
        return None

    plan = dict(rows[0])
    cplan_id = int(plan["id"])
    subplan = int(plan["subplan"])
    return {
        "kind": "contingency_plan",
        "title": f"Plan {cplan_id}.{subplan}: {plan['name_of_line']}",
        "blockage_type": plan.get("blockage_type") or "Contingency",
        "source": plan.get("source_file") or "contingency plan deck",
        "service_alteration": fetch_plan_text(conn, "ServiceAlteration", "alteration", cplan_id, subplan),
        "alternative_passenger_journey": fetch_plan_text(conn, "AltPassengerJourney", "altjourney", cplan_id, subplan),
        "signaller_info": fetch_plan_text(conn, "SignallerInfo", "info", cplan_id, subplan),
        "station_staff_info": fetch_plan_text(conn, "StationStaffInfo", "info", cplan_id, subplan),
        "passenger_info": fetch_plan_text(conn, "PassengerInfo", "info", cplan_id, subplan),
        "requested_detail": slots.get("requested_detail", "all"),
    }


def station_disruption_advice(conn: sqlite3.Connection, query: str, slots: dict[str, Any]) -> dict[str, Any] | None:
    station = str(slots.get("station") or "").strip() or find_station_name(conn, query)
    if not station:
        return None

    plan = conn.execute(
        """
        SELECT id, station_name, source_file
        FROM StationDisruptionPlan
        WHERE lower(station_name) = lower(?)
        LIMIT 1
        """,
        (station,),
    ).fetchone()
    if plan is None:
        return None

    dplan_id = int(plan["id"])
    requested_detail = slots.get("requested_detail", "all")
    tips = station_staff_tips(conn, dplan_id, str(requested_detail))
    alt_transport = conn.execute(
        "SELECT route_info FROM AltTransportInfo WHERE dplan_id = ? LIMIT 2",
        (dplan_id,),
    ).fetchall()
    alternative_text = "\n".join(shorten(row["route_info"], 300) for row in alt_transport)
    if not alternative_text:
        alternative_tips = conn.execute(
            """
            SELECT issue, tip
            FROM StationTips
            WHERE dplan_id = ?
              AND (
                lower(issue) LIKE '%alternative%'
                OR lower(issue) LIKE '%top%'
                OR lower(tip) LIKE '%bus%'
                OR lower(tip) LIKE '%route%'
              )
            ORDER BY
              CASE
                WHEN lower(issue) LIKE '%top%' THEN 1
                WHEN lower(issue) LIKE '%alternative%' THEN 2
                WHEN lower(tip) LIKE '%bus%' THEN 3
                ELSE 4
              END
            LIMIT 3
            """,
            (dplan_id,),
        ).fetchall()
        top_tips = [row for row in alternative_tips if "top" in str(row["issue"]).casefold()]
        if top_tips:
            alternative_tips = top_tips
        alternative_text = "\n".join(
            f"{row['issue']}: {shorten(row['tip'], 350)}" for row in alternative_tips
        )

    return {
        "kind": "station_disruption_plan",
        "title": f"Station disruption plan: {plan['station_name']}",
        "source": plan["source_file"],
        "station_staff_info": format_station_staff_tips(tips),
        "alternative_passenger_journey": alternative_text,
        "passenger_info": "Use this station disruption plan alongside local station emergency and crowd-control plans.",
        "signaller_info": "",
        "service_alteration": "",
        "requested_detail": requested_detail,
    }


def station_staff_tips(
    conn: sqlite3.Connection,
    dplan_id: int,
    requested_detail: str,
) -> list[sqlite3.Row]:
    if requested_detail == "all":
        return conn.execute(
            """
            SELECT issue, tip
            FROM StationTips
            WHERE dplan_id = ?
            ORDER BY
              CASE
                WHEN issue LIKE '%Communication With Control%' THEN 1
                WHEN issue LIKE '%Local Communication%' THEN 2
                WHEN issue LIKE '%Practical%' THEN 3
                WHEN issue LIKE '%Overall Communication%' THEN 4
                WHEN issue LIKE '%Top%' THEN 5
                ELSE 6
              END
            LIMIT 5
            """,
            (dplan_id,),
        ).fetchall()

    return conn.execute(
        """
        SELECT issue, tip
        FROM StationTips
        WHERE dplan_id = ?
          AND lower(issue) NOT LIKE '%top%'
          AND lower(issue) NOT LIKE '%overview%'
          AND (
            lower(issue) LIKE '%communication with control%'
            OR lower(issue) LIKE '%local communication%'
            OR lower(issue) LIKE '%practical operation%'
          )
        ORDER BY
          CASE
            WHEN lower(issue) LIKE '%local communication%' THEN 1
            WHEN lower(issue) LIKE '%communication with control%' THEN 2
            WHEN lower(issue) LIKE '%practical operation%' THEN 3
            ELSE 4
          END
        LIMIT 3
        """,
        (dplan_id,),
    ).fetchall()


def format_station_staff_tips(tips: list[sqlite3.Row]) -> str:
    if not tips:
        return "Staff should keep passengers informed, coordinate with Control, and manage the station safely during the disruption."
    formatted = []
    for row in tips:
        issue = str(row["issue"] or "Staff action").strip()
        tip = shorten(str(row["tip"] or ""), 320)
        if tip:
            formatted.append(f"- {issue}: {tip}")
    return "\n".join(formatted)


def semantic_fallback(query: str) -> dict[str, Any] | None:
    if not query:
        return None
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(ROOT / CHROMA_DB_PATH))
        collection = client.get_collection(name=TASK3_CHROMA_COLLECTION_NAME)
        result = collection.query(query_texts=[query], n_results=2)
    except Exception:
        return None

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    if not documents:
        return None
    source = metadatas[0].get("source", "Task 3 expert KB") if metadatas else "Task 3 expert KB"
    title = metadatas[0].get("title", "Relevant contingency guidance") if metadatas else "Relevant contingency guidance"
    return {
        "kind": "semantic_guidance",
        "title": title,
        "source": source,
        "service_alteration": "",
        "alternative_passenger_journey": "",
        "signaller_info": "",
        "station_staff_info": shorten(documents[0], 900),
        "passenger_info": "",
    }


def merge_semantic_matches(
    advice: dict[str, Any],
    matches: list[dict[str, Any]],
    slots: dict[str, Any],
) -> dict[str, Any]:
    if not matches:
        advice["retrieval_matches"] = []
        return advice

    merged = dict(advice)
    used_matches: list[dict[str, Any]] = []
    relevant_matches = relevant_semantic_matches(advice, matches, slots)
    for match in relevant_matches[:4]:
        field = field_for_match(match)
        action = shorten(str(match.get("recommended_action") or match.get("text") or ""), 700)
        if not action:
            continue
        existing = str(merged.get(field) or "").strip()
        if not existing:
            merged[field] = action
            used_matches.append(match)
    merged["retrieval_matches"] = retrieval_trace(used_matches)
    merged["source"] = combine_sources(str(merged.get("source", "")), used_matches)
    merged["requested_detail"] = slots.get("requested_detail", merged.get("requested_detail", "all"))
    return merged


def relevant_semantic_matches(
    advice: dict[str, Any],
    matches: list[dict[str, Any]],
    slots: dict[str, Any],
) -> list[dict[str, Any]]:
    terms = [
        str(slots.get(key) or "").casefold()
        for key in ("station", "first_station", "last_station")
        if slots.get(key)
    ]
    if not terms:
        return matches
    relevant = []
    for match in matches:
        haystack = " ".join(
            str(match.get(key) or "") for key in ("title", "location", "text", "recommended_action")
        ).casefold()
        if all(term in haystack for term in terms):
            relevant.append(match)
    return relevant


def semantic_guidance(matches: list[dict[str, Any]], slots: dict[str, Any]) -> dict[str, Any]:
    best = matches[0]
    advice = {
        "kind": "semantic_guidance",
        "title": best.get("title", "Relevant contingency guidance"),
        "source": combine_sources("", matches),
        "service_alteration": "",
        "alternative_passenger_journey": "",
        "signaller_info": "",
        "station_staff_info": "",
        "passenger_info": "",
        "requested_detail": slots.get("requested_detail", "all"),
        "retrieval_matches": retrieval_trace(matches),
    }
    return merge_semantic_matches(advice, matches, slots)


def field_for_match(match: dict[str, Any]) -> str:
    audience = str(match.get("audience") or "").casefold()
    text = f"{match.get('title', '')} {match.get('text', '')}".casefold()
    if "signaller" in audience or "signaller" in text:
        return "signaller_info"
    if "station_staff" in audience or "station staff" in text or "practical operation" in text:
        return "station_staff_info"
    if "service_controller" in audience or "service alteration" in text:
        return "service_alteration"
    if "alternative" in text or "transport" in text or "bus" in text or "taxi" in text:
        return "alternative_passenger_journey"
    if "passenger" in audience or "customer" in text:
        return "passenger_info"
    return "station_staff_info"


def retrieval_trace(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "title": match.get("title", ""),
            "source": match.get("source", ""),
            "category": match.get("category", ""),
            "location": match.get("location", ""),
            "audience": match.get("audience", ""),
            "distance": round(float(match.get("distance", 0.0)), 3),
        }
        for match in matches[:5]
    ]


def combine_sources(existing: str, matches: list[dict[str, Any]]) -> str:
    sources = [existing] if existing else []
    for match in matches[:4]:
        source = str(match.get("source") or "").strip()
        if source and source not in sources:
            sources.append(source)
    return "; ".join(sources) or "Task 3 expert KB"


def fetch_plan_text(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    cplan_id: int,
    subplan: int,
) -> str:
    rows = conn.execute(
        f"SELECT {column} FROM {table} WHERE cplan_id = ? AND subplan = ? ORDER BY important DESC LIMIT 3",
        (cplan_id, subplan),
    ).fetchall()
    return "\n".join(shorten(str(row[column]), 550) for row in rows if row[column])


def find_station_name(conn: sqlite3.Connection, query: str) -> str | None:
    stations = conn.execute(
        "SELECT station_name FROM StationDisruptionPlan ORDER BY length(station_name) DESC"
    ).fetchall()
    query_lower = query.casefold()
    for row in stations:
        station = str(row["station_name"])
        if station.casefold() in query_lower:
            return station
    return None


def line_pair(query: str, slots: dict[str, Any] | None = None) -> tuple[str, str]:
    slots = slots or {}
    if slots.get("first_station") and slots.get("last_station"):
        return clean_name(str(slots["first_station"])), clean_name(str(slots["last_station"]))
    match = re.search(r"\bbetween\s+([A-Za-z ]+?)\s+and\s+([A-Za-z ]+?)(?:[,.?]|$)", query, re.I)
    if match:
        return clean_name(match.group(1)), clean_name(match.group(2))
    match = re.search(
        r"\b(?:for|on|at)\s+([A-Za-z ]+?)\s+(?:-|to)\s+([A-Za-z ]+?)(?:\s+disruption|[,.?]|$)",
        query,
        re.I,
    )
    if match:
        return clean_name(match.group(1)), clean_name(match.group(2))
    return "", ""


def blockage_phrase(query: str, slots: dict[str, Any] | None = None) -> str:
    slots = slots or {}
    slot_blockage = str(slots.get("blockage_type") or "").casefold()
    if "both" in slot_blockage:
        return "both"
    if "one" in slot_blockage or "single" in slot_blockage:
        return "one"
    text = query.casefold()
    if "both line" in text or "both lines" in text:
        return "both"
    if "one line" in text or "single line" in text:
        return "one"
    return ""


def clean_name(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().title()


def shorten(value: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def unavailable(reason: str) -> dict[str, Any]:
    return {
        "kind": "unavailable",
        "title": "No contingency plan found",
        "source": "Task 3 expert system",
        "service_alteration": "",
        "alternative_passenger_journey": "",
        "signaller_info": "",
        "station_staff_info": reason,
        "passenger_info": "",
    }
