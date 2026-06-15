from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import CHROMA_DB_PATH, TASK3_CHROMA_COLLECTION_NAME


ROOT = Path(__file__).resolve().parents[1]
MAX_DISTANCE = 1.65


def search_operational_rules(query: str, slots: dict[str, Any], top_k: int = 6) -> list[dict[str, Any]]:
    if not query.strip():
        return []

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(ROOT / CHROMA_DB_PATH))
        collection = client.get_collection(name=TASK3_CHROMA_COLLECTION_NAME)
        result = collection.query(query_texts=[query_with_slots(query, slots)], n_results=top_k)
    except Exception:
        return []

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    matches: list[dict[str, Any]] = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        numeric_distance = float(distance) if distance is not None else 0.0
        if numeric_distance > MAX_DISTANCE:
            continue
        metadata = metadata or {}
        matches.append(
            {
                "text": str(text),
                "distance": numeric_distance,
                "score": score_match(metadata, slots, numeric_distance),
                "source": metadata.get("source_file") or metadata.get("source") or "Task 3 expert KB",
                "title": metadata.get("source_title") or metadata.get("title") or "Relevant contingency rule",
                "category": metadata.get("category", "general"),
                "location": metadata.get("location", ""),
                "blockage_type": metadata.get("blockage_type", ""),
                "audience": metadata.get("audience", "all"),
                "recommended_action": metadata.get("recommended_action") or str(text),
            }
        )

    return sorted(matches, key=lambda item: (-float(item["score"]), float(item["distance"])))


def query_with_slots(query: str, slots: dict[str, Any]) -> str:
    parts = [query]
    for key in ("event_type", "first_station", "last_station", "station", "blockage_type", "requested_detail", "role"):
        value = slots.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def score_match(metadata: dict[str, Any], slots: dict[str, Any], distance: float) -> float:
    score = max(0.0, 2.0 - distance)
    requested_audience = requested_audience_from_slots(slots)
    audience = str(metadata.get("audience") or "").casefold()
    category = str(metadata.get("category") or "").casefold()
    location = str(metadata.get("location") or "").casefold()
    blockage = str(metadata.get("blockage_type") or "").casefold()

    if requested_audience and requested_audience in audience:
        score += 1.0
    if audience == "all":
        score += 0.2
    if slots.get("event_type") and str(slots["event_type"]).casefold() == category:
        score += 0.8
    for key in ("station", "first_station", "last_station"):
        value = str(slots.get(key) or "").casefold()
        if value and value in location:
            score += 0.7
    slot_blockage = str(slots.get("blockage_type") or "").casefold()
    if slot_blockage and any(word in blockage for word in slot_blockage.split()):
        score += 0.5
    return score


def requested_audience_from_slots(slots: dict[str, Any]) -> str:
    requested = str(slots.get("requested_detail") or slots.get("role") or "").casefold()
    if "signaller" in requested:
        return "signaller"
    if "station_staff" in requested or "station staff" in requested or requested == "staff":
        return "station_staff"
    if "passenger" in requested or "alternative" in requested:
        return "passenger"
    if "service" in requested or "control" in requested:
        return "service_controller"
    return ""

