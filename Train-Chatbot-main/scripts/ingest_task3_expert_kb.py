from __future__ import annotations

import csv
import html
import re
import sqlite3
import sys
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import (
    CHROMA_DB_PATH,
    TASK3_CHROMA_COLLECTION_NAME,
    TASK3_DOWNLOADS_PATH,
    TASK3_EXPERT_DB_PATH,
    TASK3_KB_PATH,
)
from app.rag import split_text


DOWNLOADS = Path(TASK3_DOWNLOADS_PATH)
EXPERT_DB = ROOT / TASK3_EXPERT_DB_PATH
TASK3_KB = ROOT / TASK3_KB_PATH
REPORT_PATH = ROOT / "y" / "task3_extraction_report.csv"
W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

PPTM_FILES = [
    DOWNLOADS / "20 Batch 2 LOR 2-4 CPT (19th May 21).pptm",
    DOWNLOADS / "Mainline CPT (8th May 21).pptm",
]
DISRUPTION_ZIP = DOWNLOADS / "SWR Station Disruption Plans.zip"
STATIONS_CSV = DOWNLOADS / "stations.csv"

SECTION_HEADINGS = [
    "Principles of Service Alteration",
    "Alternative Passenger Journey",
    "Information for Signallers",
    "Additional Information for Station Staff",
    "Passenger Information",
]


@dataclass
class ExtractedChunk:
    source: str
    title: str
    text: str
    category: str = "general"
    location: str = ""
    blockage_type: str = "general disruption"
    audience: str = "all"
    recommended_action: str = ""


def main() -> None:
    REPORT_PATH.parent.mkdir(exist_ok=True)
    TASK3_KB.mkdir(exist_ok=True)

    with sqlite3.connect(EXPERT_DB) as conn:
        initialise_schema(conn)
        clear_existing(conn)
        report_rows: list[dict[str, Any]] = []
        rule_cards: list[ExtractedChunk] = []

        if STATIONS_CSV.exists():
            count = import_stations(conn, STATIONS_CSV)
            report_rows.append(report("stations.csv", "stations", "processed", count, "Imported TIPLOC records"))

        for pptm in PPTM_FILES:
            if not pptm.exists():
                report_rows.append(report(pptm.name, "pptm", "missing", 0, "File not found"))
                continue
            plan_count, rule_count = ingest_pptm(conn, pptm, rule_cards)
            report_rows.append(report(pptm.name, "pptm", "processed", plan_count, f"{rule_count} rule cards"))

        if DISRUPTION_ZIP.exists():
            totals = ingest_station_zip(conn, DISRUPTION_ZIP, rule_cards)
            report_rows.extend(totals)
        else:
            report_rows.append(report(DISRUPTION_ZIP.name, "zip", "missing", 0, "File not found"))

        write_chunks(rule_cards)
        chroma_count = ingest_chroma(rule_cards)
        report_rows.append(report("swr_contingency_expert_kb", "chroma", "processed", chroma_count, "Rule cards upserted"))
        write_report(report_rows)

    print(f"Task 3 expert DB: {EXPERT_DB}")
    print(f"Task 3 KB chunks: {TASK3_KB}")
    print(f"Extraction report: {REPORT_PATH}")


def initialise_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS Tiploc (
          name TEXT,
          longname TEXT,
          name_alias TEXT,
          alpha3 TEXT,
          tiploc TEXT PRIMARY KEY
        );

        CREATE TABLE IF NOT EXISTS blockage_types (
          id INTEGER PRIMARY KEY,
          description TEXT
        );

        CREATE TABLE IF NOT EXISTS ContingencyPlan (
          id INTEGER,
          subplan INTEGER,
          name_of_line TEXT,
          first_station TEXT,
          last_station TEXT,
          blockage_type INTEGER,
          source_file TEXT,
          PRIMARY KEY (id, subplan)
        );

        CREATE TABLE IF NOT EXISTS ServiceAlteration (
          cplan_id INTEGER,
          subplan INTEGER,
          section TEXT,
          alteration TEXT,
          important INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS AltPassengerJourney (
          cplan_id INTEGER,
          subplan INTEGER,
          section TEXT,
          altjourney TEXT,
          important INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS SignallerInfo (
          cplan_id INTEGER,
          subplan INTEGER,
          section TEXT,
          info TEXT,
          important INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS StationStaffInfo (
          cplan_id INTEGER,
          subplan INTEGER,
          section TEXT,
          info TEXT,
          important INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS PassengerInfo (
          cplan_id INTEGER,
          subplan INTEGER,
          section TEXT,
          info TEXT,
          important INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS ContactDetails (
          name_of_line TEXT,
          contact_name TEXT,
          phone_numbers TEXT
        );

        CREATE TABLE IF NOT EXISTS StationDisruptionPlan (
          id INTEGER PRIMARY KEY,
          station_name TEXT,
          dateof TEXT,
          doc_owner TEXT,
          source_file TEXT
        );

        CREATE TABLE IF NOT EXISTS OtherStationSupport (
          dplan_id INTEGER,
          other_station_name TEXT,
          phone_numbers TEXT,
          time_info TEXT
        );

        CREATE TABLE IF NOT EXISTS StationMapInfo (
          dplan_id INTEGER,
          location_id INTEGER,
          location_info TEXT
        );

        CREATE TABLE IF NOT EXISTS AltTransportInfo (
          dplan_id INTEGER,
          destination TEXT,
          route_type INTEGER,
          route_info TEXT
        );

        CREATE TABLE IF NOT EXISTS StationTips (
          dplan_id INTEGER,
          issue TEXT,
          tip TEXT
        );
        """
    )


def clear_existing(conn: sqlite3.Connection) -> None:
    for table in [
        "Tiploc",
        "blockage_types",
        "ContingencyPlan",
        "ServiceAlteration",
        "AltPassengerJourney",
        "SignallerInfo",
        "StationStaffInfo",
        "PassengerInfo",
        "ContactDetails",
        "StationDisruptionPlan",
        "OtherStationSupport",
        "StationMapInfo",
        "AltTransportInfo",
        "StationTips",
    ]:
        conn.execute(f"DELETE FROM {table}")
    conn.executemany(
        "INSERT INTO blockage_types (id, description) VALUES (?, ?)",
        [(1, "Both lines blocked"), (2, "One line blocked"), (3, "Line or platform blocked")],
    )


def import_stations(conn: sqlite3.Connection, path: Path) -> int:
    count = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tiploc = clean_null(row.get("tiploc"))
            if not tiploc:
                continue
            conn.execute(
                """
                INSERT OR REPLACE INTO Tiploc (name, longname, name_alias, alpha3, tiploc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    clean_null(row.get("name")),
                    clean_null(row.get("longname.name_alias")),
                    clean_null(row.get("name_alias")),
                    clean_null(row.get("alpha3")),
                    tiploc,
                ),
            )
            count += 1
    return count


def ingest_pptm(conn: sqlite3.Connection, path: Path, chunks: list[ExtractedChunk]) -> tuple[int, int]:
    plan_count = 0
    chunk_count = 0
    with zipfile.ZipFile(path) as deck:
        slides = sorted(
            [name for name in deck.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", name)],
            key=lambda name: int(re.search(r"(\d+)", name).group(1)),
        )
        current_index_line = ""
        for slide in slides:
            text = ppt_slide_text(deck.read(slide))
            if not text:
                continue
            if not text.startswith("Plan "):
                current_index_line = text
                continue

            plan = parse_plan_slide(text, current_index_line, path.name)
            if not plan:
                continue
            insert_contingency_plan(conn, plan)
            title = f"Plan {plan['id']}.{plan['subplan']} {plan['first_station']} - {plan['last_station']}"
            rule_cards = contingency_rule_cards(path.name, title, plan)
            chunks.extend(rule_cards)
            plan_count += 1
            chunk_count += len(rule_cards)
    return plan_count, chunk_count


def ppt_slide_text(blob: bytes) -> str:
    raw = blob.decode("utf-8", "ignore")
    values = re.findall(r"<a:t>(.*?)</a:t>", raw)
    return normalise_space(" ".join(html.unescape(value) for value in values))


def parse_plan_slide(text: str, index_line: str, source_file: str) -> dict[str, Any] | None:
    match = re.match(r"Plan\s+(\d+)\.(\d+)\s+(.+?)\s+Status:\s+(.+)", text)
    if not match:
        return None

    plan_id = int(match.group(1))
    subplan = int(match.group(2))
    line_name = normalise_space(match.group(3))
    status_and_sections = match.group(4)
    status, section_text = split_status_and_sections(status_and_sections)
    first_station, last_station = split_line_stations(line_name)
    sections = extract_sections(section_text)

    return {
        "id": plan_id,
        "subplan": subplan,
        "name_of_line": line_name,
        "first_station": first_station,
        "last_station": last_station,
        "blockage_type": blockage_type_id(status),
        "status": status,
        "sections": sections,
        "source_file": source_file,
        "summary_text": (
            f"{index_line}\nPlan {plan_id}.{subplan}: {line_name}. Status: {status}.\n"
            f"Service alteration: {sections.get('Principles of Service Alteration', '')}\n"
            f"Alternative passenger journey: {sections.get('Alternative Passenger Journey', '')}\n"
            f"Signaller info: {sections.get('Information for Signallers', '')}\n"
            f"Station staff info: {sections.get('Additional Information for Station Staff', '')}\n"
            f"Passenger info: {sections.get('Passenger Information', '')}"
        ),
    }


def split_status_and_sections(text: str) -> tuple[str, str]:
    first_heading = min([text.find(h) for h in SECTION_HEADINGS if h in text] or [-1])
    if first_heading == -1:
        return normalise_space(text), ""
    return normalise_space(text[:first_heading]), text[first_heading:]


def extract_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for index, heading in enumerate(SECTION_HEADINGS):
        start = text.find(heading)
        if start == -1:
            continue
        end_candidates = [text.find(next_heading, start + len(heading)) for next_heading in SECTION_HEADINGS[index + 1 :]]
        end_candidates = [candidate for candidate in end_candidates if candidate != -1]
        end = min(end_candidates) if end_candidates else len(text)
        sections[heading] = normalise_space(text[start + len(heading) : end])
    return sections


def insert_contingency_plan(conn: sqlite3.Connection, plan: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO ContingencyPlan
          (id, subplan, name_of_line, first_station, last_station, blockage_type, source_file)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            plan["id"],
            plan["subplan"],
            plan["name_of_line"],
            plan["first_station"],
            plan["last_station"],
            plan["blockage_type"],
            plan["source_file"],
        ),
    )
    section_map = [
        ("Principles of Service Alteration", "ServiceAlteration", "alteration"),
        ("Alternative Passenger Journey", "AltPassengerJourney", "altjourney"),
        ("Information for Signallers", "SignallerInfo", "info"),
        ("Additional Information for Station Staff", "StationStaffInfo", "info"),
        ("Passenger Information", "PassengerInfo", "info"),
    ]
    for heading, table, column in section_map:
        text = plan["sections"].get(heading)
        if not text:
            continue
        conn.execute(
            f"INSERT INTO {table} (cplan_id, subplan, section, {column}, important) VALUES (?, ?, ?, ?, ?)",
            (plan["id"], plan["subplan"], plan["status"], text[:1024], int(is_important(text))),
        )


def ingest_station_zip(conn: sqlite3.Connection, path: Path, chunks: list[ExtractedChunk]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        rows.append(report(path.name, "zip", "discovered", len(names), "Station disruption plan entries"))
        dplan_id = 1
        processed = skipped = 0
        for name in names:
            suffix = Path(name).suffix.lower()
            if suffix not in {".docx", ".docm"}:
                skipped += 1
                rows.append(report(name, suffix.lstrip("."), "skipped", 0, "Legacy Word format not parsed"))
                continue
            try:
                paragraphs = docx_paragraphs(archive.read(name))
            except Exception as exc:
                skipped += 1
                rows.append(report(name, suffix.lstrip("."), "error", 0, type(exc).__name__))
                continue
            if not paragraphs:
                skipped += 1
                rows.append(report(name, suffix.lstrip("."), "skipped", 0, "No text extracted"))
                continue
            station = station_from_doc_name(name, paragraphs)
            sections = station_sections(paragraphs)
            insert_station_plan(conn, dplan_id, station, name, sections)
            chunks.extend(station_rule_cards(name, station, sections))
            processed += 1
            dplan_id += 1
        rows.append(report("SWR Station Disruption Plans.zip", "station_plans", "processed", processed, f"{skipped} skipped"))
    return rows


def docx_paragraphs(blob: bytes) -> list[str]:
    with zipfile.ZipFile(BytesIO(blob)) as docx:
        xml = docx.read("word/document.xml")
    root = ET.fromstring(xml)
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", W_NS):
        parts = [node.text or "" for node in paragraph.findall(".//w:t", W_NS)]
        text = normalise_space(" ".join(parts))
        if text and not text.startswith("PAGEREF "):
            paragraphs.append(text)
    return paragraphs


def station_sections(paragraphs: list[str]) -> dict[str, str]:
    wanted = [
        "INTRODUCTION",
        "COLLEAGUE WELFARE",
        "OVERALL COMMUNICATIONS",
        "COMMUNICATION WITH CONTROL",
        "LOCAL COMMUNICATIONS",
        "PRACTICAL OPERATION OF THE STATION",
        "ALTERNATIVE TRANSPORT",
        "CSL2 OVERVIEW",
        "USEFUL WEBSITES",
        "TOP TIPS",
        "ROLE ABBREVIATIONS",
        "ROLE DESCRIPTIONS",
    ]
    sections: dict[str, list[str]] = {}
    current = "GENERAL"
    for paragraph in paragraphs:
        upper = paragraph.upper()
        matched = next((heading for heading in wanted if upper.startswith(heading)), None)
        if matched:
            current = matched
            sections.setdefault(current, [])
            continue
        if current in {"GENERAL", *wanted} and len(paragraph) > 15:
            sections.setdefault(current, []).append(paragraph)
    return {heading: normalise_space(" ".join(values))[:4000] for heading, values in sections.items() if values}


def insert_station_plan(
    conn: sqlite3.Connection,
    dplan_id: int,
    station: str,
    source_file: str,
    sections: dict[str, str],
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO StationDisruptionPlan (id, station_name, dateof, doc_owner, source_file)
        VALUES (?, ?, ?, ?, ?)
        """,
        (dplan_id, station, None, None, source_file),
    )
    for heading, text in sections.items():
        if "ALTERNATIVE TRANSPORT" in heading:
            conn.execute(
                "INSERT INTO AltTransportInfo (dplan_id, destination, route_type, route_info) VALUES (?, ?, ?, ?)",
                (dplan_id, station, 1, text[:255]),
            )
        elif "SUPPORT" in heading:
            conn.execute(
                "INSERT INTO OtherStationSupport (dplan_id, other_station_name, phone_numbers, time_info) VALUES (?, ?, ?, ?)",
                (dplan_id, station, "", text),
            )
        else:
            conn.execute(
                "INSERT INTO StationTips (dplan_id, issue, tip) VALUES (?, ?, ?)",
                (dplan_id, heading.title(), text),
            )


def station_summary_text(station: str, sections: dict[str, str]) -> str:
    lines = [f"Station disruption plan for {station}."]
    for heading in [
        "COMMUNICATION WITH CONTROL",
        "LOCAL COMMUNICATIONS",
        "PRACTICAL OPERATION OF THE STATION",
        "ALTERNATIVE TRANSPORT",
        "CSL2 OVERVIEW",
        "TOP TIPS",
    ]:
        if heading in sections:
            lines.append(f"{heading.title()}: {sections[heading]}")
    return "\n".join(lines)


def contingency_rule_cards(source: str, title: str, plan: dict[str, Any]) -> list[ExtractedChunk]:
    section_map = [
        ("Principles of Service Alteration", "service_controller", "service_alteration"),
        ("Alternative Passenger Journey", "passenger", "alternative_passenger_journey"),
        ("Information for Signallers", "signaller", "signaller_info"),
        ("Additional Information for Station Staff", "station_staff", "station_staff_info"),
        ("Passenger Information", "passenger", "passenger_info"),
    ]
    location = f"{plan['first_station']} - {plan['last_station']}".strip(" -")
    blockage = str(plan.get("status") or "general disruption")
    cards: list[ExtractedChunk] = []
    for heading, audience, action_type in section_map:
        action = normalise_space(plan["sections"].get(heading, ""))
        if not action:
            continue
        action = shorten_text(action, 1600)
        text = (
            f"{title}. Category: line blockage. Location: {location}. "
            f"Blockage type: {blockage}. Audience: {audience}. "
            f"Recommended action ({action_type}): {action}"
        )
        cards.append(
            ExtractedChunk(
                source=source,
                title=f"{title} - {heading}",
                text=text,
                category="line_blockage",
                location=location,
                blockage_type=blockage,
                audience=audience,
                recommended_action=action,
            )
        )
    if not cards:
        cards.append(
            ExtractedChunk(
                source=source,
                title=title,
                text=plan["summary_text"],
                category="line_blockage",
                location=location,
                blockage_type=blockage,
                audience="all",
                recommended_action=plan["summary_text"],
            )
        )
    return cards


def station_rule_cards(source: str, station: str, sections: dict[str, str]) -> list[ExtractedChunk]:
    audience_map = {
        "COMMUNICATION WITH CONTROL": "service_controller",
        "LOCAL COMMUNICATIONS": "station_staff",
        "PRACTICAL OPERATION OF THE STATION": "station_staff",
        "ALTERNATIVE TRANSPORT": "passenger",
        "CSL2 OVERVIEW": "station_staff",
        "TOP TIPS": "station_staff",
    }
    cards: list[ExtractedChunk] = []
    for heading, text in sections.items():
        action = shorten_text(normalise_space(text), 1600)
        if not action:
            continue
        audience = audience_map.get(heading, "all")
        title = f"Station disruption plan: {station} - {heading.title()}"
        card_text = (
            f"Station disruption plan for {station}. Category: station disruption. "
            f"Location: {station}. Blockage type: station or platform disruption. "
            f"Audience: {audience}. Recommended action ({heading.title()}): {action}"
        )
        cards.append(
            ExtractedChunk(
                source=source,
                title=title,
                text=card_text,
                category="station_disruption",
                location=station,
                blockage_type="station or platform disruption",
                audience=audience,
                recommended_action=action,
            )
        )
    return cards


def write_chunks(chunks: list[ExtractedChunk]) -> None:
    for existing in TASK3_KB.glob("*.txt"):
        existing.unlink()
    for index, chunk in enumerate(chunks, start=1):
        safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", chunk.title).strip("_")[:90]
        path = TASK3_KB / f"{index:04d}_{safe_title}.txt"
        path.write_text(
            "\n".join(
                [
                    chunk.title,
                    f"Source: {chunk.source}",
                    f"Category: {chunk.category}",
                    f"Location: {chunk.location}",
                    f"Blockage type: {chunk.blockage_type}",
                    f"Audience: {chunk.audience}",
                    "",
                    shorten_text(chunk.text, 1800),
                ]
            ),
            encoding="utf-8",
        )


def ingest_chroma(chunks: list[ExtractedChunk]) -> int:
    import chromadb

    client = chromadb.PersistentClient(path=str(ROOT / CHROMA_DB_PATH))
    try:
        client.delete_collection(name=TASK3_CHROMA_COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=TASK3_CHROMA_COLLECTION_NAME,
        metadata={"description": "Task 3 operational contingency rule cards"},
    )
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        for part_index, text in enumerate(split_text(chunk.text, chunk_size=1100, overlap=150)):
            ids.append(f"task3:{chunk_index}:{part_index}")
            documents.append(text)
            metadatas.append(
                {
                    "source": chunk.source,
                    "source_file": chunk.source,
                    "source_title": chunk.title,
                    "title": chunk.title,
                    "category": chunk.category,
                    "location": chunk.location,
                    "blockage_type": chunk.blockage_type,
                    "audience": chunk.audience,
                    "recommended_action": shorten_text(chunk.recommended_action, 1600),
                    "chunk": part_index,
                }
            )
    if documents:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(documents)


def write_report(rows: list[dict[str, Any]]) -> None:
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "kind", "status", "records", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def report(file: str, kind: str, status: str, records: int, notes: str) -> dict[str, Any]:
    return {"file": file, "kind": kind, "status": status, "records": records, "notes": notes}


def split_line_stations(line_name: str) -> tuple[str, str]:
    parts = re.split(r"\s+-\s+", line_name, maxsplit=1)
    if len(parts) == 2:
        return clean_station(parts[0]), clean_station(parts[1].split()[0] if len(parts[1].split()) == 1 else parts[1])
    return line_name, ""


def station_from_doc_name(name: str, paragraphs: list[str]) -> str:
    first = paragraphs[0] if paragraphs else ""
    if first and len(first) < 80 and not first.startswith("Station Disruption"):
        return first
    base = Path(name).stem
    match = re.search(r"Station Disruption Plan - (.+?)(?: Issue|$)", base)
    return match.group(1).strip() if match else base


def blockage_type_id(status: str) -> int:
    status_lower = status.casefold()
    if "both" in status_lower:
        return 1
    if "one" in status_lower or "single line" in status_lower:
        return 2
    return 3


def is_important(text: str) -> bool:
    return any(word in text.casefold() for word in ("do not", "cancel", "cancelled", "blocked", "required"))


def clean_station(value: str) -> str:
    return normalise_space(value.replace("\t", " "))


def clean_null(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text == r"\N" else text


def shorten_text(value: str, max_chars: int) -> str:
    text = normalise_space(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def normalise_space(value: str) -> str:
    replacements = {
        "â": "-",
        "â": "-",
        "â": "'",
        "â": "'",
        "â": '"',
        "â": '"',
        "â¢": "-",
        "Â": "",
    }
    for bad, good in replacements.items():
        value = value.replace(bad, good)
    return re.sub(r"\s+", " ", value).strip()


if __name__ == "__main__":
    main()
