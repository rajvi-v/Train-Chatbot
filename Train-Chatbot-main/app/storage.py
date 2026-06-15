from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class ChatStorage:
    def __init__(self, db_path: str | Path = "chatbot.db") -> None:
        self.db_path = Path(db_path)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialise(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    intent TEXT NOT NULL,
                    state TEXT NOT NULL,
                    slots_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stations (
                    code TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    region TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS routes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    origin TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    usual_operator TEXT NOT NULL,
                    average_journey_minutes INTEGER NOT NULL,
                    congestion_level INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(origin, destination)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS delay_rules (
                    code TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    points INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS service_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    origin TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    status_type TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    travel_datetime TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._seed_reference_data(conn)

    def _seed_reference_data(self, conn: sqlite3.Connection) -> None:
        stations = [
            ("NRW", "Norwich", "East of England"),
            ("LST", "London", "London"),
            ("CBG", "Cambridge", "East of England"),
            ("MAN", "Manchester", "North West"),
            ("BHM", "Birmingham", "West Midlands"),
        ]
        routes = [
            ("Norwich", "London", "Greater Anglia", 112, 2),
            ("Cambridge", "London", "Great Northern", 65, 1),
            ("Manchester", "London", "Avanti West Coast", 130, 2),
            ("Birmingham", "London", "Avanti West Coast", 85, 1),
        ]
        rules = [
            ("peak_time", "Peak travel time increases delay risk.", 2),
            ("route_congestion", "This route has known congestion or capacity pressure.", 2),
            ("recent_disruption", "Recent route disruption is affecting reliability.", 3),
            ("weather_warning", "Weather warnings can slow services and cause knock-on delays.", 3),
            ("operator_reliability", "The operator has moderate recent reliability risk.", 1),
        ]
        statuses = [
            (
                "Norwich",
                "London",
                "recent_disruption",
                2,
                "Earlier signalling problems near London Liverpool Street may cause residual delays.",
            ),
            (
                "Manchester",
                "London",
                "weather_warning",
                3,
                "High winds on parts of the route may extend journey times.",
            ),
        ]

        conn.executemany(
            "INSERT OR IGNORE INTO stations (code, name, region) VALUES (?, ?, ?)",
            stations,
        )
        conn.executemany(
            """
            INSERT OR IGNORE INTO routes
                (origin, destination, usual_operator, average_journey_minutes, congestion_level)
            VALUES (?, ?, ?, ?, ?)
            """,
            routes,
        )
        conn.executemany(
            "INSERT OR IGNORE INTO delay_rules (code, description, points) VALUES (?, ?, ?)",
            rules,
        )
        conn.executemany(
            """
            INSERT INTO service_status (origin, destination, status_type, severity, message)
            SELECT ?, ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM service_status
                WHERE origin = ? AND destination = ? AND status_type = ?
            )
            """,
            [(o, d, t, s, m, o, d, t) for o, d, t, s, m in statuses],
        )

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, intent, state, slots_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row["session_id"],
            "intent": row["intent"],
            "state": row["state"],
            "slots": json.loads(row["slots_json"]),
        }

    def save_session(self, session_id: str, intent: str, state: str, slots: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, intent, state, slots_json, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    intent = excluded.intent,
                    state = excluded.state,
                    slots_json = excluded.slots_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, intent, state, json.dumps(slots)),
            )

    def log_message(self, session_id: str, sender: str, message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversation_logs (session_id, sender, message) VALUES (?, ?, ?)",
                (session_id, sender, message),
            )

    def find_route(self, origin: str, destination: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT origin, destination, usual_operator, average_journey_minutes, congestion_level
                FROM routes
                WHERE lower(origin) = lower(?) AND lower(destination) = lower(?)
                """,
                (origin, destination),
            ).fetchone()
        return dict(row) if row else None

    def get_delay_rules(self) -> dict[str, dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT code, description, points FROM delay_rules").fetchall()
        return {row["code"]: dict(row) for row in rows}

    def get_route_status(self, origin: str, destination: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status_type, severity, message, updated_at
                FROM service_status
                WHERE lower(origin) = lower(?) AND lower(destination) = lower(?)
                ORDER BY severity DESC, updated_at DESC
                """,
                (origin, destination),
            ).fetchall()
        return [dict(row) for row in rows]

    def log_prediction(
        self,
        session_id: str,
        slots: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_logs
                    (session_id, origin, destination, travel_datetime, result_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    slots.get("origin", ""),
                    slots.get("destination", ""),
                    slots.get("travel_datetime", ""),
                    json.dumps(result),
                ),
            )
