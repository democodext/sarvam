from datetime import date, datetime
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional
import uuid

DEFAULT_DB_PATH = "mean_ai.db"

VALID_STATUSES = {
    "explicit",
    "unclear",
    "pending",
    "completed",
    "overdue",
    "cancelled",
    "renegotiated",
}

# Allowed status transitions
ALLOWED_TRANSITIONS = {
    "explicit": {"completed", "overdue", "cancelled", "renegotiated"},
    "unclear": {"completed", "overdue", "cancelled", "renegotiated"},
    "pending": {"completed", "overdue", "cancelled", "renegotiated"},
    "renegotiated": {"completed", "overdue", "cancelled", "renegotiated"},
    "overdue": {"completed", "cancelled", "renegotiated"},
    "completed": {"renegotiated", "cancelled"},
    "cancelled": {"renegotiated"},
}


def load_env(filepath: str = ".env") -> None:
    """Loads key-value pairs from .env file into os.environ if present."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip("'\"")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes the database schema if tables do not exist."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                context TEXT,
                transcript TEXT NOT NULL,
                summary TEXT,
                clarification_question TEXT,
                created_at TEXT NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS commitments (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                description TEXT NOT NULL,
                responsible_person TEXT,
                deadline TEXT,
                deadline_date TEXT,
                status TEXT NOT NULL,
                commitment_type TEXT NOT NULL DEFAULT 'explicit',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS commitment_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commitment_id TEXT NOT NULL,
                quote TEXT NOT NULL,
                FOREIGN KEY (commitment_id) REFERENCES commitments(id) ON DELETE CASCADE
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS commitment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commitment_id TEXT NOT NULL,
                previous_status TEXT,
                new_status TEXT NOT NULL,
                previous_deadline TEXT,
                new_deadline TEXT,
                reason_or_notes TEXT,
                conversation_id TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (commitment_id) REFERENCES commitments(id) ON DELETE CASCADE
            );
        """)

        # Create Indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_commitments_conv_id ON commitments(conversation_id);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_commitments_status ON commitments(status);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_evidence_comm_id ON commitment_evidence(commitment_id);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_comm_id ON commitment_history(commitment_id);"
        )

        conn.commit()


def parse_deadline_date(deadline_str: Optional[str]) -> Optional[str]:
    """Attempts to parse a deadline string into an ISO YYYY-MM-DD format if explicit date mentioned."""
    if not deadline_str:
        return None

    clean_str = deadline_str.strip().lower()

    # Match YYYY-MM-DD
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", clean_str)
    if match:
        return match.group(1)

    # Match DD-MM-YYYY or DD/MM/YYYY
    match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", clean_str)
    if match:
        day, month, year = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        return f"{year:04d}-{month:02d}-{day:02d}"

    return None


def save_conversation_analysis(
    context: str,
    transcript: str,
    analysis_data: Dict[str, Any],
    conversation_id: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Saves a conversation and its extracted commitments/analysis to SQLite."""
    init_db(db_path)

    if not conversation_id:
        conversation_id = f"conv_{uuid.uuid4().hex[:12]}"

    now_iso = datetime.now().isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO conversations (id, context, transcript, summary, clarification_question, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                conversation_id,
                context,
                transcript,
                analysis_data.get("summary", ""),
                analysis_data.get("clarification_question", ""),
                now_iso,
            ),
        )

        commitments = analysis_data.get("commitments", [])
        saved_commitments = []

        for c in commitments:
            comm_id = c.get("id") or f"comm_{uuid.uuid4().hex[:12]}"
            desc = c.get("description", "")
            person = c.get("responsible_person")
            deadline = c.get("deadline")
            deadline_date = parse_deadline_date(deadline)
            status = c.get("status", "explicit")
            if status not in VALID_STATUSES:
                status = "explicit"
            comm_type = c.get("commitment_type", "explicit")

            cursor.execute(
                """
                INSERT INTO commitments (id, conversation_id, description, responsible_person, deadline, deadline_date, status, commitment_type, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    comm_id,
                    conversation_id,
                    desc,
                    person,
                    deadline,
                    deadline_date,
                    status,
                    comm_type,
                    now_iso,
                    now_iso,
                ),
            )

            # Record initial history
            cursor.execute(
                """
                INSERT INTO commitment_history (commitment_id, previous_status, new_status, previous_deadline, new_deadline, reason_or_notes, conversation_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    comm_id,
                    None,
                    status,
                    None,
                    deadline,
                    "Created during conversation analysis",
                    conversation_id,
                    now_iso,
                ),
            )

            # Save evidence quotes
            evidence_list = c.get("evidence", [])
            for quote in evidence_list:
                cursor.execute(
                    """
                    INSERT INTO commitment_evidence (commitment_id, quote)
                    VALUES (?, ?)
                """,
                    (comm_id, quote),
                )

            c["id"] = comm_id
            c["conversation_id"] = conversation_id
            saved_commitments.append(c)

        conn.commit()

    analysis_data["conversation_id"] = conversation_id
    analysis_data["commitments"] = saved_commitments
    return conversation_id


def update_commitment_status(
    commitment_id: str,
    new_status: str,
    reason_or_notes: str = "",
    new_deadline: Optional[str] = None,
    conversation_id: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """Updates commitment status and/or deadline, maintaining change audit history.

    Raises:
        ValueError: If status transition is invalid or commitment not found.
    """
    init_db(db_path)
    new_status = new_status.strip().lower()

    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{new_status}'. Allowed statuses: {sorted(VALID_STATUSES)}"
        )

    now_iso = datetime.now().isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM commitments WHERE id = ?", (commitment_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Commitment '{commitment_id}' not found.")

        current_status = row["status"]
        current_deadline = row["deadline"]

        # Validate transition
        if current_status != new_status:
            allowed = ALLOWED_TRANSITIONS.get(current_status, set())
            if new_status not in allowed and current_status != new_status:
                raise ValueError(
                    f"Invalid status transition from '{current_status}' to '{new_status}'."
                )

        target_deadline = (
            new_deadline if new_deadline is not None else current_deadline
        )
        target_deadline_date = parse_deadline_date(target_deadline)

        cursor.execute(
            """
            UPDATE commitments
            SET status = ?, deadline = ?, deadline_date = ?, updated_at = ?
            WHERE id = ?
        """,
            (
                new_status,
                target_deadline,
                target_deadline_date,
                now_iso,
                commitment_id,
            ),
        )

        # Record History
        cursor.execute(
            """
            INSERT INTO commitment_history (commitment_id, previous_status, new_status, previous_deadline, new_deadline, reason_or_notes, conversation_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                commitment_id,
                current_status,
                new_status,
                current_deadline,
                target_deadline,
                reason_or_notes,
                conversation_id,
                now_iso,
            ),
        )

        conn.commit()

    return get_commitment_by_id(commitment_id, db_path=db_path)


def update_overdue_commitments(
    current_date: Optional[date] = None, db_path: str = DEFAULT_DB_PATH
) -> int:
    """Evaluates commitments with explicit deadline dates and marks them overdue if past deadline.

    Note: Commitments with missing/unknown deadlines are never marked overdue.
    """
    init_db(db_path)
    if current_date is None:
        current_date = date.today()

    today_iso = current_date.isoformat()
    now_iso = datetime.now().isoformat()
    count = 0

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, status, deadline FROM commitments
            WHERE deadline_date IS NOT NULL
              AND deadline_date < ?
              AND status IN ('explicit', 'pending', 'unclear', 'renegotiated')
        """,
            (today_iso,),
        )
        overdue_rows = cursor.fetchall()

        for row in overdue_rows:
            comm_id = row["id"]
            prev_status = row["status"]

            cursor.execute(
                """
                UPDATE commitments
                SET status = 'overdue', updated_at = ?
                WHERE id = ?
            """,
                (now_iso, comm_id),
            )

            cursor.execute(
                """
                INSERT INTO commitment_history (commitment_id, previous_status, new_status, previous_deadline, new_deadline, reason_or_notes, timestamp)
                VALUES (?, ?, 'overdue', ?, ?, 'Automatically marked overdue as deadline date passed', ?)
            """,
                (
                    comm_id,
                    prev_status,
                    row["deadline"],
                    row["deadline"],
                    now_iso,
                ),
            )
            count += 1

        conn.commit()
    return count


def get_commitment_by_id(
    commitment_id: str, db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM commitments WHERE id = ?", (commitment_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Commitment '{commitment_id}' not found.")

        c_dict = dict(row)

        cursor.execute(
            "SELECT quote FROM commitment_evidence WHERE commitment_id = ?",
            (commitment_id,),
        )
        c_dict["evidence"] = [r["quote"] for r in cursor.fetchall()]

        cursor.execute(
            "SELECT * FROM commitment_history WHERE commitment_id = ? ORDER BY id ASC",
            (commitment_id,),
        )
        c_dict["history"] = [dict(r) for r in cursor.fetchall()]

        return c_dict


def get_all_commitments(
    status_filter: Optional[str] = None, db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    init_db(db_path)
    update_overdue_commitments(db_path=db_path)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        if status_filter and status_filter.lower() != "all":
            cursor.execute(
                "SELECT * FROM commitments WHERE status = ? ORDER BY created_at DESC",
                (status_filter.lower(),),
            )
        else:
            cursor.execute("SELECT * FROM commitments ORDER BY created_at DESC")

        rows = cursor.fetchall()
        results = []

        for row in rows:
            c_dict = dict(row)
            cursor.execute(
                "SELECT quote FROM commitment_evidence WHERE commitment_id = ?",
                (row["id"],),
            )
            c_dict["evidence"] = [r["quote"] for r in cursor.fetchall()]
            results.append(c_dict)

        return results


def get_conversation_by_id(
    conversation_id: str, db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Conversation '{conversation_id}' not found.")

        conv_dict = dict(row)

        cursor.execute(
            "SELECT * FROM commitments WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )
        comm_rows = cursor.fetchall()

        commitments = []
        for cr in comm_rows:
            cd = dict(cr)
            cursor.execute(
                "SELECT quote FROM commitment_evidence WHERE commitment_id = ?",
                (cr["id"],),
            )
            cd["evidence"] = [r["quote"] for r in cursor.fetchall()]
            commitments.append(cd)

        conv_dict["commitments"] = commitments
        return conv_dict


def get_all_conversations(
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations ORDER BY created_at DESC")
        return [dict(r) for r in cursor.fetchall()]


def get_active_commitments_context(
    max_items: int = 5, db_path: str = DEFAULT_DB_PATH
) -> str:
    """Retrieves a bounded list of active prior commitments for memory prompting."""
    init_db(db_path)
    update_overdue_commitments(db_path=db_path)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, description, responsible_person, deadline, status
            FROM commitments
            WHERE status IN ('explicit', 'pending', 'unclear', 'overdue', 'renegotiated')
            ORDER BY updated_at DESC
            LIMIT ?
        """,
            (max_items,),
        )
        rows = cursor.fetchall()

    if not rows:
        return ""

    lines = ["\n[PREVIOUS ACTIVE COMMITMENTS IN MEMORY]"]
    for r in rows:
        resp = r["responsible_person"] or "Unspecified"
        dl = r["deadline"] or "No explicit deadline"
        lines.append(
            f"- Commitment ID: {r['id']} | Status: [{r['status'].upper()}] | "
            f"Responsible: {resp} | Deadline: {dl} | Description: {r['description']}"
        )

    return "\n".join(lines)
