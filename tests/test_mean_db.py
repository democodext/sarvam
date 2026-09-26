from datetime import date, timedelta
import os
import tempfile
from unittest.mock import MagicMock
import pytest

from mean_db import (
    ALLOWED_TRANSITIONS,
    VALID_STATUSES,
    get_active_commitments_context,
    get_all_commitments,
    get_all_conversations,
    get_commitment_by_id,
    get_connection,
    get_conversation_by_id,
    init_db,
    parse_deadline_date,
    save_conversation_analysis,
    update_commitment_status,
    update_overdue_commitments,
)
from mean_engine import analyze_conversation


def get_temp_db_path() -> str:
    tf = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tf.close()
    return tf.name


def cleanup_db(path: str):
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def test_db_initialization_and_reconnect():
    db_path = get_temp_db_path()
    try:
        init_db(db_path)
        # Check tables exist
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN"
                " ('conversations', 'commitments', 'commitment_evidence', 'commitment_history');"
            )
            tables = [r["name"] for r in cursor.fetchall()]
            assert len(tables) == 4

        # Reconnect check
        with get_connection(db_path) as conn2:
            cursor2 = conn2.cursor()
            cursor2.execute("SELECT COUNT(*) as cnt FROM conversations;")
            assert cursor2.fetchone()["cnt"] == 0
    finally:
        cleanup_db(db_path)


def test_parse_deadline_date():
    assert parse_deadline_date("2026-10-15") == "2026-10-15"
    assert parse_deadline_date("Due on 15/10/2026") == "2026-10-15"
    assert parse_deadline_date("Friday morning") is None
    assert parse_deadline_date(None) is None


def test_save_and_retrieve_conversation():
    db_path = get_temp_db_path()
    try:
        analysis = {
            "summary": "Project deadline discussion.",
            "explicit_statements": ["Editor promises Friday."],
            "commitments": [
                {
                    "description": "Deliver final edit",
                    "responsible_person": "Editor",
                    "deadline": "2026-10-20",
                    "status": "explicit",
                    "evidence": ["Haan, Friday tak bhej dunga."],
                    "commitment_type": "explicit",
                }
            ],
            "ambiguities": [],
            "clarification_question": "None needed.",
            "limitations": [],
        }

        conv_id = save_conversation_analysis(
            context="Video delivery context",
            transcript="Client: Friday?\nEditor: Haan",
            analysis_data=analysis,
            db_path=db_path,
        )

        conv = get_conversation_by_id(conv_id, db_path=db_path)
        assert conv["id"] == conv_id
        assert conv["context"] == "Video delivery context"
        assert conv["summary"] == "Project deadline discussion."
        assert len(conv["commitments"]) == 1
        assert conv["commitments"][0]["description"] == "Deliver final edit"
        assert conv["commitments"][0]["evidence"] == ["Haan, Friday tak bhej dunga."]
    finally:
        cleanup_db(db_path)


def test_commitment_creation_and_evidence():
    db_path = get_temp_db_path()
    try:
        analysis = {
            "summary": "Test summary",
            "commitments": [
                {
                    "description": "Submit report",
                    "responsible_person": "Alice",
                    "deadline": "Monday",
                    "status": "explicit",
                    "evidence": ["I will submit report.", "Guaranteed."],
                }
            ],
            "clarification_question": "None",
        }
        save_conversation_analysis(
            context="Test",
            transcript="...",
            analysis_data=analysis,
            db_path=db_path,
        )

        commitments = get_all_commitments(db_path=db_path)
        assert len(commitments) == 1
        comm = commitments[0]
        assert comm["description"] == "Submit report"
        assert comm["responsible_person"] == "Alice"
        assert len(comm["evidence"]) == 2
        assert "I will submit report." in comm["evidence"]
    finally:
        cleanup_db(db_path)


def test_status_transition_validation():
    db_path = get_temp_db_path()
    try:
        analysis = {
            "summary": "Test",
            "commitments": [
                {
                    "description": "Draft document",
                    "status": "explicit",
                    "evidence": [],
                }
            ],
            "clarification_question": "None",
        }
        save_conversation_analysis(
            context="Test",
            transcript="...",
            analysis_data=analysis,
            db_path=db_path,
        )
        comm = get_all_commitments(db_path=db_path)[0]
        comm_id = comm["id"]

        # Valid transition to completed
        updated = update_commitment_status(
            comm_id, "completed", reason_or_notes="Done!", db_path=db_path
        )
        assert updated["status"] == "completed"

        # Check invalid transition raises ValueError
        with pytest.raises(ValueError) as excinfo:
            update_commitment_status(comm_id, "explicit", db_path=db_path)
        assert "Invalid status transition" in str(excinfo.value)
    finally:
        cleanup_db(db_path)


def test_completed_and_cancelled_commitments():
    db_path = get_temp_db_path()
    try:
        analysis = {
            "summary": "Test",
            "commitments": [
                {
                    "description": "Task 1",
                    "status": "explicit",
                    "evidence": [],
                },
                {
                    "description": "Task 2",
                    "status": "explicit",
                    "evidence": [],
                },
            ],
            "clarification_question": "None",
        }
        save_conversation_analysis(
            context="Test",
            transcript="...",
            analysis_data=analysis,
            db_path=db_path,
        )
        comms = get_all_commitments(db_path=db_path)
        id1, id2 = comms[0]["id"], comms[1]["id"]

        update_commitment_status(id1, "completed", db_path=db_path)
        update_commitment_status(id2, "cancelled", db_path=db_path)

        completed_list = get_all_commitments(
            status_filter="completed", db_path=db_path
        )
        cancelled_list = get_all_commitments(
            status_filter="cancelled", db_path=db_path
        )

        assert len(completed_list) == 1
        assert completed_list[0]["id"] == id1

        assert len(cancelled_list) == 1
        assert cancelled_list[0]["id"] == id2
    finally:
        cleanup_db(db_path)


def test_overdue_calculation():
    db_path = get_temp_db_path()
    try:
        past_date = (date.today() - timedelta(days=5)).isoformat()
        analysis = {
            "summary": "Test overdue",
            "commitments": [
                {
                    "description": "Overdue task",
                    "deadline": past_date,
                    "status": "explicit",
                    "evidence": [],
                },
                {
                    "description": "Task without date",
                    "deadline": "Friday morning",
                    "status": "explicit",
                    "evidence": [],
                },
            ],
            "clarification_question": "None",
        }
        save_conversation_analysis(
            context="Test",
            transcript="...",
            analysis_data=analysis,
            db_path=db_path,
        )

        updated_count = update_overdue_commitments(db_path=db_path)
        assert updated_count == 1

        all_comms = get_all_commitments(db_path=db_path)
        status_map = {c["description"]: c["status"] for c in all_comms}

        assert status_map["Overdue task"] == "overdue"
        # Essential requirement: missing explicit date is NEVER marked overdue
        assert status_map["Task without date"] == "explicit"
    finally:
        cleanup_db(db_path)


def test_renegotiation_and_history():
    db_path = get_temp_db_path()
    try:
        analysis = {
            "summary": "Original agreement",
            "commitments": [
                {
                    "description": "Fix feature",
                    "deadline": "2026-10-01",
                    "status": "explicit",
                    "evidence": [],
                }
            ],
            "clarification_question": "None",
        }
        save_conversation_analysis(
            context="Test",
            transcript="...",
            analysis_data=analysis,
            db_path=db_path,
        )
        comm_id = get_all_commitments(db_path=db_path)[0]["id"]

        # Renegotiate deadline
        updated = update_commitment_status(
            commitment_id=comm_id,
            new_status="renegotiated",
            new_deadline="2026-10-10",
            reason_or_notes="Extended scope",
            db_path=db_path,
        )

        assert updated["status"] == "renegotiated"
        assert updated["deadline"] == "2026-10-10"
        assert len(updated["history"]) == 2  # Created + Renegotiated
        assert updated["history"][-1]["previous_deadline"] == "2026-10-01"
        assert updated["history"][-1]["new_deadline"] == "2026-10-10"
    finally:
        cleanup_db(db_path)


def test_retrieval_of_relevant_earlier_context():
    db_path = get_temp_db_path()
    try:
        analysis = {
            "summary": "Earlier discussion",
            "commitments": [
                {
                    "description": "Video render deliverable",
                    "responsible_person": "Editor",
                    "deadline": "Friday",
                    "status": "explicit",
                    "evidence": [],
                }
            ],
            "clarification_question": "None",
        }
        save_conversation_analysis(
            context="Test",
            transcript="...",
            analysis_data=analysis,
            db_path=db_path,
        )

        mem_context = get_active_commitments_context(
            max_items=5, db_path=db_path
        )
        assert "[PREVIOUS ACTIVE COMMITMENTS IN MEMORY]" in mem_context
        assert "Video render deliverable" in mem_context
        assert "Responsible: Editor" in mem_context
    finally:
        cleanup_db(db_path)


def test_conversation_isolation():
    db_path = get_temp_db_path()
    try:
        conv1 = save_conversation_analysis(
            context="Project A",
            transcript="Transcript A",
            analysis_data={"summary": "Sum A", "commitments": []},
            db_path=db_path,
        )
        conv2 = save_conversation_analysis(
            context="Project B",
            transcript="Transcript B",
            analysis_data={"summary": "Sum B", "commitments": []},
            db_path=db_path,
        )

        res1 = get_conversation_by_id(conv1, db_path=db_path)
        res2 = get_conversation_by_id(conv2, db_path=db_path)

        assert res1["context"] == "Project A"
        assert res2["context"] == "Project B"
        assert res1["id"] != res2["id"]
    finally:
        cleanup_db(db_path)


def test_sql_parameterization_and_errors():
    db_path = get_temp_db_path()
    try:
        init_db(db_path)
        malicious_text = "'; DROP TABLE conversations; --"
        analysis = {
            "summary": malicious_text,
            "commitments": [],
            "clarification_question": "None",
        }

        conv_id = save_conversation_analysis(
            context=malicious_text,
            transcript=malicious_text,
            analysis_data=analysis,
            db_path=db_path,
        )

        conv = get_conversation_by_id(conv_id, db_path=db_path)
        assert conv["summary"] == malicious_text

        # Verify table was not dropped
        conversations = get_all_conversations(db_path=db_path)
        assert len(conversations) == 1
    finally:
        cleanup_db(db_path)


def test_existing_analysis_workflow_persists_to_db():
    db_path = get_temp_db_path()
    try:
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = """{
            "summary": "Persisted analysis summary",
            "explicit_statements": ["Statement 1"],
            "commitments": [
                {
                    "description": "Mock commitment",
                    "responsible_person": "Alice",
                    "deadline": "Tomorrow",
                    "status": "explicit",
                    "evidence": ["Evidence 1"],
                    "commitment_type": "explicit"
                }
            ],
            "ambiguities": [],
            "clarification_question": "None",
            "limitations": []
        }"""
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.return_value = mock_response

        res = analyze_conversation(
            context="DB integration test",
            conversation="Client: Hi\nEditor: Hello",
            client=mock_client,
            db_path=db_path,
        )

        assert res["summary"] == "Persisted analysis summary"
        assert "conversation_id" in res
        assert len(res["commitments"]) == 1

        # Check database persistence
        saved_conv = get_conversation_by_id(res["conversation_id"], db_path=db_path)
        assert saved_conv["summary"] == "Persisted analysis summary"
        assert len(saved_conv["commitments"]) == 1
    finally:
        cleanup_db(db_path)
