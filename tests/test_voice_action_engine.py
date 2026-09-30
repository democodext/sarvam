"""Unit tests for Voice Action Engine (Intent Extraction, Confirmation Gate, and SQLite execution)."""

import os
import sqlite3
import pytest

from mean_db import get_all_commitments, init_db, save_conversation_analysis
from voice_action_engine import (
    PENDING_ACTIONS,
    cancel_pending_action,
    confirm_pending_action,
    extract_core_tokens,
    extract_task_target,
    parse_voice_intent,
    process_voice_transcript,
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_voice_actions.db")
    init_db(db_file)
    yield db_file


def test_intent_parsing_list_commitments():
    intent = parse_voice_intent("show all my pending commitments")
    assert intent.action == "list_commitments"
    assert intent.status_filter == "pending"


def test_intent_parsing_create_commitment():
    intent = parse_voice_intent("I promise to deliver the video editing by Friday afternoon")
    assert intent.action == "create_commitment"
    assert intent.description == "I promise to deliver the video editing by Friday afternoon"
    assert intent.deadline is not None


def test_intent_parsing_complete_commitment_matching(temp_db):
    # Seed a commitment in temp_db
    conv_data = {
        "summary": "Setup test",
        "commitments": [
            {
                "id": "comm_test_1",
                "description": "Finalize video render",
                "responsible_person": "Editor",
                "deadline": "Friday",
                "status": "explicit"
            }
        ]
    }
    save_conversation_analysis("Context", "Transcript", conv_data, db_path=temp_db)

    intent = parse_voice_intent("Mark finalize video render as completed", db_path=temp_db)
    assert intent.action == "complete_commitment"
    assert intent.commitment_id == "comm_test_1"


def test_process_voice_transcript_create_requires_confirmation(temp_db):
    res = process_voice_transcript("I will send the report by tomorrow", db_path=temp_db)
    assert res["requires_confirmation"] is True
    assert res["status"] == "pending_confirmation"
    assert "action_id" in res
    
    # Verify DB has NOT been mutated before confirmation!
    comms = get_all_commitments(db_path=temp_db)
    assert len(comms) == 0


def test_confirm_pending_create_commitment(temp_db):
    res = process_voice_transcript("I will send the report by tomorrow", db_path=temp_db)
    action_id = res["action_id"]

    # Confirm action
    confirm_res = confirm_pending_action(action_id, db_path=temp_db)
    assert confirm_res["status"] == "success"
    assert "Successfully created commitment" in confirm_res["voice_response"]

    # Verify DB NOW has the created commitment!
    comms = get_all_commitments(db_path=temp_db)
    assert len(comms) == 1
    assert "send the report" in comms[0]["description"]


def test_cancel_pending_action(temp_db):
    res = process_voice_transcript("I will send the report by tomorrow", db_path=temp_db)
    action_id = res["action_id"]

    cancel_res = cancel_pending_action(action_id)
    assert cancel_res["status"] == "cancelled"

    # Verify DB has NOT been mutated
    comms = get_all_commitments(db_path=temp_db)
    assert len(comms) == 0

    # Attempting to confirm cancelled action raises error
    with pytest.raises(ValueError) as exc:
        confirm_pending_action(action_id, db_path=temp_db)
    assert "cancelled" in str(exc.value)


def test_duplicate_confirmation_prevention(temp_db):
    res = process_voice_transcript("I will send the report by tomorrow", db_path=temp_db)
    action_id = res["action_id"]

    confirm_pending_action(action_id, db_path=temp_db)

    # Second confirmation attempt must fail
    with pytest.raises(ValueError) as exc:
        confirm_pending_action(action_id, db_path=temp_db)
    assert "already been executed" in str(exc.value)


def test_invalid_action_id(temp_db):
    with pytest.raises(ValueError) as exc:
        confirm_pending_action("act_non_existent", db_path=temp_db)
    assert "Invalid or expired" in str(exc.value)


def test_ambiguous_transcript_handling(temp_db):
    res = process_voice_transcript("hello there", db_path=temp_db)
    assert res["requires_confirmation"] is False
    assert res["status"] == "clarification_needed"
    assert "Could not determine intended action" in res["voice_response"]


def test_multiple_commitment_matching_ambiguity(temp_db):
    conv_data = {
        "summary": "Setup multiple commitments",
        "commitments": [
            {"id": "comm_m1", "description": "Send video preview", "status": "explicit"},
            {"id": "comm_m2", "description": "Send video final render", "status": "explicit"}
        ]
    }
    save_conversation_analysis("Context", "Transcript", conv_data, db_path=temp_db)

    intent = parse_voice_intent("Mark send video as completed", db_path=temp_db)
    assert intent.action == "complete_commitment"
    assert intent.clarification_needed is not None
    assert "Multiple commitments match" in intent.clarification_needed


def test_expired_pending_action_ttl(temp_db):
    from datetime import datetime, timedelta
    res = process_voice_transcript("I will send the report by tomorrow", db_path=temp_db)
    action_id = res["action_id"]

    # Simulate expiration (11 minutes ago)
    PENDING_ACTIONS[action_id]["created_at"] = datetime.now() - timedelta(minutes=11)

    with pytest.raises(ValueError) as exc:
        confirm_pending_action(action_id, db_path=temp_db)
    assert "Invalid or expired" in str(exc.value)


def test_database_failure_during_confirmation(monkeypatch, temp_db):
    res = process_voice_transcript("I will send the report by tomorrow", db_path=temp_db)
    action_id = res["action_id"]

    def mock_db_fail(*args, **kwargs):
        raise sqlite3.OperationalError("Database disk I/O error")

    monkeypatch.setattr("voice_action_engine.save_conversation_analysis", mock_db_fail)

    with pytest.raises(sqlite3.OperationalError):
        confirm_pending_action(action_id, db_path=temp_db)


def test_cross_session_unauthorized_confirmation(temp_db):
    res = process_voice_transcript("I will send the report by tomorrow", session_id="sess_user_A", db_path=temp_db)
    action_id = res["action_id"]

    # Unauthorized session B attempting to confirm User A's action
    with pytest.raises(ValueError) as exc:
        confirm_pending_action(action_id, session_id="sess_user_B", db_path=temp_db)
    assert "Unauthorized" in str(exc.value)

    # Valid session A confirming
    confirm_res = confirm_pending_action(action_id, session_id="sess_user_A", db_path=temp_db)
    assert confirm_res["status"] == "success"


def test_user_reported_bug_remind_me_to_finish_is_create(temp_db):
    """'Remind me to finish my maths assignment tomorrow at 10 AM' must be classified as CREATE_COMMITMENT."""
    intent = parse_voice_intent("Remind me to finish my maths assignment tomorrow at 10 AM", db_path=temp_db)
    assert intent.action == "create_commitment"
    assert "maths assignment" in intent.description.lower()
    assert intent.deadline is not None


def test_intent_parsing_create_commitment_with_complete_in_description(temp_db):
    """'Create a commitment to complete my Maths assignment tomorrow at 10 AM' must be CREATE_COMMITMENT."""
    intent = parse_voice_intent("Create a commitment to complete my Maths assignment tomorrow at 10 AM", db_path=temp_db)
    assert intent.action == "create_commitment"
    assert "maths assignment" in intent.description.lower()
    assert intent.deadline is not None



def test_extract_task_target_helper():
    assert extract_task_target("Mark my Maths assignment as completed") == "Maths assignment"
    assert extract_task_target("Complete the quarterly financial report") == "quarterly financial report"
    assert extract_task_target("Mark commitment finish video edit as done") == "finish video edit"


def test_intent_parsing_mark_as_completed_matching_deadline_commitment(temp_db):
    """'Mark my Maths assignment as completed' must match 'Finish my maths assignment tomorrow at 10 AM'."""
    conv_data = {
        "summary": "Maths assignment commitment",
        "commitments": [
            {
                "id": "comm_maths_999",
                "description": "Finish my maths assignment tomorrow at 10 AM",
                "responsible_person": "Student",
                "deadline": "Tomorrow at 10 AM",
                "status": "explicit"
            }
        ]
    }
    save_conversation_analysis("Context", "Transcript", conv_data, db_path=temp_db)

    intent = parse_voice_intent("Mark my Maths assignment as completed", db_path=temp_db)
    assert intent.action == "complete_commitment"
    assert intent.commitment_id == "comm_maths_999"
    assert intent.description == "Finish my maths assignment tomorrow at 10 AM"


def test_intent_parsing_mark_as_completed_non_matching(temp_db):
    """Attempting to complete a non-existent commitment returns clarification_needed."""
    conv_data = {
        "summary": "Maths assignment commitment",
        "commitments": [
            {
                "id": "comm_maths_101",
                "description": "Complete maths assignment",
                "responsible_person": "Student",
                "deadline": "Tomorrow",
                "status": "explicit"
            }
        ]
    }
    save_conversation_analysis("Context", "Transcript", conv_data, db_path=temp_db)

    intent = parse_voice_intent("Mark physics lab report as completed", db_path=temp_db)
    assert intent.action == "complete_commitment"
    assert intent.commitment_id is None
    assert intent.clarification_needed is not None
    assert "Could not find an active commitment" in intent.clarification_needed


def test_intent_parsing_mark_as_completed_ambiguous_matches(temp_db):
    """When multiple active commitments match the target, ask for clarification."""
    conv_data = {
        "summary": "Multiple maths commitments",
        "commitments": [
            {"id": "comm_m1", "description": "Finish maths algebra homework", "status": "explicit"},
            {"id": "comm_m2", "description": "Finish maths geometry homework", "status": "explicit"}
        ]
    }
    save_conversation_analysis("Context", "Transcript", conv_data, db_path=temp_db)

    intent = parse_voice_intent("Mark maths homework as completed", db_path=temp_db)
    assert intent.action == "complete_commitment"
    assert intent.commitment_id is None
    assert intent.clarification_needed is not None
    assert "Multiple commitments match" in intent.clarification_needed


def test_complete_commitment_confirmation_gate_non_mutation(temp_db):
    """Complete commitment flow must require confirmation and NOT mutate DB until confirmed."""
    conv_data = {
        "summary": "Setup commitment",
        "commitments": [
            {
                "id": "comm_gate_1",
                "description": "Finish my maths assignment tomorrow at 10 AM",
                "responsible_person": "Student",
                "deadline": "Tomorrow",
                "status": "explicit"
            }
        ]
    }
    save_conversation_analysis("Context", "Transcript", conv_data, db_path=temp_db)

    # Process completion transcript
    res = process_voice_transcript("Mark my Maths assignment as completed", db_path=temp_db)
    assert res["requires_confirmation"] is True
    assert res["status"] == "pending_confirmation"
    action_id = res["action_id"]

    # Verify DB commitment status is STILL 'explicit' before confirmation!
    comms = get_all_commitments(db_path=temp_db)
    target_comm = [c for c in comms if c["id"] == "comm_gate_1"][0]
    assert target_comm["status"] == "explicit"

    # Confirm action
    confirm_res = confirm_pending_action(action_id, db_path=temp_db)
    assert confirm_res["status"] == "success"

    # Verify DB commitment status NOW is 'completed'!
    comms_after = get_all_commitments(db_path=temp_db)
    target_comm_after = [c for c in comms_after if c["id"] == "comm_gate_1"][0]
    assert target_comm_after["status"] == "completed"



