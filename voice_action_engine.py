"""Voice Action Engine for MEAN AI.

Parses finalized voice transcripts into structured intents:
- create_commitment
- list_commitments
- complete_commitment

Manages server-side pending action confirmation states with TTL expiry,
prevents duplicate execution, and executes database mutations upon user confirmation.
"""

from datetime import datetime, timedelta
import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from mean_db import (
    DEFAULT_DB_PATH,
    get_all_commitments,
    get_commitment_by_id,
    save_conversation_analysis,
    update_commitment_status,
)
from mean_engine import load_env

# Server-side pending action cache: {action_id: PendingActionDict}
PENDING_ACTIONS: Dict[str, Dict[str, Any]] = {}
ACTION_TTL_SECONDS = 600  # 10 minutes expiry


class VoiceIntent(BaseModel):
    action: str = Field(..., description="Detected action: create_commitment, list_commitments, complete_commitment, or unknown")
    description: Optional[str] = Field(None, description="Commitment description")
    responsible_person: Optional[str] = Field(None, description="Person responsible")
    deadline: Optional[str] = Field(None, description="Deadline or timeline")
    status_filter: Optional[str] = Field("all", description="Status filter for list_commitments")
    commitment_id: Optional[str] = Field(None, description="Target commitment ID for complete_commitment")
    target_query: Optional[str] = Field(None, description="Target search description for commitment matching")
    reason: Optional[str] = Field(None, description="Reason or notes")
    clarification_needed: Optional[str] = Field(None, description="Clarification message if intent is ambiguous")


def clean_expired_pending_actions() -> None:
    """Removes expired pending actions from server memory."""
    now = datetime.now()
    expired_ids = []
    for action_id, item in PENDING_ACTIONS.items():
        created_at = item.get("created_at")
        if isinstance(created_at, datetime) and (now - created_at).total_seconds() > ACTION_TTL_SECONDS:
            expired_ids.append(action_id)
    for aid in expired_ids:
        PENDING_ACTIONS.pop(aid, None)


STOPWORDS_AND_NOISE = {
    "a", "an", "the", "my", "our", "your", "his", "her", "their", "this", "that",
    "task", "tasks", "commitment", "commitments", "deliverable", "deliverables",
    "to", "is", "as", "for", "of", "in", "on", "at", "by", "tak", "before",
    "finish", "finishing", "complete", "completed", "completing", "do", "done", "doing",
    "mark", "marking", "close", "closing", "set", "setting",
    "tomorrow", "today", "yesterday", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "morning", "afternoon", "evening", "night",
    "am", "pm", "oclock", "o'clock",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"
}


def normalize_word(word: str) -> str:
    """Normalizes word variations such as math/maths and plurals."""
    w = word.lower()
    if w in ("math", "maths", "mathematics"):
        return "math"
    if w in ("assignment", "assignments"):
        return "assignment"
    if w in ("task", "tasks"):
        return "task"
    if w in ("commitment", "commitments"):
        return "commitment"
    return w


def extract_core_tokens(text: str) -> set:
    """Extracts normalized core topic tokens from text, filtering action verbs, possessives, and deadline noise."""
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    tokens = set()
    for w in words:
        norm = normalize_word(w)
        if norm not in STOPWORDS_AND_NOISE and len(norm) > 1:
            tokens.add(norm)
    return tokens


def extract_task_target(transcript: str) -> str:
    """Extracts target task description from completion command transcript."""
    text = transcript.strip()
    # Clean leading action prefix
    text = re.sub(
        r"^(please\s+)?(mark|set|complete|finish|close|done)\s+(the\s+|my\s+|our\s+|your\s+|a\s+|an\s+)?(commitment\s+|task\s+|deliverable\s+)?",
        "",
        text,
        flags=re.IGNORECASE
    ).strip()
    # Clean trailing completion suffix
    text = re.sub(
        r"\s+(as|to|is)?\s*(completed|done|finished)$",
        "",
        text,
        flags=re.IGNORECASE
    ).strip()
    # Clean leading possessives/articles if still present
    text = re.sub(r"^(my|the|a|an|our|your)\s+", "", text, flags=re.IGNORECASE).strip()
    return text


def parse_voice_intent(transcript: str, db_path: str = DEFAULT_DB_PATH) -> VoiceIntent:
    """Parses a transcript into a structured VoiceIntent using deterministic pattern matching and fuzzy commitment matching."""
    text = transcript.strip().lower()
    if not text:
        return VoiceIntent(action="unknown", clarification_needed="Transcript is empty. Please speak your command.")

    # 1. LIST COMMITMENTS intent patterns
    list_patterns = [
        r"\b(list|show|get|view|display|fetch)\b.*\b(commitment|commitments|task|tasks|deliverable|deliverables)\b",
        r"\bwhat are (my|the) (commitments|tasks|deliverables)\b",
        r"\bshow my tasks\b",
    ]
    for pattern in list_patterns:
        if re.search(pattern, text):
            status_filter = "all"
            if "pending" in text or "unclear" in text:
                status_filter = "pending"
            elif "completed" in text or "done" in text:
                status_filter = "completed"
            elif "overdue" in text:
                status_filter = "overdue"
            elif "explicit" in text:
                status_filter = "explicit"
            return VoiceIntent(action="list_commitments", status_filter=status_filter)

    # 2. CREATE / REMIND INTENT (High Priority)
    # If the command explicitly requests creating a task, setting a reminder, or making a commitment
    create_intent_patterns = [
        r"\b(remind me|set a reminder|create a commitment|create a task|add a commitment|add a task|create commitment|add commitment)\b",
        r"\b(i promise to|i agree to|i commit to)\b",
        r"^(please\s+)?(create|add|remind|set)\b",
    ]
    is_create_action = False
    for pattern in create_intent_patterns:
        if re.search(pattern, text):
            is_create_action = True
            break

    # 3. COMPLETE COMMITMENT INTENT (Strict completion triggers)
    complete_patterns = [
        r"\bmark\b.+\b(as|to|is)?\s*(completed|done|finished)\b",
        r"\b(mark|set)\b.*\b(as completed|as done|as finished)\b",
        r"\b(complete|close|finish)\s+(the\s+)?(commitment|task|deliverable)\b",
        r"\b(complete|close|finish)\s+(comm_[a-zA-Z0-9_]+)\b",
        r"\bmark\s+(comm_[a-zA-Z0-9_]+)\s+as\s+completed\b",
    ]
    is_complete_action = False
    if not is_create_action:
        for pattern in complete_patterns:
            if re.search(pattern, text):
                is_complete_action = True
                break

    # Handle COMPLETE COMMITMENT Action
    if is_complete_action:
        target_query = extract_task_target(transcript)

        # Check if direct commitment_id mentioned (e.g. comm_12345)
        comm_match = re.search(r"\b(comm_[a-zA-Z0-9_]+)\b", transcript, re.IGNORECASE)
        if comm_match:
            comm_id = comm_match.group(1)
            comm = get_commitment_by_id(comm_id, db_path=db_path)
            if comm:
                return VoiceIntent(
                    action="complete_commitment",
                    commitment_id=comm["id"],
                    description=comm["description"],
                    reason="Completed via voice agent"
                )
            else:
                return VoiceIntent(
                    action="complete_commitment",
                    commitment_id=comm_id,
                    clarification_needed=f"Commitment with ID '{comm_id}' was not found in the database."
                )

        # Match target_query against active commitments in database
        active_commitments = get_all_commitments(status_filter="all", db_path=db_path)
        non_completed = [c for c in active_commitments if c.get("status") not in ("completed", "cancelled")]
        search_pool = non_completed if non_completed else active_commitments

        target_tokens = extract_core_tokens(target_query)
        exact_matches = []
        token_matches = []
        substring_matches = []

        query_clean = target_query.lower()
        for c in search_pool:
            desc = c["description"]
            desc_lower = desc.lower()

            # 1. Direct or normalized string equality
            if query_clean and (query_clean == desc_lower or query_clean in desc_lower):
                exact_matches.append(c)
                continue

            # 2. Core token set matching
            desc_tokens = extract_core_tokens(desc)
            if target_tokens and (target_tokens == desc_tokens or target_tokens.issubset(desc_tokens)):
                token_matches.append(c)
                continue

            # 3. Substring match
            if query_clean and desc_lower in query_clean:
                substring_matches.append(c)
                continue

        # Priority resolution
        if len(exact_matches) > 0:
            candidates = exact_matches
        elif len(token_matches) > 0:
            candidates = token_matches
        else:
            candidates = substring_matches

        # Deduplicate
        unique_matches = []
        seen_ids = set()
        for m in candidates:
            if m["id"] not in seen_ids:
                seen_ids.add(m["id"])
                unique_matches.append(m)

        if len(unique_matches) == 1:
            matched = unique_matches[0]
            return VoiceIntent(
                action="complete_commitment",
                commitment_id=matched["id"],
                description=matched["description"],
                reason="Completed via voice command"
            )
        elif len(unique_matches) > 1:
            match_descriptions = ", ".join([f"'{c['description']}' ({c['id']})" for c in unique_matches[:3]])
            return VoiceIntent(
                action="complete_commitment",
                target_query=target_query,
                clarification_needed=f"Multiple commitments match '{target_query}': {match_descriptions}. Please specify the exact commitment ID or description."
            )
        else:
            return VoiceIntent(
                action="complete_commitment",
                target_query=target_query,
                clarification_needed=f"Could not find an active commitment matching '{target_query or transcript}'. Please check your commitment list."
            )

    # Handle CREATE COMMITMENT Action
    create_keywords = ["will", "promise", "agree", "commit", "send", "deliver", "finish", "complete", "create", "add", "remind", "by", "deadline", "tomorrow", "today"]
    if is_create_action or any(kw in text for kw in create_keywords):
        deadline = None
        time_match = re.search(r"\b(tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)(\s+at\s+\d{1,2}(:\d{2})?\s*(am|pm)?)?\b", text, re.IGNORECASE)
        if time_match:
            deadline = time_match.group(0).strip().title()
        else:
            dl_match = re.search(r"\b(by|tak|before|on|at)\s+([a-zA-Z0-9\s:]+?)(?=\.|$)", text, re.IGNORECASE)
            if dl_match:
                deadline = dl_match.group(0).strip().title()

        responsible_person = "User"
        person_match = re.search(r"\b(rahul|editor|client|alex|priya|john|me|i)\b", text, re.IGNORECASE)
        if person_match:
            p = person_match.group(1).title()
            responsible_person = "Me" if p.lower() in ("i", "me") else p

        clean_desc = transcript.strip()
        clean_desc = re.sub(r"^(please\s+)?(remind me to|create a commitment to|create a task to|add a commitment to|add a task to|create commitment to|add commitment to|set a reminder to)\s+", "", clean_desc, flags=re.IGNORECASE).strip()
        if clean_desc:
            clean_desc = clean_desc[0].upper() + clean_desc[1:]

        return VoiceIntent(
            action="create_commitment",
            description=clean_desc or transcript.strip(),
            responsible_person=responsible_person,
            deadline=deadline
        )

    # Fallback / Ambiguous
    return VoiceIntent(
        action="unknown",
        clarification_needed=f"Could not determine intended action from transcript: '{transcript}'. Please specify if you want to create, list, or complete a commitment."
    )


def process_voice_transcript(
    transcript: str,
    context: str = "Voice Command",
    session_id: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """Processes a finalized transcript, identifies intent, and prepares response or confirmation gate."""
    clean_expired_pending_actions()
    intent = parse_voice_intent(transcript, db_path=db_path)

    if intent.action == "unknown" or intent.clarification_needed:
        return {
            "status": "clarification_needed",
            "action_type": intent.action,
            "voice_response": intent.clarification_needed or "Please clarify your request.",
            "requires_confirmation": False
        }

    # Action 1: LIST COMMITMENTS (Read-only, no confirmation needed)
    if intent.action == "list_commitments":
        commitments = get_all_commitments(status_filter=intent.status_filter or "all", db_path=db_path)
        count = len(commitments)
        status_str = intent.status_filter if intent.status_filter != "all" else "total"
        msg = f"Found {count} {status_str} commitment{'s' if count != 1 else ''} in your tracker."
        return {
            "status": "executed",
            "action_type": "list_commitments",
            "commitments": commitments,
            "voice_response": msg,
            "requires_confirmation": False
        }

    # Action 2: CREATE COMMITMENT (Mutation requires confirmation)
    if intent.action == "create_commitment":
        if not intent.description:
            return {
                "status": "clarification_needed",
                "action_type": "create_commitment",
                "voice_response": "Commitment description is missing. Please provide what task needs to be created.",
                "requires_confirmation": False
            }

        action_id = f"act_{uuid.uuid4().hex[:10]}"
        pending_payload = {
            "action_id": action_id,
            "session_id": session_id,
            "action_type": "create_commitment",
            "transcript": transcript,
            "context": context,
            "description": intent.description,
            "responsible_person": intent.responsible_person or "User",
            "deadline": intent.deadline,
            "created_at": datetime.now(),
            "status": "pending"
        }
        PENDING_ACTIONS[action_id] = pending_payload

        deadline_str = f" with deadline {intent.deadline}" if intent.deadline else ""
        voice_resp = f"Proposed new commitment: '{intent.description}'{deadline_str}. Would you like to confirm and save this?"

        return {
            "status": "pending_confirmation",
            "action_id": action_id,
            "action_type": "create_commitment",
            "proposed_action": {
                "description": intent.description,
                "responsible_person": intent.responsible_person or "User",
                "deadline": intent.deadline
            },
            "voice_response": voice_resp,
            "requires_confirmation": True
        }

    # Action 3: COMPLETE COMMITMENT (Mutation requires confirmation)
    if intent.action == "complete_commitment":
        if not intent.commitment_id:
            return {
                "status": "clarification_needed",
                "action_type": "complete_commitment",
                "voice_response": intent.clarification_needed or "Please specify which commitment to complete.",
                "requires_confirmation": False
            }

        comm = get_commitment_by_id(intent.commitment_id, db_path=db_path)
        if not comm:
            return {
                "status": "clarification_needed",
                "action_type": "complete_commitment",
                "voice_response": f"Commitment '{intent.commitment_id}' not found.",
                "requires_confirmation": False
            }

        action_id = f"act_{uuid.uuid4().hex[:10]}"
        pending_payload = {
            "action_id": action_id,
            "session_id": session_id,
            "action_type": "complete_commitment",
            "commitment_id": comm["id"],
            "description": comm["description"],
            "current_status": comm["status"],
            "new_status": "completed",
            "reason": intent.reason or "Completed via voice command",
            "created_at": datetime.now(),
            "status": "pending"
        }
        PENDING_ACTIONS[action_id] = pending_payload

        voice_resp = f"Proposed change: Mark commitment '{comm['description']}' as COMPLETED. Confirm to execute."

        return {
            "status": "pending_confirmation",
            "action_id": action_id,
            "action_type": "complete_commitment",
            "proposed_action": {
                "commitment_id": comm["id"],
                "description": comm["description"],
                "current_status": comm["status"],
                "new_status": "completed"
            },
            "voice_response": voice_resp,
            "requires_confirmation": True
        }

    return {
        "status": "clarification_needed",
        "action_type": "unknown",
        "voice_response": "Unsupported action.",
        "requires_confirmation": False
    }


def confirm_pending_action(
    action_id: str,
    session_id: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """Confirms and executes a pending action server-side.

    Validates action token, verifies session authorization, prevents duplicate execution, checks expiry, and mutates SQLite DB.
    """
    clean_expired_pending_actions()

    pending = PENDING_ACTIONS.get(action_id)
    if not pending:
        raise ValueError("Invalid or expired action_id. Please re-submit your command.")

    if pending.get("session_id") and session_id and pending["session_id"] != session_id:
        raise ValueError("Unauthorized: Action token belongs to another session.")

    if pending.get("status") in ("executed", "executing"):
        raise ValueError("Action has already been executed.")

    if pending.get("status") == "cancelled":
        raise ValueError("Action was cancelled and cannot be executed.")

    # Mark as executing immediately to prevent concurrent execution
    pending["status"] = "executing"
    action_type = pending.get("action_type")

    try:
        if action_type == "create_commitment":
            analysis_data = {
                "summary": f"Voice commitment created: {pending.get('description')}",
                "explicit_statements": [pending.get("transcript")],
                "commitments": [
                    {
                        "description": pending.get("description"),
                        "responsible_person": pending.get("responsible_person"),
                        "deadline": pending.get("deadline"),
                        "status": "explicit",
                        "commitment_type": "explicit",
                        "evidence": [pending.get("transcript")]
                    }
                ],
                "ambiguities": [],
                "clarification_question": ""
            }
            conv_id = save_conversation_analysis(
                context=pending.get("context", "Voice Command"),
                transcript=pending.get("transcript", ""),
                analysis_data=analysis_data,
                db_path=db_path
            )
            pending["status"] = "executed"
            pending["executed_at"] = datetime.now()

            commitments = get_all_commitments(status_filter="all", db_path=db_path)
            return {
                "status": "success",
                "action_id": action_id,
                "action_type": "create_commitment",
                "conversation_id": conv_id,
                "commitments": commitments,
                "voice_response": f"Successfully created commitment '{pending.get('description')}'."
            }

        elif action_type == "complete_commitment":
            comm_id = pending.get("commitment_id")
            reason = pending.get("reason", "Completed via voice command")
            updated = update_commitment_status(
                commitment_id=comm_id,
                new_status="completed",
                reason_or_notes=reason,
                db_path=db_path
            )
            pending["status"] = "executed"
            pending["executed_at"] = datetime.now()

            commitments = get_all_commitments(status_filter="all", db_path=db_path)
            return {
                "status": "success",
                "action_id": action_id,
                "action_type": "complete_commitment",
                "updated_commitment": updated,
                "commitments": commitments,
                "voice_response": f"Successfully marked commitment '{pending.get('description')}' as completed."
            }
        else:
            raise ValueError(f"Unknown action type '{action_type}'.")

    except Exception as e:
        pending["status"] = "failed"
        pending["error"] = str(e)
        raise


def cancel_pending_action(action_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Cancels a pending action without mutating the database."""
    clean_expired_pending_actions()
    pending = PENDING_ACTIONS.get(action_id)
    if not pending:
        return {"status": "cancelled", "action_id": action_id, "voice_response": "Action was already cleared."}

    if pending.get("session_id") and session_id and pending["session_id"] != session_id:
        raise ValueError("Unauthorized: Action token belongs to another session.")

    pending["status"] = "cancelled"
    return {
        "status": "cancelled",
        "action_id": action_id,
        "voice_response": "Action cancelled. No database changes were made."
    }
