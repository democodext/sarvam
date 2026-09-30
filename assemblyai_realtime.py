"""AssemblyAI Realtime Speech-to-Text V3 Streaming Integration.

Provides secure short-lived streaming token generation and WebSocket event parsing
for AssemblyAI Universal-3.5 Pro Realtime STT (wss://streaming.assemblyai.com/v3/ws).
"""

import json
import os
from typing import Any, Dict, Optional
import requests

from mean_db import load_env

ASSEMBLYAI_TOKEN_URL = "https://streaming.assemblyai.com/v3/token"
ASSEMBLYAI_WS_BASE_URL = "wss://streaming.assemblyai.com/v3/ws"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SPEECH_MODEL = "universal-3-5-pro"
DEFAULT_EXPIRES_IN_SECONDS = 480


class AssemblyAIConfigError(Exception):
    """Raised when AssemblyAI API key is missing or invalid."""
    pass


class AssemblyAITokenError(Exception):
    """Raised when temporary token creation fails."""
    pass


def get_assemblyai_api_key() -> str:
    """Retrieves AssemblyAI API Key securely from environment."""
    if "ASSEMBLYAI_API_KEY" not in os.environ:
        load_env()
    api_key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
    if not api_key:
        raise AssemblyAIConfigError("ASSEMBLYAI_API_KEY environment variable is not configured.")
    return api_key


def create_temporary_token(
    api_key: Optional[str] = None,
    expires_in_seconds: int = DEFAULT_EXPIRES_IN_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    speech_model: str = DEFAULT_SPEECH_MODEL,
) -> Dict[str, Any]:
    """Generates a short-lived AssemblyAI streaming token for browser WebSocket connections.

    Official v3 Endpoint: GET https://streaming.assemblyai.com/v3/token?expires_in_seconds=480
    Official WebSocket: wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&speech_model=universal-3-5-pro&token=<token>

    Args:
        api_key: Optional API key override. If None, loaded from env.
        expires_in_seconds: Expiration time in seconds (default: 480s / 8 min).
        sample_rate: Audio sampling rate in Hz (default: 16000 Hz).
        speech_model: Speech model identifier (default: universal-3-5-pro).

    Returns:
        Dict containing token, expires_in_seconds, sample_rate, speech_model, and constructed WebSocket URL.
    """
    if not api_key:
        api_key = get_assemblyai_api_key()

    headers = {
        "Authorization": api_key,
    }
    params = {
        "expires_in_seconds": expires_in_seconds
    }

    try:
        response = requests.get(
            ASSEMBLYAI_TOKEN_URL,
            headers=headers,
            params=params,
            timeout=10
        )
    except Exception as e:
        raise AssemblyAITokenError(f"Network error while connecting to AssemblyAI token service: {str(e)}")

    if response.status_code != 200:
        error_msg = response.text or f"HTTP {response.status_code}"
        raise AssemblyAITokenError(f"AssemblyAI token generation failed ({response.status_code}): {error_msg}")

    data = response.json()
    token = data.get("token")
    if not token:
        raise AssemblyAITokenError("AssemblyAI token response did not contain a valid token.")

    websocket_url = (
        f"{ASSEMBLYAI_WS_BASE_URL}"
        f"?sample_rate={sample_rate}"
        f"&speech_model={speech_model}"
        f"&token={token}"
    )

    return {
        "token": token,
        "expires_in_seconds": expires_in_seconds,
        "sample_rate": sample_rate,
        "speech_model": speech_model,
        "websocket_url": websocket_url
    }


def parse_realtime_event(data: Any) -> Dict[str, Any]:
    """Parses incoming WebSocket frames from AssemblyAI V3 Realtime stream.

    Supports V3 'Begin', 'Turn' (partials & finals), 'Error', 'Termination' / 'SessionTerminated'.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return {"event_type": "unknown", "text": "", "raw": data}

    if not isinstance(data, dict):
        return {"event_type": "unknown", "text": "", "raw": data}

    msg_type = data.get("message_type") or data.get("type")

    if msg_type in ("Begin", "SessionBegins"):
        return {
            "event_type": "session_begins",
            "session_id": data.get("session_id") or data.get("id"),
            "expires_at": data.get("expires_at"),
            "text": "",
            "raw": data
        }

    if msg_type == "Turn":
        transcript = data.get("transcript", "").strip()
        end_of_turn = bool(data.get("end_of_turn", False))
        event_type = "turn_final" if end_of_turn else "turn_partial"
        return {
            "event_type": event_type,
            "text": transcript,
            "end_of_turn": end_of_turn,
            "turn_is_formatted": data.get("turn_is_formatted", True),
            "raw": data
        }

    if msg_type == "PartialTranscript":
        return {
            "event_type": "turn_partial",
            "text": data.get("text", "").strip(),
            "end_of_turn": False,
            "raw": data
        }

    if msg_type == "FinalTranscript":
        return {
            "event_type": "turn_final",
            "text": data.get("text", "").strip(),
            "end_of_turn": True,
            "raw": data
        }

    if msg_type in ("Error", "SessionError"):
        return {
            "event_type": "error",
            "error": data.get("error") or data.get("message") or "Unknown AssemblyAI stream error",
            "text": "",
            "raw": data
        }

    if msg_type in ("Termination", "SessionTerminated"):
        return {
            "event_type": "session_terminated",
            "text": "",
            "raw": data
        }

    return {
        "event_type": "unknown",
        "text": data.get("transcript") or data.get("text") or "",
        "raw": data
    }
