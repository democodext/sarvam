"""Unit tests for AssemblyAI Realtime STT module and API routes."""

import os
from unittest.mock import MagicMock, patch
import pytest

from assemblyai_realtime import (
    ASSEMBLYAI_TOKEN_URL,
    ASSEMBLYAI_WS_BASE_URL,
    AssemblyAIConfigError,
    AssemblyAITokenError,
    create_temporary_token,
    get_assemblyai_api_key,
    parse_realtime_event,
)
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_get_assemblyai_api_key_success(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test_key_12345")
    key = get_assemblyai_api_key()
    assert key == "test_key_12345"


def test_get_assemblyai_api_key_missing(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "")
    with pytest.raises(AssemblyAIConfigError) as exc_info:
        get_assemblyai_api_key()
    assert "ASSEMBLYAI_API_KEY environment variable is not configured" in str(exc_info.value)


@patch("assemblyai_realtime.requests.get")
def test_create_temporary_token_success(mock_get, monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "dummy_key")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"token": "sample_token_abc"}
    mock_get.return_value = mock_response

    result = create_temporary_token()
    assert result["token"] == "sample_token_abc"
    assert result["expires_in_seconds"] == 480
    assert result["speech_model"] == "universal-3-5-pro"
    
    expected_ws = "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&speech_model=universal-3-5-pro&token=sample_token_abc"
    assert result["websocket_url"] == expected_ws

    # Verify GET method, URL, params, and Authorization header sent securely
    mock_get.assert_called_once_with(
        ASSEMBLYAI_TOKEN_URL,
        headers={"Authorization": "dummy_key"},
        params={"expires_in_seconds": 480},
        timeout=10
    )


@patch("assemblyai_realtime.requests.get")
def test_create_temporary_token_api_error(mock_get, monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "dummy_key")
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized API key"
    mock_get.return_value = mock_response

    with pytest.raises(AssemblyAITokenError) as exc_info:
        create_temporary_token()
    assert "AssemblyAI token generation failed (401)" in str(exc_info.value)


def test_parse_realtime_event_session_begins():
    event = parse_realtime_event({
        "type": "Begin",
        "session_id": "sess_123",
        "expires_at": "2026-09-30T00:00:00Z"
    })
    assert event["event_type"] == "session_begins"
    assert event["session_id"] == "sess_123"


def test_parse_realtime_event_turn_partial():
    event = parse_realtime_event({
        "message_type": "Turn",
        "transcript": "Hello world",
        "end_of_turn": False
    })
    assert event["event_type"] == "turn_partial"
    assert event["text"] == "Hello world"
    assert event["end_of_turn"] is False


def test_parse_realtime_event_turn_final():
    event = parse_realtime_event({
        "message_type": "Turn",
        "transcript": "Finalized sentence here.",
        "end_of_turn": True
    })
    assert event["event_type"] == "turn_final"
    assert event["text"] == "Finalized sentence here."
    assert event["end_of_turn"] is True


def test_parse_realtime_event_error():
    event = parse_realtime_event({
        "type": "Error",
        "error": "Audio decoding error"
    })
    assert event["event_type"] == "error"
    assert event["error"] == "Audio decoding error"


def test_parse_realtime_event_termination():
    event = parse_realtime_event({
        "type": "Termination"
    })
    assert event["event_type"] == "session_terminated"


@patch("app.create_temporary_token")
def test_api_assemblyai_token_route_success(mock_create_token, client):
    mock_create_token.return_value = {
        "token": "tok_123",
        "expires_in_seconds": 480,
        "websocket_url": "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&speech_model=universal-3-5-pro&token=tok_123",
        "sample_rate": 16000,
        "speech_model": "universal-3-5-pro"
    }
    response = client.post("/api/assemblyai/token")
    assert response.status_code == 200
    data = response.get_json()
    assert data["token"] == "tok_123"
    assert "streaming.assemblyai.com/v3/ws" in data["websocket_url"]


def test_api_assemblyai_token_missing_key(client, monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "")
    response = client.post("/api/assemblyai/token")
    assert response.status_code == 400
    data = response.get_json()
    assert "ASSEMBLYAI_API_KEY" in data["error"]
