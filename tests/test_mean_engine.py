import os
from unittest.mock import MagicMock, patch
import pytest

from mean_engine import (
    AnalysisResult,
    ConfigurationError,
    ProviderError,
    ResponseParsingError,
    SchemaValidationError,
    analyze_conversation,
    extract_json_from_text,
)


def create_mock_sarvam_client(response_text: str):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = response_text
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.return_value = mock_response
    return mock_client


def test_extract_json_from_text():
    raw = '{"summary": "Test"}'
    assert extract_json_from_text(raw) == '{"summary": "Test"}'

    fenced = '```json\n{"summary": "Test"}\n```'
    assert extract_json_from_text(fenced) == '{"summary": "Test"}'

    preamble = 'Here is the response:\n{"summary": "Test"}\nHope this helps!'
    assert extract_json_from_text(preamble) == '{"summary": "Test"}'


def test_valid_model_response_parsing():
    valid_json = """{
        "summary": "Client and editor agreed on Friday delivery.",
        "explicit_statements": ["Editor promised Friday delivery."],
        "commitments": [
            {
                "description": "Video delivery by Friday",
                "responsible_person": "Editor",
                "deadline": "Friday",
                "status": "explicit",
                "evidence": ["Haan, Friday tak bhej dunga."],
                "commitment_type": "explicit"
            }
        ],
        "ambiguities": [
            {
                "description": "Unclear if morning delivery was accepted.",
                "why_it_matters": "Client asked for morning delivery, editor gave vague response.",
                "evidence": ["Haan bhai, dekh lenge."],
                "alternative_interpretations": ["Editor will try for morning", "Editor only commits to end of day Friday"]
            }
        ],
        "clarification_question": "Kya tum Friday morning tak video deliver kar paoge?",
        "limitations": ["No prior discussion context available."]
    }"""
    mock_client = create_mock_sarvam_client(valid_json)
    result = analyze_conversation(
        context="Video editing discussion",
        conversation="Client: Friday?\nEditor: Haan",
        client=mock_client,
    )

    assert result["summary"] == "Client and editor agreed on Friday delivery."
    assert len(result["commitments"]) == 1
    assert result["commitments"][0]["deadline"] == "Friday"
    assert result["commitments"][0]["status"] == "explicit"
    assert len(result["ambiguities"]) == 1
    assert (
        result["clarification_question"]
        == "Kya tum Friday morning tak video deliver kar paoge?"
    )


def test_invalid_or_malformed_output():
    invalid_json = "This is completely invalid and not JSON."
    mock_client = create_mock_sarvam_client(invalid_json)

    with pytest.raises(ResponseParsingError):
        analyze_conversation(
            context="Test", conversation="Test", client=mock_client
        )


def test_schema_validation_error():
    incomplete_json = """{
        "summary": "Missing required clarification_question field",
        "explicit_statements": [],
        "commitments": [],
        "ambiguities": [],
        "limitations": []
    }"""
    mock_client = create_mock_sarvam_client(incomplete_json)

    with pytest.raises(SchemaValidationError):
        analyze_conversation(
            context="Test", conversation="Test", client=mock_client
        )


def test_explicit_vs_unclear_commitments():
    data = """{
        "summary": "Discussion with explicit and vague commitments",
        "explicit_statements": [],
        "commitments": [
            {
                "description": "Submit report",
                "responsible_person": "Alice",
                "deadline": "Monday",
                "status": "explicit",
                "evidence": ["I will submit the report on Monday."],
                "commitment_type": "explicit"
            },
            {
                "description": "Fix minor bugs",
                "responsible_person": "Bob",
                "deadline": null,
                "status": "unclear",
                "evidence": ["I might look into it later."],
                "commitment_type": "implicit"
            }
        ],
        "ambiguities": [],
        "clarification_question": "When will Bob fix the minor bugs?",
        "limitations": []
    }"""
    mock_client = create_mock_sarvam_client(data)
    result = analyze_conversation(
        context="Project update", conversation="...", client=mock_client
    )

    statuses = [c["status"] for c in result["commitments"]]
    assert "explicit" in statuses
    assert "unclear" in statuses


def test_preservation_of_original_deadline():
    """Verifies that a vague follow-up response does not overwrite or cancel the original explicit deadline."""
    data = """{
        "summary": "Editor confirmed Friday delivery, vague on morning request.",
        "explicit_statements": ["Editor promises Friday delivery."],
        "commitments": [
            {
                "description": "Deliver final video edit",
                "responsible_person": "Editor",
                "deadline": "Friday",
                "status": "explicit",
                "evidence": ["Haan, Friday tak bhej dunga."],
                "commitment_type": "explicit"
            }
        ],
        "ambiguities": [
            {
                "description": "Specific time of day on Friday is unconfirmed.",
                "why_it_matters": "Friday morning delivery vs end of day Friday.",
                "evidence": ["Haan bhai, dekh lenge."],
                "alternative_interpretations": [
                    "Editor agrees to attempt morning delivery",
                    "Editor will deliver by end of day Friday"
                ]
            }
        ],
        "clarification_question": "Kya tum Friday morning tak video deliver kar paoge?",
        "limitations": []
    }"""
    mock_client = create_mock_sarvam_client(data)
    result = analyze_conversation(
        context="Delivery check", conversation="...", client=mock_client
    )

    commitment = result["commitments"][0]
    assert commitment["deadline"] == "Friday"
    assert commitment["status"] == "explicit"
    assert "Haan, Friday tak bhej dunga." in commitment["evidence"]


def test_no_meaningful_ambiguity():
    data = """{
        "summary": "Clear agreement with no unresolved points.",
        "explicit_statements": ["Meeting scheduled for 3 PM."],
        "commitments": [
            {
                "description": "Attend meeting",
                "responsible_person": "Both",
                "deadline": "3 PM today",
                "status": "explicit",
                "evidence": ["Let's meet at 3 PM today.", "Sounds good, see you at 3 PM."],
                "commitment_type": "explicit"
            }
        ],
        "ambiguities": [],
        "clarification_question": "None needed.",
        "limitations": []
    }"""
    mock_client = create_mock_sarvam_client(data)
    result = analyze_conversation(
        context="Scheduling", conversation="...", client=mock_client
    )

    assert len(result["ambiguities"]) == 0
    assert result["clarification_question"] == "None needed."


def test_missing_api_credentials():
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.path.exists", return_value=False):
            with pytest.raises(ConfigurationError):
                analyze_conversation(
                    context="Test", conversation="Test", client=None
                )


def test_provider_errors():
    mock_client = MagicMock()
    mock_client.chat.completions.side_effect = Exception("API Connection Timeout")

    with pytest.raises(ProviderError):
        analyze_conversation(
            context="Test", conversation="Test", client=mock_client
        )
