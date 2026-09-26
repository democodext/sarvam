import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from mean_engine import analyze_conversation
from main import handle_audio_workflow
from sarvam_transcription import (
    AudioValidationError,
    TranscriptionAuthError,
    TranscriptionProviderError,
    transcribe_audio,
    validate_audio_file,
)


def create_mock_stt_response(
    transcript="Client: Video Friday tak mil jayega? Editor: Haan.",
    language_code="hi-IN",
    language_probability=0.98,
    request_id="req_12345",
):
    mock_response = MagicMock()
    mock_response.transcript = transcript
    mock_response.language_code = language_code
    mock_response.language_probability = language_probability
    mock_response.request_id = request_id
    return mock_response


def create_temp_wav_file(content=b"RIFF1234WAVEfmt ", extension=".wav"):
    tf = tempfile.NamedTemporaryFile(suffix=extension, delete=False)
    tf.write(content)
    tf.close()
    return tf.name


def test_successful_transcription_mock():
    wav_path = create_temp_wav_file()
    try:
        mock_client = MagicMock()
        mock_client.speech_to_text.transcribe.return_value = (
            create_mock_stt_response()
        )

        result = transcribe_audio(file_path=wav_path, client=mock_client)

        assert (
            result["transcript"]
            == "Client: Video Friday tak mil jayega? Editor: Haan."
        )
        assert result["language_code"] == "hi-IN"
        assert result["language_probability"] == 0.98
        assert result["request_id"] == "req_12345"
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def test_empty_transcript():
    wav_path = create_temp_wav_file()
    try:
        mock_client = MagicMock()
        mock_client.speech_to_text.transcribe.return_value = (
            create_mock_stt_response(transcript="")
        )

        result = transcribe_audio(file_path=wav_path, client=mock_client)

        assert result["transcript"] == ""
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def test_unsupported_file_extension():
    txt_path = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    txt_path.write(b"Hello world")
    txt_path.close()

    try:
        with pytest.raises(AudioValidationError) as excinfo:
            validate_audio_file(txt_path.name)
        assert "Unsupported file extension '.txt'" in str(excinfo.value)
    finally:
        if os.path.exists(txt_path.name):
            os.remove(txt_path.name)


def test_missing_or_invalid_file():
    with pytest.raises(AudioValidationError) as excinfo:
        validate_audio_file("non_existent_file_xyz.wav")
    assert "Audio file not found" in str(excinfo.value)


def test_oversized_file():
    wav_path = create_temp_wav_file()
    try:
        with patch("os.path.getsize", return_value=30 * 1024 * 1024):
            with pytest.raises(AudioValidationError) as excinfo:
                validate_audio_file(wav_path)
            assert "exceeds maximum supported limit" in str(excinfo.value)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def test_authentication_provider_failure():
    wav_path = create_temp_wav_file()
    try:
        mock_client = MagicMock()
        mock_client.speech_to_text.transcribe.side_effect = Exception(
            "401 Unauthorized"
        )

        with pytest.raises(TranscriptionAuthError) as excinfo:
            transcribe_audio(file_path=wav_path, client=mock_client)
        assert "Authentication failed" in str(excinfo.value)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def test_network_timeout_or_rate_limit():
    wav_path = create_temp_wav_file()
    try:
        mock_client = MagicMock()
        mock_client.speech_to_text.transcribe.side_effect = Exception(
            "Rate limit exceeded / Timeout"
        )

        with pytest.raises(TranscriptionProviderError) as excinfo:
            transcribe_audio(file_path=wav_path, client=mock_client)
        assert "Sarvam Speech-to-Text call failed" in str(excinfo.value)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def test_user_cancellation_after_transcript_review():
    wav_path = create_temp_wav_file()
    try:
        mock_stt_result = {
            "transcript": "Original transcript",
            "language_code": "en-IN",
            "language_probability": 0.95,
        }
        with patch(
            "main.transcribe_audio", return_value=mock_stt_result
        ) as mock_tr:
            with patch("main.analyze_conversation") as mock_anal:
                # Inputs: file path, context, review choice (3 = Cancel)
                inputs = iter([wav_path, "Test context", "3"])
                with patch("builtins.input", lambda prompt="": next(inputs)):
                    handle_audio_workflow()

                mock_tr.assert_called_once()
                mock_anal.assert_not_called()
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def test_corrected_transcript_passed_to_engine():
    wav_path = create_temp_wav_file()
    try:
        mock_stt_result = {
            "transcript": "Original raw transcript",
            "language_code": "hi-IN",
            "language_probability": 0.9,
        }
        with patch(
            "main.transcribe_audio", return_value=mock_stt_result
        ) as mock_tr:
            with patch("main.run_analysis") as mock_run_anal:
                # Inputs: file path, context, review choice 2 (Edit), corrected text, empty line to complete multiline input
                inputs = iter([
                    wav_path,
                    "Video deadline context",
                    "2",
                    "Corrected transcript text line",
                    "",
                ])
                with patch("builtins.input", lambda prompt="": next(inputs)):
                    handle_audio_workflow()

                mock_tr.assert_called_once()
                mock_run_anal.assert_called_once_with(
                    "Video deadline context", "Corrected transcript text line"
                )
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def test_existing_text_workflow_still_works():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = """{
        "summary": "Text workflow summary",
        "explicit_statements": ["Statement 1"],
        "commitments": [],
        "ambiguities": [],
        "clarification_question": "None needed",
        "limitations": []
    }"""
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.return_value = mock_response

    result = analyze_conversation(
        context="Text workflow test",
        conversation="User: Hello\nAssistant: Hi",
        client=mock_client,
    )

    assert result["summary"] == "Text workflow summary"
    assert result["explicit_statements"] == ["Statement 1"]
