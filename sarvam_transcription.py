import os
from typing import Any, Dict, Optional
from sarvamai import SarvamAI
from mean_engine import ConfigurationError, load_env

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".aac",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".webm",
    ".amr",
}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


class TranscriptionError(Exception):
    """Base exception for transcription errors."""

    pass


class AudioValidationError(TranscriptionError):
    """Raised when an audio file fails validation checks."""

    pass


class TranscriptionAuthError(TranscriptionError):
    """Raised when Sarvam API credentials are missing or invalid."""

    pass


class TranscriptionProviderError(TranscriptionError):
    """Raised when Sarvam Speech-to-Text service call fails."""

    pass


def validate_audio_file(file_path: str) -> None:
    """Validates existence, non-emptiness, size, extension, and header bytes of an audio file."""
    if not os.path.exists(file_path):
        raise AudioValidationError(f"Audio file not found: '{file_path}'")

    if not os.path.isfile(file_path):
        raise AudioValidationError(f"Specified path is not a file: '{file_path}'")

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise AudioValidationError(f"Audio file is empty (0 bytes): '{file_path}'")

    if file_size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        file_mb = file_size / (1024 * 1024)
        raise AudioValidationError(
            f"Audio file size ({file_mb:.2f} MB) exceeds maximum supported limit of {max_mb:.0f} MB."
        )

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise AudioValidationError(
            f"Unsupported file extension '{ext}'. Supported formats: {supported_str}"
        )

    # Check magic bytes for common formats where feasible
    try:
        with open(file_path, "rb") as f:
            header = f.read(12)
            if ext == ".wav" and not (
                header.startswith(b"RIFF") or header.startswith(b"RIFX")
            ):
                raise AudioValidationError(
                    f"File '{file_path}' has .wav extension but invalid WAV header."
                )
            elif ext == ".flac" and not header.startswith(b"fLaC"):
                raise AudioValidationError(
                    f"File '{file_path}' has .flac extension but invalid FLAC header."
                )
            elif ext == ".ogg" and not header.startswith(b"OggS"):
                raise AudioValidationError(
                    f"File '{file_path}' has .ogg extension but invalid OGG header."
                )
    except AudioValidationError:
        raise
    except Exception as e:
        raise AudioValidationError(f"Unable to read audio file header: {e}")


def transcribe_audio(
    file_path: str,
    client: Optional[Any] = None,
    model: str = "saaras:v4",
    language_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribes an audio file using Sarvam Speech-to-Text API.

    Returns a dict containing:
        - transcript: str
        - language_code: Optional[str]
        - language_probability: Optional[float]
        - request_id: Optional[str]

    Raises:
        AudioValidationError: If file fails validation.
        TranscriptionAuthError: If API key is missing.
        TranscriptionProviderError: If service request fails.
    """
    validate_audio_file(file_path)

    load_env()
    api_key = os.getenv("SARVAM_API_KEY")

    if client is None:
        if not api_key:
            raise TranscriptionAuthError(
                "SARVAM_API_KEY is missing. Please set it in your .env file."
            )
        try:
            client = SarvamAI(api_subscription_key=api_key)
        except Exception as e:
            raise TranscriptionAuthError(f"Failed to initialize SarvamAI client: {e}")

    try:
        with open(file_path, "rb") as audio_file:
            kwargs = {"file": audio_file, "model": model}
            if language_code:
                kwargs["language_code"] = language_code

            response = client.speech_to_text.transcribe(**kwargs)
    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "Unauthorized" in err_str or "api_key" in err_str:
            raise TranscriptionAuthError(f"Authentication failed with Sarvam API: {e}")
        raise TranscriptionProviderError(f"Sarvam Speech-to-Text call failed: {e}")

    transcript = getattr(response, "transcript", "") or ""
    lang_code = getattr(response, "language_code", None)
    lang_prob = getattr(response, "language_probability", None)
    req_id = getattr(response, "request_id", None)

    return {
        "transcript": transcript.strip(),
        "language_code": lang_code,
        "language_probability": lang_prob,
        "request_id": req_id,
    }
