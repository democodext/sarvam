import json
import os
import sys
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError
from sarvamai import SarvamAI

from mean_db import (
    DEFAULT_DB_PATH,
    get_active_commitments_context,
    load_env,
    save_conversation_analysis,
)


class Commitment(BaseModel):
    id: Optional[str] = Field(None, description="Unique commitment ID")
    conversation_id: Optional[str] = Field(
        None, description="Source conversation ID"
    )
    description: str = Field(..., description="Description of the commitment made")
    responsible_person: Optional[str] = Field(
        None, description="Person responsible for the commitment"
    )
    deadline: Optional[str] = Field(None, description="Agreed deadline if specified")
    status: str = Field(
        ..., description="Status of commitment, e.g. explicit, unclear, pending"
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Exact transcript excerpts supporting this commitment",
    )
    commitment_type: str = Field(
        default="explicit", description="Type of commitment"
    )
    updated_at: Optional[str] = Field(None, description="Last status update timestamp")


class Ambiguity(BaseModel):
    description: str = Field(
        ..., description="Description of the ambiguity or unresolved point"
    )
    why_it_matters: str = Field(
        ..., description="Why this ambiguity or unresolved detail matters"
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Exact transcript excerpts demonstrating the ambiguity",
    )
    alternative_interpretations: List[str] = Field(
        default_factory=list, description="Plausible alternative interpretations"
    )


class AnalysisResult(BaseModel):
    conversation_id: Optional[str] = Field(
        None, description="Associated conversation ID"
    )
    summary: str = Field(
        ..., description="Brief conversation summary focusing on key outcomes"
    )
    explicit_statements: List[str] = Field(
        default_factory=list, description="List of clear, explicit statements made"
    )
    commitments: List[Commitment] = Field(
        default_factory=list, description="List of commitments identified"
    )
    ambiguities: List[Ambiguity] = Field(
        default_factory=list,
        description="List of ambiguities or unresolved questions",
    )
    clarification_question: str = Field(
        ..., description="One high-value clarification question to resolve key ambiguity"
    )
    limitations: List[str] = Field(
        default_factory=list, description="Analytical limitations (e.g. missing context)"
    )
    created_at: Optional[str] = Field(None, description="Creation timestamp")


class MeanEngineError(Exception):
    """Base exception for MEAN engine errors."""

    pass


class ConfigurationError(MeanEngineError):
    """Raised when environment/API credentials are missing or invalid."""

    pass


class ProviderError(MeanEngineError):
    """Raised when the Sarvam AI service call fails."""

    pass


class ResponseParsingError(MeanEngineError):
    """Raised when the model response is malformed or invalid JSON."""

    pass


class SchemaValidationError(MeanEngineError):
    """Raised when model output fails Pydantic schema validation."""

    pass


def extract_json_from_text(text: str) -> str:
    """Extracts JSON object string from text, stripping markdown fences if present."""
    text = text.strip()
    if "```" in text:
        lines = text.splitlines()
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                json_lines.append(line)
        if json_lines:
            text = "\n".join(json_lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


SYSTEM_PROMPT = """You are MEAN AI, a context-aware conversation intelligence engine.
Your task is to analyze human conversations objectively and extract structured insights based strictly on transcript evidence and context.

Analysis Rules:
1. Extract:
   - Summary: Brief overall summary of the exchange.
   - Explicit Statements: Clear, objective factual statements made by speakers.
   - Commitments: Deliverables or promises made. Include description, responsible person, deadline (if specified), status ('explicit' or 'unclear'), evidence (exact quotes), and commitment_type ('explicit' or 'implicit').
   - Ambiguities: Unresolved details, vague responses (e.g. "dekh lenge"), missing exact timing/conditions, or potential misunderstandings. Explain why it matters, evidence quotes, and alternative interpretations.
   - Clarification Question: Exactly ONE actionable, high-value clarification question to resolve the primary ambiguity.
   - Limitations: Note any limitations in the analysis (e.g. lack of prior context).
2. DO NOT claim to read minds, detect lies, or infer hidden intentions.
3. DO NOT treat vague wording (such as "dekh lenge") as cancelling a prior explicit commitment (such as "Friday tak bhej dunga"). Keep the original commitment recorded as explicit, and flag the specific unresolved detail (e.g. exact morning delivery time) under ambiguities.
4. If prior active commitments are included in memory, check if current transcript updates, confirms, or renegotiates an existing commitment without hallucinating new IDs.
5. Use exact quotes from the transcript for evidence.
6. Return ONLY valid JSON adhering strictly to this schema:

{
  "summary": "Brief summary",
  "explicit_statements": ["statement1"],
  "commitments": [
    {
      "description": "Commitment description",
      "responsible_person": "Name or role",
      "deadline": "Deadline if any",
      "status": "explicit",
      "evidence": ["exact quote"],
      "commitment_type": "explicit"
    }
  ],
  "ambiguities": [
    {
      "description": "Ambiguity description",
      "why_it_matters": "Reason why it matters",
      "evidence": ["exact quote"],
      "alternative_interpretations": ["interpretation 1"]
    }
  ],
  "clarification_question": "One focused question",
  "limitations": ["Limitation note"]
}
"""


def analyze_conversation(
    context: str,
    conversation: str,
    client: Optional[Any] = None,
    model: str = "sarvam-105b-conversations",
    conversation_id: Optional[str] = None,
    include_history: bool = True,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """Analyzes a conversation given context and returns a validated dictionary, persisting results to SQLite.

    Raises:
        ConfigurationError: If API key is missing.
        ProviderError: If the Sarvam API call fails.
        ResponseParsingError: If model output is not valid JSON.
        SchemaValidationError: If JSON fails Pydantic schema validation.
    """
    load_env()
    api_key = os.getenv("SARVAM_API_KEY")

    if client is None:
        if not api_key:
            raise ConfigurationError(
                "SARVAM_API_KEY is missing. Please configure it in your .env file."
            )
        try:
            client = SarvamAI(api_subscription_key=api_key)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize SarvamAI client: {e}")

    # Retrieve bounded memory context if enabled
    memory_context = ""
    if include_history:
        memory_context = get_active_commitments_context(max_items=5, db_path=db_path)

    full_context = context
    if memory_context:
        full_context += f"\n{memory_context}"

    user_prompt = (
        f"Context:\n{full_context}\n\n"
        f"Conversation Transcript:\n{conversation}\n\n"
        "Analyze this conversation following all system instructions and return strictly valid JSON."
    )

    try:
        response = client.chat.completions(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise ProviderError(f"Sarvam AI service request failed: {e}")

    try:
        raw_content = response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        raise ProviderError(f"Unexpected response structure from Sarvam AI: {e}")

    json_str = extract_json_from_text(raw_content)

    try:
        data = json.loads(json_str)
    except Exception as e:
        raise ResponseParsingError(
            f"Failed to parse model response as JSON: {e}\nRaw response:\n{raw_content}"
        )

    try:
        validated_model = AnalysisResult.model_validate(data)
    except ValidationError as e:
        raise SchemaValidationError(f"Model output failed schema validation: {e}")

    analysis_dict = validated_model.model_dump()

    # Save to SQLite persistence layer
    conv_id = save_conversation_analysis(
        context=context,
        transcript=conversation,
        analysis_data=analysis_dict,
        conversation_id=conversation_id,
        db_path=db_path,
    )

    analysis_dict["conversation_id"] = conv_id
    return analysis_dict
