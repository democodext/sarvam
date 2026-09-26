import os
import sys
from mean_db import (
    get_all_commitments,
    get_all_conversations,
    get_commitment_by_id,
    get_conversation_by_id,
    load_env,
    update_commitment_status,
)
from mean_engine import (
    ConfigurationError,
    MeanEngineError,
    ProviderError,
    ResponseParsingError,
    SchemaValidationError,
    analyze_conversation,
)
from sarvam_transcription import (
    AudioValidationError,
    TranscriptionAuthError,
    TranscriptionProviderError,
    transcribe_audio,
)

# Ensure UTF-8 output formatting for Windows console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEMO_CONTEXT = "A client is discussing a video delivery deadline with an editor."
DEMO_CONVERSATION = """Client: "Video Friday tak mil jayega?"
Editor: "Haan, Friday tak bhej dunga."
Client: "Friday morning?"
Editor: "Haan bhai, dekh lenge." """


def print_divider(character: str = "=", length: int = 60):
    print(character * length)


def print_section(title: str):
    print_divider("=")
    print(f"  {title.upper()}")
    print_divider("=")


def display_analysis(analysis: dict):
    print_section("MEAN Conversation Analysis")

    conv_id = analysis.get("conversation_id")
    if conv_id:
        print(f"🆔 Conversation ID: {conv_id}\n")

    print("📌 SUMMARY:")
    print(f"   {analysis.get('summary', 'N/A')}")

    print("\n📢 EXPLICIT STATEMENTS:")
    statements = analysis.get("explicit_statements", [])
    if statements:
        for stmt in statements:
            print(f"   • {stmt}")
    else:
        print("   (None recorded)")

    print("\n🤝 COMMITMENTS:")
    commitments = analysis.get("commitments", [])
    if commitments:
        for c in commitments:
            status_tag = f"[{c.get('status', 'unclear').upper()}]"
            person = c.get("responsible_person") or "Unspecified"
            deadline = c.get("deadline") or "No explicit deadline"
            comm_id = c.get("id", "")
            id_str = f" ({comm_id})" if comm_id else ""
            print(f"   • {status_tag}{id_str} {c.get('description')}")
            print(f"     - Responsible: {person} | Deadline: {deadline}")
            evidence = c.get("evidence", [])
            if evidence:
                print(f"     - Evidence: {', '.join(evidence)}")
    else:
        print("   (None recorded)")

    print("\n❓ AMBIGUITIES & UNRESOLVED POINTS:")
    ambiguities = analysis.get("ambiguities", [])
    if ambiguities:
        for a in ambiguities:
            print(f"   • {a.get('description')}")
            print(f"     - Why it matters: {a.get('why_it_matters')}")
            evidence = a.get("evidence", [])
            if evidence:
                print(f"     - Evidence: {', '.join(evidence)}")
            alts = a.get("alternative_interpretations", [])
            if alts:
                print(f"     - Plausible interpretations: {'; '.join(alts)}")
    else:
        print("   (None recorded)")

    print("\n💡 RECOMMENDED CLARIFICATION QUESTION:")
    print(f"   👉 \"{analysis.get('clarification_question', 'N/A')}\"")

    print("\n⚠️ LIMITATIONS:")
    limitations = analysis.get("limitations", [])
    if limitations:
        for lim in limitations:
            print(f"   • {lim}")
    else:
        print("   (None noted)")

    print_divider("=", 60)
    print()


def handle_text_workflow():
    print_section("Text Conversation Analysis")
    print("1. Run Demonstration Scenario (Video Editor Deadline)")
    print("2. Enter Custom Context and Conversation Text")
    choice = input("\nEnter choice (1/2) [Default: 1]: ").strip()

    if choice == "2":
        print("\nEnter Context (e.g. background of the discussion):")
        context = input("> ").strip()
        print("\nEnter Conversation (press Enter on an empty line when done):")
        lines = []
        while True:
            try:
                line = input()
                if not line and lines:
                    break
                lines.append(line)
            except EOFError:
                break
        conversation = "\n".join(lines).strip()
    else:
        print("\nRunning Demonstration Scenario...")
        context = DEMO_CONTEXT
        conversation = DEMO_CONVERSATION
        print(f"\nContext:\n{context}")
        print(f"\nConversation:\n{conversation}")

    run_analysis(context, conversation)


def handle_audio_workflow():
    print_section("Audio Transcription & Analysis")
    file_path = input("Enter path to audio file: ").strip().strip("'\"")

    if not file_path:
        print("❌ Error: Audio file path cannot be empty.")
        return

    context = input("Enter conversation context (background info): ").strip()
    if not context:
        context = "Audio conversation analysis"

    print("\nTranscribing audio file with Sarvam AI...")
    try:
        transcription_result = transcribe_audio(file_path)
    except AudioValidationError as e:
        print(f"\n❌ File Validation Error: {e}")
        return
    except TranscriptionAuthError as e:
        print(f"\n❌ Authentication Error: {e}")
        return
    except TranscriptionProviderError as e:
        print(f"\n❌ Sarvam Transcription Provider Error: {e}")
        return
    except Exception as e:
        print(f"\n❌ Unexpected Transcription Error: {e}")
        return

    raw_transcript = transcription_result.get("transcript", "")
    lang = transcription_result.get("language_code") or "Detected"
    prob = transcription_result.get("language_probability")

    print_section("Sarvam Speech-to-Text Output")
    lang_info = f"{lang}"
    if prob is not None:
        lang_info += f" (Confidence: {prob * 100:.1f}%)"
    print(f"Language Metadata: {lang_info}")
    print("\nRAW TRANSCRIPT:")
    if raw_transcript:
        print(f"\"{raw_transcript}\"")
    else:
        print("⚠️ Warning: Sarvam Speech-to-Text returned an empty transcript.")

    print("\nReview Transcript:")
    print("1. Confirm transcript as-is and run analysis")
    print("2. Edit/correct transcript before analysis")
    print("3. Cancel analysis")
    review_choice = input("\nEnter choice (1/2/3) [Default: 1]: ").strip()

    if review_choice == "3":
        print("\nAnalysis cancelled by user.")
        return

    final_transcript = raw_transcript
    if review_choice == "2":
        print("\nEnter corrected transcript (press Enter on an empty line when done):")
        lines = []
        while True:
            try:
                line = input()
                if not line and lines:
                    break
                lines.append(line)
            except EOFError:
                break
        final_transcript = "\n".join(lines).strip()
        print("\nUsing corrected transcript for analysis.")
    else:
        print("\nUsing raw transcript for analysis.")

    if not final_transcript:
        print("❌ Error: Transcript is empty. Cannot run analysis.")
        return

    run_analysis(context, final_transcript)


def handle_commitments_view():
    print_section("Tracked Commitments")
    filter_choice = input(
        "Enter status filter (all / explicit / pending / overdue / completed / renegotiated / cancelled) [Default: all]: "
    ).strip().lower() or "all"

    try:
        commitments = get_all_commitments(status_filter=filter_choice)
        if not commitments:
            print("\nNo commitments found matching filter.")
            return

        print(f"\nFound {len(commitments)} commitment(s):\n")
        for c in commitments:
            status_tag = f"[{c['status'].upper()}]"
            print(f"• ID: {c['id']} | {status_tag} {c['description']}")
            print(f"  Responsible: {c['responsible_person'] or 'Unspecified'} | Deadline: {c['deadline'] or 'None'}")
            if c.get("evidence"):
                print(f"  Evidence: \"{', '.join(c['evidence'])}\"")
            print()
    except Exception as e:
        print(f"\n❌ Error retrieving commitments: {e}")


def handle_update_commitment():
    print_section("Update Commitment Status")
    comm_id = input("Enter Commitment ID (e.g. comm_...): ").strip()
    if not comm_id:
        print("❌ Commitment ID is required.")
        return

    try:
        current = get_commitment_by_id(comm_id)
        print(f"\nCurrent Status: [{current['status'].upper()}] | {current['description']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    new_status = input(
        "Enter New Status (completed / renegotiated / cancelled / overdue): "
    ).strip().lower()
    if not new_status:
        print("❌ Status is required.")
        return

    new_deadline = None
    if new_status == "renegotiated":
        new_deadline = input("Enter New Deadline (e.g. 2026-10-15 or Friday 4 PM): ").strip() or None

    reason = input("Enter Reason / Notes for change: ").strip()

    confirm = input(f"Confirm changing status of '{comm_id}' to {new_status.upper()}? (y/N): ").strip().lower()
    if confirm != "y":
        print("Status update cancelled.")
        return

    try:
        updated = update_commitment_status(
            commitment_id=comm_id,
            new_status=new_status,
            reason_or_notes=reason,
            new_deadline=new_deadline,
        )
        print(f"\n✅ Commitment updated successfully! New status: [{updated['status'].upper()}]")
    except Exception as e:
        print(f"\n❌ Status update failed: {e}")


def handle_history_view():
    print_section("Conversation Audit History")
    try:
        conversations = get_all_conversations()
        if not conversations:
            print("\nNo conversations recorded yet.")
            return

        print(f"\nFound {len(conversations)} recorded conversation(s):\n")
        for conv in conversations:
            print(f"• ID: {conv['id']} | Date: {conv['created_at']}")
            print(f"  Context: {conv['context'] or 'N/A'}")
            print(f"  Summary: {conv['summary'] or 'N/A'}")
            print()
    except Exception as e:
        print(f"\n❌ Error retrieving conversation history: {e}")


def run_analysis(context: str, conversation: str):
    print("\nAnalyzing conversation with Sarvam AI...")
    try:
        analysis = analyze_conversation(context, conversation)
        display_analysis(analysis)
    except ConfigurationError as e:
        print(f"\n❌ Configuration Error: {e}")
    except ProviderError as e:
        print(f"\n❌ Sarvam Provider Error: {e}")
        print("Please check your network connection, API key, or rate limits.")
    except (ResponseParsingError, SchemaValidationError) as e:
        print(f"\n❌ Response Validation Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected Analysis Error: {e}")


def main():
    load_env()
    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        print("\n❌ ERROR: SARVAM_API_KEY environment variable is not configured.")
        print("Please set your API key in the `.env` file:")
        print("  SARVAM_API_KEY=your_actual_key_here\n")
        sys.exit(1)

    # Non-interactive mode fallback
    if not sys.stdin.isatty():
        print("Running non-interactive demonstration mode...")
        run_analysis(DEMO_CONTEXT, DEMO_CONVERSATION)
        return

    while True:
        print_section("MEAN AI — Main Menu")
        print("1. Analyze text conversation")
        print("2. Transcribe and analyze an audio file")
        print("3. View Tracked Commitments")
        print("4. Update Commitment Status")
        print("5. View Conversation History")
        print("6. Exit")
        choice = input("\nEnter choice (1-6): ").strip()

        if choice == "1":
            handle_text_workflow()
        elif choice == "2":
            handle_audio_workflow()
        elif choice == "3":
            handle_commitments_view()
        elif choice == "4":
            handle_update_commitment()
        elif choice == "5":
            handle_history_view()
        elif choice == "6" or choice.lower() in ("exit", "q", "quit"):
            print("\nExiting MEAN AI. Goodbye!")
            break
        else:
            print("\n❌ Invalid choice. Please enter a number between 1 and 6.")


if __name__ == "__main__":
    main()
