import os
import sys
import tempfile
import webbrowser
from flask import Flask, jsonify, render_template_string, request, send_from_directory

# Ensure UTF-8 output formatting for Windows console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from mean_db import (
    DEFAULT_DB_PATH,
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
from assemblyai_realtime import (
    AssemblyAIConfigError,
    AssemblyAITokenError,
    create_temporary_token,
)
from voice_action_engine import (
    cancel_pending_action,
    confirm_pending_action,
    process_voice_transcript,
)

app = Flask(__name__)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = (
        "no-cache, no-store, must-revalidate, max-age=0"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MEAN AI — Conversational Intelligence Command Center</title>

    <!-- Open Graph / Discord Link Preview -->
    <meta property="og:title" content="MEAN AI — Indian Conversational Intelligence">
    <meta property="og:description" content="Understand what was said. Clarify what remains unsaid. Extract commitments, deadlines, dependencies, and ambiguities from conversations.">
    <meta property="og:url" content="https://mean-ai.onrender.com/">
    <meta property="og:type" content="website">
    <meta property="og:image" content="https://mean-ai.onrender.com/static/og-preview.png">

    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="MEAN AI — Indian Conversational Intelligence">
    <meta name="twitter:description" content="Understand what was said. Clarify what remains unsaid. Extract commitments, deadlines, dependencies, and ambiguities from conversations.">
    <meta name="twitter:image" content="https://mean-ai.onrender.com/static/og-preview.png">

    <!-- Bootstrap 5 & Fonts & Lucide Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/lucide@latest"></script>

    <style>
        :root {
            --bg-main: #08090D;
            --bg-sidebar: #0B0C12;
            --bg-panel: #11131C;
            --bg-elevated: #161927;
            --border-color: rgba(255, 255, 255, 0.09);
            --border-subtle: #232636;

            --text-primary: #F5F5FA;
            --text-secondary: #A4A9BB;
            --text-muted: #81879A;

            --accent-violet: #7865FF;
            --accent-blue-violet: #6657E8;
            --accent-cyan: #38D9F5;
            --accent-success: #34D399;
            --accent-warning: #FBBF24;
            --accent-error: #F87171;

            --cta-gradient: linear-gradient(135deg, #8875FF 0%, #6D5AE8 52%, #5744CF 100%);
            --mic-gradient: linear-gradient(145deg, #8877FF 0%, #6754E8 60%, #4937B8 100%);

            --radius-panel: 16px;
            --radius-control: 10px;
            --radius-sm: 6px;
        }

        body {
            background-color: var(--bg-main) !important;
            background-image: radial-gradient(circle at 50% 20%, rgba(120, 101, 255, 0.04) 0%, transparent 60%);
            color: var(--text-primary) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
            margin: 0;
            padding: 0;
            overflow-x: hidden;
        }

        /* Typography */
        h1, .h1 { font-size: 26px; font-weight: 600; color: var(--text-primary); letter-spacing: -0.02em; }
        h5, .h5 { font-size: 18px; font-weight: 600; color: var(--text-primary); letter-spacing: -0.01em; }
        h6, .h6 { font-size: 15px; font-weight: 600; color: var(--text-primary); }
        body, p, label { font-size: 14px; color: var(--text-secondary); line-height: 1.5; }
        .small, small { font-size: 13px; color: var(--text-muted); }
        .code-font { font-family: 'JetBrains Mono', monospace; font-size: 13px; }

        /* Custom Scrollbars */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-main); }
        ::-webkit-scrollbar-thumb { background: var(--border-subtle); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent-violet); }

        /* Sidebar */
        .sidebar {
            width: 228px;
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border-color);
            min-height: 100vh;
            flex-shrink: 0;
        }

        .nav-link-item {
            color: var(--text-secondary);
            border-radius: var(--radius-control);
            padding: 10px 14px;
            font-weight: 500;
            font-size: 14px;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            gap: 12px;
            width: 100%;
            border: 1px solid transparent;
            background: transparent;
            text-align: left;
            cursor: pointer;
        }
        .nav-link-item:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.03);
            border-color: rgba(255, 255, 255, 0.05);
        }
        .nav-link-item.active {
            color: #ffffff;
            background: rgba(120, 101, 255, 0.12);
            border: 1px solid rgba(120, 101, 255, 0.3);
            font-weight: 600;
        }

        /* Header */
        .header-bar {
            height: 64px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
        }

        /* Panels */
        .main-panel {
            background: var(--bg-panel);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-panel);
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }
        .elevated-panel {
            background: var(--bg-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-panel);
            padding: 20px;
        }

        /* Hero Microphone Button */
        .mic-hero-btn {
            width: 72px;
            height: 72px;
            border-radius: 50%;
            background: var(--mic-gradient);
            border: none;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 6px 24px rgba(103, 84, 232, 0.35);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            flex-shrink: 0;
        }
        .mic-hero-btn:hover {
            transform: scale(1.04);
            box-shadow: 0 8px 30px rgba(136, 117, 255, 0.45);
        }
        .mic-hero-btn.listening {
            background: linear-gradient(145deg, #f87171 0%, #ef4444 60%, #dc2626 100%);
            box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.6);
            animation: pulse-ring 1.8s infinite cubic-bezier(0.4, 0, 0.6, 1);
        }
        @keyframes pulse-ring {
            0% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.6); }
            70% { box-shadow: 0 0 0 20px rgba(248, 113, 113, 0); }
            100% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0); }
        }

        /* Buttons */
        .btn-cta {
            background: var(--cta-gradient);
            color: white !important;
            border: none;
            font-weight: 600;
            font-size: 14px;
            border-radius: var(--radius-control);
            padding: 10px 22px;
            box-shadow: 0 4px 16px rgba(109, 90, 232, 0.35);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .btn-cta:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(136, 117, 255, 0.45);
        }
        .btn-secondary-dark {
            background: var(--bg-elevated);
            color: var(--text-primary) !important;
            border: 1px solid var(--border-subtle);
            font-weight: 500;
            font-size: 14px;
            border-radius: var(--radius-control);
            padding: 10px 20px;
            transition: background-color 0.15s ease, border-color 0.15s ease;
        }
        .btn-secondary-dark:hover:not(:disabled) {
            background: #1C2032;
            border-color: rgba(255, 255, 255, 0.15);
        }
        .btn-secondary-dark:disabled {
            opacity: 0.45;
            cursor: not-allowed;
        }

        /* Audio Waveform Bars */
        .waveform-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 3px;
            height: 52px;
            width: 100%;
        }
        .wave-bar {
            width: 3px;
            border-radius: 3px;
            background: var(--text-muted);
            opacity: 0.3;
            transition: height 0.12s ease, background-color 0.12s ease, opacity 0.12s ease;
        }

        /* Status Badges */
        .badge-status {
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.04em;
            padding: 4px 10px;
            border-radius: 6px;
            text-transform: uppercase;
        }
        .badge-pending { background: rgba(251, 191, 36, 0.12); color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.25); }
        .badge-completed { background: rgba(52, 211, 153, 0.12); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.25); }
        .badge-overdue { background: rgba(248, 113, 113, 0.12); color: #F87171; border: 1px solid rgba(248, 113, 113, 0.25); }
        .badge-commitment { background: rgba(120, 101, 255, 0.12); color: #A4A9BB; border: 1px solid rgba(120, 101, 255, 0.25); }

        /* Form Elements */
        .form-control, .form-select {
            background-color: var(--bg-main) !important;
            border: 1px solid var(--border-subtle) !important;
            color: var(--text-primary) !important;
            border-radius: var(--radius-control);
            font-size: 14px;
        }
        .form-control:focus, .form-select:focus {
            border-color: var(--accent-violet) !important;
            box-shadow: 0 0 0 0.2rem rgba(120, 101, 255, 0.2) !important;
        }

        .spinner-overlay {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(8, 9, 13, 0.92); backdrop-filter: blur(8px); z-index: 2000;
            align-items: center; justify-content: center;
        }
    </style>
</head>
<body>
    <div class="d-flex w-100">
        <!-- LEFT SIDEBAR (228px) -->
        <div class="sidebar p-3 d-flex flex-column justify-content-between">
            <div>
                <!-- Brand Header -->
                <div class="d-flex align-items-center gap-2 mb-4 ps-2 pt-2">
                    <div class="rounded-2 d-flex align-items-center justify-content-center" style="width: 32px; height: 32px; background: var(--cta-gradient);">
                        <i data-lucide="cpu" style="color: white; width: 18px; height: 18px;"></i>
                    </div>
                    <div>
                        <h2 class="h6 fw-bold mb-0 text-white tracking-wider">MEAN <span style="color:var(--accent-violet)">AI</span></h2>
                    </div>
                </div>

                <!-- Navigation Menu -->
                <nav class="d-flex flex-column gap-1 mb-4">
                    <button class="nav-link-item active" onclick="showTab('workspace')">
                        <i data-lucide="layout-dashboard" style="width: 16px; height: 16px;"></i> Command Center
                    </button>
                    <button class="nav-link-item" onclick="showTab('commitments'); loadCommitments();">
                        <i data-lucide="check-square" style="width: 16px; height: 16px;"></i> Commitments
                    </button>
                    <button class="nav-link-item" onclick="showTab('history'); loadConversations();">
                        <i data-lucide="history" style="width: 16px; height: 16px;"></i> History
                    </button>
                </nav>
            </div>

            <!-- Compact System Status -->
            <div class="p-3 elevated-panel" style="background: var(--bg-main);">
                <small class="text-secondary code-font tracking-wider d-block mb-2" style="font-size: 11px; font-weight: 600;">SYSTEM STATUS</small>
                <div class="d-flex flex-column gap-2" style="font-size: 12px;">
                    <div class="d-flex align-items-center justify-content-between">
                        <span class="text-secondary">AssemblyAI STT</span>
                        <span class="d-flex align-items-center gap-1 text-success fw-medium"><span style="width:6px; height:6px; background:var(--accent-success); border-radius:50%;"></span> Online</span>
                    </div>
                    <div class="d-flex align-items-center justify-content-between">
                        <span class="text-secondary">Database Engine</span>
                        <span class="d-flex align-items-center gap-1 text-success fw-medium"><span style="width:6px; height:6px; background:var(--accent-success); border-radius:50%;"></span> Connected</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- MAIN CONTENT CONTAINER -->
        <div class="flex-grow-1" style="min-width: 0;">
            <!-- HEADER BAR (64px) -->
            <header class="header-bar">
                <h1 class="h5 mb-0">Conversational Intelligence</h1>
                <div class="d-flex align-items-center gap-2">
                    <small class="text-secondary code-font" style="font-size: 12px;">STT STREAM:</small>
                    <span id="topBarStatusBadge" class="badge badge-pending code-font">DISCONNECTED</span>
                </div>
            </header>

            <!-- WORKSPACE AREA (24px padding & gaps) -->
            <div class="p-4">
                <div class="row g-4">
                    <!-- CENTRAL MAIN WORKSPACE (~72%) -->
                    <div class="col-lg-8 col-md-12">
                        <div id="tab-workspace">
                            <p class="text-secondary mb-4" style="font-size: 15px;">Turn conversations into clear, actionable commitments using AssemblyAI.</p>

                            <!-- Dominant Realtime Voice Agent Hero Panel -->
                            <div class="main-panel mb-4">
                                <div class="d-flex align-items-center justify-content-between mb-4">
                                    <div class="d-flex align-items-center gap-3">
                                        <div class="rounded-circle d-flex align-items-center justify-content-center" style="width: 40px; height: 40px; background: rgba(120, 101, 255, 0.12); border: 1px solid rgba(120, 101, 255, 0.3);">
                                            <i data-lucide="mic" style="color: var(--accent-violet); width: 20px; height: 20px;"></i>
                                        </div>
                                        <div>
                                            <h5 class="mb-0">Realtime Voice Agent</h5>
                                            <p class="text-secondary small mb-0">Speak naturally. MEAN AI listens, extracts intent, and manages commitments.</p>
                                        </div>
                                    </div>
                                    <span class="badge badge-commitment code-font">AssemblyAI v3 Pro</span>
                                </div>

                                <!-- Center Microphone Focal Point & Waveform -->
                                <div class="d-flex flex-column align-items-center justify-content-center py-4 my-2">
                                    <div class="d-flex align-items-center justify-content-center w-100 gap-3 mb-3">
                                        <div class="waveform-container flex-grow-1" id="leftWaveform"></div>

                                        <button class="mic-hero-btn mx-2" id="centralMicBtn" onclick="toggleVoiceAgent()">
                                            <i data-lucide="mic" style="width: 32px; height: 32px;" id="centralMicIcon"></i>
                                        </button>

                                        <div class="waveform-container flex-grow-1" id="rightWaveform"></div>
                                    </div>

                                    <h6 class="fw-semibold text-white mb-1" id="voiceAgentStatusTitle">Ready to listen</h6>
                                    <p class="text-secondary small mb-0" id="voiceAgentStatusSubtitle">Click the microphone to start voice streaming</p>
                                </div>

                                <!-- Control Bar -->
                                <div class="d-flex align-items-center justify-content-between pt-3 border-top border-secondary border-opacity-10">
                                    <div class="d-flex align-items-center gap-2">
                                        <button id="btnStartVoiceAgent" class="btn btn-cta d-flex align-items-center gap-2" onclick="startVoiceAgent()">
                                            <i data-lucide="mic" style="width: 16px; height: 16px;"></i> Start Listening
                                        </button>
                                        <button id="btnStopVoiceAgent" class="btn btn-outline-danger btn-sm px-3" onclick="stopVoiceAgent()" style="display: none;">
                                            Stop Session
                                        </button>
                                        <button id="btnProcessCommand" class="btn btn-secondary-dark d-flex align-items-center gap-2" onclick="submitVoiceAction()" disabled>
                                            <i data-lucide="zap" style="width: 16px; height: 16px;"></i> Process Command
                                        </button>
                                    </div>
                                    <button class="btn btn-secondary-dark btn-sm text-secondary" onclick="clearRealtimeTranscript()">Clear</button>
                                </div>

                                <!-- Live Partial Streaming Box -->
                                <div id="aaiErrorAlert" class="alert alert-danger py-2 px-3 small code-font mt-3" style="display: none;"></div>

                                <div id="liveVoiceArea" class="mt-3 p-3 rounded-3" style="background: var(--bg-main); border: 1px solid var(--border-subtle); display: none;">
                                    <small class="text-secondary code-font d-block mb-1">LIVE PARTIAL TRANSCRIPT:</small>
                                    <div id="aaiPartialBox" class="code-font text-info" style="min-height: 28px; font-style: italic;">
                                        <span class="text-muted opacity-50">Listening to voice stream...</span>
                                    </div>
                                </div>

                                <!-- Voice Action Result & Formatted Confirmation Gate -->
                                <div id="voiceActionPanel" style="display: none;" class="elevated-panel p-3 mt-3">
                                    <div class="d-flex align-items-center justify-content-between mb-2">
                                        <h6 class="small fw-bold text-info code-font mb-0 d-flex align-items-center gap-1">
                                            <i data-lucide="bot" style="width: 15px; height: 15px;"></i> VOICE ACTION INTENT
                                        </h6>
                                        <span id="voiceActionBadge" class="badge bg-primary code-font">DETECTED</span>
                                    </div>
                                    <p id="voiceResponseText" class="fs-6 text-white mb-3"></p>

                                    <!-- Confirmation Gate -->
                                    <div id="voiceConfirmGate" style="display: none;" class="p-3 rounded-3" style="background: rgba(120, 101, 255, 0.08); border: 1px solid rgba(120, 101, 255, 0.3);">
                                        <h6 class="small fw-bold text-warning mb-2 code-font d-flex align-items-center gap-1">
                                            <i data-lucide="shield-alert" style="width: 15px; height: 15px;"></i> ACTION CONFIRMATION REQUIRED
                                        </h6>
                                        <div id="proposedActionDetails" class="small text-slate-300 mb-3 code-font" style="background: var(--bg-main); padding: 12px; border-radius: 8px; border: 1px solid var(--border-subtle);"></div>
                                        <div class="d-flex gap-2">
                                            <button class="btn btn-cta btn-sm d-flex align-items-center gap-1" onclick="confirmVoiceAction()">
                                                <i data-lucide="check" style="width: 14px; height: 14px;"></i> Confirm Action
                                            </button>
                                            <button class="btn btn-secondary-dark btn-sm d-flex align-items-center gap-1" onclick="cancelVoiceAction()">
                                                <i data-lucide="x" style="width: 14px; height: 14px;"></i> Cancel
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Input Mode Selection Section -->
                            <div class="main-panel">
                                <div class="d-flex align-items-center justify-content-between mb-3">
                                    <div class="d-flex align-items-center gap-2">
                                        <button class="btn btn-cta btn-sm px-3" id="mode-realtime-btn" onclick="switchInputMode('realtime')">Realtime Voice</button>
                                        <button class="btn btn-secondary-dark btn-sm px-3" id="mode-text-btn" onclick="switchInputMode('text')">Text Input</button>
                                        <button class="btn btn-secondary-dark btn-sm px-3" id="mode-audio-btn" onclick="switchInputMode('audio')">Audio File</button>
                                    </div>
                                </div>

                                <!-- Realtime Mode Container -->
                                <div id="input-mode-realtime">
                                    <div class="mb-3">
                                        <textarea id="aaiFinalTranscript" class="form-control code-font p-3" rows="3" placeholder="Finalized speech transcript will accumulate here..." oninput="checkTranscriptInput()"></textarea>
                                    </div>
                                    <button class="btn btn-secondary-dark btn-sm d-flex align-items-center gap-1" onclick="useRealtimeTranscriptForAnalysis()">
                                        <i data-lucide="sparkles" style="width: 14px; height: 14px;"></i> Deep Intelligence Analysis
                                    </button>
                                </div>

                                <!-- Text Mode Container -->
                                <div id="input-mode-text" style="display: none;">
                                    <div class="mb-2 d-none">
                                        <input type="text" id="textContext" class="form-control" value="A client is discussing a video delivery deadline with an editor.">
                                    </div>
                                    <div class="position-relative mb-2">
                                        <textarea id="textTranscript" class="form-control code-font p-3" rows="4" placeholder="Type or paste transcript here..." oninput="updateCharCount(this); checkTranscriptInput();">Client: "Video Friday tak mil jayega?"&#10;Editor: "Haan, Friday tak bhej dunga."&#10;Client: "Friday morning?"&#10;Editor: "Haan bhai, dekh lenge."</textarea>
                                    </div>
                                    <div class="d-flex align-items-center justify-content-between">
                                        <span class="small text-secondary code-font" id="charCountLabel">118/2000</span>
                                        <button class="btn btn-cta btn-sm" onclick="runTextAnalysis()">Run Analysis</button>
                                    </div>
                                </div>

                                <!-- Audio Mode Container -->
                                <div id="input-mode-audio" style="display: none;">
                                    <div class="mb-3">
                                        <input type="file" id="audioFileInput" class="form-control" accept=".mp3,.wav,.m4a,.aac,.flac,.ogg,.opus,.webm,.amr">
                                    </div>
                                    <div class="mb-3">
                                        <input type="text" id="audioContext" class="form-control" value="Audio transcript review">
                                    </div>
                                    <button class="btn btn-cta btn-sm mb-3" onclick="transcribeAudio()">Transcribe Audio</button>

                                    <div id="transcriptReviewArea" style="display: none;" class="p-3 elevated-panel border-info">
                                        <div class="badge bg-info text-dark mb-2 code-font" id="langMetadata">Language Detected: hi-IN</div>
                                        <textarea id="reviewedTranscript" class="form-control code-font mb-3" rows="3"></textarea>
                                        <button class="btn btn-cta btn-sm" onclick="runAudioAnalysis()">Confirm & Analyze</button>
                                    </div>
                                </div>
                            </div>

                            <!-- Extraction Analysis Results Container -->
                            <div id="resultsContainer" style="display: none;" class="mt-4">
                                <div class="main-panel border-success">
                                    <h6 class="fw-bold mb-3 text-success">Extraction Results</h6>
                                    <div class="mb-3">
                                        <small class="text-secondary code-font d-block mb-1">SUMMARY</small>
                                        <p id="summaryText" class="text-white mb-0"></p>
                                    </div>
                                    <div class="mb-3">
                                        <small class="text-secondary code-font d-block mb-1">COMMITMENTS</small>
                                        <div id="commitmentsContainer"></div>
                                    </div>
                                    <div class="mb-3">
                                        <small class="text-secondary code-font d-block mb-1">AMBIGUITIES</small>
                                        <div id="ambiguitiesContainer"></div>
                                    </div>
                                    <div class="p-3 rounded-3" style="background: var(--bg-main); border: 1px solid var(--border-subtle);">
                                        <small class="text-info code-font d-block mb-1">RECOMMENDED CLARIFICATION</small>
                                        <p id="clarificationText" class="fw-bold text-white mb-0"></p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Commitments Tab Pane -->
                        <div id="tab-commitments" style="display:none;">
                            <div class="main-panel">
                                <div class="d-flex justify-content-between align-items-center mb-4">
                                    <h5 class="fw-bold mb-0 text-white">Tracked Commitments</h5>
                                    <div class="btn-group">
                                        <button class="btn btn-sm btn-secondary-dark active" onclick="filterCommitments('all', this)">All</button>
                                        <button class="btn btn-sm btn-secondary-dark" onclick="filterCommitments('explicit', this)">Explicit</button>
                                        <button class="btn btn-sm btn-secondary-dark" onclick="filterCommitments('pending', this)">Pending</button>
                                        <button class="btn btn-sm btn-secondary-dark" onclick="filterCommitments('overdue', this)">Overdue</button>
                                        <button class="btn btn-sm btn-secondary-dark" onclick="filterCommitments('completed', this)">Completed</button>
                                    </div>
                                </div>
                                <div id="commitmentsList"></div>
                            </div>
                        </div>

                        <!-- History Tab Pane -->
                        <div id="tab-history" style="display:none;">
                            <div class="main-panel">
                                <h5 class="fw-bold mb-4 text-white">Conversation History</h5>
                                <div id="conversationsList"></div>
                            </div>
                        </div>
                    </div>

                    <!-- RIGHT RAIL (~280px / 4 columns) -->
                    <div class="col-lg-4 col-md-12 d-flex flex-column gap-4">
                        <!-- Section 1: Recent Commitments -->
                        <div class="main-panel">
                            <div class="d-flex align-items-center justify-content-between mb-3">
                                <h6 class="fw-bold text-white mb-0">Recent Commitments</h6>
                                <span class="badge badge-pending code-font" id="briefPendingCount">0 Pending</span>
                            </div>
                            <div id="recentCommitmentsList" class="d-flex flex-column gap-2 mb-2"></div>
                            <button class="btn btn-link text-decoration-none p-0 text-secondary small code-font" onclick="showTab('commitments'); loadCommitments();">
                                View All Commitments →
                            </button>
                        </div>

                        <!-- Section 2: Recent Activity -->
                        <div class="main-panel">
                            <div class="d-flex align-items-center justify-content-between mb-3">
                                <h6 class="fw-bold text-white mb-0">Recent Activity</h6>
                            </div>
                            <div id="recentConversationsList" class="d-flex flex-column gap-2 mb-2"></div>
                            <button class="btn btn-link text-decoration-none p-0 text-secondary small code-font" onclick="showTab('history'); loadConversations();">
                                View History →
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Status Update Modal -->
    <div class="modal fade" id="statusModal" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content main-panel text-white">
                <div class="modal-header border-bottom border-secondary border-opacity-10">
                    <h5 class="modal-title fw-bold">Update Commitment Status</h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <input type="hidden" id="modalCommId">
                    <div class="mb-3">
                        <label class="form-label small text-secondary">NEW STATUS:</label>
                        <select id="modalNewStatus" class="form-select">
                            <option value="completed">Completed</option>
                            <option value="renegotiated">Renegotiated</option>
                            <option value="cancelled">Cancelled</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label class="form-label small text-secondary">NEW DEADLINE (IF RENEGOTIATED):</label>
                        <input type="text" id="modalNewDeadline" class="form-control" placeholder="e.g. 2026-10-20">
                    </div>
                    <div class="mb-3">
                        <label class="form-label small text-secondary">REASON / NOTES:</label>
                        <textarea id="modalReason" class="form-control" rows="2"></textarea>
                    </div>
                </div>
                <div class="modal-footer border-top border-secondary border-opacity-10">
                    <button type="button" class="btn btn-secondary-dark" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-cta" onclick="submitStatusUpdate()">Confirm Change</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Loading Spinner Overlay -->
    <div class="spinner-overlay" id="spinnerOverlay">
        <div class="text-center">
            <div class="spinner-border mb-3" style="width: 2.5rem; height: 2.5rem; color: var(--accent-violet);"></div>
            <h6 class="fw-bold text-white code-font" id="spinnerMessage">Processing...</h6>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function refreshIcons() {
            if (window.lucide) {
                window.lucide.createIcons();
            }
        }

        function generateWaveformBars() {
            const leftContainer = document.getElementById('leftWaveform');
            const rightContainer = document.getElementById('rightWaveform');
            if (!leftContainer || !rightContainer) return;

            leftContainer.innerHTML = '';
            rightContainer.innerHTML = '';

            const heights = [10, 18, 14, 28, 36, 18, 30, 40, 24, 32, 16, 24, 14, 20, 10];

            heights.forEach(h => {
                const barLeft = document.createElement('div');
                barLeft.className = 'wave-bar';
                barLeft.style.height = h + 'px';
                leftContainer.appendChild(barLeft);

                const barRight = document.createElement('div');
                barRight.className = 'wave-bar';
                barRight.style.height = h + 'px';
                rightContainer.appendChild(barRight);
            });
        }

        function updateCharCount(textarea) {
            const len = textarea.value.length;
            document.getElementById('charCountLabel').innerText = `${len}/2000`;
        }

        function checkTranscriptInput() {
            const realVal = document.getElementById('aaiFinalTranscript').value.trim();
            const textVal = document.getElementById('textTranscript').value.trim();
            const processBtn = document.getElementById('btnProcessCommand');
            if (processBtn) {
                processBtn.disabled = !(realVal || textVal);
            }
        }

        function switchInputMode(mode) {
            ['text', 'audio', 'realtime'].forEach(m => {
                document.getElementById(`input-mode-${m}`).style.display = (m === mode) ? 'block' : 'none';
                const btn = document.getElementById(`mode-${m}-btn`);
                if (btn) {
                    if (m === mode) {
                        btn.className = 'btn btn-cta btn-sm px-3';
                    } else {
                        btn.className = 'btn btn-secondary-dark btn-sm px-3';
                    }
                }
            });
            checkTranscriptInput();
            refreshIcons();
        }

        function setEngineState(state) {
            const label = document.getElementById('engineStateLabel');
            if (!label) return;
            label.innerText = state.toUpperCase();
        }

        function showSpinner(msg) {
            document.getElementById('spinnerMessage').innerText = msg;
            document.getElementById('spinnerOverlay').style.display = 'flex';
        }
        function hideSpinner() {
            document.getElementById('spinnerOverlay').style.display = 'none';
        }

        function showTab(name) {
            ['workspace', 'commitments', 'history'].forEach(t => {
                const el = document.getElementById('tab-' + t);
                if (el) el.style.display = (t === name) ? 'block' : 'none';
            });
            document.querySelectorAll('.sidebar .nav-link-item').forEach(b => b.classList.remove('active'));
            if (event && event.currentTarget) {
                event.currentTarget.classList.add('active');
            }
            refreshIcons();
        }

        async function runTextAnalysis() {
            const context = document.getElementById('textContext').value.trim();
            const conversation = document.getElementById('textTranscript').value.trim();
            if (!conversation) return alert('Please enter a transcript.');

            setEngineState('ANALYZING');
            showSpinner('Extracting conversation intelligence...');
            try {
                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ context, conversation })
                });
                const data = await res.json();
                hideSpinner();
                setEngineState('COMPLETE');
                if (data.error) return alert('Error: ' + data.error);
                renderResults(data);
                updateBrief();
            } catch(e) {
                hideSpinner(); setEngineState('READY');
                alert('Failed: ' + e.message);
            }
        }

        async function transcribeAudio() {
            const fileInput = document.getElementById('audioFileInput');
            if (!fileInput.files.length) return alert('Select audio file.');
            const formData = new FormData();
            formData.append('audio', fileInput.files[0]);

            setEngineState('TRANSCRIBING');
            showSpinner('Transcribing audio with Sarvam Speech-to-Text...');
            try {
                const res = await fetch('/api/transcribe', { method: 'POST', body: formData });
                const data = await res.json();
                hideSpinner(); setEngineState('READY');
                if (data.error) return alert('Transcription Error: ' + data.error);
                document.getElementById('langMetadata').innerText = `Language Detected: ${data.language_code || 'Unknown'}`;
                document.getElementById('reviewedTranscript').value = data.transcript || '';
                document.getElementById('transcriptReviewArea').style.display = 'block';
            } catch(e) {
                hideSpinner(); setEngineState('READY');
                alert('Failed: ' + e.message);
            }
        }

        async function runAudioAnalysis() {
            const context = document.getElementById('audioContext').value.trim();
            const conversation = document.getElementById('reviewedTranscript').value.trim();
            if (!conversation) return alert('Transcript is empty.');

            setEngineState('ANALYZING');
            showSpinner('Analyzing transcript...');
            try {
                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ context, conversation })
                });
                const data = await res.json();
                hideSpinner(); setEngineState('COMPLETE');
                if (data.error) return alert('Error: ' + data.error);
                renderResults(data);
                updateBrief();
            } catch(e) {
                hideSpinner(); setEngineState('READY');
                alert('Failed: ' + e.message);
            }
        }

        /* --- ASSEMBLYAI REALTIME VOICE AGENT (V3 STREAMING + REAL AUDIO ANALYSER) --- */
        let aaiWebSocket = null;
        let aaiAudioContext = null;
        let aaiMediaStream = null;
        let aaiScriptProcessor = null;
        let aaiAnalyser = null;
        let aaiAnimFrame = null;

        function toggleVoiceAgent() {
            if (aaiWebSocket && aaiWebSocket.readyState === WebSocket.OPEN) {
                stopVoiceAgent();
            } else {
                startVoiceAgent();
            }
        }

        async function startVoiceAgent() {
            const topBadge = document.getElementById('topBarStatusBadge');
            const errorAlert = document.getElementById('aaiErrorAlert');
            const btnStart = document.getElementById('btnStartVoiceAgent');
            const btnStop = document.getElementById('btnStopVoiceAgent');
            const partialBox = document.getElementById('aaiPartialBox');
            const liveVoiceArea = document.getElementById('liveVoiceArea');
            const micBtn = document.getElementById('centralMicBtn');
            const statusTitle = document.getElementById('voiceAgentStatusTitle');
            const statusSubtitle = document.getElementById('voiceAgentStatusSubtitle');

            errorAlert.style.display = 'none';
            topBadge.className = 'badge badge-pending code-font';
            topBadge.innerText = 'CONNECTING...';
            btnStart.disabled = true;
            statusTitle.innerText = 'Connecting...';
            statusSubtitle.innerText = 'Establishing AssemblyAI streaming session';

            try {
                const tokenRes = await fetch('/api/assemblyai/token', { method: 'POST' });
                const tokenData = await tokenRes.json();

                if (!tokenRes.ok || tokenData.error) {
                    throw new Error(tokenData.error || 'Failed to obtain AssemblyAI streaming token');
                }

                const wsUrl = tokenData.websocket_url;
                const targetSampleRate = tokenData.sample_rate || 16000;

                try {
                    aaiMediaStream = await navigator.mediaDevices.getUserMedia({
                        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
                    });
                } catch (micErr) {
                    throw new Error('Microphone access denied or unavailable: ' + micErr.message);
                }

                aaiWebSocket = new WebSocket(wsUrl);

                aaiWebSocket.onopen = () => {
                    topBadge.className = 'badge badge-completed code-font';
                    topBadge.innerText = 'CONNECTED (LIVE)';
                    btnStart.style.display = 'none';
                    btnStop.style.display = 'inline-flex';
                    liveVoiceArea.style.display = 'block';
                    micBtn.classList.add('listening');

                    statusTitle.innerText = 'Listening to voice stream...';
                    statusSubtitle.innerText = 'Speak naturally into microphone';
                    partialBox.innerHTML = '<span class="text-info">Listening to voice stream...</span>';
                    setEngineState('LISTENING');

                    startPcm16AudioStreaming(aaiMediaStream, targetSampleRate);
                };

                aaiWebSocket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        handleAssemblyAIEvent(data);
                    } catch (err) {
                        console.error('Error parsing AssemblyAI WS message:', err);
                    }
                };

                aaiWebSocket.onerror = (wsErr) => {
                    console.error('AssemblyAI WS error:', wsErr);
                    errorAlert.innerText = 'WebSocket Connection Error with AssemblyAI Realtime API.';
                    errorAlert.style.display = 'block';
                    topBadge.className = 'badge badge-overdue code-font';
                    topBadge.innerText = 'ERROR';
                    setEngineState('ERROR');
                };

                aaiWebSocket.onclose = () => {
                    topBadge.className = 'badge badge-pending code-font';
                    topBadge.innerText = 'DISCONNECTED';
                    btnStart.disabled = false;
                    btnStart.style.display = 'inline-flex';
                    btnStop.style.display = 'none';
                    micBtn.classList.remove('listening');

                    statusTitle.innerText = 'Ready to listen';
                    statusSubtitle.innerText = 'Click the microphone to start voice streaming';
                    setEngineState('READY');
                    stopAudioCapture();
                };

            } catch (err) {
                console.error('Start Voice Agent failed:', err);
                errorAlert.innerText = err.message;
                errorAlert.style.display = 'block';
                topBadge.className = 'badge badge-overdue code-font';
                topBadge.innerText = 'ERROR';
                btnStart.disabled = false;
                btnStart.style.display = 'inline-flex';
                btnStop.style.display = 'none';
                micBtn.classList.remove('listening');

                statusTitle.innerText = 'Connection Error';
                statusSubtitle.innerText = err.message;
                setEngineState('READY');
                stopAudioCapture();
            }
        }

        function handleAssemblyAIEvent(data) {
            const msgType = data.message_type || data.type;
            const partialBox = document.getElementById('aaiPartialBox');
            const finalTextArea = document.getElementById('aaiFinalTranscript');

            if (msgType === 'Turn') {
                const text = (data.transcript || '').trim();
                const endOfTurn = Boolean(data.end_of_turn);

                if (!endOfTurn) {
                    if (text) {
                        partialBox.innerText = '💬 ' + text;
                    }
                } else {
                    partialBox.innerHTML = '<span class="text-info">Listening to voice stream...</span>';
                    if (text) {
                        const currentText = finalTextArea.value.trim();
                        finalTextArea.value = currentText ? currentText + ' ' + text : text;
                        checkTranscriptInput();
                    }
                }
            } else if (msgType === 'PartialTranscript') {
                if (data.text) {
                    partialBox.innerText = '💬 ' + data.text;
                }
            } else if (msgType === 'FinalTranscript') {
                partialBox.innerHTML = '<span class="text-info">Listening to voice stream...</span>';
                if (data.text) {
                    const currentText = finalTextArea.value.trim();
                    finalTextArea.value = currentText ? currentText + ' ' + data.text : data.text;
                    checkTranscriptInput();
                }
            } else if (msgType === 'Begin' || msgType === 'SessionBegins') {
                console.log('AssemblyAI Realtime Session Started:', data.session_id || data.id);
            } else if (msgType === 'Termination' || msgType === 'SessionTerminated') {
                console.log('AssemblyAI Realtime Session Terminated Cleanly.');
                if (aaiWebSocket) aaiWebSocket.close();
            } else if (msgType === 'Error' || msgType === 'SessionError') {
                const errorAlert = document.getElementById('aaiErrorAlert');
                errorAlert.innerText = 'AssemblyAI Error: ' + (data.error || data.message || 'Unknown streaming error');
                errorAlert.style.display = 'block';
            }
        }

        function resampleTo16k(inputBuffer, inputSampleRate) {
            if (inputSampleRate === 16000) return inputBuffer;
            const ratio = inputSampleRate / 16000;
            const newLength = Math.round(inputBuffer.length / ratio);
            const result = new Float32Array(newLength);
            for (let i = 0; i < newLength; i++) {
                const originIndex = i * ratio;
                const indexFloor = Math.floor(originIndex);
                const indexCeil = Math.min(inputBuffer.length - 1, Math.ceil(originIndex));
                const interpolationFraction = originIndex - indexFloor;
                result[i] = inputBuffer[indexFloor] * (1 - interpolationFraction) + inputBuffer[indexCeil] * interpolationFraction;
            }
            return result;
        }

        function startPcm16AudioStreaming(stream, targetSampleRate) {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            aaiAudioContext = new AudioContextClass();

            const source = aaiAudioContext.createMediaStreamSource(stream);

            // Web Audio API AnalyserNode for real audio frequencies
            aaiAnalyser = aaiAudioContext.createAnalyser();
            aaiAnalyser.fftSize = 64;
            source.connect(aaiAnalyser);

            aaiScriptProcessor = aaiAudioContext.createScriptProcessor(4096, 1, 1);

            aaiScriptProcessor.onaudioprocess = (e) => {
                if (!aaiWebSocket || aaiWebSocket.readyState !== WebSocket.OPEN) return;

                const rawInput = e.inputBuffer.getChannelData(0);
                const resampled = resampleTo16k(rawInput, e.inputBuffer.sampleRate);
                const pcm16Buffer = new Int16Array(resampled.length);
                for (let i = 0; i < resampled.length; i++) {
                    const s = Math.max(-1, Math.min(1, resampled[i]));
                    pcm16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }

                aaiWebSocket.send(pcm16Buffer.buffer);
            };

            source.connect(aaiScriptProcessor);
            aaiScriptProcessor.connect(aaiAudioContext.destination);

            visualizeRealMicrophoneAudio();
        }

        function visualizeRealMicrophoneAudio() {
            if (!aaiAnalyser) return;
            const dataArray = new Uint8Array(aaiAnalyser.frequencyBinCount);

            function renderFrame() {
                if (!aaiAnalyser || !aaiWebSocket || aaiWebSocket.readyState !== WebSocket.OPEN) {
                    resetWaveformHeights();
                    return;
                }
                aaiAnalyser.getByteFrequencyData(dataArray);

                const leftBars = document.querySelectorAll('#leftWaveform .wave-bar');
                const rightBars = document.querySelectorAll('#rightWaveform .wave-bar');

                const barCount = leftBars.length;
                for (let i = 0; i < barCount; i++) {
                    const val = dataArray[i % dataArray.length] || 0;
                    const h = Math.max(6, Math.min(48, Math.round(6 + (val / 255) * 42)));
                    if (leftBars[i]) {
                        leftBars[i].style.height = h + 'px';
                        leftBars[i].style.backgroundColor = val > 20 ? 'var(--accent-cyan)' : 'var(--text-muted)';
                        leftBars[i].style.opacity = val > 20 ? '0.95' : '0.3';
                    }
                    if (rightBars[i]) {
                        rightBars[i].style.height = h + 'px';
                        rightBars[i].style.backgroundColor = val > 20 ? 'var(--accent-cyan)' : 'var(--text-muted)';
                        rightBars[i].style.opacity = val > 20 ? '0.95' : '0.3';
                    }
                }

                aaiAnimFrame = requestAnimationFrame(renderFrame);
            }

            renderFrame();
        }

        function resetWaveformHeights() {
            if (aaiAnimFrame) cancelAnimationFrame(aaiAnimFrame);
            aaiAnimFrame = null;
            const leftBars = document.querySelectorAll('#leftWaveform .wave-bar');
            const rightBars = document.querySelectorAll('#rightWaveform .wave-bar');
            const heights = [10, 18, 14, 28, 36, 18, 30, 40, 24, 32, 16, 24, 14, 20, 10];

            leftBars.forEach((b, idx) => {
                b.style.height = (heights[idx] || 10) + 'px';
                b.style.backgroundColor = 'var(--text-muted)';
                b.style.opacity = '0.3';
            });
            rightBars.forEach((b, idx) => {
                b.style.height = (heights[idx] || 10) + 'px';
                b.style.backgroundColor = 'var(--text-muted)';
                b.style.opacity = '0.3';
            });
        }

        function stopVoiceAgent() {
            if (aaiWebSocket && aaiWebSocket.readyState === WebSocket.OPEN) {
                aaiWebSocket.send(JSON.stringify({ type: "Terminate" }));
            } else {
                stopAudioCapture();
            }
        }

        function stopAudioCapture() {
            resetWaveformHeights();
            if (aaiScriptProcessor) {
                aaiScriptProcessor.disconnect();
                aaiScriptProcessor = null;
            }
            if (aaiAnalyser) {
                aaiAnalyser.disconnect();
                aaiAnalyser = null;
            }
            if (aaiAudioContext) {
                aaiAudioContext.close().catch(() => {});
                aaiAudioContext = null;
            }
            if (aaiMediaStream) {
                aaiMediaStream.getTracks().forEach(track => track.stop());
                aaiMediaStream = null;
            }
        }

        function useRealtimeTranscriptForAnalysis() {
            const realtimeText = document.getElementById('aaiFinalTranscript').value.trim();
            if (!realtimeText) {
                alert('No finalized transcript captured yet. Please speak into the microphone first.');
                return;
            }
            switchInputMode('text');
            document.getElementById('textTranscript').value = realtimeText;
            document.getElementById('textContext').value = 'Realtime AssemblyAI Voice Session';
            updateCharCount(document.getElementById('textTranscript'));
            checkTranscriptInput();
            runTextAnalysis();
        }

        let currentPendingActionId = null;

        function formatProposedAction(action) {
            if (!action) return '';
            const actType = (action.action || '').replace('_', ' ').toUpperCase();
            let html = `<div class="d-flex flex-column gap-2">`;
            html += `<div><span class="text-secondary">ACTION INTENT:</span> <strong class="text-white ms-1">${actType}</strong></div>`;
            if (action.description) html += `<div><span class="text-secondary">DESCRIPTION:</span> <span class="text-white ms-1">${action.description}</span></div>`;
            if (action.responsible_person) html += `<div><span class="text-secondary">RESPONSIBLE:</span> <span class="text-white ms-1">${action.responsible_person}</span></div>`;
            if (action.deadline) html += `<div><span class="text-secondary">DEADLINE:</span> <span class="text-white ms-1">${action.deadline}</span></div>`;
            if (action.commitment_id) html += `<div><span class="text-secondary">TARGET COMMITMENT ID:</span> <code class="text-info ms-1">${action.commitment_id}</code></div>`;
            html += `</div>`;
            return html;
        }

        async function submitVoiceAction() {
            const transcript = document.getElementById('aaiFinalTranscript').value.trim() || document.getElementById('textTranscript').value.trim();
            if (!transcript) {
                alert('Please capture or type a transcript first.');
                return;
            }
            showSpinner('Extracting voice intent & action...');
            try {
                const res = await fetch('/api/voice-agent/submit', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ transcript, context: 'Realtime Voice Agent' })
                });
                const data = await res.json();
                hideSpinner();

                if (data.error) return alert('Error: ' + data.error);

                speakVoiceResponse(data.voice_response);

                const panel = document.getElementById('voiceActionPanel');
                const badge = document.getElementById('voiceActionBadge');
                const respText = document.getElementById('voiceResponseText');
                const gate = document.getElementById('voiceConfirmGate');
                const details = document.getElementById('proposedActionDetails');

                badge.innerText = (data.action_type || 'INTENT').toUpperCase();
                respText.innerText = data.voice_response || '';

                if (data.requires_confirmation && data.action_id) {
                    currentPendingActionId = data.action_id;
                    details.innerHTML = formatProposedAction(data.proposed_action);
                    gate.style.display = 'block';
                } else {
                    currentPendingActionId = null;
                    gate.style.display = 'none';
                }

                panel.style.display = 'block';

                if (data.status === 'executed' && data.commitments) {
                    loadCommitments();
                    updateBrief();
                }
            } catch(e) {
                hideSpinner();
                alert('Action Failed: ' + e.message);
            }
        }

        async function confirmVoiceAction() {
            if (!currentPendingActionId) return;
            showSpinner('Executing voice action...');
            try {
                const res = await fetch('/api/voice-agent/confirm', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ action_id: currentPendingActionId })
                });
                const data = await res.json();
                hideSpinner();

                if (data.error) return alert('Confirmation Error: ' + data.error);

                speakVoiceResponse(data.voice_response);

                document.getElementById('voiceConfirmGate').style.display = 'none';
                document.getElementById('voiceResponseText').innerText = '✓ ' + data.voice_response;
                currentPendingActionId = null;

                loadCommitments();
                loadConversations();
                updateBrief();
            } catch(e) {
                hideSpinner();
                alert('Execution Failed: ' + e.message);
            }
        }

        async function cancelVoiceAction() {
            if (!currentPendingActionId) return;
            try {
                const res = await fetch('/api/voice-agent/cancel', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ action_id: currentPendingActionId })
                });
                const data = await res.json();
                speakVoiceResponse(data.voice_response);
                document.getElementById('voiceConfirmGate').style.display = 'none';
                document.getElementById('voiceResponseText').innerText = '✕ ' + (data.voice_response || 'Action cancelled.');
                currentPendingActionId = null;
            } catch(e) {
                console.error(e);
            }
        }

        function speakVoiceResponse(text) {
            if ('speechSynthesis' in window && text) {
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 1.0;
                window.speechSynthesis.speak(utterance);
            }
        }

        function clearRealtimeTranscript() {
            document.getElementById('aaiFinalTranscript').value = '';
            document.getElementById('aaiPartialBox').innerHTML = '<span class="text-muted opacity-50">Listening to voice stream...</span>';
            document.getElementById('voiceActionPanel').style.display = 'none';
            checkTranscriptInput();
        }

        function renderResults(data) {
            document.getElementById('summaryText').innerText = data.summary || 'N/A';

            const commDiv = document.getElementById('commitmentsContainer');
            commDiv.innerHTML = '';
            (data.commitments || []).forEach(c => {
                const badge = getStatusBadge(c.status);
                const card = document.createElement('div');
                card.className = 'main-panel p-3 mb-2';
                card.innerHTML = `<div>${badge} <strong class="fs-6 text-white">${c.description}</strong></div>
                                  <div class="text-secondary small mt-1">Responsible: ${c.responsible_person || 'Unspecified'} | Deadline: ${c.deadline || 'None'}</div>`;
                commDiv.appendChild(card);
            });

            const ambDiv = document.getElementById('ambiguitiesContainer');
            ambDiv.innerHTML = '';
            (data.ambiguities || []).forEach(a => {
                const card = document.createElement('div');
                card.className = 'main-panel p-3 mb-2 border-warning';
                card.innerHTML = `<div class="fw-bold text-warning mb-1">⚠️ ${a.description}</div><div class="text-secondary small">Why it matters: ${a.why_it_matters}</div>`;
                ambDiv.appendChild(card);
            });

            document.getElementById('clarificationText').innerText = data.clarification_question ? `"${data.clarification_question}"` : 'N/A';
            document.getElementById('resultsContainer').style.display = 'block';
            document.getElementById('resultsContainer').scrollIntoView({ behavior: 'smooth' });
            refreshIcons();
        }

        function getStatusBadge(status) {
            status = (status || 'explicit').toLowerCase();
            let colorClass = 'badge-completed';
            if (status === 'overdue') colorClass = 'badge-overdue';
            else if (status === 'pending' || status === 'unclear' || status === 'explicit') colorClass = 'badge-pending';
            return `<span class="badge ${colorClass} badge-status me-2">${status.toUpperCase()}</span>`;
        }

        async function loadCommitments(filter = 'all') {
            try {
                const res = await fetch(`/api/commitments?status=${filter}`);
                const data = await res.json();
                const container = document.getElementById('commitmentsList');
                container.innerHTML = '';
                if (!data.length) { container.innerHTML = '<div class="text-secondary p-3">No commitments found.</div>'; return; }
                data.forEach(c => {
                    const card = document.createElement('div'); card.className = 'main-panel p-3 mb-3';
                    card.innerHTML = `
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                ${getStatusBadge(c.status)}
                                <strong class="fs-6 text-white">${c.description}</strong>
                                <div class="text-secondary small mt-1">Responsible: <b class="text-white">${c.responsible_person || 'Unspecified'}</b> | Deadline: <b class="text-white">${c.deadline || 'None'}</b></div>
                                ${c.evidence && c.evidence.length ? `<div class="p-2 rounded mt-2 small text-secondary" style="background:var(--bg-main); border-left:3px solid var(--accent-cyan);">Quote: "${c.evidence.join('", "')}"</div>` : ''}
                            </div>
                            <button class="btn btn-secondary-dark btn-sm code-font" onclick="openStatusModal('${c.id}')">Update</button>
                        </div>
                    `;
                    container.appendChild(card);
                });
                refreshIcons();
            } catch(e) { console.error(e); }
        }

        function filterCommitments(filter, btn) {
            document.querySelectorAll('#tab-commitments .btn-group button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadCommitments(filter);
        }

        function openStatusModal(id) {
            document.getElementById('modalCommId').value = id;
            document.getElementById('modalReason').value = '';
            document.getElementById('modalNewDeadline').value = '';
            new bootstrap.Modal(document.getElementById('statusModal')).show();
        }

        async function submitStatusUpdate() {
            const id = document.getElementById('modalCommId').value;
            const new_status = document.getElementById('modalNewStatus').value;
            const reason = document.getElementById('modalReason').value;
            const new_deadline = document.getElementById('modalNewDeadline').value;

            if (!confirm(`Confirm status change to ${new_status.toUpperCase()}?`)) return;

            showSpinner('Updating status...');
            try {
                const res = await fetch(`/api/commitments/${id}/status`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ new_status, reason, new_deadline: new_deadline || null })
                });
                const data = await res.json();
                hideSpinner();
                bootstrap.Modal.getInstance(document.getElementById('statusModal')).hide();
                if (data.error) return alert('Error: ' + data.error);
                loadCommitments();
                updateBrief();
            } catch(e) { hideSpinner(); alert('Failed: ' + e.message); }
        }

        async function loadConversations() {
            try {
                const res = await fetch('/api/conversations');
                const data = await res.json();
                const container = document.getElementById('conversationsList');
                container.innerHTML = '';
                if (!data.length) { container.innerHTML = '<div class="text-secondary p-3">No recorded conversations.</div>'; return; }
                data.forEach(c => {
                    const div = document.createElement('div'); div.className = 'main-panel p-3 mb-3';
                    div.innerHTML = `
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="badge bg-secondary code-font">ID: ${c.id}</span>
                            <small class="text-secondary">${new Date(c.created_at).toLocaleString()}</small>
                        </div>
                        <h6 class="fw-bold mb-1 text-white">${c.context || 'Conversation'}</h6>
                        <p class="text-secondary small mb-2">${c.summary || 'No summary available.'}</p>
                        <div class="p-2 rounded small text-break text-slate-300 code-font" style="background: var(--bg-main);"><strong>Transcript:</strong> ${c.transcript}</div>
                    `;
                    container.appendChild(div);
                });
                refreshIcons();
            } catch(e) { console.error(e); }
        }

        async function updateBrief() {
            try {
                const resComms = await fetch('/api/commitments?status=all');
                const comms = await resComms.json();

                const pendingCount = (comms || []).filter(c => ['pending', 'explicit', 'unclear'].includes((c.status||'').toLowerCase())).length;
                document.getElementById('briefPendingCount').innerText = `${pendingCount} Pending`;

                const recentCommContainer = document.getElementById('recentCommitmentsList');
                recentCommContainer.innerHTML = '';

                const validComms = (comms || []).filter(c => c && c.description).slice(0, 3);

                if (validComms.length > 0) {
                    validComms.forEach(c => {
                        const st = (c.status || 'pending').toUpperCase();
                        const stClass = (st === 'COMPLETED') ? 'badge-completed' : (st === 'OVERDUE' ? 'badge-overdue' : 'badge-pending');
                        const card = document.createElement('div');
                        card.className = 'p-2 rounded-3 border border-secondary border-opacity-10 d-flex align-items-center justify-content-between';
                        card.style.background = 'var(--bg-elevated)';
                        card.innerHTML = `
                            <div class="pe-2 overflow-hidden">
                                <h6 class="mb-0 text-white fw-medium text-truncate" style="font-size: 13px;">${c.description}</h6>
                                <small class="text-secondary d-block text-truncate" style="font-size: 11px;">${c.responsible_person || 'Unspecified'} • ${c.deadline || 'No deadline'}</small>
                            </div>
                            <span class="badge ${stClass} badge-status flex-shrink-0">${st}</span>
                        `;
                        recentCommContainer.appendChild(card);
                    });
                } else {
                    recentCommContainer.innerHTML = `<small class="text-secondary" style="font-size:12px;">No commitments tracked yet.</small>`;
                }

                const recentConvContainer = document.getElementById('recentConversationsList');
                recentConvContainer.innerHTML = '';

                const resConvs = await fetch('/api/conversations');
                const convs = await resConvs.json();

                const validConvs = (convs || []).filter(c => c && c.transcript).slice(0, 3);

                if (validConvs.length > 0) {
                    validConvs.forEach(c => {
                        const snippet = c.transcript ? `"${c.transcript.substring(0, 36)}..."` : '"Voice interaction"';
                        const card = document.createElement('div');
                        card.className = 'p-2 rounded-3 border border-secondary border-opacity-10 d-flex align-items-center justify-content-between';
                        card.style.background = 'var(--bg-elevated)';
                        card.innerHTML = `
                            <div class="pe-2 overflow-hidden">
                                <h6 class="mb-1 text-white fw-medium text-truncate code-font" style="font-size: 12px;">${snippet}</h6>
                                <small class="text-secondary d-block" style="font-size: 11px;">${new Date(c.created_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</small>
                            </div>
                            <i data-lucide="chevron-right" class="text-secondary flex-shrink-0" style="width: 14px; height: 14px;"></i>
                        `;
                        recentConvContainer.appendChild(card);
                    });
                } else {
                    recentConvContainer.innerHTML = `<small class="text-secondary" style="font-size:12px;">No recent activity.</small>`;
                }

                refreshIcons();

            } catch(e) { console.error(e); }
        }

        document.addEventListener('DOMContentLoaded', () => {
            generateWaveformBars();
            checkTranscriptInput();
            updateBrief();
            refreshIcons();
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)


@app.route("/api/assemblyai/token", methods=["POST", "GET"])
def api_assemblyai_token():
    try:
        token_info = create_temporary_token()
        return jsonify(token_info)
    except AssemblyAIConfigError as e:
        return jsonify({"error": str(e)}), 400
    except AssemblyAITokenError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error generating AssemblyAI token: {e}"}), 500


@app.route("/api/conversations", methods=["GET"])
def api_get_conversations():
    try:
        conversations = get_all_conversations()
        return jsonify(conversations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/conversations/<conv_id>", methods=["GET"])
def api_get_conversation_detail(conv_id):
    try:
        conv = get_conversation_by_id(conv_id)
        return jsonify(conv)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/commitments", methods=["GET"])
def api_get_commitments():
    status = request.args.get("status", "all")
    try:
        commitments = get_all_commitments(status_filter=status)
        return jsonify(commitments)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/commitments/<comm_id>/status", methods=["POST"])
def api_update_commitment_status(comm_id):
    data = request.get_json() or {}
    new_status = data.get("new_status")
    reason = data.get("reason", "")
    new_deadline = data.get("new_deadline")

    if not new_status:
        return jsonify({"error": "new_status is required"}), 400

    try:
        updated = update_commitment_status(
            commitment_id=comm_id,
            new_status=new_status,
            reason_or_notes=reason,
            new_deadline=new_deadline,
        )
        return jsonify(updated)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided in request."}), 400

    file = request.files["audio"]
    if not file.filename:
        return jsonify({"error": "Selected file has no filename."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        file.save(temp_file.name)
        temp_file.close()

        result = transcribe_audio(temp_file.name)
        return jsonify(result)
    except (
        AudioValidationError,
        TranscriptionAuthError,
        TranscriptionProviderError,
    ) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected transcription error: {e}"}), 500
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json() or {}
    context = data.get("context", "")
    conversation = data.get("conversation", "")

    if not conversation.strip():
        return jsonify({"error": "Conversation transcript cannot be empty."}), 400

    try:
        analysis = analyze_conversation(context, conversation)
        return jsonify(analysis)
    except (
        ConfigurationError,
        ProviderError,
        ResponseParsingError,
        SchemaValidationError,
    ) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected analysis error: {e}"}), 500


def get_client_session_id():
    sid = request.cookies.get("mean_session_id") or request.headers.get("X-Session-ID")
    if not sid:
        import uuid
        sid = f"sess_{uuid.uuid4().hex[:12]}"
    return sid


@app.route("/api/voice-agent/submit", methods=["POST"])
def api_voice_agent_submit():
    data = request.get_json() or {}
    transcript = data.get("transcript", "")
    context = data.get("context", "Voice Command")
    if not transcript.strip():
        return jsonify({"error": "Transcript cannot be empty."}), 400
    try:
        session_id = get_client_session_id()
        result = process_voice_transcript(transcript, context, session_id=session_id)
        resp = jsonify(result)
        resp.set_cookie("mean_session_id", session_id, max_age=86400, httponly=True, samesite="Lax")
        return resp
    except Exception as e:
        return jsonify({"error": f"Failed to process voice transcript: {e}"}), 500


@app.route("/api/voice-agent/confirm", methods=["POST"])
def api_voice_agent_confirm():
    data = request.get_json() or {}
    action_id = data.get("action_id")
    if not action_id:
        return jsonify({"error": "action_id is required."}), 400
    try:
        session_id = get_client_session_id()
        result = confirm_pending_action(action_id, session_id=session_id)
        resp = jsonify(result)
        resp.set_cookie("mean_session_id", session_id, max_age=86400, httponly=True, samesite="Lax")
        return resp
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to execute action: {e}"}), 500


@app.route("/api/voice-agent/cancel", methods=["POST"])
def api_voice_agent_cancel():
    data = request.get_json() or {}
    action_id = data.get("action_id")
    if not action_id:
        return jsonify({"error": "action_id is required."}), 400
    try:
        session_id = get_client_session_id()
        result = cancel_pending_action(action_id, session_id=session_id)
        resp = jsonify(result)
        resp.set_cookie("mean_session_id", session_id, max_age=86400, httponly=True, samesite="Lax")
        return resp
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    load_env()
    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        print("\n❌ ERROR: SARVAM_API_KEY environment variable is missing in .env")
        sys.exit(1)

    port = 5000
    url = f"http://127.0.0.1:{port}"
    print("\n==================================================")
    print("  🚀 MEAN AI Command Center Web Server Running!")
    print(f"  Open in browser: {url}")
    print("==================================================\n")

    webbrowser.open(url)
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
